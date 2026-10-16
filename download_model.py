"""Explicit one-time download of the official Argos English -> Chinese model.

Run: python download_model.py [--model-dir PATH]
No model is downloaded merely by importing or starting the translator.

Source: https://github.com/argosopentech/argospm-index
The model and its training data belong to their respective copyright holders;
the upstream package's README/license files are preserved without alteration.
See models/en_zh/MODEL_SOURCE.json and the package README for provenance.
This script uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import zipfile

INDEX_URL = "https://raw.githubusercontent.com/argosopentech/argospm-index/main/index.json"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models"
MAX_DOWNLOAD = 512 * 1024 * 1024
MAX_EXTRACTED = 1024 * 1024 * 1024
PACKAGE_VERSION = "1.9"  # Validated with this script's SentencePiece/CT2 backend.


def _https_open(url: str):
    if urlparse(url).scheme != "https":
        raise ValueError("Only HTTPS model sources are accepted")
    response = urlopen(Request(url, headers={"User-Agent": "ScreenTranslator/1.0"}), timeout=60)
    if urlparse(response.geturl()).scheme != "https":
        response.close()
        raise ValueError("The model server redirected to a non-HTTPS URL")
    return response


def _fetch_index() -> list[dict]:
    for attempt in range(3):
        try:
            with _https_open(INDEX_URL) as response:
                data = response.read(4 * 1024 * 1024 + 1)
            if len(data) > 4 * 1024 * 1024:
                raise ValueError("Model index exceeds the expected size")
            index = json.loads(data)
            if not isinstance(index, list):
                raise ValueError("The official model index has an unexpected format")
            return index
        except (OSError, ValueError) as exc:
            if attempt == 2:
                raise RuntimeError(f"Cannot read the official Argos index: {exc}") from exc
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def _download(url: str, destination: Path) -> str:
    digest = hashlib.sha256()
    with _https_open(url) as response, destination.open("wb") as output:
        expected = int(response.headers.get("Content-Length", "0"))
        if expected > MAX_DOWNLOAD:
            raise ValueError("Model download exceeds the 512 MiB safety limit")
        received = 0
        next_report = 8 * 1024 * 1024
        while block := response.read(1024 * 1024):
            received += len(block)
            if received > MAX_DOWNLOAD:
                raise ValueError("Model download exceeds the 512 MiB safety limit")
            output.write(block)
            digest.update(block)
            if received >= next_report:
                total = f" / {expected / 1024**2:.1f}" if expected else ""
                print(f"Downloaded {received / 1024**2:.1f}{total} MiB", flush=True)
                next_report = received + 8 * 1024 * 1024
        if expected and received != expected:
            raise OSError(f"Incomplete model download: {received} of {expected} bytes")
    return digest.hexdigest()


def _safe_extract(archive: Path, destination: Path) -> None:
    """Extract only regular relative members inside this temporary directory."""
    root = destination.resolve()
    with zipfile.ZipFile(archive) as package:
        members = package.infolist()
        if len(members) > 10000 or sum(m.file_size for m in members) > MAX_EXTRACTED:
            raise ValueError("The model archive exceeds the extraction limits")
        for member in members:
            relative = PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "\\" in member.filename
                or ":" in member.filename
                or stat.S_ISLNK(mode)
                or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR))
            ):
                raise ValueError(f"Unsafe model archive entry: {member.filename!r}")
            target = root.joinpath(*relative.parts).resolve()
            if not target.is_relative_to(root):
                raise ValueError("Model archive entry escapes the destination")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with package.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)


def _validate_package(path: Path) -> dict:
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("from_code") != "en" or metadata.get("to_code") != "zh":
        raise ValueError("The downloaded model is not English -> Chinese")
    if metadata.get("package_version") != PACKAGE_VERSION:
        raise ValueError(f"Expected the tested Argos model version {PACKAGE_VERSION}")
    for relative in ("model/model.bin", "model/config.json", "sentencepiece.model"):
        if not (path / relative).is_file():
            raise ValueError(f"The downloaded model is missing {relative}")
    return metadata


def install_model(model_dir: Path) -> Path:
    model_dir = model_dir.expanduser().resolve()
    destination = model_dir / "en_zh"
    if destination.exists():
        _validate_package(destination)
        print(f"Model already installed: {destination}")
        return destination
    index = _fetch_index()
    candidates = [entry for entry in index if entry.get("from_code") == "en" and entry.get("to_code") == "zh" and entry.get("package_version") == PACKAGE_VERSION]
    if not candidates:
        raise RuntimeError(f"The official index no longer contains the tested English -> Chinese {PACKAGE_VERSION} model")
    entry = candidates[0]
    links = [url for url in entry.get("links", []) if isinstance(url, str) and urlparse(url).scheme == "https"]
    if not links:
        raise RuntimeError("The official model entry contains no HTTPS download link")
    print(f"Argos English -> Chinese {entry.get('package_version', '')}")
    print("The first download is about 68 MiB; later translation works offline.")
    model_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".argos-download-", dir=model_dir) as temporary:
        staging = Path(temporary)
        archive = staging / "model.argosmodel"
        errors = []
        checksum = ""
        selected_url = ""
        for attempt in range(3):
            selected_url = links[attempt % len(links)]
            try:
                checksum = _download(selected_url, archive)
                if not zipfile.is_zipfile(archive):
                    raise ValueError("The downloaded file is not an Argos ZIP model")
                break
            except (OSError, ValueError) as exc:
                errors.append(str(exc))
                if attempt == 2:
                    raise RuntimeError("Model download failed after 3 attempts: " + errors[-1]) from exc
                print(f"Download failed, retrying ({attempt + 1}/3): {exc}", flush=True)
                time.sleep(attempt + 1)
        extracted = staging / "extracted"
        extracted.mkdir()
        _safe_extract(archive, extracted)
        package_roots = [path.parent for path in extracted.rglob("metadata.json")]
        compatible = []
        for path in package_roots:
            try:
                metadata = _validate_package(path)
            except (OSError, ValueError):
                continue
            compatible.append((path, metadata))
        if len(compatible) != 1:
            raise RuntimeError("The archive does not contain exactly one supported translation model")
        package_root, metadata = compatible[0]
        provenance = {
            "project": "Argos Translate", "index_url": INDEX_URL,
            "download_url": selected_url, "package_version": metadata.get("package_version"),
            "download_sha256": checksum,
            "notice": "Upstream model/README/license files are preserved. SHA-256 records this download; it is not an upstream signature.",
        }
        (package_root / "MODEL_SOURCE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        # Rename only our staged package; never overwrite an existing installation.
        if destination.exists():
            raise FileExistsError(f"Destination appeared during installation: {destination}")
        package_root.rename(destination)
    print(f"Model installed: {destination}")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the official Argos English -> Chinese offline model")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Parent model folder (default: ./models beside this script)")
    args = parser.parse_args()
    try:
        install_model(args.model_dir)
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        print(f"Model installation failed: {exc}\nCheck your connection and run this command again.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
