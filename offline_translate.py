"""Fast English -> Chinese translation with a local Argos/CTranslate2 model.

Install ctranslate2 and sentencepiece, then explicitly run download_model.py.
Create one OfflineTranslator in the background worker and reuse it. No network
requests occur here. The model is distributed by Argos Open Technologies;
copyright/license information remains in the installed model package. The
official en_zh 1.9 package credits OPUS-MT (Jörg Tiedemann and Santhosh
Thottingal) and lists the original model's license as CC-BY 4.0.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models"


class OfflineTranslator:
    """Reusable CPU/int8 translator; initialize outside the Tk UI thread."""

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR, threads: int = 2):
        import ctranslate2
        import sentencepiece

        package = Path(model_dir).expanduser().resolve()
        if not (package / "metadata.json").is_file():
            package = package / "en_zh"
        if not (package / "sentencepiece.model").is_file() or not (package / "model/model.bin").is_file():
            raise FileNotFoundError(
                f"Offline English -> Chinese model not found: {package}\n"
                "Run python download_model.py once before starting offline translation."
            )
        metadata = json.loads((package / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("from_code") != "en" or metadata.get("to_code") != "zh":
            raise ValueError("This offline backend requires the English -> Chinese Argos model")
        if metadata.get("package_version") != "1.9":
            raise ValueError("This backend is validated with Argos English -> Chinese 1.9; run download_model.py.")
        self._target_prefix = metadata.get("target_prefix", "")
        self._tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(package / "sentencepiece.model"))
        self._translator = ctranslate2.Translator(
            str(package / "model"), device="cpu", compute_type="int8",
            inter_threads=1, intra_threads=max(1, min(threads, os.cpu_count() or 1)),
        )

    def _chunks(self, text: str) -> list[list[str]]:
        """Bound input length without silently dropping long OCR paragraphs."""
        chunks = []
        # A lightweight splitter avoids pulling in a second neural model/stanza.
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            tokens = self._tokenizer.encode(sentence, out_type=str)
            while tokens:
                end = min(128, len(tokens))
                if end < len(tokens):
                    # Prefer a SentencePiece word boundary in the last 32 tokens.
                    for candidate in range(end, max(1, end - 32), -1):
                        if tokens[candidate].startswith("▁"):
                            end = candidate
                            break
                chunks.append(tokens[:end])
                tokens = tokens[end:]
        return chunks

    def translate(self, text: str) -> str:
        if not text or not text.strip():
            return ""
        lines = text.splitlines()
        inputs = []
        line_counts = []
        for line in lines:
            chunks = self._chunks(line.strip()) if line.strip() else []
            inputs.extend(chunks)
            line_counts.append(len(chunks))
        if not inputs:
            return ""
        prefix = [[self._target_prefix]] * len(inputs) if self._target_prefix else None
        results = self._translator.translate_batch(
            inputs, target_prefix=prefix, beam_size=2, num_hypotheses=1,
            max_batch_size=16, batch_type="examples", replace_unknowns=True,
            length_penalty=0.2, max_input_length=0, max_decoding_length=256,
        )
        translated = []
        for index, result in enumerate(results):
            tokens = result.hypotheses[0]
            # Some short inputs can trigger private-use garbage in this model.
            # Retry only those rare outputs with a wider beam.
            if any("\ue000" <= char <= "\uf8ff" for token in tokens for char in token):
                tokens = self._translator.translate_batch(
                    [inputs[index]], target_prefix=[prefix[index]] if prefix else None,
                    beam_size=4, replace_unknowns=True, length_penalty=0.2,
                    max_input_length=0, max_decoding_length=256,
                )[0].hypotheses[0]
            if len(tokens) >= 256:
                raise RuntimeError("Translation reached its output limit; select a smaller screen region.")
            if any("\ue000" <= char <= "\uf8ff" for token in tokens for char in token):
                raise RuntimeError("The offline model returned unreadable text; try a larger text context.")
            if self._target_prefix and tokens and tokens[0] == self._target_prefix:
                tokens = tokens[1:]
            decoded = self._tokenizer.decode_pieces(tokens).replace("▁", " ").strip()
            translated.append(re.sub(r"\s+([，。！？；：、])", r"\1", decoded))
        output = []
        offset = 0
        for count in line_counts:
            output.append(" ".join(translated[offset:offset + count]))
            offset += count
        return "\n".join(output)
