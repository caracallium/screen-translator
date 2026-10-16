"""Windows 屏幕区域翻译：Python 3.12，英语 → 简体中文。

安装：py -3.12 -m pip install -r requirements.txt
运行：py -3.12 screen_translator.py
默认离线翻译；可选 Google 联网翻译（会发送识别出的文本）。
"""
from __future__ import annotations

import asyncio
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

# 改这里可以调整性能；轮询间隔不等于翻译延迟。
FORCE_OCR_SECONDS = 1.5  # 定期复查，避免差分漏掉很小的文字变化
NETWORK_TIMEOUT = 5.0
MIN_REQUEST_INTERVAL = 0.6
CACHE_SIZE = 256
MAX_TEXT_CHARS = 4000
SOURCE_LANGUAGES = {"英语": "en", "自动检测（中英文字）": "auto"}
TARGET_LANGUAGE = "zh-cn"
BACKENDS = {"本地离线（推荐）": "offline", "Google 联网翻译": "google"}
MODEL_DIR = Path(__file__).resolve().parent / "models"
SETTINGS_FILE = Path(__file__).resolve().parent / "settings.json"


def enable_dpi_awareness():
    """必须在创建 Tk 窗口前调用，让鼠标/截图使用同一物理像素坐标。"""
    if os.name == "nt":
        try:
            if ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
                return
        except (AttributeError, OSError):
            pass
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()


@dataclass(frozen=True)
class Region:
    left: int
    top: int
    width: int
    height: int

    def as_dict(self):
        return dict(left=self.left, top=self.top, width=self.width, height=self.height)


@dataclass(frozen=True)
class Session:
    number: int
    region: Region
    interval: float
    source: str
    backend: str = "offline"


@dataclass(frozen=True)
class Job:
    session: Session
    revision: int
    text: str
    captured_at: float
    ocr_ms: float


def overlaps(a: Region, b: Region) -> bool:
    return (a.left < b.left + b.width and a.left + a.width > b.left
            and a.top < b.top + b.height and a.top + a.height > b.top)


def normalize_text(text: str) -> str:
    return "\n".join(" ".join(line.split()) for line in text.splitlines() if line.strip())


class Pipeline:
    """线程之间只共享受锁保护的状态，不从工作线程操作 Tk。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.closed = threading.Event()
        self.generation = 0
        self.revision = 0
        self.session = None
        self.job = None
        self.translation = ""
        self.ocr_status = "尚未开始"
        self.translate_status = ""
        self.fatal_error = None

    def start(self, region, interval=0.3, source="en", backend="offline"):
        with self.lock:
            if self.fatal_error:
                self.ocr_status = self.fatal_error
                return None
            self.generation += 1
            self.session = Session(self.generation, region, interval, source, backend)
            self.job = None
            self.translation = ""
            self.ocr_status = "正在准备 OCR…"
            self.translate_status = ""
            return self.session

    def stop(self, reason="已暂停"):
        with self.lock:
            self.generation += 1
            self.session = None
            self.job = None
            self.translation = ""
            self.ocr_status = self.fatal_error or reason
            self.translate_status = ""

    def fail(self, message):
        with self.lock:
            self.fatal_error = message
        self.stop(message)

    def snapshot(self):
        with self.lock:
            return (self.session, self.job, self.translation,
                    self.ocr_status, self.translate_status)

    def is_current(self, job):
        with self.lock:
            return self.session == job.session and self.job == job

    def publish_ocr(self, session, text, captured_at, ocr_ms):
        text = normalize_text(text)
        with self.lock:
            if self.session != session:
                return None
            self.ocr_status = f"OCR {ocr_ms:.0f} ms"
            if self.job is not None and self.job.text == text:
                return self.job
            self.revision += 1  # A → B → A 也必须生成三个不同的版本
            self.job = Job(session, self.revision, text, captured_at, ocr_ms)
            self.translation = ""
            self.translate_status = "等待翻译…" if text else "区域内暂无文字"
            return self.job

    def set_ocr_error(self, session, message):
        with self.lock:
            if self.session == session:
                self.ocr_status = message
                # 截图失败后，不再展示/提交上一次内容的翻译。
                self.job = None
                self.translation = ""
                self.translate_status = ""

    def publish_translation(self, job, text, status):
        with self.lock:
            if self.session != job.session or self.job != job:
                return False
            self.translation = text
            self.translate_status = status
            return True


def image_changed(previous, current):
    import cv2
    import numpy as np
    if previous is None or previous.shape != current.shape:
        return True
    delta = cv2.absdiff(previous, current)
    return np.count_nonzero(delta > 18) >= max(8, current.size * 0.001)


def ocr_worker(pipe: Pipeline):
    try:
        import cv2
        import mss
        import numpy as np
        from rapidocr_onnxruntime import RapidOCR

        # 模型只初始化一次；限制 CPU 线程，避免后台过度占用。
        engine = RapidOCR(intra_op_num_threads=min(4, os.cpu_count() or 1),
                          inter_op_num_threads=1, det_limit_side_len=960,
                          det_limit_type="max", text_score=0.55)
        session_number = None
        previous = None
        last_ocr = 0.0
        with mss.mss() as capture:  # MSS 在使用它的线程里创建
            while not pipe.closed.is_set():
                session = pipe.snapshot()[0]
                if session is None:
                    pipe.closed.wait(0.1)
                    continue
                if session.number != session_number:
                    session_number = session.number
                    previous = None
                    last_ocr = 0.0
                started = time.monotonic()
                try:
                    frame = np.asarray(capture.grab(session.region.as_dict()))[:, :, :3].copy()
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    scale = min(1.0, 640 / max(gray.shape))
                    if scale < 1:
                        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                                          interpolation=cv2.INTER_AREA)
                    if (image_changed(previous, gray)
                            or started - last_ocr >= FORCE_OCR_SECONDS):
                        tick = time.monotonic()
                        result, _ = engine(frame, use_cls=False)  # 正向屏幕文字无需方向分类
                        text = "\n".join(str(row[1]) for row in (result or []))
                        pipe.publish_ocr(session, text, started, (time.monotonic() - tick) * 1000)
                        previous = gray
                        last_ocr = time.monotonic()
                except Exception as exc:
                    previous = None
                    pipe.set_ocr_error(session, f"截图/识别失败：{type(exc).__name__}；1 秒后重试")
                    pipe.closed.wait(1.0)
                # 不积压截图：本次识别完成后直接获取最新画面。
                pipe.closed.wait(max(0.02, session.interval - (time.monotonic() - started)))
    except Exception as exc:
        pipe.fail(f"OCR 初始化失败：{type(exc).__name__}: {str(exc)[:160]}。请重启程序。")


async def translate_current(pipe, client, job):
    """新文字到来时取消旧请求；总超时不会因连续收到小块数据而延长。"""
    cancel_stale = getattr(client, "cancel_stale", True)
    operation = client.translate(job.text, src=job.session.source, dest=TARGET_LANGUAGE)
    # CPU 推理不能靠取消协程真正停止；必须等本次结束，避免堆积推理线程。
    task = asyncio.create_task(asyncio.wait_for(operation, NETWORK_TIMEOUT)
                               if cancel_stale else operation)
    try:
        while not task.done():
            await asyncio.wait({task}, timeout=0.05)
            if pipe.closed.is_set() or (cancel_stale and not pipe.is_current(job)):
                return None
        return task.result()
    finally:
        if not task.done():
            task.cancel()
        # 取走异常；取消旧请求不应留下未处理任务。
        await asyncio.gather(task, return_exceptions=True)


async def translation_loop(pipe, client):
    cache = OrderedDict()
    done = None
    failed = None
    attempts = 0
    retry_at = 0.0
    next_request_at = 0.0
    while not pipe.closed.is_set():
        job = pipe.snapshot()[1]
        if job is None or not job.text or job == done:
            await asyncio.sleep(0.05)
            continue
        if len(job.text) > MAX_TEXT_CHARS:
            pipe.publish_translation(job, "", f"文字超过 {MAX_TEXT_CHARS} 字符，请缩小识别区域。")
            done = job
            continue
        key = (job.session.backend, job.session.source, TARGET_LANGUAGE, job.text)
        if key in cache:
            cache.move_to_end(key)
            total = (time.monotonic() - job.captured_at) * 1000
            pipe.publish_translation(job, cache[key], f"缓存命中 · 本次处理 {total:.0f} ms")
            done = job
            continue
        if failed != job:
            attempts, retry_at, failed = 0, 0.0, job
        network_due = next_request_at if job.session.backend == "google" else 0.0
        if time.monotonic() < max(retry_at, network_due):
            await asyncio.sleep(0.05)
            continue
        pipe.publish_translation(job, "", "正在翻译…")
        began = time.monotonic()
        if job.session.backend == "google":
            next_request_at = began + MIN_REQUEST_INTERVAL
        try:
            active_client = client.client_for(job.session.backend) if hasattr(client, "client_for") else client
            result = await translate_current(pipe, active_client, job)
            if result is None:
                continue
            translated = result.text.strip()
            if not translated:
                raise ValueError("empty translation")
            cache[key] = translated
            cache.move_to_end(key)
            if len(cache) > CACHE_SIZE:
                cache.popitem(last=False)
            elapsed = (time.monotonic() - began) * 1000
            total = (time.monotonic() - job.captured_at) * 1000
            pipe.publish_translation(job, translated,
                                     f"翻译 {elapsed:.0f} ms · 本次处理 {total:.0f} ms")
            done = job
        except Exception as exc:
            attempts += 1
            delay = min(30, 2 ** min(attempts, 5))
            retry_at = time.monotonic() + delay
            hint = (str(exc)[:160] if job.session.backend == "offline"
                    else "请检查 Google 翻译连接，或改用本地离线模式")
            pipe.publish_translation(job, "", f"翻译失败（{type(exc).__name__}），"
                                     f"{delay} 秒后重试；{hint}。")


class OfflineClient:
    cancel_stale = False

    def __init__(self):
        from offline_translate import OfflineTranslator
        self.engine = OfflineTranslator(MODEL_DIR)

    async def translate(self, text, src="en", dest="zh-cn"):
        from types import SimpleNamespace
        if src != "en" or dest != "zh-cn":
            raise ValueError("离线模型仅支持英语 → 简体中文，请把原文语言设为英语")
        translated = await asyncio.to_thread(self.engine.translate, text)
        return SimpleNamespace(text=translated)


class TranslationClients:
    def __init__(self):
        self.clients = {}

    def client_for(self, backend):
        if backend not in self.clients:
            if backend == "offline":
                self.clients[backend] = OfflineClient()
            else:
                import httpx
                from googletrans import Translator
                self.clients[backend] = Translator(service_urls=["translate.googleapis.com"],
                                                   timeout=httpx.Timeout(4.0, connect=2.0),
                                                   raise_exception=True)
        return self.clients[backend]

    async def close(self):
        if "google" in self.clients:
            await self.clients["google"].client.aclose()


def translation_worker(pipe):
    async def run():
        client = TranslationClients()
        try:
            await translation_loop(pipe, client)
        finally:
            await client.close()
    try:
        asyncio.run(run())
    except Exception as exc:
        pipe.fail(f"翻译组件初始化失败：{type(exc).__name__}: {str(exc)[:160]}。请重启程序。")


def native_window(window):
    user32 = ctypes.windll.user32
    user32.GetParent.argtypes = [wintypes.HWND]
    user32.GetParent.restype = wintypes.HWND
    return user32.GetParent(window.winfo_id()) or window.winfo_id()


def window_region(window):
    if os.name == "nt":
        rect = wintypes.RECT()
        get_rect = ctypes.windll.user32.GetWindowRect
        get_rect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        if get_rect(native_window(window), ctypes.byref(rect)):
            return Region(rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top)
    return Region(window.winfo_rootx(), window.winfo_rooty(),
                  window.winfo_width(), window.winfo_height())


def monitor_layout():
    import mss
    with mss.mss() as capture:
        return [Region(**monitor).as_dict() for monitor in capture.monitors[1:]]


def read_saved_region():
    """仅保存像素坐标；显示器布局变化后让用户重新框选。"""
    try:
        saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        if saved.get("monitors") != monitor_layout():
            return None
        region = Region(**saved["region"])
        if any(type(v) is not int for v in region.as_dict().values()):
            return None
        if region.width < 20 or region.height < 20:
            return None
        return region
    except Exception:
        # 坐标文件损坏或显示器枚举失败不应阻止程序启动。
        return None


def save_region(region):
    saved = {"region": region.as_dict(), "monitors": monitor_layout()}
    temporary = SETTINGS_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(SETTINGS_FILE)


class RegionSelector:
    def __init__(self, root, callback):
        import mss
        self.root, self.callback = root, callback
        with mss.mss() as capture:
            self.bounds = Region(**capture.monitors[0])
        self.start = None
        self.rectangle = None
        self.window = tk.Toplevel(root)
        self.window.overrideredirect(True)
        self.window.attributes("-topmost", True)
        self.window.attributes("-alpha", 0.30)
        self.window.geometry(f"{self.bounds.width}x{self.bounds.height}+0+0")
        self.canvas = tk.Canvas(self.window, bg="black", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.window.update_idletasks()
        if os.name == "nt":
            move = ctypes.windll.user32.SetWindowPos
            move.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                             ctypes.c_int, ctypes.c_int, wintypes.UINT]
            move(native_window(self.window), wintypes.HWND(-1), self.bounds.left,
                 self.bounds.top, self.bounds.width, self.bounds.height, 0x0040)
        else:
            self.window.geometry(f"+{self.bounds.left}+{self.bounds.top}")
        self.canvas.bind("<ButtonPress-1>", self.press)
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<ButtonRelease-1>", self.release)
        self.window.bind("<Escape>", lambda _e: self.finish(None))
        self.window.bind("<Button-3>", lambda _e: self.finish(None))
        self.window.grab_set()
        self.window.focus_force()

    def point(self, event):
        x, y = event.x_root, event.y_root
        if os.name == "nt":
            cursor = wintypes.POINT()
            if ctypes.windll.user32.GetCursorPos(ctypes.byref(cursor)):
                x, y = cursor.x, cursor.y
        return (max(self.bounds.left, min(x, self.bounds.left + self.bounds.width)),
                max(self.bounds.top, min(y, self.bounds.top + self.bounds.height)))

    def press(self, event):
        self.start = self.point(event)
        if self.rectangle:
            self.canvas.delete(self.rectangle)
        x, y = self.start[0] - self.bounds.left, self.start[1] - self.bounds.top
        self.rectangle = self.canvas.create_rectangle(x, y, x, y, outline="#00ff88", width=4)

    def drag(self, event):
        if self.start:
            x, y = self.point(event)
            self.canvas.coords(self.rectangle, self.start[0] - self.bounds.left,
                               self.start[1] - self.bounds.top,
                               x - self.bounds.left, y - self.bounds.top)

    def release(self, event):
        if not self.start:
            return
        x, y = self.point(event)
        left, right = sorted((self.start[0], x))
        top, bottom = sorted((self.start[1], y))
        if right - left < 20 or bottom - top < 20:
            self.finish(None)
        else:
            self.finish(Region(left, top, right - left, bottom - top))

    def finish(self, region):
        self.window.grab_release()
        self.window.destroy()
        self.callback(region)


class App:
    def __init__(self, root, start_workers=True):
        self.root = root
        self.pipe = Pipeline()
        self.region = read_saved_region()
        self.selector = None
        self.last_display = None
        self.threads = []
        root.title("屏幕区域翻译 · 英语 → 简体中文")
        root.geometry("730x540")
        root.minsize(560, 420)
        root.attributes("-topmost", True)
        root.protocol("WM_DELETE_WINDOW", self.close)
        style = ttk.Style(root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        body = ttk.Frame(root, padding=14)
        body.pack(fill="both", expand=True)
        controls = ttk.Frame(body)
        controls.pack(fill="x")
        ttk.Button(controls, text="① 框选区域", command=self.select_region).pack(side="left")
        self.run_button = ttk.Button(controls, text="② 开始翻译", command=self.toggle)
        self.run_button.pack(side="left", padx=8)
        self.topmost = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="窗口置顶", variable=self.topmost,
                        command=lambda: root.attributes("-topmost", self.topmost.get())).pack(side="right")
        options = ttk.Frame(body)
        options.pack(fill="x", pady=(10, 4))
        ttk.Label(options, text="原文语言").pack(side="left")
        self.language = ttk.Combobox(options, values=list(SOURCE_LANGUAGES), state="readonly", width=21)
        self.language.set("英语")
        self.language.pack(side="left", padx=(6, 16))
        ttk.Label(options, text="检测间隔（秒）").pack(side="left")
        self.interval = ttk.Combobox(options, values=["0.2", "0.3", "0.5", "1.0"],
                                     state="readonly", width=5)
        self.interval.set("0.3")
        self.interval.pack(side="left", padx=6)
        mode_row = ttk.Frame(body)
        mode_row.pack(fill="x", pady=(4, 2))
        ttk.Label(mode_row, text="翻译方式").pack(side="left")
        self.backend = ttk.Combobox(mode_row, values=list(BACKENDS), state="readonly", width=23)
        self.backend.set("本地离线（推荐）")
        self.backend.pack(side="left", padx=6)
        self.region_label = ttk.Label(body, text="先框选英文区域；拖动本窗口，将译文放在你想看的位置。")
        self.region_label.pack(anchor="w", pady=6)
        if self.region:
            r = self.region
            self.region_label.config(text=f"已记住区域：({r.left}, {r.top}) · {r.width} × {r.height} 像素")
            self.pipe.stop("已载入上次选区。将译文窗口移到选区外，点击开始。")
        ttk.Label(body, text="识别到的原文").pack(anchor="w")
        self.original = ScrolledText(body, height=4, wrap="word", font=("Segoe UI", 11), state="disabled")
        self.original.pack(fill="x", pady=(4, 8))
        ttk.Label(body, text="简体中文译文").pack(anchor="w")
        self.translated = ScrolledText(body, height=7, wrap="word", font=("Microsoft YaHei UI", 15),
                                       state="disabled", bg="#f2f7fb", relief="flat")
        self.translated.pack(fill="both", expand=True, pady=(4, 8))
        self.status = ttk.Label(body, text="就绪", wraplength=650)
        self.status.pack(anchor="w")
        ttk.Label(body, text="离线模式在本机处理；Google 模式会发送原文。译文窗口请放在选区外。",
                  foreground="#666666").pack(anchor="w", pady=(6, 0))
        root.bind("<Control-space>", lambda _e: self.toggle())
        if start_workers:
            for target in (ocr_worker, translation_worker):
                worker = threading.Thread(target=target, args=(self.pipe,), daemon=True)
                worker.start()
                self.threads.append(worker)
        self.poll_id = root.after(80, self.poll)

    def select_region(self):
        self.pipe.stop("按住鼠标左键框选；Esc 或右键取消")
        self.root.withdraw()
        self.root.after(150, self.open_selector)

    def open_selector(self):
        try:
            self.selector = RegionSelector(self.root, self.selected)
        except Exception as exc:
            if self.selector and self.selector.window.winfo_exists():
                self.selector.window.destroy()
            self.root.deiconify()
            self.pipe.stop(f"无法打开选择层：{type(exc).__name__}: {str(exc)[:120]}")

    def selected(self, region):
        self.selector = None
        if region:
            self.region = region
            self.region_label.config(text=f"区域：({region.left}, {region.top}) · "
                                          f"{region.width} × {region.height} 像素")
        self.root.deiconify()
        self.root.lift()
        status = "已取消框选"
        if region:
            try:
                save_region(region)
                status = "区域已记住。将译文窗口移到选区外，再点击开始。"
            except Exception:
                status = "区域已选好，但无法保存坐标。移开译文窗口后点击开始。"
        self.pipe.stop(status)

    def toggle(self):
        if self.pipe.snapshot()[0]:
            self.pipe.stop()
            return
        if self.region is None:
            self.select_region()
            return
        if overlaps(window_region(self.root), self.region):
            self.pipe.stop("请把译文窗口移到识别区域外，再点击开始。")
            return
        source = SOURCE_LANGUAGES[self.language.get()]
        backend = BACKENDS[self.backend.get()]
        if backend == "offline" and source != "en":
            self.pipe.stop("离线模型仅支持英译中，请将原文语言设为英语。")
            return
        self.pipe.start(self.region, float(self.interval.get()), source, backend)

    @staticmethod
    def set_text(widget, value):
        widget.config(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.config(state="disabled")

    def poll(self):
        session, job, translated, ocr_status, translation_status = self.pipe.snapshot()
        if session and self.root.state() == "normal" and overlaps(window_region(self.root), session.region):
            self.pipe.stop("译文窗口进入了识别区域，已暂停。移开后点击开始。")
            session, job, translated, ocr_status, translation_status = self.pipe.snapshot()
        display = (job.text if job else "", translated)
        if display != self.last_display:
            self.set_text(self.original, display[0])
            self.set_text(self.translated, display[1])
            self.last_display = display
        self.status.config(text=" · ".join(s for s in (ocr_status, translation_status) if s))
        self.run_button.config(text="暂停翻译" if session else "② 开始翻译")
        for combo in (self.interval, self.language, self.backend):
            combo.config(state="disabled" if session else "readonly")
        self.poll_id = self.root.after(80, self.poll)

    def close(self):
        self.pipe.stop()
        self.pipe.closed.set()
        self.root.after_cancel(self.poll_id)
        self.root.destroy()


def main():
    import argparse
    global MODEL_DIR
    parser = argparse.ArgumentParser(description="框选屏幕区域，实时英译中；默认离线。")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR, help="离线模型目录")
    args = parser.parse_args()
    MODEL_DIR = args.model_dir.resolve()
    enable_dpi_awareness()
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
