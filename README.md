# 屏幕区域实时翻译

Windows 桌面上的英语 → 简体中文屏幕区域翻译工具。程序可持续识别用户框选区域中的英文，并通过本地离线模型或 Google 翻译输出中文。

## 功能

- 自由框选屏幕区域，适合固定位置的字幕或文字
- 本地 OCR 与离线英译中，截图和文本无需上传
- 可选 Google 联网翻译
- 仅在画面变化时识别，并缓存重复结果
- 保存选区坐标，支持窗口置顶

## 环境要求

- Windows
- 64 位 Python 3.12
- 首次安装依赖及下载离线模型时需要联网

## 快速开始

1. 下载或克隆本项目。
2. 双击 `start.bat`。
3. 首次运行会自动创建 `.venv`、安装依赖，并下载约 71 MB 的英中翻译模型。
4. 在程序中点击“① 框选区域”，选择需要识别的英文区域。
5. 点击“② 开始翻译”。

也可以在项目目录中手动运行：

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe download_model.py
.venv\Scripts\python.exe screen_translator.py
```

完整使用说明请见 [使用说明.txt](使用说明.txt)。

## 技术栈

- Tkinter：桌面界面
- MSS：屏幕截图
- RapidOCR / ONNX Runtime：文字识别
- CTranslate2 / SentencePiece / Argos 英中 1.9：离线翻译
- googletrans：可选联网翻译

## 隐私说明

使用“本地离线”模式时，截图、识别和翻译均在本机完成。程序不会保存截图、识别原文或翻译历史，只会在 `settings.json` 中保存框选坐标。

使用“Google 联网翻译”模式时，识别出的文字会发送给 Google 翻译；截图不会上传。

## 模型来源

离线翻译模型来自 Argos 官方模型索引，使用 `en_zh` 1.9。下载后，上游 README、许可文件及 `MODEL_SOURCE.json` 会保存在模型目录中。
