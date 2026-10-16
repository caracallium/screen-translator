@echo off
setlocal
cd /d "%~dp0"
if not exist "..\..\work\translator-venv\Scripts\python.exe" goto setup
if not exist "..\..\work\offline-model\en_zh\model\model.bin" goto setup
"..\..\work\translator-venv\Scripts\python.exe" screen_translator.py --model-dir "..\..\work\offline-model"
if errorlevel 1 pause
exit /b
:setup
call start.bat
