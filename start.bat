@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Creating a Python 3.12 environment...
    py -3.12 -m venv .venv
    if errorlevel 1 goto error
)
echo Checking dependencies...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto error
echo Preparing the offline English-to-Chinese model...
".venv\Scripts\python.exe" download_model.py
if errorlevel 1 goto error
".venv\Scripts\python.exe" screen_translator.py
if errorlevel 1 goto error
exit /b 0
:error
echo.
echo Setup or startup failed. Read the error above.
echo This program needs 64-bit Python 3.12 and internet access for the first setup.
pause
exit /b 1
