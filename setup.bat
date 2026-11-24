@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if not errorlevel 1 (
    set "PYTHON=py -3"
) else (
    set "PYTHON=python"
)

%PYTHON% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if errorlevel 1 (
    echo Python 3.10 or newer was not found.
    echo Install Python and make sure the Python launcher or python command is available.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    %PYTHON% -m venv .venv
    if errorlevel 1 goto error
)

".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto error

echo.
echo Starting Face Access. Keep this window open while using the app.
echo Open http://127.0.0.1:8000 in your browser.
".venv\Scripts\python.exe" -m uvicorn server:app --host 127.0.0.1 --port 8000
if errorlevel 1 goto error
exit /b 0

:error
echo.
echo Setup or startup failed. Check the error above, then try again.
pause
exit /b 1
