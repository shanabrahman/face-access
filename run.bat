@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Face Access is not set up yet. Double-click setup.bat first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -c "import cv2" >nul 2>nul
if errorlevel 1 (
    echo OpenCV could not start. Install the Microsoft Visual C++ Redistributable (x64),
    echo restart Windows if requested, then try run.bat again.
    echo https://aka.ms/vs/17/release/vc_redist.x64.exe
    pause
    exit /b 1
)

echo Waiting for Face Access to start. Your browser will open automatically.
start "" powershell.exe -NoProfile -WindowStyle Hidden -Command "$url='http://127.0.0.1:8000'; for ($i=0; $i -lt 120; $i++) { try { $response=Invoke-WebRequest -Uri $url -TimeoutSec 2; if ($response.StatusCode -eq 200) { Start-Process $url; exit } } catch {}; Start-Sleep -Seconds 1 }"
".venv\Scripts\python.exe" -m uvicorn server:app --host 127.0.0.1 --port 8000
if errorlevel 1 (
    echo.
    echo Face Access stopped or could not start. Check the error above.
    pause
    exit /b 1
)
