@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo AdamLM virtual environment was not found. Follow the README setup steps first.
  pause
  exit /b 1
)
start "AdamLM Dashboard" "%~dp0.venv\Scripts\python.exe" -u -m adamlm.web
timeout /t 2 /nobreak >nul
start "" "http://127.0.0.1:8765/"
