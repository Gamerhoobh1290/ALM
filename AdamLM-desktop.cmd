@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo AdamLM virtual environment was not found. Follow the README setup steps first.
  pause
  exit /b 1
)
rem python.exe, not pythonw.exe: the console window stays visible so the app can
rem always be stopped from it (Ctrl+C, or close the window) without Task Manager.
start "AdamLM Desktop Fallback" ".venv\Scripts\python.exe" -u -m adamlm.gui
