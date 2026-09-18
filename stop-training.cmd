@echo off
setlocal
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" -m adamlm.control stop %*
set "RESULT=%ERRORLEVEL%"
pause
exit /b %RESULT%
