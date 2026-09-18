@echo off
setlocal
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" -u -m adamlm.bpe_train --auto-resume %*
set "RESULT=%ERRORLEVEL%"
echo.
echo Training exited with code %RESULT%.
pause
exit /b %RESULT%
