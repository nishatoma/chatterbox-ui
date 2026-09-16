@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo.
  echo The Chatterbox UI virtual environment was not found here:
  echo %CD%\.venv
  echo.
  echo Follow the Windows installation steps in README.md,
  echo then run START_UI.bat again.
  echo.
  pause
  exit /b 1
)

echo Starting Chatterbox Voice Studio...
echo Keep this window open while using the interface.
echo.
".venv\Scripts\python.exe" app.py

if errorlevel 1 (
  echo.
  echo The interface stopped with an error. Copy the error text and send it to ChatGPT.
  pause
)
