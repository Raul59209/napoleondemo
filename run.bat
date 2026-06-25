@echo off
cd /d "%~dp0"

echo Checking virtual environment...
if not exist "venv\Scripts\python.exe" (
    echo ❌ Virtual environment not found.
    echo Run install.py again.
    pause
    exit /b
)

echo Launching Napoleon...
venv\Scripts\python.exe launcher.py
pause
