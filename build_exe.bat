@echo off
cd /d "%~dp0"

echo ============================================================
echo   NAPOLEON — PyInstaller Build
echo ============================================================
echo.

pyinstaller ^
  --noconfirm ^
  --onedir ^
  --name napoleon ^
  --collect-all streamlit ^
  --collect-all openai ^
  --hidden-import=dotenv ^
  --hidden-import=reportlab ^
  --add-data "app_demo.py;." ^
  --add-data "prompts.py;." ^
  launcher.py

echo.
echo ============================================================
echo   Build complete — dist\napoleon\
echo   Copy your .env file into dist\napoleon\ before sharing.
echo ============================================================
echo.
pause