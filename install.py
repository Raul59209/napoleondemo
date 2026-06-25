"""
install.py — Napoleon Installer
================================
Sets up a virtual environment and installs all dependencies.

Transcription now runs LOCALLY using Kyutai STT models (Hugging Face).
No API key required for STT.

The Scaleway API key is still required for LLM calls (DPI, CR, review).

Usage:
    Double-click install.py, or run: python install.py
"""

import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
VENV_DIR = BASE_DIR / "venv"

# Updated dependency list for Kyutai + Streamlit + LLM
REQUIREMENTS = [
    "streamlit",
    "openai",
    "python-dotenv",
    "reportlab",
    "kyutai-streaming-client",
    "numpy",
    "soundfile",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def banner(text: str):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def step(text: str):
    print(f"\n>>> {text}")

def run(cmd: list):
    print(f"    {' '.join(cmd)}")
    subprocess.check_call(cmd)


# ── Checks ────────────────────────────────────────────────────────────────────

def check_python_version():
    if sys.version_info < (3, 10):
        print("\n❌  Python 3.10 or newer is required.")
        print("    Download from https://www.python.org/downloads/")
        print("    Make sure to check 'Add Python to PATH' during install.")
        input("\nPress Enter to exit...")
        sys.exit(1)
    print(f"✓  Python {sys.version_info.major}.{sys.version_info.minor} detected.")


def check_env_file():
    """
    Warn if .env is missing or has no Scaleway API key.
    STT does NOT require any API key.
    """
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        print("\n⚠️  No .env file found.")
        print("    Kyutai STT works without any API key.")
        print("    For LLM (DPI/CR), Napoleon will ask for your Scaleway API key on first launch.")
        print("    Or create a .env file now with:")
        print("        SCW_API_KEY=scw-your-key-here")
        return

    content = env_path.read_text()
    if "SCW_API_KEY" not in content:
        print("\n⚠️  .env exists but SCW_API_KEY is not set.")
        print("    Kyutai STT works without any API key.")
        print("    Napoleon will ask for your Scaleway key on first launch.")
    else:
        print("✓  .env file found with SCW_API_KEY.")


# ── Venv & dependencies ───────────────────────────────────────────────────────

def create_venv():
    if VENV_DIR.exists():
        print("✓  Virtual environment already exists, skipping creation.")
    else:
        step("Creating virtual environment...")
        run([sys.executable, "-m", "venv", str(VENV_DIR)])
        print("✓  Virtual environment created.")


def get_python() -> str:
    if os.name == "nt":
        return str(VENV_DIR / "Scripts" / "python.exe")
    return str(VENV_DIR / "bin" / "python")


def install_requirements(python_exe: str):
    step("Upgrading pip...")
    run([python_exe, "-m", "pip", "install", "--upgrade", "pip", "-q"])

    step("Installing dependencies...")
    for package in REQUIREMENTS:
        print(f"    Installing {package}...")
        run([python_exe, "-m", "pip", "install", package, "-q"])

    print("\n✓  All dependencies installed.")


# ── run.bat ───────────────────────────────────────────────────────────────────

def create_run_bat():
    if os.name != "nt":
        return  # Windows only

    content = "@echo off\r\ncd /d \"%~dp0\"\r\nvenv\\Scripts\\python.exe launcher.py\r\n"
    run_bat = BASE_DIR / "run.bat"
    run_bat.write_text(content)
    print("✓  run.bat created.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    banner("NAPOLEON INSTALLER")
    print("  Medical audio pipeline — local Kyutai STT + Scaleway LLM.")
    print("  No GPU required. Kyutai models run locally via Hugging Face.")

    print("\n── System checks ──")
    check_python_version()
    check_env_file()

    print("\n── Environment setup ──")
    create_venv()
    python_exe = get_python()
    install_requirements(python_exe)

    if os.name == "nt":
        print("\n── Launcher ──")
        create_run_bat()

    banner("Installation complete!")
    if os.name == "nt":
        print("  → Double-click run.bat to launch Napoleon.")
    else:
        print("  → Run:  venv/bin/python launcher.py")
    print()
    input("Press Enter to exit...")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\n❌  A command failed: {e}")
        input("Press Enter to exit...")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInstallation cancelled.")
        sys.exit(0)
