"""
launcher.py — Napoleon Launcher
================================
Starts the Streamlit app and opens it in the browser.
No GPU or CUDA setup needed — all inference runs on Scaleway.

Usage:
    python launcher.py
    (or via run.bat on Windows)
"""

import subprocess
import webbrowser
import time
import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent


def main():
    app_path = BASE_DIR / "app_demo.py"

    if not app_path.exists():
        print(f"❌  Could not find app_demo.py at {app_path}")
        input("Press Enter to exit...")
        sys.exit(1)

    print("Starting Napoleon...")

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ]

    subprocess.Popen(cmd, env=os.environ.copy())

    time.sleep(4)

    webbrowser.open("http://localhost:8501")


if __name__ == "__main__":
    main()