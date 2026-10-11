"""
Chaos Bot PC - Graphical User Interface Entry Point.
Run this file to start the modern Fluent UI client:
    python run_gui.py
"""

import sys
import os
import ctypes
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

def ensure_admin():
    """Ensure process is elevated with Administrator rights for Win32 PostMessage UIPI."""
    try:
        if os.environ.get("NO_ELEVATE"):
            return
        if not bool(ctypes.windll.shell32.IsUserAnAdmin()):
            script = str(Path(__file__).resolve())
            params = f'"{script}"'
            ret = ctypes.windll.shell32.ShellExecuteW(
                None, "runas", sys.executable, params, str(root_dir), 1
            )
            # If ShellExecuteW triggered successfully, terminate non-elevated caller
            if ret > 32:
                sys.exit(0)
    except Exception:
        pass

from lucid_chaos.ui import launch_app

if __name__ == "__main__":
    ensure_admin()
    launch_app()
