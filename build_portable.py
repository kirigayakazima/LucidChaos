"""
Local Build Script for LucidChaos Portable Release.
Packages the project into a standalone, portable directory and zip archive.
"""
import sys
import os
import shutil
import subprocess
from pathlib import Path

VERSION = "v0.1.0"
PROJECT_ROOT = Path(__file__).resolve().parent

def build():
    print(f"=== Building LucidChaos {VERSION} Portable Release ===")
    os.chdir(PROJECT_ROOT)

    # 1. Run PyInstaller
    cmd = [sys.executable, "-m", "PyInstaller", "LucidChaos.spec", "--noconfirm"]
    print(f"Running command: {' '.join(cmd)}")
    ret = subprocess.call(cmd)
    if ret != 0:
        print(f"PyInstaller build failed with exit code {ret}")
        return False

    dist_dir = PROJECT_ROOT / "dist" / "LucidChaos"
    if not dist_dir.exists():
        print(f"Error: Build directory not found: {dist_dir}")
        return False

    # 2. Copy supplementary files
    print("Copying documentation and startup scripts...")
    for f in ["README.md", "start_gui.bat"]:
        src = PROJECT_ROOT / f
        if src.exists():
            shutil.copy2(src, dist_dir / f)

    # 3. Create zip archive
    zip_base = PROJECT_ROOT / "dist" / f"LucidChaos-{VERSION}-windows-x64"
    print(f"Creating portable zip archive: {zip_base}.zip ...")
    shutil.make_archive(str(zip_base), "zip", root_dir=str(dist_dir))

    zip_file = PROJECT_ROOT / "dist" / f"LucidChaos-{VERSION}-windows-x64.zip"
    if zip_file.exists():
        size_mb = zip_file.stat().st_size / (1024 * 1024)
        print(f"\n[SUCCESS] Portable release package generated!")
        print(f"Archive: {zip_file}")
        print(f"Size: {size_mb:.2f} MB")
        return True

    return False

if __name__ == "__main__":
    success = build()
    sys.exit(0 if success else 1)
