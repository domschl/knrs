#!/usr/bin/env python3
"""
knrs install script.

- Checks for 'uv'.
- Runs 'uv sync' in the root.
- Runs 'uv sync' in all subprocess directories with a pyproject.toml.
- Skips macOS-specific backends on non-macOS platforms.
"""

import os
import sys
import shutil
import subprocess
import platform
from pathlib import Path

import argparse

def run_command(cmd, cwd=None):
    """Run a command and print output."""
    print(f"Running: {' '.join(cmd)} in {cwd or '.'}")
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with exit code {e.returncode}")
        sys.exit(e.returncode)

def has_torch_installed(cwd):
    """Check if torch is installed in the local environment."""
    try:
        # We use 'uv pip show torch' which is fast.
        result = subprocess.run(["uv", "pip", "show", "torch"], cwd=cwd, capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser(description="knrs installation and update script.")
    parser.add_argument("--xpu", action="store_true", help="Install Intel XPU optimized torch versions.")
    args = parser.parse_args()

    root = Path(__file__).parent.resolve()
    
    # 1. Check for uv
    if not shutil.which("uv"):
        print("Error: 'uv' is not installed. Please install it first (https://github.com/astral-sh/uv).")
        sys.exit(1)

    print("--- Syncing root project ---")
    run_command(["uv", "sync"], cwd=root)

    # 2. Iterate subprocesses
    sub_dir = root / "knrs" / "subprocesses"
    if not sub_dir.exists():
        print(f"Warning: Subprocesses directory not found at {sub_dir}")
        return

    is_macos = platform.system() == "Darwin"
    
    print("\n--- Syncing subprocesses ---")
    for item in sorted(sub_dir.iterdir()):
        if not item.is_dir():
            continue
        
        pyproject = item / "pyproject.toml"
        if not pyproject.exists():
            continue

        # Skip macOS-specific backends on other platforms
        if item.name.endswith("_macos") and not is_macos:
            print(f"Skipping macOS-specific backend: {item.name}")
            continue

        print(f"\nSyncing {item.name}...")
        run_command(["uv", "sync"], cwd=item)

        # 3. Handle XPU overlay
        if args.xpu and has_torch_installed(item):
            print(f"Overlaying XPU-optimized torch for {item.name}...")
            run_command([
                "uv", "pip", "install", "torch", "--upgrade", 
                "--index-url", "https://download.pytorch.org/whl/xpu"
            ], cwd=item)

    print("\nInstallation/Update complete!")

if __name__ == "__main__":
    main()
