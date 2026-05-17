from __future__ import annotations

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class BackendManager:
    def __init__(self, subprocesses_dir: Path) -> None:
        self.subprocesses_dir: Path = subprocesses_dir
        self.backends: Dict[str, Dict[str, Any]] = {}
        self.discover()

    def discover(self) -> None:
        self.backends.clear()
        if not self.subprocesses_dir.exists():
            return
            
        current_platform = sys.platform
        if current_platform.startswith("linux"):
            os_name = "linux"
        elif current_platform == "darwin":
            os_name = "macos"
        else:
            os_name = "any"

        for entry in self.subprocesses_dir.iterdir():
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            
            script_path = entry / f"{entry.name}.py"
            if not script_path.exists():
                continue
                
            venv_python = entry / ".venv" / "bin" / "python"
            python_exe = str(venv_python) if venv_python.exists() else sys.executable
            
            try:
                result = subprocess.run(
                    [python_exe, str(script_path), "--capabilities"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    try:
                        cap: Dict[str, Any] = json.loads(result.stdout.strip())
                        # Check platform compatibility
                        plat: str = cap.get("platform", "any")
                        if plat == "any" or plat == os_name:
                            self.backends[entry.name] = cap
                    except json.JSONDecodeError as e:
                        logger.debug("Could not parse capabilities for %s: %s", entry.name, e)
            except Exception as e:
                logger.debug("Failed to query capabilities for %s: %s", entry.name, e)

    def get_backends(self, backend_type: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        if backend_type:
            return {k: v for k, v in self.backends.items() if v.get("type") == backend_type}
        return self.backends
        
    def get_backend(self, name: str) -> Optional[Dict[str, Any]]:
        return self.backends.get(name)
