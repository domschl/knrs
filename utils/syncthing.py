from __future__ import annotations

import os
import logging
import urllib.request
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

def find_st_root(path: Path) -> Path | None:
    """Climb up from path to find the Syncthing folder root (containing .stfolder)."""
    current = path.resolve()
    while True:
        if (current / ".stfolder").exists():
            return current
        if current.parent == current:
            break
        current = current.parent
    return None

def get_syncthing_info() -> tuple[str | None, str | None, dict[str, str]]:
    """
    Parse Syncthing config to get API key, GUI address, and path->id mapping.
    
    Returns:
        (api_key, gui_address, {expanded_path: folder_id})
    """
    candidates = [
        Path.home() / "Library/Application Support/Syncthing/config.xml",
        Path.home() / ".config/syncthing/config.xml",
        Path.home() / ".local/state/syncthing/config.xml",
    ]
    config_path = None
    for p in candidates:
        if p.exists():
            config_path = p
            break
            
    if not config_path:
        return None, None, {}

    try:
        tree = ET.parse(config_path)
        root = tree.getroot()
        
        gui = root.find("gui")
        if gui is None:
            return None, None, {}
            
        api_key = gui.findtext("apikey")
        address = gui.findtext("address")
        
        folder_map = {}
        for folder in root.findall("folder"):
            f_id = folder.get("id")
            f_path = folder.get("path")
            if f_id and f_path:
                # Expand ~ in path
                expanded = os.path.expanduser(f_path)
                folder_map[str(Path(expanded).resolve())] = f_id
                
        return api_key, address, folder_map
    except Exception as e:
        logger.error("Failed to parse Syncthing config: %s", e)
        return None, None, {}

def get_syncthing_status(path: Path) -> dict[str, Any] | None:
    """
    Check if a path is in Syncthing and return its sync status.
    
    Returns:
        A dict with status info, or None if not a Syncthing folder.
    """
    st_root = find_st_root(path)
    if not st_root:
        return None
        
    api_key, address, folder_map = get_syncthing_info()
    if not api_key or not address:
        return {"root": st_root, "error": "Syncthing config not found or invalid"}
        
    folder_id = folder_map.get(str(st_root))
    if not folder_id:
        return {"root": st_root, "error": f"Folder root {st_root} not found in Syncthing config"}
        
    # Query Syncthing API
    url = f"http://{address}/rest/db/status?folder={folder_id}"
    try:
        req = urllib.request.Request(url)
        req.add_header("X-API-Key", api_key)
        
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            
        return {
            "root": st_root,
            "folder_id": folder_id,
            "state": data.get("state"),
            "is_idle": data.get("state") == "idle",
            "need_bytes": data.get("needBytes", 0),
            "in_sync": data.get("state") == "idle" and data.get("needBytes", 0) == 0
        }
    except Exception as e:
        return {"root": st_root, "folder_id": folder_id, "error": f"API request failed: {e}"}
