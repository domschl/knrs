from __future__ import annotations

import os
import json
import time
import logging
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger("summarizer_core.utils")

# ── Runtime config validation ──────────────────────────────────────────────────

def validate_config(cfg: Dict[str, Any], schema: Dict[str, str]) -> List[str]:
    """Validate a config dict against a simple type schema.

    Args:
        cfg:    The loaded config dictionary.
        schema: Maps key names to expected Python type names, e.g.
                {"model_name": "str", "chunk_size": "int", "temperature": "float"}.

    Returns:
        A list of error strings (empty = valid).
    """
    errors: List[str] = []
    type_map: Dict[str, Type[Any]] = {"str": str, "int": int, "float": float, "bool": bool}

    for key, type_name in schema.items():
        if key not in cfg:
            errors.append(f"Missing required config key: '{key}'")
            continue
        expected = type_map.get(type_name)
        if expected is None:
            continue  # Unknown type — skip validation
        val = cfg[key]
        # Allow int where float is expected (JSON numbers are always int or float)
        if expected is float and isinstance(val, int):
            continue
        if not isinstance(val, expected):
            errors.append(
                f"Config key '{key}': expected {type_name}, got {type(val).__name__!r} (value: {val!r})"
            )
    return errors


# ── Platform config helpers ────────────────────────────────────────────────────

def get_platform_config(config_name: str, default_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load a backend config from ~/.config/knrs/<config_name>.

    If the file does not exist, it is created from *default_config*.
    If the file exists but is malformed JSON, the default is returned and an
    error is logged (the file is NOT overwritten to avoid data loss).

    Args:
        config_name:    Filename, e.g. "agent_config_api.json".
        default_config: Values to use and write when no config file exists.

    Returns:
        The loaded (or default) config dict.
    """
    default = default_config or {"chunk_size": 50000}
    config_file = os.path.expanduser(f"~/.config/knrs/{config_name}")

    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(
                "Config file %s contains invalid JSON: %s — using defaults.", config_file, e
            )
            return default.copy()
        except Exception as e:
            logger.error("Failed to read config %s: %s — using defaults.", config_file, e)
            return default.copy()

    # File does not exist — write the default so the user can see and edit it.
    try:
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, "w") as f:
            json.dump(default, f, indent=4)
        logger.info("Default config written to %s", config_file)
    except Exception as e:
        logger.error("Failed to create default config at %s: %s", config_file, e)

    return default.copy()


def update_platform_config(config_name: str, key: str, value: Any, default_config: Optional[Dict[str, Any]] = None) -> bool:
    """Update a single key in ~/.config/knrs/<config_name>.

    If the file does not exist, it is created from *default_config* first.

    Returns True on success, False on failure.
    """
    config_file = os.path.expanduser(f"~/.config/knrs/{config_name}")
    try:
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                raw: Dict[str, Any] = json.load(f)
        else:
            raw = (default_config or {}).copy()
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            logger.info("Creating config file %s with defaults before update.", config_file)

        raw[key] = value

        temp_path = config_file + ".tmp"
        with open(temp_path, "w") as f:
            json.dump(raw, f, indent=4)
        os.replace(temp_path, config_file)
        return True
    except Exception as e:
        logger.error("Failed to update %s: %s", config_file, e)
        return False


def get_llm_server_config() -> Dict[str, Any]:
    """Returns the shared LLM server configuration."""
    return get_platform_config("llm_server.json", {
        "url": "http://localhost:8180",
        "api_type": "llama-server",
        "api_key": None
    })


def watchdog() -> None:
    """Exits the process if the parent process dies (PPID becomes 1)."""
    while True:
        if os.getppid() == 1:
            logger.warning("Parent process died. Exiting...")
            os._exit(1)
        time.sleep(2)
