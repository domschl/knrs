import os
import json
import time
import logging

logger = logging.getLogger("summarizer_core.utils")

def get_platform_config(config_name: str, default_config: dict = None):
    config_file = os.path.expanduser(f"~/.config/knrs/{config_name}")
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
    except Exception:
        pass

    # If not found or error, create default if possible
    default = default_config or {"chunk_size": 50000}
    try:
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(default, f, indent=4)
        logger.warning(f"Default config created at {config_file}")
    except Exception as e:
        logger.error(f"Failed to create default config at {config_file}: {e}")

    return default.copy() if hasattr(default, "copy") else default

def get_llm_server_config():
    """Returns the shared LLM server configuration."""
    return get_platform_config("llm_server.json", {
        "url": "http://localhost:8180",
        "api_type": "llama-server",
        "api_key": None
    })

def watchdog():
    """Exits the process if the parent process dies (PPID becomes 1)."""
    while True:
        if os.getppid() == 1:
            logger.warning("Parent process died. Exiting...")
            os._exit(1)
        time.sleep(2)
