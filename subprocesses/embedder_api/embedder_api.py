from __future__ import annotations

import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import requests

# Setup logging (to stderr so stdout stays clean for protocol/capabilities)
from rich.logging import RichHandler
from rich.console import Console

logging.basicConfig(
    level=logging.INFO if os.environ.get("KNRS_VERBOSE") == "1" else logging.WARNING,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            show_path=False,
            markup=False,
            console=Console(stderr=True),
        )
    ],
)
logger = logging.getLogger("embedder_api")

from summarizer_core.utils import get_platform_config, get_llm_server_config, watchdog

# Constants
DEFAULT_LOCAL_CONFIG: dict[str, Any] = {
    "model_name": "embeddinggemma-300M-Q8_0",
    "batch_size": 1024
}

def _embed(url: str, api_key: str | None, model: str, input_path: Path, output_path: Path) -> None:
    with input_path.open("r", encoding="utf-8") as f:
        texts: list[str] = json.load(f)
    if not texts:
        np.save(str(output_path), np.array([], dtype=np.float32))
        return

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    # Use standard OpenAI-compatible embeddings endpoint
    payload: dict[str, Any] = {
        "model": model,
        "input": texts
    }
    
    try:
        response = requests.post(f"{url}/v1/embeddings", json=payload, headers=headers)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        
        embeddings: list[list[float]]
        # OpenAI format: {"data": [{"embedding": [...], "index": 0}, ...]}
        if "data" in data and isinstance(data["data"], list):
            # Sort by index to ensure order if not guaranteed
            sorted_data = sorted(data["data"], key=lambda x: x.get("index", 0))
            embeddings = [item["embedding"] for item in sorted_data]
            np.save(str(output_path), np.array(embeddings, dtype=np.float32))
        else:
            logger.error("Unexpected API response format: %s", data)
            raise ValueError("Malformed API response")
            
    except Exception as e:
        logger.error("Embedding request failed: %s", e)
        raise

def server_mode() -> None:
    """Persistent server: load config once, serve many batches."""
    server_cfg = get_llm_server_config()
    local_cfg = get_platform_config("embedder_config_api.json", DEFAULT_LOCAL_CONFIG)
    
    url = server_cfg["url"].rstrip("/")
    api_key = server_cfg.get("api_key")
    model = local_cfg["model_name"]
    
    # Suppress INFO-level noise during serving so it doesn't fight rich bars.
    logging.getLogger().setLevel(logging.WARNING)
    logger.setLevel(logging.WARNING)
    # Signal readiness to parent before entering the loop.
    print("READY", flush=True)

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 3:
            # We ignore mode (parts[0]) here as it's always standard embeddings for now
            print(f"ERROR: expected 'MODE INPUT OUTPUT', got {line!r}", flush=True)
            continue
        input_path, output_path = Path(parts[1]), Path(parts[2])
        try:
            _embed(url, api_key, model, input_path, output_path)
            print("DONE", flush=True)
        except Exception as exc:
            print(f"ERROR: {exc}", flush=True)

def main() -> None:
    w = threading.Thread(target=watchdog, daemon=True)
    w.start()

    if len(sys.argv) == 2 and sys.argv[1] == "--capabilities":
        server_config = get_llm_server_config()
        url = server_config.get("url", "http://localhost:8180").rstrip("/")
        api_key = server_config.get("api_key")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        available_models: list[str] = []
        try:
            response = requests.get(f"{url}/v1/models", headers=headers, timeout=2)
            response.raise_for_status()
            data = response.json()
            available_models = [m["id"] for m in data.get("data", [])]
        except Exception as e:
            logger.error(f"Failed to query {url}/v1/models: {e}")

        cap: dict[str, Any] = {
            "name": "embedder_api",
            "type": "embedder",
            "config_file": "embedder_config_api.json",
            "platform": "any",
            "validated_models": [DEFAULT_LOCAL_CONFIG["model_name"]],
            "available_models": available_models,
            "parameters": {
                "model_name": {"type": "str"},
                "batch_size": {"type": "int", "min": 1, "max": 512}
            },
        }
        print(json.dumps(cap))
        sys.exit(0)
    elif len(sys.argv) == 2 and sys.argv[1] == "--server":
        server_mode()
    elif len(sys.argv) == 5 and sys.argv[1] == "--mode":
        server_cfg = get_llm_server_config()
        local_cfg = get_platform_config("embedder_config_api.json", DEFAULT_LOCAL_CONFIG)
        url = server_cfg["url"].rstrip("/")
        api_key = server_cfg.get("api_key")
        model = local_cfg["model_name"]
        
        _embed(url, api_key, model, Path(sys.argv[3]), Path(sys.argv[4]))
    else:
        print("Usage:")
        print("  embedder_api.py --capabilities                     # print capabilities as JSON")
        print("  embedder_api.py --server                           # persistent server mode")
        print("  embedder_api.py --mode query|document in.json out.npy # one-shot mode")
        sys.exit(1)

if __name__ == "__main__":
    main()
