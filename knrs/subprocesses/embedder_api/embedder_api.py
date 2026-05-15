from __future__ import annotations

import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            show_path=False,
            markup=False,
        )
    ],
)
logger = logging.getLogger("embedder_api")

from summarizer_core.utils import get_platform_config, get_llm_server_config, watchdog

# Constants
DEFAULT_LOCAL_CONFIG: Dict[str, Any] = {
    "model_name": "embeddinggemma-300M-Q8_0",
    "batch_size": 32
}

def _embed(url: str, api_key: Optional[str], model: str, input_path: Path, output_path: Path) -> None:
    with input_path.open("r", encoding="utf-8") as f:
        texts: List[str] = json.load(f)
    if not texts:
        np.save(str(output_path), np.array([], dtype=np.float32))
        return

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    # Use standard OpenAI-compatible embeddings endpoint
    payload: Dict[str, Any] = {
        "model": model,
        "input": texts
    }
    
    try:
        response = requests.post(f"{url}/v1/embeddings", json=payload, headers=headers)
        response.raise_for_status()
        data: Dict[str, Any] = response.json()
        
        embeddings: List[List[float]]
        # OpenAI format: {"data": [{"embedding": [...], "index": 0}, ...]}
        if "data" in data:
            # Sort by index to ensure correct order
            results = sorted(data["data"], key=lambda x: x["index"])
            embeddings = [r["embedding"] for r in results]
        elif "results" in data:
            # llama.cpp legacy format
            embeddings = [r["embedding"] for r in data["results"]]
        elif "embedding" in data:
            if isinstance(data["embedding"][0], list):
                embeddings = data["embedding"]
            else:
                embeddings = [data["embedding"]]
        else:
            raise ValueError(f"Unexpected response format from server: {data}")
            
        embeddings_np = np.array(embeddings, dtype=np.float32)
        np.save(str(output_path), embeddings_np)
    except Exception as e:
        logger.error(f"Embedding request failed: {e}")
        raise

def server_mode(url: str, api_key: Optional[str], model: str) -> None:
    """Persistent server: handles requests via stdin/stdout."""
    # Suppress noise during serving
    logging.getLogger().setLevel(logging.WARNING)
    logger.setLevel(logging.WARNING)

    # Signal readiness
    print("READY", flush=True)

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 3:
            print(f"ERROR: expected 'MODE INPUT OUTPUT', got {line!r}", flush=True)
            continue
        mode, input_path, output_path = parts[0], Path(parts[1]), Path(parts[2])
        try:
            # Note: llama-server typically handles the "query" vs "document" prefix 
            # internally if configured, or we'd need to add it to texts.
            # For gemma-300m, it's often better to just send as-is unless specific 
            # prefixes are required by the server setup.
            _embed(url, api_key, model, input_path, output_path)
            print("DONE", flush=True)
        except Exception as exc:
            print(f"ERROR: {exc}", flush=True)

def main() -> None:
    w = threading.Thread(target=watchdog, daemon=True)
    w.start()

    server_cfg = get_llm_server_config()
    local_cfg = get_platform_config("embedder_config_api.json", DEFAULT_LOCAL_CONFIG)
    
    url: str = server_cfg["url"].rstrip("/")
    api_key: Optional[str] = server_cfg.get("api_key")
    model: str = local_cfg["model_name"]

    if len(sys.argv) == 2 and sys.argv[1] == "--capabilities":
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            response = requests.get(f"{url}/v1/models", headers=headers, timeout=2)
            response.raise_for_status()
            data = response.json()
            available_models = [m["id"] for m in data.get("data", [])]
        except Exception as e:
            logger.error(f"Failed to query {url}/v1/models: {e}")
            available_models = []

        cap: Dict[str, Any] = {
            "name": "embedder_api",
            "type": "embedder",
            "config_file": "embedder_config_api.json",
            "platform": "any",
            "validated_models": ["embeddinggemma-300M-Q8_0"],
            "available_models": available_models,
            "parameters": {
                "model_name": {"type": "str"},
                "batch_size":  {"type": "int", "min": 1, "max": 512},
            },
        }
        print(json.dumps(cap))
        sys.exit(0)
    elif len(sys.argv) == 2 and sys.argv[1] == "--server":
        server_mode(url, api_key, model)
    elif len(sys.argv) == 5 and sys.argv[1] == "--mode":
        mode = sys.argv[2]
        _embed(url, api_key, model, Path(sys.argv[3]), Path(sys.argv[4]))
    else:
        print("Usage:")
        print("  embedder_api.py --capabilities")
        print("  embedder_api.py --server")
        print("  embedder_api.py --mode query|document in.json out.npy")
        sys.exit(1)

if __name__ == "__main__":
    main()
