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
    "model_name": "EmbeddingGemma-2_Q8",
    "batch_size": 64
}

# Persistent HTTP session with retry & pooling
_http_session: requests.Session | None = None

def _get_http_session() -> requests.Session:
    global _http_session
    if _http_session is None:
        from requests.adapters import HTTPAdapter
        from urllib3.util import Retry

        _http_session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[502, 503, 504])
        adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=retries)
        _http_session.mount("http://", adapter)
        _http_session.mount("https://", adapter)
    return _http_session


def _apply_prefix(text: str, mode: str, model_name: str) -> str:
    """Prepend task instruction prefix for models that require asymmetric retrieval formatting."""
    m = model_name.lower()
    if any(k in m for k in ("embeddinggemma-2", "embedding-gemma-2", "gemma-embedding2", "gemma2", "gemma_2")):
        # EmbeddingGemma 2 official retrieval prefixes
        if mode == "query":
            if not text.startswith("task:"):
                return f"task: search result | query: {text}"
        else:
            if not text.startswith("title:") and not text.startswith("task:"):
                return f"title: none | text: {text}"
    elif any(k in m for k in ("embeddinggemma", "gemma-embedding")):
        # EmbeddingGemma 1 prefixes
        if mode == "query":
            if not text.startswith("task:"):
                return f"task: search query | input: {text}"
        else:
            if not text.startswith("task:"):
                return f"task: search document | input: {text}"
    return text


def _embed(
    url: str,
    api_key: str | None,
    model: str,
    input_path: Path,
    output_path: Path,
    mode: str = "document",
    batch_size: int = 64,
) -> None:
    with input_path.open("r", encoding="utf-8") as f:
        texts: list[str] = json.load(f)
    if not texts:
        np.save(str(output_path), np.array([], dtype=np.float32))
        return

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    session = _get_http_session()
    all_embeddings: list[list[float]] = []

    step = max(1, batch_size)
    for start_idx in range(0, len(texts), step):
        batch = texts[start_idx : start_idx + step]
        formatted = [_apply_prefix(t, mode, model) for t in batch]
        payload: dict[str, Any] = {
            "model": model,
            "input": formatted,
        }

        try:
            response = session.post(f"{url}/v1/embeddings", json=payload, headers=headers, timeout=120)
            if response.status_code != 200:
                err_text = response.text
                try:
                    err_json = response.json()
                    if isinstance(err_json, dict) and "error" in err_json:
                        err_text = err_json["error"].get("message", response.text)
                except Exception:
                    pass
                raise RuntimeError(f"HTTP {response.status_code}: {err_text}")

            data: dict[str, Any] = response.json()

            if "data" in data and isinstance(data["data"], list):
                sorted_data = sorted(data["data"], key=lambda x: x.get("index", 0))
                for item in sorted_data:
                    all_embeddings.append(item["embedding"])
            else:
                logger.error("Unexpected API response format: %s", data)
                raise ValueError("Malformed API response")
        except Exception as e:
            logger.error("Embedding request failed (slice %d-%d): %s", start_idx, start_idx + len(batch), e)
            raise

    np.save(str(output_path), np.array(all_embeddings, dtype=np.float32))


def server_mode() -> None:
    """Persistent server: load config once, serve many batches."""
    server_cfg = get_llm_server_config()
    local_cfg = get_platform_config("embedder_config_api.json", DEFAULT_LOCAL_CONFIG)

    url = server_cfg["url"].rstrip("/")
    api_key = server_cfg.get("api_key")
    model = local_cfg["model_name"]
    batch_size = int(local_cfg.get("batch_size", DEFAULT_LOCAL_CONFIG["batch_size"]))

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
            print(f"ERROR: expected 'MODE INPUT OUTPUT', got {line!r}", flush=True)
            continue
        mode, input_path, output_path = parts[0], Path(parts[1]), Path(parts[2])
        try:
            _embed(url, api_key, model, input_path, output_path, mode=mode, batch_size=batch_size)
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
            "validated_models": ["EmbeddingGemma-2_Q8", "EmbeddingGemma-2_BF16", "EmbeddingGemma-300M"],
            "available_models": available_models,
            "parameters": {
                "model_name": {"type": "str"},
                "batch_size": {"type": "int", "min": 1, "max": 512},
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
        batch_size = int(local_cfg.get("batch_size", DEFAULT_LOCAL_CONFIG["batch_size"]))
        mode = sys.argv[2]
        input_path = Path(sys.argv[3])
        output_path = Path(sys.argv[4])

        _embed(url, api_key, model, input_path, output_path, mode=mode, batch_size=batch_size)
    else:
        print("Usage:")
        print("  embedder_api.py --capabilities                     # print capabilities as JSON")
        print("  embedder_api.py --server                           # persistent server mode")
        print("  embedder_api.py --mode query|document in.json out.npy # one-shot mode")
        sys.exit(1)

if __name__ == "__main__":
    main()
