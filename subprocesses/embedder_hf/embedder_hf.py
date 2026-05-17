from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Setup logging (to stderr so stdout stays clean for capabilities/protocol)
from rich.console import Console
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            console=Console(stderr=True),
            rich_tracebacks=True,
            show_path=False,
            markup=False,
        )
    ],
)
logger = logging.getLogger("embedder_hf")

MODEL_NAME = "google/embeddinggemma-300m"
# 32 is the sentence-transformers default and keeps per-call KV-cache
# allocations small.  256 caused large sliding-window KV caches with
# Gemma3 that accumulated across server-mode encode() calls → OOM.
ENCODE_BATCH_SIZE = 32

DEFAULT_CONFIG: Dict[str, Any] = {
    "device": "auto"
}

def get_platform_config() -> Dict[str, Any]:
    config_file = os.path.expanduser("~/.config/knrs/embedder_config_hf.json")
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading platform config: {e}")

    try:
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
    except Exception as e:
        logger.error(f"Failed to create default config at {config_file}: {e}")

    return DEFAULT_CONFIG.copy()


def _get_device(config_device: str = "auto") -> str:
    import torch
    if config_device and config_device != "auto":
        return config_device

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_model() -> Any:
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    
    # Attempt to find the model in the local HuggingFace cache to avoid 
    # the delay caused by checking for updates online (ETag requests).
    load_path: str = MODEL_NAME
    try:
        cached_path: str = snapshot_download(MODEL_NAME, local_files_only=True)
        if cached_path:
            logger.info("Found cached model at %s", cached_path)
            load_path = cached_path
    except Exception:
        # Fallback to the original MODEL_NAME if not cached or on error.
        pass

    config = get_platform_config()
    device = _get_device(config.get("device", "auto"))
    logger.info("Using device: %s", device)
    logger.info("Loading SentenceTransformer model %s...", load_path)
    model = SentenceTransformer(load_path, trust_remote_code=True, device=device)
    logger.info("Model loaded.")
    return model


def _embed(model: Any, input_path: Path, output_path: Path, mode: str = "document") -> None:
    import torch
    import numpy as np
    
    with input_path.open("r", encoding="utf-8") as f:
        texts: List[str] = json.load(f)
    if not texts:
        np.save(str(output_path), np.array([], dtype=np.float32))
        return
    
    encode_kwargs: Dict[str, Any] = {
        "batch_size": ENCODE_BATCH_SIZE,
        "show_progress_bar": False,   # parent's rich bar covers overall progress
        "convert_to_numpy": True,
    }
    
    if mode == "query":
        embeddings = model.encode_query(texts, **encode_kwargs)
    else:
        embeddings = model.encode_document(texts, **encode_kwargs)
    np.save(str(output_path), embeddings)
    # Release PyTorch's reserved-but-unallocated CUDA memory after each call.
    # Without this, the allocator retains freed KV-cache tensors in its pool
    # across server-mode calls, causing progressive OOM on large corpora.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()


def server_mode() -> None:
    """Persistent server: load model once, serve many batches."""
    # Must be set before PyTorch is imported so the CUDA allocator picks it up.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    
    model = _load_model()

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
            _embed(model, input_path, output_path, mode)
            print("DONE", flush=True)
        except Exception as exc:
            print(f"ERROR: {exc}", flush=True)


def one_shot_mode(mode: str, input_path: Path, output_path: Path) -> None:
    """Legacy one-shot mode for standalone / backward-compat use."""
    # Must be set before PyTorch is imported so the CUDA allocator picks it up.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    
    model = _load_model()
    logger.info("Computing embeddings (mode=%s)...", mode)
    _embed(model, input_path, output_path, mode)
    logger.info("Saved embeddings to %s", output_path)


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] == "--capabilities":
        cap: Dict[str, Any] = {
            "name": "embedder_hf",
            "type": "embedder",
            "config_file": "embedder_config_hf.json",
            "platform": "any",
            "validated_models": [MODEL_NAME],
            "available_models": [MODEL_NAME],
            "parameters": {
                "device": {"type": "str"}
            },
        }
        print(json.dumps(cap))
        sys.exit(0)
    elif len(sys.argv) == 2 and sys.argv[1] == "--server":
        server_mode()
    elif len(sys.argv) == 5 and sys.argv[1] == "--mode":
        mode = sys.argv[2]
        one_shot_mode(mode, Path(sys.argv[3]), Path(sys.argv[4]))
    else:
        print("Usage:")
        print("  embedder_hf.py --capabilities                     # print capabilities as JSON")
        print("  embedder_hf.py --server                           # persistent server mode")
        print("  embedder_hf.py --mode query|document in.json out.npy # one-shot mode")
        sys.exit(1)


if __name__ == "__main__":
    main()
