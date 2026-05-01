"""
embedder_hf — HuggingFace SentenceTransformer embedding subprocess.

Two modes
---------
One-shot (legacy / standalone):
    python embedder_hf.py input.json output.npy

Server (persistent, used by KnrsIndexer):
    python embedder_hf.py --server

In server mode the model is loaded once.  The parent process communicates
over stdin/stdout using a simple line protocol:

  stdin  ← "INPUT_PATH OUTPUT_PATH\\n"   (parent sends a batch)
  stdout → "READY\\n"                    (after model load, once)
  stdout → "DONE\\n"                     (after each batch)
  stdout → "ERROR: <msg>\\n"             (on failure)

The parent closes stdin to signal shutdown.
"""

import json
import logging
import os
import sys
from pathlib import Path

# Must be set before PyTorch is imported so the CUDA allocator picks it up.
# expandable_segments reduces fragmentation when many small tensors are
# allocated and freed repeatedly (e.g. KV-cache per encode() call).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("embedder_hf")

MODEL_NAME = "google/embeddinggemma-300m"
# 32 is the sentence-transformers default and keeps per-call KV-cache
# allocations small.  256 caused large sliding-window KV caches with
# Gemma3 that accumulated across server-mode encode() calls → OOM.
ENCODE_BATCH_SIZE = 32


def _load_model() -> SentenceTransformer:
    logger.info("Loading SentenceTransformer model %s...", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    logger.info("Model loaded.")
    return model


def _embed(model: SentenceTransformer, input_path: Path, output_path: Path) -> None:
    with input_path.open("r", encoding="utf-8") as f:
        texts = json.load(f)
    if not texts:
        np.save(str(output_path), np.array([], dtype=np.float32))
        return
    embeddings = model.encode(
        texts,
        batch_size=ENCODE_BATCH_SIZE,
        show_progress_bar=False,   # parent's rich bar covers overall progress
        convert_to_numpy=True,
    )
    np.save(str(output_path), embeddings)
    # Release PyTorch's reserved-but-unallocated CUDA memory after each call.
    # Without this, the allocator retains freed KV-cache tensors in its pool
    # across server-mode calls, causing progressive OOM on large corpora.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def server_mode() -> None:
    """Persistent server: load model once, serve many batches."""
    # Suppress INFO-level noise during serving so it doesn't fight rich bars.
    logging.getLogger().setLevel(logging.WARNING)
    logger.setLevel(logging.WARNING)

    model = _load_model()
    # Signal readiness to parent before entering the loop.
    print("READY", flush=True)

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            print(f"ERROR: expected 'INPUT OUTPUT', got {line!r}", flush=True)
            continue
        input_path, output_path = Path(parts[0]), Path(parts[1])
        try:
            _embed(model, input_path, output_path)
            print("DONE", flush=True)
        except Exception as exc:
            print(f"ERROR: {exc}", flush=True)


def one_shot_mode(input_path: Path, output_path: Path) -> None:
    """Legacy one-shot mode for standalone / backward-compat use."""
    model = _load_model()
    logger.info("Computing embeddings (one-shot)…")
    _embed(model, input_path, output_path)
    logger.info("Saved embeddings to %s", output_path)


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] == "--server":
        server_mode()
    elif len(sys.argv) == 3:
        one_shot_mode(Path(sys.argv[1]), Path(sys.argv[2]))
    else:
        print("Usage:")
        print("  embedder_hf.py --server               # persistent server mode")
        print("  embedder_hf.py input.json output.npy  # one-shot mode")
        sys.exit(1)


if __name__ == "__main__":
    main()
