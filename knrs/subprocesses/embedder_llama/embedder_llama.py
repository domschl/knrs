"""
embedder_llama — llama.cpp GGUF embedding subprocess.

Two modes
---------
One-shot (legacy / standalone):
    python embedder_llama.py input.json output.npy

Server (persistent, used by KnrsIndexer):
    python embedder_llama.py --server

In server mode the model is loaded once.  The parent process communicates
over stdin/stdout using a simple line protocol:

  stdin  ← "INPUT_PATH OUTPUT_PATH\\n"   (parent sends a batch)
  stdout → "READY\\n"                    (after model load, once)
  stdout → "DONE\\n"                     (after each batch)
  stdout → "ERROR: <msg>\\n"             (on failure)

The parent closes stdin to signal shutdown.

Note: llama-cpp-python batching is broken for embeddings, so texts are
processed one at a time inside _embed().
"""

import contextlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("embedder_llama")

REPO_ID  = "ggml-org/embeddinggemma-300m-qat-q8_0-GGUF"
FILENAME = "embeddinggemma-300m-qat-Q8_0.gguf"


@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = os.dup(sys.stderr.fileno())
        os.dup2(devnull.fileno(), sys.stderr.fileno())
        try:
            yield
        finally:
            os.dup2(old_stderr, sys.stderr.fileno())
            os.close(old_stderr)


def _load_model() -> Llama:
    logger.info("Downloading/loading LlamaCpp model from %s...", REPO_ID)
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    logger.info("Initializing Llama with path %s", model_path)
    llm = Llama(
        model_path=model_path,
        embedding=True,
        verbose=False,
        n_ctx=2048,
        n_batch=512,
        n_threads=8,
        n_gpu_layers=-1,   # use CUDA if available
    )
    return llm


def _embed(llm: Llama, input_path: Path, output_path: Path, *, verbose: bool = True) -> None:
    """Embed texts from *input_path* and write the .npy to *output_path*."""
    with input_path.open("r", encoding="utf-8") as f:
        texts = json.load(f)

    if not texts:
        np.save(str(output_path), np.array([], dtype=np.float32))
        return

    all_embeddings = []
    # llama-cpp-python batching is broken for embeddings — process one at a time.
    start_time = time.time()
    for i, text in enumerate(texts):
        if verbose and i % 250 == 0:
            elapsed = time.time() - start_time
            remaining = len(texts) - i
            eta = (elapsed / (i + 1)) * remaining if i > 0 else 0.0
            logger.info(
                "Processing text %d/%d... (elapsed: %.2fs, eta: %.2fs)",
                i + 1, len(texts), elapsed, eta,
            )
        with suppress_stderr():
            res = llm.create_embedding(text)
        for item in res["data"]:
            all_embeddings.append(item["embedding"])

    np.save(str(output_path), np.array(all_embeddings, dtype=np.float32))


def server_mode() -> None:
    """Persistent server: load model once, serve many batches."""
    # Suppress INFO-level noise so it doesn't fight the parent's rich bars.
    logging.getLogger().setLevel(logging.WARNING)
    logger.setLevel(logging.WARNING)

    llm = _load_model()
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
            _embed(llm, input_path, output_path, verbose=False)
            print("DONE", flush=True)
        except Exception as exc:
            print(f"ERROR: {exc}", flush=True)


def one_shot_mode(input_path: Path, output_path: Path) -> None:
    """Legacy one-shot mode for standalone / backward-compat use."""
    llm = _load_model()
    logger.info("Computing embeddings (one-shot)…")
    _embed(llm, input_path, output_path, verbose=True)
    logger.info("Saved embeddings to %s", output_path)


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] == "--server":
        server_mode()
    elif len(sys.argv) == 3:
        one_shot_mode(Path(sys.argv[1]), Path(sys.argv[2]))
    else:
        print("Usage:")
        print("  embedder_llama.py --server               # persistent server mode")
        print("  embedder_llama.py input.json output.npy  # one-shot mode")
        sys.exit(1)


if __name__ == "__main__":
    main()
