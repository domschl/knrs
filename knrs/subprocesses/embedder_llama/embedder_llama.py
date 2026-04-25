import json
import sys
import logging
import os
import time
import contextlib
from pathlib import Path
import numpy as np
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("embedder_llama")

@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stderr = os.dup(sys.stderr.fileno())
        os.dup2(devnull.fileno(), sys.stderr.fileno())
        try:
            yield
        finally:
            os.dup2(old_stderr, sys.stderr.fileno())
            os.close(old_stderr)

REPO_ID = "ggml-org/embeddinggemma-300m-qat-q8_0-GGUF"
FILENAME = "embeddinggemma-300m-qat-Q8_0.gguf"

def main():
    if len(sys.argv) < 3:
        print("Usage: embedder_llama.py <input.json> <output.npy>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    try:
        with input_path.open("r", encoding="utf-8") as f:
            texts = json.load(f)
    except Exception as e:
        logger.error("Failed to read input JSON: %s", e)
        sys.exit(1)

    if not texts:
        logger.info("No texts to embed.")
        np.save(output_path, np.array([], dtype=np.float32))
        sys.exit(0)

    logger.info("Downloading/Loading LlamaCpp model from %s...", REPO_ID)
    try:
        model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    except Exception as e:
        logger.error("Failed to download model: %s", e)
        sys.exit(1)
        
    logger.info("Initializing Llama with path %s", model_path)
    try:
        # Initialize llama.cpp with embedding=True and GPU acceleration
        # n_gpu_layers=-1 moves all layers to GPU if CUDA is available.
        llm = Llama(
            model_path=model_path,
            embedding=True,
            verbose=False,
            n_ctx=2048,      # Large context to accommodate batches
            n_batch=512,     # Batch size for processing
            n_threads=8,     # Still useful for pre/post processing
            n_gpu_layers=-1,  # Use CUDA if available
        )
        # logger.info("Llama initialized successfully (GPU layers: %s)", 
        #             "all" if llm.n_gpu_layers != 0 else "none")
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        sys.exit(1)

    logger.info("Computing embeddings for %d texts...", len(texts))
    try:
        all_embeddings = []
        batch_size = 1 # Batching is broken for embeddings in llama-cpp-python
        start_time = time.time()
        
        for i in range(0, len(texts), batch_size):
            if batch_size == 1:
                batch = texts[i]
            else:
                batch = texts[i:i + batch_size]
            if (i // batch_size) % 250 == 0:
                elapsed = time.time() - start_time
                eta = elapsed / (i // batch_size + 1) * ((len(texts) + batch_size - 1) // batch_size - (i // batch_size + 1))
                logger.info("Processing batch %d/%d (text %d/%d)... (elapsed: %.2fs, eta: %.2fs)", 
                            i // batch_size + 1, (len(texts) + batch_size - 1) // batch_size,
                            i, len(texts), elapsed, eta)
            
            with suppress_stderr():
                # Passing a list of strings to create_embedding for batching
                res = llm.create_embedding(batch)
            
            for item in res["data"]:
                all_embeddings.append(item["embedding"])
            
        embeddings_np = np.array(all_embeddings, dtype=np.float32)
        np.save(output_path, embeddings_np)
        logger.info("Saved embeddings to %s", output_path)
    except Exception as e:
        logger.error("Embedding failed: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
