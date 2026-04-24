import json
import sys
import logging
from pathlib import Path
import numpy as np
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("embedder_llama")

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
        
        # Initialize llama.cpp with embedding=True
        llm = Llama(
            model_path=model_path,
            embedding=True,
            verbose=False,
            n_ctx=2048, # Gemma 300M context is small anyway
            n_threads=4
        )
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        sys.exit(1)

    logger.info("Computing embeddings for %d texts...", len(texts))
    try:
        all_embeddings = []
        for i, text in enumerate(texts):
            if i % 10 == 0:
                logger.info("Processing text %d/%d...", i, len(texts))
            # llama-cpp-python's create_embedding returns a dict
            res = llm.create_embedding(text)
            all_embeddings.append(res["data"][0]["embedding"])
            
        embeddings_np = np.array(all_embeddings, dtype=np.float32)
        np.save(output_path, embeddings_np)
        logger.info("Saved embeddings to %s", output_path)
    except Exception as e:
        logger.error("Embedding failed: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
