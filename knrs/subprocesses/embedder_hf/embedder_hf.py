import json
import sys
import logging
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("embedder_hf")

MODEL_NAME = "google/embeddinggemma-300m"

def main():
    if len(sys.argv) < 3:
        print("Usage: embedder_hf.py <input.json> <output.npy>")
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

    logger.info("Loading SentenceTransformer model %s...", MODEL_NAME)
    try:
        model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        sys.exit(1)

    logger.info("Computing embeddings for %d texts...", len(texts))
    try:
        embeddings = model.encode(texts, show_progress_bar=True)
        np.save(output_path, embeddings)
        logger.info("Saved embeddings to %s", output_path)
    except Exception as e:
        logger.error("Embedding failed: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
