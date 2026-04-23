"""
knrs.vector.indexer — Chunk and embed MarkdownBooks using Google's 300M Gemma embedding model.

Implementation details:
- Chunking: Sliding window (approx 512 tokens).
- Model: google/embeddinggemma-300m.
- Storage: KnrsData/VectorDB/index.npy (embeddings) and index.json (metadata).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from knrs.calibre.converter import _split_frontmatter

logger = logging.getLogger(__name__)

# The specific model requested by the user
MODEL_NAME = "google/embeddinggemma-300m"

class KnrsIndexer:
    def __init__(self, db_dir: Path):
        self.db_dir = db_dir
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.db_dir / "index.npy"
        self.meta_file = self.db_dir / "index.json"
        self._model = None

    @property
    def model(self):
        if self._model is None:
            logger.info("Loading embedding model %s...", MODEL_NAME)
            self._model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
        return self._model

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
        """Simple character-based sliding window chunking."""
        chunks = []
        if len(text) <= chunk_size:
            return [text]
            
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += (chunk_size - overlap)
        return chunks

    def run_indexing(self, markdown_books_dir: Path) -> None:
        """Scan all markdown books, chunk them, embed, and save."""
        all_chunks = []
        all_metadata = []
        
        logger.info("Scanning books for indexing...")
        for md_path in sorted(markdown_books_dir.rglob("*.md")):
            try:
                content = md_path.read_text(encoding="utf-8")
                fm_raw, body = _split_frontmatter(content)
                
                # We skip very short documents
                if len(body) < 100:
                    continue
                    
                chunks = self.chunk_text(body)
                for i, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    all_metadata.append({
                        "path": str(md_path.relative_to(markdown_books_dir)),
                        "chunk_index": i,
                        "text": chunk[:200] + "..." # Snippet for verification
                    })
            except Exception as e:
                logger.error("Failed to process %s for indexing: %s", md_path.name, e)

        if not all_chunks:
            logger.info("No content found to index.")
            return

        logger.info("Computing embeddings for %d chunks...", len(all_chunks))
        embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        
        # Save results
        np.save(self.index_file, embeddings)
        with self.meta_file.open('w', encoding='utf-8') as f:
            json.dump({
                "model": MODEL_NAME,
                "chunks": all_metadata,
                "full_texts": all_chunks # We store full text for search results
            }, f, indent=2)
            
        logger.info("Indexed %d chunks into %s", len(all_chunks), self.db_dir)
