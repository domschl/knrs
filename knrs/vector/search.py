"""
knrs.vector.search — Ranked query interface for the VectorDB.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    path: str
    text: str
    score: float

class KnrsSearcher:
    def __init__(self, db_dir: Path):
        self.db_dir = db_dir
        self.index_file = self.db_dir / "index.npy"
        self.meta_file = self.db_dir / "index.json"
        
        self.embeddings = None
        self.metadata = None
        self._model = None

    def _load(self):
        if self.embeddings is None:
            if not self.index_file.exists():
                raise FileNotFoundError("Vector index not found. Run indexer first.")
            self.embeddings = np.load(self.index_file)
            with self.meta_file.open('r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            model_name = self.metadata.get("model", "google/embeddinggemma-300m")
            logger.info("Loading model %s for search...", model_name)
            self._model = SentenceTransformer(model_name, trust_remote_code=True)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Perform a semantic search for the given query."""
        self._load()
        
        query_embedding = self._model.encode(query)
        # Compute cosine similarities
        cos_scores = util.cos_sim(query_embedding, self.embeddings)[0]
        
        # Get top-k indices
        top_results = np.argpartition(-cos_scores, range(min(top_k, len(cos_scores))))[:top_k]
        
        results = []
        for idx in top_results:
            idx = int(idx)
            score = float(cos_scores[idx])
            meta = self.metadata['chunks'][idx]
            full_text = self.metadata['full_texts'][idx]
            
            results.append(SearchResult(
                path=meta['path'],
                text=full_text,
                score=score
            ))
            
        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        return results
