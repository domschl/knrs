"""
knrs.vector.search — Ranked query interface for the VectorDB.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from knrs.vector.engine import get_embeddings
from knrs.config import KnrsConfig

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    path: str        # prefixed key, e.g. "books:Series/Book.md"
    text: str
    score: float

    @property
    def source_label(self) -> str:
        """Return 'books' or 'wiki' (or the raw label if unknown)."""
        return self.path.split(":", 1)[0] if ":" in self.path else "unknown"

    @property
    def bare_path(self) -> str:
        """Return the path without the source prefix."""
        return self.path.split(":", 1)[1] if ":" in self.path else self.path

class KnrsSearcher:
    def __init__(self, config: KnrsConfig):
        self.config = config
        self.db_dir = config.vector_db
        self.index_file = self.db_dir / "index.npy"
        self.meta_file = self.db_dir / "index.json"
        
        self.embeddings = None
        self.metadata = None

    def _load(self):
        if self.embeddings is None:
            if not self.index_file.exists():
                raise FileNotFoundError("Vector index not found. Run indexer first.")
            self.embeddings = np.load(self.index_file)
            with self.meta_file.open('r', encoding='utf-8') as f:
                self.metadata = json.load(f)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Perform a semantic search for the given query."""
        self._load()
        
        # Get query embedding through the engine
        query_embeddings = get_embeddings([query], self.config)
        query_embedding = query_embeddings[0]
        
        # Compute cosine similarities using numpy
        # cos_sim(a, b) = (a . b) / (||a|| * ||b||)
        norm_q = np.linalg.norm(query_embedding)
        norm_v = np.linalg.norm(self.embeddings, axis=1)
        
        # Avoid division by zero
        dot_product = np.dot(self.embeddings, query_embedding)
        cos_scores = dot_product / (norm_q * norm_v + 1e-9)
        
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
