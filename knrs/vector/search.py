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
    chunk_index: int = 0
    query_embedding: np.ndarray | None = None

    @property
    def source_label(self) -> str:
        """Return 'books' or 'wiki' (or the raw label if unknown)."""
        return self.path.split(":", 1)[0] if ":" in self.path else "unknown"

    @property
    def bare_path(self) -> str:
        """Return the path without the source prefix."""
        return self.path.split(":", 1)[1] if ":" in self.path else self.path

def get_context_aware_text(searcher: KnrsSearcher, result: SearchResult) -> str:
    from knrs.calibre.converter import _split_frontmatter
    
    if result.source_label == "books":
        file_path = searcher.config.markdown_books / result.bare_path
    elif result.source_label == "wiki":
        file_path = searcher.config.wiki_path / result.bare_path
    else:
        return result.text
        
    if not file_path.exists():
        return result.text
        
    try:
        content = file_path.read_text(encoding="utf-8")
        _, body = _split_frontmatter(content)
    except Exception as e:
        logger.error("Failed to read context for %s: %s", file_path, e)
        return result.text
        
    chunk_size = searcher.metadata.get("chunk_size", 3000)
    overlap = searcher.metadata.get("overlap", 600)
    step = chunk_size - overlap
    
    if len(body) <= chunk_size:
        return body
        
    start = 0
    num_chunks = 0
    while start < len(body):
        num_chunks += 1
        start += step
        
    idx = result.chunk_index
    chunk_start = idx * step
    chunk_end = chunk_start + chunk_size
    
    prev_chunk = (idx - 1) * step if idx > 0 else chunk_start
    next_chunk = (idx + 1) * step if idx < num_chunks - 1 else chunk_end
    
    extended_text = body[prev_chunk : next_chunk + chunk_size]
    
    prev_max = chunk_start - prev_chunk
    next_max = prev_max + chunk_size
    
    borders = {'.', '!', '?', '\n', '。', '！', '？'}
    act_start = prev_max
    for ind in range(prev_max - 1, -1, -1):
        if extended_text[ind] in borders:
            act_start = ind + 1
            while act_start < len(extended_text) and extended_text[act_start] in {' ', '\t', '\r'}:
                act_start += 1
            break
            
    act_end = next_max
    for ind in range(next_max, len(extended_text)):
        if extended_text[ind] in borders:
            act_end = ind + 1
            break
            
    import re
    result_text = extended_text[act_start:act_end].strip()
    return re.sub(r'\n{3,}', '\n\n', result_text)

def get_significance(text: str, query_embedding: np.ndarray, searcher: KnrsSearcher, raw: bool = False, cutoff: float = 0.5, session=None) -> str:
    context_length = 64
    context_steps = 32
    text_len = len(text)
    
    clr = []
    snippet_ranges = []
    for i in range(0, text_len, context_steps):
        i0 = max(0, i - context_length // 2)
        i1 = min(text_len, i + context_length // 2 + (context_length % 2))
        if i0 == 0 and i1 < text_len:
            i1 = min(text_len, i0 + context_length)
        elif i1 == text_len and i0 > 0:
            i0 = max(0, i1 - context_length)
            
        snippet = text[i0:i1]
        if snippet:
            clr.append(snippet)
            snippet_ranges.append((i0, i1))
            
    if not clr:
        return text
        
    if session is not None:
        snippet_embeddings = session.embed(clr, encode_mode="document")
    else:
        from knrs.vector.engine import get_embeddings
        snippet_embeddings = get_embeddings(clr, searcher.config, encode_mode="document")
    
    norm_q = np.linalg.norm(query_embedding)
    norm_v = np.linalg.norm(snippet_embeddings, axis=1)
    dot_product = np.dot(snippet_embeddings, query_embedding)
    cosines = dot_product / (norm_q * norm_v + 1e-9)
    
    min_cos = float(np.min(cosines))
    max_cos = float(np.max(cosines))
    
    if max_cos - min_cos > 0.0:
        cosines = (cosines - min_cos) / (max_cos - min_cos)
        import math
        cosines = np.array([(math.exp(c) - 1) / (math.exp(1) - 1) for c in cosines])
    else:
        cosines = np.zeros_like(cosines)
        
    char_scores = np.zeros(text_len)
    char_counts = np.zeros(text_len)
    
    for score, (i0, i1) in zip(cosines, snippet_ranges):
        char_scores[i0:i1] += score
        char_counts[i0:i1] += 1
        
    char_scores = np.divide(char_scores, char_counts, out=np.zeros_like(char_scores), where=char_counts!=0)
    
    result_parts = []
    is_highlighted = False
    current_part = []
    
    for i, char in enumerate(text):
        high = char_scores[i] >= cutoff
        if high != is_highlighted:
            if current_part:
                part_text = "".join(current_part)
                if is_highlighted:
                    result_parts.append(f"**{part_text}**")
                else:
                    result_parts.append(part_text)
                current_part = []
            is_highlighted = high
        current_part.append(char)
        
    if current_part:
        part_text = "".join(current_part)
        if is_highlighted:
            result_parts.append(f"**{part_text}**")
        else:
            result_parts.append(part_text)
            
    return "".join(result_parts)

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
        query_embeddings = get_embeddings([query], self.config, encode_mode="query")
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
                score=score,
                chunk_index=meta.get('chunk_index', 0),
                query_embedding=query_embedding
            ))
            
        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        return results
