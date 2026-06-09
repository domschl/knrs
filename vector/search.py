"""
knrs.vector.search — Ranked query interface for the VectorDB.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

from vector.engine import get_embeddings
from config import KnrsConfig

if TYPE_CHECKING:
    from vector.engine import EmbedderSession

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

def get_context_aware_text(searcher: KnrsSearcher, result: SearchResult) -> tuple[str, int, int]:
    from calibre.converter import _split_frontmatter
    
    file_path: Path | None = None
    if result.source_label == "books":
        file_path = searcher.config.markdown_books / result.bare_path
    elif result.source_label == "wiki":
        file_path = searcher.config.wiki_path / result.bare_path
    else:
        return result.text, 0, 0
        
    if file_path is None or not file_path.exists():
        return result.text, 0, 0
        
    try:
        content = file_path.read_text(encoding="utf-8")
        _, body = _split_frontmatter(content)
    except Exception as e:
        logger.error("Failed to read context for %s: %s", file_path, e)
        return result.text, 0, 0
        
    if searcher.metadata is None:
        return result.text, 0, 0
        
    chunk_size: int = searcher.metadata.get("chunk_size", 3000)
    overlap: int = searcher.metadata.get("overlap", 600)
    step = chunk_size - overlap
    
    if len(body) <= chunk_size:
        start_char_idx = content.find(body)
        start_line = content[:start_char_idx].count('\n') + 1 if start_char_idx != -1 else 0
        return body, start_line, start_line + body.count('\n')
        
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
    
    borders: set[str] = {'.', '!', '?', '\n', '。', '！', '？'}
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
    result_text = re.sub(r'\n{3,}', '\n\n', result_text)
    start_char_idx = content.find(result_text)
    if start_char_idx != -1:
        start_line = content[:start_char_idx].count('\n') + 1
        end_line = start_line + result_text.count('\n')
    else:
        start_line = end_line = 0
    return result_text, start_line, end_line

def get_significance(text: str, query_embedding: np.ndarray, searcher: KnrsSearcher, raw: bool = False, cutoff: float = 0.5, session: EmbedderSession | None = None) -> str:
    context_length = 64
    context_steps = 32
    text_len = len(text)
    
    clr: list[str] = []
    snippet_ranges: list[tuple[int, int]] = []
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
        from vector.engine import get_embeddings
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
    
    result_parts: list[str] = []
    is_highlighted = False
    current_part: list[str] = []
    
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
    def __init__(self, config: KnrsConfig) -> None:
        self.config: KnrsConfig = config
        self.db_dir: Path = config.vector_db
        self.index_file: Path = self.db_dir / "index.npy"
        self.meta_file: Path = self.db_dir / "index.json"
        
        self.embeddings: np.ndarray | None = None
        self.metadata: dict[str, Any] | None = None
        
    def _load(self) -> None:
        if self.embeddings is None:
            if not self.index_file.exists():
                raise FileNotFoundError("Vector index (index.npy) not found. Run indexer first.")
            if not self.meta_file.exists():
                raise FileNotFoundError("Vector metadata (index.json) not found. Run indexer first.")
                
            if self.meta_file.stat().st_size == 0:
                raise RuntimeError("Vector metadata file (index.json) is empty/corrupted. Please run indexer to rebuild.")

            try:
                self.embeddings = np.load(self.index_file)
                with self.meta_file.open('r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                raise RuntimeError(f"Failed to load vector index: {e}. Please run indexer to rebuild.")

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Perform a semantic search for the given query."""
        self._load()
        
        if self.embeddings is None or self.metadata is None:
            return []
            
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
        
        results: list[SearchResult] = []
        for idx_raw in top_results:
            idx = int(idx_raw)
            score = float(cos_scores[idx])
            meta: dict[str, Any] = self.metadata['chunks'][idx]
            full_text: str = self.metadata['full_texts'][idx]
            
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
