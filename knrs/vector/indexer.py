"""
knrs.vector.indexer — Differential chunk-and-embed indexer.

Sources indexed:
  - KnrsData/MarkdownBooks  (prefix ``books:``)
  - Wiki/Notes              (prefix ``wiki:``)   — AINotes excluded.

Differential strategy
---------------------
``index.json`` stores a ``file_hashes`` map of ``{prefixed_rel_path: sha256hex}``.
On each run:
  1. Load existing state (empty if first run or --force).
  2. Scan all source files; compute SHA-256.
  3. Diff → determine which files are new/changed and which are deleted.
  4. Prune chunks for deleted/changed files from existing arrays.
  5. Embed only new/changed files, extend arrays.
  6. Save merged state.

Storage layout (VectorDB/):
  index.npy   — (N, D) float32 embedding matrix
  index.json  — { model, file_hashes, chunks, full_texts }
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from knrs.calibre.converter import _split_frontmatter
from knrs.vector.engine import get_embeddings
from knrs.config import KnrsConfig

logger = logging.getLogger(__name__)

# Prefix labels for source directories
_LABEL_BOOKS = "books"
_LABEL_WIKI  = "wiki"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _hash_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's content."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _scan_source(root: Path, label: str, excludes: list[Path] | None = None) -> dict[str, Path]:
    """
    Walk *root* for ``*.md`` files and return ``{label:rel_path → abs_path}``.

    *excludes* is a list of absolute directory paths whose subtrees are skipped.
    """
    excludes = excludes or []
    result: dict[str, Path] = {}
    for md_path in sorted(root.rglob("*.md")):
        # Skip hidden / sync artefact dirs
        if any(part.startswith(".") for part in md_path.relative_to(root).parts):
            continue
        # Skip excluded subtrees
        if any(md_path.is_relative_to(ex) for ex in excludes):
            continue
        rel = str(md_path.relative_to(root))
        key = f"{label}:{rel}"
        result[key] = md_path
    return result


class KnrsIndexer:
    def __init__(self, config: KnrsConfig):
        self.config = config
        self.db_dir = config.vector_db
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.db_dir / "index.npy"
        self.meta_file  = self.db_dir / "index.json"

    # ------------------------------------------------------------------ #
    # Chunking                                                             #
    # ------------------------------------------------------------------ #

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
        """Simple character-based sliding-window chunking."""
        if len(text) <= chunk_size:
            return [text]
        chunks: list[str] = []
        start = 0
        while start < len(text):
            chunks.append(text[start:start + chunk_size])
            start += chunk_size - overlap
        return chunks

    # ------------------------------------------------------------------ #
    # State I/O                                                            #
    # ------------------------------------------------------------------ #

    def _load_state(self) -> tuple[np.ndarray, dict]:
        """Load existing index state; returns (embeddings, meta_dict)."""
        if self.index_file.exists() and self.meta_file.exists():
            embeddings = np.load(self.index_file)
            with self.meta_file.open("r", encoding="utf-8") as fh:
                meta = json.load(fh)
            # Back-compat: older indices had no file_hashes
            if "file_hashes" not in meta:
                meta["file_hashes"] = {}
            return embeddings, meta
        return np.array([], dtype=np.float32), {
            "model": self.config.embedder_name,
            "file_hashes": {},
            "chunks": [],
            "full_texts": [],
        }

    def _save_state(self, embeddings: np.ndarray, meta: dict) -> None:
        np.save(self.index_file, embeddings)
        with self.meta_file.open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

    # ------------------------------------------------------------------ #
    # Pruning                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _prune(
        embeddings: np.ndarray,
        meta: dict,
        keys_to_remove: set[str],
    ) -> tuple[np.ndarray, dict]:
        """Remove all chunks whose ``source_key`` is in *keys_to_remove*."""
        if not keys_to_remove:
            return embeddings, meta

        keep_indices = [
            i for i, ch in enumerate(meta["chunks"])
            if ch["source_key"] not in keys_to_remove
        ]

        new_embeddings = embeddings[keep_indices] if len(embeddings) else embeddings
        new_chunks     = [meta["chunks"][i]     for i in keep_indices]
        new_texts      = [meta["full_texts"][i] for i in keep_indices]

        meta = dict(meta)  # shallow copy
        meta["chunks"]     = new_chunks
        meta["full_texts"] = new_texts

        return new_embeddings, meta

    # ------------------------------------------------------------------ #
    # Main entry point                                                     #
    # ------------------------------------------------------------------ #

    def run_indexing(
        self,
        markdown_books_dir: Path,
        wiki_dir: Path,
        *,
        force: bool = False,
    ) -> None:
        """
        Differential index update.

        Args:
            markdown_books_dir: KnrsData/MarkdownBooks root.
            wiki_dir:           Wiki root (AINotes subtree is excluded).
            force:              If True, discard all existing state and re-index everything.
        """
        # ── 1. Load existing state ─────────────────────────────────────
        if force:
            logger.info("--force: discarding existing index state.")
            embeddings = np.array([], dtype=np.float32)
            meta: dict = {
                "model": self.config.embedder_name,
                "file_hashes": {},
                "chunks": [],
                "full_texts": [],
            }
        else:
            embeddings, meta = self._load_state()

        old_hashes: dict[str, str] = meta.get("file_hashes", {})

        # ── 2. Scan current sources ────────────────────────────────────
        logger.info("Scanning source directories...")
        current_files: dict[str, Path] = {}
        current_files.update(_scan_source(markdown_books_dir, _LABEL_BOOKS))
        current_files.update(
            _scan_source(
                wiki_dir,
                _LABEL_WIKI,
                excludes=[wiki_dir / "AINotes"],
            )
        )
        logger.info(
            "Found %d source files (%d books, %d wiki).",
            len(current_files),
            sum(1 for k in current_files if k.startswith(_LABEL_BOOKS + ":")),
            sum(1 for k in current_files if k.startswith(_LABEL_WIKI + ":")),
        )

        # ── 3. Compute current hashes ──────────────────────────────────
        logger.info("Hashing source files...")
        current_hashes: dict[str, str] = {}
        for key, path in current_files.items():
            try:
                current_hashes[key] = _hash_file(path)
            except OSError as exc:
                logger.error("Cannot hash %s: %s", path, exc)

        # ── 4. Diff ────────────────────────────────────────────────────
        removed  = set(old_hashes) - set(current_hashes)
        changed  = {k for k, h in current_hashes.items() if old_hashes.get(k) != h}
        to_embed = sorted(changed)
        to_prune = removed | (changed - set(to_embed))   # removed + stale

        logger.info(
            "Diff: %d new/changed, %d removed, %d unchanged.",
            len(changed), len(removed), len(current_hashes) - len(changed),
        )

        # ── 5. Prune stale / removed chunks ───────────────────────────
        if to_prune or (changed and not force):
            before = len(meta["chunks"])
            embeddings, meta = self._prune(embeddings, meta, removed | changed)
            logger.info(
                "Pruned %d chunks (was %d, now %d).",
                before - len(meta["chunks"]), before, len(meta["chunks"]),
            )

        # ── 6. Embed new / changed files ───────────────────────────────
        if not to_embed:
            logger.info("Index is up to date — nothing to embed.")
        else:
            new_chunks:    list[str]  = []
            new_chunk_meta: list[dict] = []

            for key in to_embed:
                path = current_files[key]
                try:
                    content = path.read_text(encoding="utf-8")
                    _, body = _split_frontmatter(content)
                    if len(body) < 100:
                        continue
                    chunks = self.chunk_text(body)
                    for i, chunk in enumerate(chunks):
                        new_chunks.append(chunk)
                        new_chunk_meta.append({
                            "source_key":  key,
                            "path":        key,          # kept as the display path
                            "chunk_index": i,
                            "text":        chunk[:200] + "...",
                        })
                except Exception as exc:
                    logger.error("Failed to process %s: %s", key, exc)

            if new_chunks:
                logger.info(
                    "Embedding %d new chunks from %d files using %s...",
                    len(new_chunks), len(to_embed), self.config.embedder_name,
                )
                new_embeddings = get_embeddings(new_chunks, self.config)

                # Extend arrays
                if len(embeddings) == 0:
                    embeddings = new_embeddings
                else:
                    embeddings = np.concatenate([embeddings, new_embeddings], axis=0)

                meta["chunks"]     += new_chunk_meta
                meta["full_texts"] += new_chunks

        # ── 7. Update file_hashes and save ────────────────────────────
        # Remove hashes for deleted files; update hashes for embedded files.
        new_file_hashes = {
            k: v for k, v in old_hashes.items() if k not in removed
        }
        for key in to_embed:
            if key in current_hashes:
                new_file_hashes[key] = current_hashes[key]

        meta["file_hashes"] = new_file_hashes
        meta["model"]       = self.config.embedder_name

        self._save_state(embeddings, meta)
        logger.info(
            "Index saved: %d total chunks across %d files in %s",
            len(meta["chunks"]), len(meta["file_hashes"]), self.db_dir,
        )
