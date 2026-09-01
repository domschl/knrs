"""
knrs.vector.indexer — Differential chunk-and-embed indexer.

Sources indexed:
  - KnrsData/MarkdownBooks  (prefix ``books:``)
  - Wiki/Notes & AINotes    (prefix ``wiki:``)

Differential strategy
---------------------
``index.json`` stores a ``file_hashes`` map of ``{prefixed_rel_path: sha256hex}``.
On each run:
  1. Load existing state (empty if first run or --force).
  2. Scan all source files; compute SHA-256.
  3. Diff → determine which files are new/changed and which are deleted.
  4. Prune chunks for deleted/changed files from existing arrays.
  5. Persist pruned state immediately (crash-safe baseline).
  6. Open a persistent EmbedderSession (model loads once).
  7. For each file to embed: chunk → session.embed() → extend arrays.
  8. Save a checkpoint every checkpoint_every_docs files or checkpoint_every_chunks chunks.

Storage layout (VectorDB/):
  index.npy   — (N, D) float32 embedding matrix
  index.json  — { model, file_hashes, chunks, full_texts }
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    MofNCompleteColumn, TaskProgressColumn,
    TimeElapsedColumn, TimeRemainingColumn,
)

from calibre.converter import _split_frontmatter
from vector.engine import EmbedderSession
from config import KnrsConfig

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

    def chunk_text(self, text: str) -> list[str]:
        """
        Simple character-based sliding-window chunking.

        Note on llama-server: Chunks must fit into the server's logical batch
        size (--ubatch-size, default 512). If 3000 chars exceed this (~750 tokens),
        increase --ubatch-size to 1024. --batch-size (physical batch)
        defaults to 2048 and must be >= --ubatch-size.
        """
        size = self.config.vector_chunk_size
        overlap = self.config.vector_chunk_overlap
        if len(text) <= size:
            return [text]
        chunks: list[str] = []
        start = 0
        while start < len(text):
            chunks.append(text[start:start + size])
            start += size - overlap
        return chunks

    # ------------------------------------------------------------------ #
    # State I/O                                                            #
    # ------------------------------------------------------------------ #

    def _load_state(self) -> tuple[np.ndarray, dict]:
        """Load existing index state; returns (embeddings, meta_dict)."""
        if self.index_file.exists() and self.meta_file.exists():
            try:
                if self.meta_file.stat().st_size == 0:
                    logger.warning("Index metadata file is empty. Treating as missing.")
                    raise ValueError("Empty index.json")
                
                embeddings = np.load(str(self.index_file))
                with self.meta_file.open("r", encoding="utf-8") as fh:
                    meta = json.load(fh)
                # Back-compat: older indices had no file_hashes or chunk_size info
                if "file_hashes" not in meta:
                    meta["file_hashes"] = {}
                if "chunk_size" not in meta:
                    meta["chunk_size"] = self.config.vector_chunk_size
                    meta["overlap"] = self.config.vector_chunk_overlap
                return embeddings, meta
            except Exception as e:
                logger.warning("Failed to load existing index: %s. Re-indexing will be required.", e)
                # If loading fails, we return empty state to trigger re-index
                pass
                
        return np.array([], dtype=np.float32), {
            "model": self.config.embedder_name,
            "chunk_size": self.config.vector_chunk_size,
            "overlap": self.config.vector_chunk_overlap,
            "file_hashes": {},
            "chunks": [],
            "full_texts": [],
        }

    def _save_state(self, embeddings: np.ndarray, meta: dict) -> None:
        """Atomically save the index and metadata."""
        tmp_npy = str(self.index_file) + ".tmp.npy"
        tmp_json = str(self.meta_file) + ".tmp"
        
        # 1. Write both temporary files first.
        np.save(tmp_npy, embeddings)
        with open(tmp_json, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
            
        # 2. Only if both writes succeeded, perform the atomic replaces.
        # This minimizes the window where the database could be inconsistent
        # (e.g. if writing the metadata fails due to disk space).
        os.replace(tmp_npy, str(self.index_file))
        os.replace(tmp_json, str(self.meta_file))

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
        checkpoint_every_docs: int | None = None,
        checkpoint_every_chunks: int | None = None,
    ) -> None:
        """
        Differential index update with crash-safe checkpointing.

        The embedder subprocess is launched **once** per run (via
        ``EmbedderSession``); the model is loaded into GPU memory on entry
        and stays warm for all files.  Each file's chunks are embedded
        immediately after reading, so memory usage stays bounded regardless
        of dataset size.

        A checkpoint (index.npy + index.json) is written every
        *checkpoint_every_docs* files or *checkpoint_every_chunks* chunks.
        On restart the hash-diff step automatically skips all files whose
        SHA-256 is already recorded, so the run resumes exactly where it left off.

        Args:
            markdown_books_dir:      KnrsData/MarkdownBooks root.
            wiki_dir:                Wiki root (AINotes subtree is excluded).
            force:                   Discard all existing state and re-index everything.
            checkpoint_every_docs:   Number of files between disk checkpoints.
            checkpoint_every_chunks: Number of chunks between disk checkpoints.
        """
        checkpoint_every_docs = checkpoint_every_docs or self.config.checkpoint_every_docs
        checkpoint_every_chunks = checkpoint_every_chunks or self.config.checkpoint_every_chunks
        # ── 1. Load existing state ─────────────────────────────────────
        if force:
            logger.info("--force: discarding existing index state.")
            embeddings = np.array([], dtype=np.float32)
            meta: dict = {
                "model": self.config.embedder_name,
                "chunk_size": self.config.vector_chunk_size,
                "overlap": self.config.vector_chunk_overlap,
                "file_hashes": {},
                "chunks": [],
                "full_texts": [],
            }
        else:
            embeddings, meta = self._load_state()
            # If the chunking parameters have changed, we must re-index everything
            if meta.get("chunk_size") != self.config.vector_chunk_size or meta.get("overlap") != self.config.vector_chunk_overlap:
                logger.warning(
                    "Chunking parameters changed (was %s/%s, now %s/%s). Discarding index state.",
                    meta.get("chunk_size", "unknown"), meta.get("overlap", "unknown"),
                    self.config.vector_chunk_size, self.config.vector_chunk_overlap
                )
                embeddings = np.array([], dtype=np.float32)
                meta = {
                    "model": self.config.embedder_name,
                    "chunk_size": self.config.vector_chunk_size,
                    "overlap": self.config.vector_chunk_overlap,
                    "file_hashes": {},
                    "chunks": [],
                    "full_texts": [],
                }

        old_hashes: dict[str, str] = meta.get("file_hashes", {})

        # ── 2. Scan current sources ────────────────────────────────────
        logger.info("Scanning source directories...")
        current_files: dict[str, Path] = {}
        current_files.update(_scan_source(markdown_books_dir, _LABEL_BOOKS))
        current_files.update(
            _scan_source(
                wiki_dir,
                _LABEL_WIKI,
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
        to_prune = removed | changed

        logger.info(
            "Diff: %d new/changed, %d removed, %d unchanged.",
            len(changed), len(removed), len(current_hashes) - len(changed),
        )

        # ── 5. Prune stale / removed chunks ───────────────────────────
        if to_prune:
            before = len(meta["chunks"])
            embeddings, meta = self._prune(embeddings, meta, to_prune)
            logger.info(
                "Pruned %d chunks (was %d, now %d).",
                before - len(meta["chunks"]), before, len(meta["chunks"]),
            )
            # Drop deleted-file hashes and persist immediately so a crash
            # after pruning doesn't re-add orphaned chunks on next run.
            meta["file_hashes"] = {
                k: v for k, v in meta["file_hashes"].items() if k not in removed
            }
            meta["model"] = self.config.embedder_name
            self._save_state(embeddings, meta)
            logger.info("Checkpoint: pruned state saved.")

        # ── 6. Per-file embedding with a persistent subprocess session ──
        if not to_embed:
            logger.info("Index is up to date — nothing to embed.")
            return {
                "files_indexed": 0,
                "files_failed": 0,
                "chunks_indexed": 0,
                "total_chunks": len(meta["chunks"]),
                "total_files": len(meta["file_hashes"]),
                "pruned_chunks": len(to_prune),
                "failed_items": [],
            }

        total_files     = len(to_embed)
        grand_total     = len(current_hashes)
        already_indexed = grand_total - len(changed)

        logger.info(
            "Overall progress: %d/%d files already indexed, %d to embed.",
            already_indexed, grand_total, total_files,
        )

        # ── 6. Pre-calculate chunk counts for better progress estimation ──
        to_embed_with_counts: list[tuple[str, int]] = []
        total_chunks = 0
        if to_embed:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                refresh_per_second=10,
                transient=True,
            ) as scanning:
                scan_task = scanning.add_task("Pre-scanning chunks...", total=total_files)
                for key in to_embed:
                    path = current_files[key]
                    try:
                        content = path.read_text(encoding="utf-8")
                        _, body = _split_frontmatter(content)
                        c_count = len(self.chunk_text(body)) if len(body) >= 100 else 0
                        to_embed_with_counts.append((key, c_count))
                        total_chunks += c_count
                    except Exception as exc:
                        logger.error("Failed to count chunks for %s: %s", key, exc)
                        to_embed_with_counts.append((key, 0))
                    scanning.advance(scan_task)

        total_chunks_indexed = len(meta["chunks"])
        logger.info(
            "Chunk progress: %d chunks already indexed, %d to embed (%d total).",
            total_chunks_indexed, total_chunks, total_chunks_indexed + total_chunks,
        )

        if total_chunks == 0:
            total_chunks = total_files  # Fallback for time estimation

        progress_columns = (
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )

        # EmbedderSession launches the subprocess once; the model stays
        # loaded in GPU memory for all files.  Each session.embed() call
        # sends one file's chunks via a temp file + stdin, so memory usage
        # is bounded by the largest single file, not the whole corpus.
        success_count = 0
        failed_items: list[str] = []
        new_chunks_count = 0

        with EmbedderSession(self.config) as session:
            with Progress(*progress_columns, refresh_per_second=4) as progress:
                task = progress.add_task(
                    f"Indexing [{self.config.embedder_name}]",
                    total=total_chunks,
                )

                chunks_since_checkpoint = 0
                for file_num, (key, c_count) in enumerate(to_embed_with_counts, 1):
                    path = current_files[key]
                    try:
                        content = path.read_text(encoding="utf-8")
                        _, body = _split_frontmatter(content)
                        chunks = self.chunk_text(body) if len(body) >= 100 else []

                        fname = Path(key).name
                        fname = (fname[:37] + "…") if len(fname) > 38 else fname.ljust(38)
                        progress.update(
                            task,
                            description=f"[{file_num:{len(str(total_files))}}/{total_files}] {fname}",
                        )

                        if chunks:
                            # Process in batches for smoother progress updates on large files
                            file_embs = []
                            INTERNAL_BATCH = 1024
                            for i in range(0, len(chunks), INTERNAL_BATCH):
                                batch = chunks[i : i + INTERNAL_BATCH]
                                batch_emb = session.embed(batch, encode_mode="document")
                                file_embs.append(batch_emb)
                                
                                progress.advance(task, len(batch))
                                chunks_since_checkpoint += len(batch)

                            new_emb = np.concatenate(file_embs, axis=0)
                            if len(embeddings) == 0:
                                embeddings = new_emb
                            else:
                                embeddings = np.concatenate(
                                    [embeddings, new_emb], axis=0
                                )
                            for i, chunk in enumerate(chunks):
                                meta["chunks"].append({
                                    "source_key":  key,
                                    "path":        key,
                                    "chunk_index": i,
                                    "text":        chunk[:200] + "...",
                                })
                                meta["full_texts"].append(chunk)
                            new_chunks_count += len(chunks)
                        else:
                            # Empty file or too small, just advance by 1 if we are in fallback mode
                            if total_chunks == total_files:
                                progress.advance(task, 1)

                        if key in current_hashes:
                            meta["file_hashes"][key] = current_hashes[key]

                        success_count += 1

                    except Exception as exc:
                        logger.error("Failed to process %s: %s", key, exc)
                        failed_items.append(f"{key}: {exc}")

                    # ── Checkpoint every N files OR every M chunks ──────────
                    if (file_num % checkpoint_every_docs == 0 or 
                        file_num == total_files or 
                        chunks_since_checkpoint >= checkpoint_every_chunks):
                        
                        meta["model"] = self.config.embedder_name
                        self._save_state(embeddings, meta)
                        chunks_since_checkpoint = 0
                        
                        pct = file_num * 100 // total_files if total_files else 100
                        progress.console.log(
                            f"Checkpoint: {file_num}/{total_files} files "
                            f"({pct}%), {len(meta['chunks'])} total chunks."
                        )

        logger.info(
            "Indexing complete: %d total chunks across %d files in %s",
            len(meta["chunks"]), len(meta["file_hashes"]), self.db_dir,
        )
        return {
            "files_indexed": success_count,
            "files_failed": len(failed_items),
            "chunks_indexed": new_chunks_count,
            "total_chunks": len(meta["chunks"]),
            "total_files": len(meta["file_hashes"]),
            "failed_items": failed_items,
        }
