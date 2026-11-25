"""
knrs.external_lib.sync — Synchronize Calibre library EPUB/PDF to an external folder.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import stat
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from calibre.library import scan_calibre_library
from config import KnrsConfig
from naming import generate_external_lib_filename

logger = logging.getLogger(__name__)

CATALOG_FILENAME = ".knrs-catalog.json"


def filename_md5(filename: str) -> str:
    """Calculate the standard MD5 hash of a UTF-8 filename string (matching KOReader)."""
    return hashlib.md5(filename.encode("utf-8")).hexdigest()


def partial_koreader_md5(filepath: Path) -> str | None:
    """Calculate the partial MD5 used by KOReader to identify document content.

    Matches KOReader frontend/util.lua: reads 1024-byte chunks at offset 0 and
    1024 << (2 * i) for i in range(0, 11).
    """
    try:
        st = filepath.stat()
        if not stat.S_ISREG(st.st_mode):
            return None
        with open(filepath, "rb") as f:
            step = 1024
            size = 1024
            md5_hash = hashlib.md5()
            for i in range(-1, 11):
                exponent = 2 * i
                offset = 0 if exponent < 0 else (step << exponent)
                f.seek(offset, os.SEEK_SET)
                sample = f.read(size)
                if sample:
                    md5_hash.update(sample)
                else:
                    break
            return md5_hash.hexdigest()
    except OSError as exc:
        logger.warning("Failed to compute KOReader partial MD5 for '%s': %s", filepath, exc)
        return None


def _write_catalog_if_changed(catalog_path: Path, books_map: dict[str, dict[str, Any]]) -> bool:
    """Atomically write .knrs-catalog.json only when the books payload has changed."""
    if catalog_path.exists():
        try:
            existing = json.loads(catalog_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict) and existing.get("version") == 1 and existing.get("books") == books_map:
                logger.debug("Catalog %s is already up to date (%d books).", catalog_path, len(books_map))
                return False
        except Exception:
            pass

    payload = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "books": books_map,
    }
    tmp_path = catalog_path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    os.replace(tmp_path, catalog_path)
    logger.info("Wrote MetaLibrary catalog %s (%d books).", catalog_path, len(books_map))
    return True


def _find_book_files(book_dir: Path) -> list[Path]:
    """Find EPUB and/or PDF files in the Calibre book directory."""
    files = {f.suffix.lower(): f for f in book_dir.iterdir() if f.is_file()}
    found = []
    if ".epub" in files:
        found.append(files[".epub"])
    if ".pdf" in files:
        found.append(files[".pdf"])
    return found

def run_external_sync(cfg: KnrsConfig, *, dry_run: bool = False) -> dict[str, Any]:
    """
    Sync EPUB/PDF files from Calibre to the external library and write .knrs-catalog.json.
    Target format: <external_library>/<series>/<Author> - <Title>.<ext>
    """
    if not cfg.external_library.exists():
        logger.warning("External library directory does not exist: %s", cfg.external_library)
        logger.warning("Aborting /sync-external-lib.")
        return {
            "success_count": 0,
            "failure_count": 0,
            "total": 0,
            "copied": 0,
            "updated": 0,
            "removed_files": 0,
            "removed_dirs": 0,
            "catalog_books": 0,
            "failed_items": ["External library directory does not exist"],
        }

    logger.info("Scanning Calibre library at %s", cfg.calibre_path)
    calibre_index = scan_calibre_library(cfg.calibre_path, cfg.target_series)
    
    catalog_path = cfg.external_library / CATALOG_FILENAME
    valid_paths: set[str] = {
        unicodedata.normalize("NFC", str(catalog_path)),
    }
    actions = []
    books_map: dict[str, dict[str, Any]] = {}

    for uuid, book in sorted(calibre_index.items()):
        source_files = _find_book_files(book.book_dir)
        file_entries: list[dict[str, Any]] = []
        for source_file in source_files:
            ext = source_file.suffix.lower()
            base_name = generate_external_lib_filename(book.title, book.first_author)
            target_name = unicodedata.normalize("NFC", f"{base_name}{ext}")
            target_path = cfg.external_library / book.series_dir / target_name
            
            valid_paths.add(unicodedata.normalize("NFC", str(target_path)))
            valid_paths.add(unicodedata.normalize("NFC", str(target_path.parent)))  # Ensure series dir is valid
            
            if not target_path.exists():
                actions.append(("COPY", source_file, target_path))
            elif target_path.stat().st_size != source_file.stat().st_size:
                actions.append(("UPDATE", source_file, target_path))

            rel_path = unicodedata.normalize(
                "NFC", target_path.relative_to(cfg.external_library).as_posix()
            )
            file_entries.append({
                "rel_path": rel_path,
                "format": ext.lstrip("."),
                "filename_md5": filename_md5(target_name),
                "koreader_partial_md5": partial_koreader_md5(source_file),
            })

        if file_entries:
            books_map[uuid] = {
                "uuid": uuid,
                "title": book.title,
                "authors": list(book.authors),
                "series": book.series_dir,
                "tags": list(book.tags),
                "calibre_id": book.calibre_id,
                "files": file_entries,
            }

    # Cleanup phase
    for p in cfg.external_library.rglob("*"):
        if ".stfolder" in p.parts:
            continue
        norm_p = unicodedata.normalize("NFC", str(p))
        if p.is_file() and norm_p not in valid_paths:
            actions.append(("REMOVE_FILE", None, p))

    stats = {"COPY": 0, "UPDATE": 0, "REMOVE_FILE": 0, "REMOVE_DIR": 0}

    if dry_run:
        logger.info("[dry-run] Syncing external library (no files written)...")
        for action, src, dst in actions:
            stats[action] += 1
            if action in ("COPY", "UPDATE") and src:
                logger.info("  %s %s to %s", action, src.name, dst)
            elif action == "REMOVE_FILE":
                logger.info("  %s %s", action, dst)
                
        logger.info(
            "[dry-run] Sync plan complete. Would Add: %d, Update: %d, Delete Files: %d, Catalog Books: %d",
            stats["COPY"], stats["UPDATE"], stats["REMOVE_FILE"], len(books_map),
        )
        return {
            "success_count": len(actions),
            "failure_count": 0,
            "total": len(actions),
            "copied": stats["COPY"],
            "updated": stats["UPDATE"],
            "removed_files": stats["REMOVE_FILE"],
            "removed_dirs": 0,
            "catalog_books": len(books_map),
            "failed_items": [],
        }

    success_count = 0
    failure_count = 0
    failed_items: list[str] = []

    if actions:
        logger.info("Executing %d action(s)...", len(actions))
    for action, src, dst in actions:
        try:
            if action in ("COPY", "UPDATE") and src:
                dst.parent.mkdir(parents=True, exist_ok=True)
                logger.info("Copying %s to %s", src.name, dst)
                shutil.copy2(src, dst)
                stats[action] += 1
                success_count += 1
            elif action == "REMOVE_FILE":
                logger.info("Removing debris file: %s", dst)
                dst.unlink(missing_ok=True)
                stats[action] += 1
                success_count += 1
        except Exception as exc:
            logger.error("Error executing %s for %s: %s", action, dst, exc)
            failure_count += 1
            failed_items.append(f"{action}: {dst.name} ({exc})")

    # Clean empty directories
    for p in sorted(cfg.external_library.rglob("*"), key=lambda x: len(x.parts), reverse=True):
        if ".stfolder" in p.parts:
            continue
        if p.is_dir() and not any(p.iterdir()):
            try:
                stats["REMOVE_DIR"] += 1
                logger.info("Removing empty directory: %s", p)
                p.rmdir()
            except Exception as exc:
                logger.error("Error removing empty dir %s: %s", p, exc)

    try:
        _write_catalog_if_changed(catalog_path, books_map)
    except Exception as exc:
        logger.error("Failed to write MetaLibrary catalog %s: %s", catalog_path, exc)
        failure_count += 1
        failed_items.append(f"CATALOG: {CATALOG_FILENAME} ({exc})")

    logger.info(
        "External library sync complete. Added: %d, Updated: %d, Deleted Files: %d, Deleted Dirs: %d, Catalog Books: %d",
        stats["COPY"], stats["UPDATE"], stats["REMOVE_FILE"], stats["REMOVE_DIR"], len(books_map),
    )
    return {
        "success_count": success_count,
        "failure_count": failure_count,
        "total": len(actions),
        "copied": stats["COPY"],
        "updated": stats["UPDATE"],
        "removed_files": stats["REMOVE_FILE"],
        "removed_dirs": stats["REMOVE_DIR"],
        "catalog_books": len(books_map),
        "failed_items": failed_items,
    }
