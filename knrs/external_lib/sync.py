"""
knrs.external_lib.sync — Synchronize Calibre library EPUB/PDF to an external folder.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from knrs.calibre.library import scan_calibre_library
from knrs.config import KnrsConfig
from knrs.naming import generate_external_lib_filename

logger = logging.getLogger(__name__)

def _find_book_file(book_dir: Path) -> Path | None:
    """Find an EPUB or PDF file in the Calibre book directory."""
    files = {f.suffix.lower(): f for f in book_dir.iterdir() if f.is_file()}
    if ".epub" in files:
        return files[".epub"]
    if ".pdf" in files:
        return files[".pdf"]
    return None

def run_external_sync(cfg: KnrsConfig, *, dry_run: bool = False) -> None:
    """
    Sync EPUB/PDF files from Calibre to the external library.
    Target format: <external_library>/<series>/<Author> - <Title>.<ext>
    """
    if not cfg.external_library.exists():
        logger.warning("External library directory does not exist: %s", cfg.external_library)
        logger.warning("Aborting /sync-external-lib.")
        return

    logger.info("Scanning Calibre library at %s", cfg.calibre_path)
    calibre_index = scan_calibre_library(cfg.calibre_path, cfg.target_series)
    
    valid_paths: set[Path] = set()
    actions = []

    for uuid, book in calibre_index.items():
        source_file = _find_book_file(book.book_dir)
        if not source_file:
            continue
            
        ext = source_file.suffix.lower()
        base_name = generate_external_lib_filename(book.title, book.first_author)
        target_name = f"{base_name}{ext}"
        target_path = cfg.external_library / book.series_dir / target_name
        
        valid_paths.add(target_path)
        valid_paths.add(target_path.parent)  # Ensure series dir is valid
        
        if not target_path.exists():
            actions.append(("COPY", source_file, target_path))
        elif target_path.stat().st_size != source_file.stat().st_size:
            actions.append(("UPDATE", source_file, target_path))

    # Cleanup phase
    for p in cfg.external_library.rglob("*"):
        if ".stfolder" in p.parts:
            continue
        if p.is_file() and p not in valid_paths:
            actions.append(("REMOVE_FILE", None, p))

    stats = {"COPY": 0, "UPDATE": 0, "REMOVE_FILE": 0, "REMOVE_DIR": 0}

    if dry_run:
        logger.info("[dry-run] Syncing external library (no files written)...")
        for action, src, dst in actions:
            stats[action] += 1
            if action in ("COPY", "UPDATE"):
                logger.info("  %s %s to %s", action, src.name, dst)
            elif action == "REMOVE_FILE":
                logger.info("  %s %s", action, dst)
                
        logger.info(
            "[dry-run] Sync plan complete. Would Add: %d, Update: %d, Delete Files: %d",
            stats["COPY"], stats["UPDATE"], stats["REMOVE_FILE"]
        )
        return

    if actions:
        logger.info("Executing %d action(s)...", len(actions))
    for action, src, dst in actions:
        stats[action] += 1
        if action in ("COPY", "UPDATE"):
            dst.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Copying %s to %s", src.name, dst)
            shutil.copy2(src, dst)
        elif action == "REMOVE_FILE":
            logger.info("Removing debris file: %s", dst)
            dst.unlink(missing_ok=True)

    # Clean empty directories
    for p in sorted(cfg.external_library.rglob("*"), key=lambda x: len(x.parts), reverse=True):
        if ".stfolder" in p.parts:
            continue
        if p.is_dir() and not any(p.iterdir()):
            stats["REMOVE_DIR"] += 1
            logger.info("Removing empty directory: %s", p)
            p.rmdir()

    logger.info(
        "External library sync complete. Added: %d, Updated: %d, Deleted Files: %d, Deleted Dirs: %d",
        stats["COPY"], stats["UPDATE"], stats["REMOVE_FILE"], stats["REMOVE_DIR"]
    )
