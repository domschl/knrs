"""
knrs.migration.migrate — One-time migration from Summarizer to KnrsData.

Reads ~/.config/knrs/summarizer_config.json and plans moves of
MarkdownBooks and BookSummaries to the new KnrsData structure.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

from knrs.config import KnrsConfig
from knrs.naming import capitalize_series
from knrs.paths import resolve

logger = logging.getLogger(__name__)

def run_migration(cfg: KnrsConfig, dry_run: bool = True):
    """
    Migrate data from legacy Summarizer project.
    
    Args:
        cfg:     New KnrsConfig.
        dry_run: If True, only log planned moves.
    """
    old_config_path = resolve("~/.config/knrs/summarizer_config.json")
    if not old_config_path.exists():
        logger.error("Legacy summarizer config not found at %s. Migration aborted.", old_config_path)
        return

    try:
        with old_config_path.open('r', encoding='utf-8') as f:
            old_cfg = json.load(f)
    except Exception as e:
        logger.error("Failed to parse legacy config: %s", e)
        return

    old_md_root = resolve(old_cfg.get("markdown_path", ""))
    old_sum_root = resolve(old_cfg.get("summaries_path", ""))
    
    if not old_md_root.exists():
        logger.warning("Old markdown path %s does not exist.", old_md_root)
    else:
        logger.info("Planning migration of MarkdownBooks from %s to %s", old_md_root, cfg.markdown_books)
        _migrate_dir(old_md_root, cfg.markdown_books, dry_run)

    if not old_sum_root.exists():
        logger.warning("Old summaries path %s does not exist.", old_sum_root)
    else:
        logger.info("Planning migration of BookSummaries from %s to %s", old_sum_root, cfg.book_summaries)
        _migrate_dir(old_sum_root, cfg.book_summaries, dry_run)

    if dry_run:
        logger.info("Dry-run complete. No files moved. Run with --execute to perform migration.")
    else:
        logger.info("Migration complete.")

def _migrate_dir(src_root: Path, dst_root: Path, dry_run: bool):
    """Helper to migrate all files from one directory to another, preserving structure."""
    for src_path in src_root.rglob("*"):
        if src_path.is_dir():
            continue
            
        rel_path = src_path.relative_to(src_root)
        parts = list(rel_path.parts)
        if len(parts) > 1:
            # Capitalize the first folder (the series name)
            parts[0] = capitalize_series(parts[0])
            rel_path = Path(*parts)
            
        dst_path = dst_root / rel_path
        
        if dry_run:
            logger.info("[MIGRATE] git mv '%s' '%s'", src_path, dst_path)
        else:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            if dst_path.exists():
                logger.warning("Destination exists, skipping: %s", dst_path)
                continue
            shutil.move(src_path, dst_path)
            logger.debug("Moved %s -> %s", src_path, dst_path)
            
            # Post-move cleanup for migrated markdowns: remove legacy base64 'icon'
            if dst_path.suffix == ".md":
                from knrs.calibre.converter import update_frontmatter_inplace
                try:
                    update_frontmatter_inplace(dst_path, {}, remove_fields=["icon"])
                except Exception as e:
                    logger.warning("Failed to clean frontmatter for %s: %s", dst_path, e)
