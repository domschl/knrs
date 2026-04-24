#!/usr/bin/env python3
"""
One-time migration: Upgrade BookSummaries to use body-only hashes.

Recalculates the content-only (body) hash for all MarkdownBooks and updates the
source_md_hash field in corresponding BookSummaries. Also ensures tags are
in-sync from the book to the summary.
"""

import logging
import sys
from pathlib import Path

# Add the project root to sys.path so we can import knrs
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from knrs.config import load_config
from knrs.summarizer.sync import scan_existing_summaries, scan_markdown_sources
from knrs.calibre.converter import update_frontmatter_inplace

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("migration")


def run_migration(dry_run: bool = False):
    try:
        cfg = load_config()
    except Exception as e:
        logger.error("Could not load config: %s", e)
        return

    logger.info("Scanning MarkdownBooks at %s...", cfg.markdown_books)
    # This now uses compute_markdown_content_hash internally
    md_index = scan_markdown_sources(cfg.markdown_books, [])

    logger.info("Scanning BookSummaries at %s...", cfg.book_summaries)
    sum_index = scan_existing_summaries(cfg.book_summaries)

    updated_count = 0
    common_uuids = set(md_index) & set(sum_index)

    logger.info("Found %d matching books/summaries to check.", len(common_uuids))

    for uuid in sorted(common_uuids):
        mi = md_index[uuid]
        si = sum_index[uuid]

        new_hash = mi["content_hash"]
        old_hash = si["source_md_hash"]
        
        md_tags = mi["metadata"].get("tags", [])
        si_tags = si["metadata"].get("tags", [])

        needs_update = (new_hash != old_hash) or (md_tags != si_tags)

        if needs_update:
            logger.info("Updating [%s]: %s", uuid[:8], mi["title"])
            if new_hash != old_hash:
                logger.info("  Hash: %s -> %s", old_hash[:8] if old_hash else "None", new_hash[:8])
            if md_tags != si_tags:
                logger.info("  Tags: %s -> %s", si_tags, md_tags)

            if not dry_run:
                updates = {
                    "source_md_hash": new_hash,
                    "tags": md_tags,
                }
                update_frontmatter_inplace(si["path"], updates)
            updated_count += 1

    if dry_run:
        logger.info("[dry-run] Would have updated %d summaries.", updated_count)
    else:
        logger.info("Migration complete. Updated %d summaries.", updated_count)


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    run_migration(dry_run=dry)
