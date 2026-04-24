"""
knrs.__main__ — Entry point for `uv run python -m knrs`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from knrs import __version__
from knrs.logging_setup import setup_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="knrs",
        description="LLM-enabled knowledge-base wiki — knrs",
    )
    parser.add_argument(
        "--version", action="version", version=f"knrs {__version__}"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging."
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="Override config file path (default: ~/.config/knrs/knrs.json).",
    )
    
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # config
    subparsers.add_parser("config", help="Print the resolved configuration and exit.")

    # sync-calibre
    sync_cal_p = subparsers.add_parser("sync-calibre", help="Sync Calibre to MarkdownBooks.")
    sync_cal_p.add_argument("--dry-run", action="store_true")
    sync_cal_p.add_argument("--concurrency", type=int, default=1)

    # sync-summaries
    sync_sum_p = subparsers.add_parser("sync-summaries", help="Sync MarkdownBooks to Summaries.")
    sync_sum_p.add_argument("--dry-run", action="store_true")
    sync_sum_p.add_argument("--concurrency", type=int, default=1)

    # sync-wiki
    subparsers.add_parser("sync-wiki", help="Sync KnrsData to Wiki/AINotes.")

    # timeline
    subparsers.add_parser("timeline", help="Extract timelines from Wiki/Notes.")

    # index
    subparsers.add_parser("index", help="Update VectorDB index.")

    # search
    search_p = subparsers.add_parser("search", help="Search VectorDB.")
    search_p.add_argument("query", nargs="+")

    # migrate
    migrate_p = subparsers.add_parser("migrate", help="Migrate data from legacy Summarizer project.")
    migrate_p.add_argument("--execute", action="store_true", help="Perform the migration (default is dry-run).")

    # repl
    subparsers.add_parser("repl", help="Start the interactive REPL (default).")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    setup_logging(verbose=args.verbose)

    from knrs.config import load_config
    cfg_path = Path(args.config) if args.config else None
    
    try:
        cfg = load_config(cfg_path)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)


    if args.command == "config":
        from knrs.config import print_config
        print_config(cfg)
    
    elif args.command == "sync-calibre":
        from knrs.calibre.sync import run_sync
        run_sync(cfg, dry_run=args.dry_run, concurrency=args.concurrency)
    
    elif args.command == "sync-summaries":
        from knrs.summarizer.sync import run_summary_sync
        run_summary_sync(cfg, dry_run=args.dry_run, concurrency=args.concurrency)
    
    elif args.command == "sync-wiki":
        from knrs.wiki.sync import run_wiki_sync, inject_uuids_in_notes
        inject_uuids_in_notes(cfg.notes_path)
        run_wiki_sync(cfg)
        
    elif args.command == "timeline":
        from knrs.timelines.extractor import run_extraction
        run_extraction(cfg.notes_path, cfg.timelines / "timelines.json")
        
    elif args.command == "index":
        from knrs.vector.indexer import KnrsIndexer
        KnrsIndexer(cfg).run_indexing(cfg.markdown_books)
        
    elif args.command == "search":
        from knrs.vector.search import KnrsSearcher
        query = " ".join(args.query)
        results = KnrsSearcher(cfg).search(query)
        for r in results:
            print(f"[{r.score:.4f}] {r.path}\n{r.text[:200]}...\n")

    elif args.command == "migrate":
        from knrs.migration.migrate import run_migration
        run_migration(cfg, dry_run=not args.execute)
            
    elif args.command in (None, "repl"):
        from knrs.repl.repl import run_repl
        run_repl(cfg)

if __name__ == "__main__":
    main()
