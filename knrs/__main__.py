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
    sync_cal_p.add_argument("--force", action="store_true", help="Override git safety checks.")
    sync_cal_p.add_argument("--concurrency", type=int, default=1)

    # sync-summaries
    sync_sum_p = subparsers.add_parser("sync-summaries", help="Sync MarkdownBooks to Summaries.")
    sync_sum_p.add_argument("--dry-run", action="store_true")
    sync_sum_p.add_argument("--force", action="store_true", help="Override git safety checks.")
    sync_sum_p.add_argument("--concurrency", type=int, default=1)

    # sync-wiki
    sync_wiki_p = subparsers.add_parser("sync-wiki", help="Sync KnrsData to Wiki/AINotes.")
    sync_wiki_p.add_argument("--force", action="store_true", help="Override git safety checks.")

    # timeline
    timeline_p = subparsers.add_parser("timeline", help="Extract timelines from Wiki/Notes.")
    timeline_p.add_argument("--force", action="store_true", help="Override git safety checks.")

    # index
    index_p = subparsers.add_parser("index", help="Update VectorDB index.")
    index_p.add_argument(
        "--force", action="store_true",
        help="Discard existing index and re-embed everything from scratch (also overrides git safety)."
    )
    index_p.add_argument(
        "--checkpoint-every", type=int, default=50, metavar="N",
        help="Save a checkpoint after every N files (default: 50)."
    )

    # search
    search_p = subparsers.add_parser("search", help="Search VectorDB.")
    search_p.add_argument("query", nargs="+")
    search_p.add_argument("--raw", action="store_true", help="Output raw markdown without formatting")
    search_p.add_argument("--highlight", action="store_true", help="Highlight significant text passages")

    # sync (full pipeline)
    sync_p = subparsers.add_parser("sync", help="Run full sync pipeline (calibre -> summaries -> wiki -> timeline -> index -> external-lib -> check-wiki).")
    sync_p.add_argument("--dry-run", action="store_true")
    sync_p.add_argument("--force", action="store_true", help="Override git safety checks.")
    sync_p.add_argument("--concurrency", type=int, default=1)
    sync_p.add_argument("--checkpoint-every", type=int, default=50, metavar="N")

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
        from knrs.repl.commands import init_git_state, GIT_STATE
        init_git_state(cfg)
        if not args.dry_run and not args.force and not GIT_STATE["knrs_data_safe_local"]:
            print(f"Error: {cfg.knrs_data} is a git repo but not up-to-date. Use --force to override.", file=sys.stderr)
            sys.exit(1)
        run_sync(cfg, dry_run=args.dry_run, concurrency=args.concurrency)
    
    elif args.command == "sync-summaries":
        from knrs.summarizer.sync import run_summary_sync
        from knrs.repl.commands import init_git_state, GIT_STATE
        init_git_state(cfg)
        if not args.dry_run and not args.force and not GIT_STATE["knrs_data_safe_local"]:
            print(f"Error: {cfg.knrs_data} is a git repo but not up-to-date. Use --force to override.", file=sys.stderr)
            sys.exit(1)
        run_summary_sync(cfg, dry_run=args.dry_run, concurrency=args.concurrency)
    
    elif args.command == "sync-wiki":
        from knrs.wiki.sync import run_wiki_sync, inject_frontmatter_in_notes
        from knrs.repl.commands import init_git_state, GIT_STATE
        init_git_state(cfg)
        if not args.force and not GIT_STATE["wiki_path_safe_local"]:
            print(f"Error: {cfg.wiki_path} is a git repo but not up-to-date. Use --force to override.", file=sys.stderr)
            sys.exit(1)
        inject_frontmatter_in_notes(cfg.notes_path, cfg.wiki_path)
        run_wiki_sync(cfg)
        
    elif args.command == "timeline":
        from knrs.timelines.extractor import run_extraction
        from knrs.repl.commands import init_git_state, GIT_STATE
        init_git_state(cfg)
        if not args.force and not GIT_STATE["knrs_data_safe_local"]:
            print(f"Error: {cfg.knrs_data} is a git repo but not up-to-date. Use --force to override.", file=sys.stderr)
            sys.exit(1)
        run_extraction(cfg.notes_path, cfg.timelines / "timelines.json")
        
    elif args.command == "index":
        from knrs.vector.indexer import KnrsIndexer
        from knrs.repl.commands import init_git_state, GIT_STATE
        init_git_state(cfg)
        if not args.force and not GIT_STATE["knrs_data_safe_remote"]:
            print(f"Error: {cfg.knrs_data} is a git repo but not up-to-date (remote check enabled). Use --force to override.", file=sys.stderr)
            sys.exit(1)
        if not args.force and not GIT_STATE["wiki_path_safe_remote"]:
            print(f"Error: {cfg.wiki_path} is a git repo but not up-to-date (remote check enabled). Use --force to override.", file=sys.stderr)
            sys.exit(1)
        KnrsIndexer(cfg).run_indexing(
            cfg.markdown_books,
            cfg.wiki_path,
            force=args.force,
            checkpoint_every=args.checkpoint_every,
        )
        
    elif args.command == "search":
        from knrs.vector.search import KnrsSearcher, get_context_aware_text, get_significance
        query = " ".join(args.query)
        searcher = KnrsSearcher(cfg)
        try:
            results = searcher.search(query)
            processed_results = []
            
            for i, r in enumerate(results, 1):
                doc_name = Path(r.bare_path).name
                header = f"### {i}. {doc_name}\n"
                header += f"**Path:** `{r.bare_path}` | **Source:** `{r.source_label}` | **Score:** `{r.score:.4f}`\n"
                chunk_text = get_context_aware_text(searcher, r)
                processed_results.append([header, chunk_text, r])
                
            if args.highlight and results:
                from knrs.vector.engine import EmbedderSession
                with EmbedderSession(cfg) as session:
                    for idx, (header, chunk_text, r) in enumerate(processed_results):
                        if r.query_embedding is not None:
                            chunk_text = get_significance(chunk_text, r.query_embedding, searcher, raw=args.raw, session=session)
                            processed_results[idx][1] = chunk_text
                            
            for header, chunk_text, r in processed_results:
                print(header)
                print(chunk_text)
                print("\n" + "-" * 40 + "\n")
                
        except FileNotFoundError:
            print("Error: Index not found. Run indexer first.", file=sys.stderr)
            sys.exit(1)

    elif args.command == "sync":
        from knrs.repl.commands import init_git_state, GIT_STATE, cmd_sync
        init_git_state(cfg)
        
        # Prepare mock args list from argparse to pass to cmd_sync
        mock_args = []
        if args.dry_run:
            mock_args.append("--dry-run")
        if args.force:
            mock_args.append("--force")
        mock_args.extend(["--checkpoint-every", str(args.checkpoint_every)])
        
        cmd_sync(mock_args, cfg)

    elif args.command in (None, "repl"):
        from knrs.repl.repl import run_repl
        run_repl(cfg)

if __name__ == "__main__":
    main()
