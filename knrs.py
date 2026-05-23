"""
knrs.__main__ — Entry point for `uv run python -m knrs`.
"""

from __future__ import annotations

__version__ = "0.1.0"

import argparse
import sys
from pathlib import Path

from logging_setup import setup_logging


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
    timeline_p = subparsers.add_parser("timeline", help="Extract and filter timelines from Wiki/Notes.")
    timeline_p.add_argument("--force", action="store_true", help="Override git safety checks.")
    timeline_p.add_argument("--from", dest="from_date", help="Filter events starting from this date.")
    timeline_p.add_argument("--to", dest="to_date", help="Filter events ending at this date.")
    timeline_p.add_argument("--context", action="append", help="Filter by context (can be specified multiple times).")
    timeline_p.add_argument("--raw", action="store_true", help="Output raw markdown table.")
    timeline_p.add_argument("keywords", nargs="*", help="Keyword search patterns.")

    # index
    index_p = subparsers.add_parser("index", help="Update VectorDB index.")
    index_p.add_argument(
        "--force", action="store_true",
        help="Discard existing index and re-embed everything from scratch (also overrides git safety)."
    )
    index_p.add_argument(
        "--checkpoint-every-docs", type=int, metavar="N",
        help="Save a checkpoint after every N files (default: from config)."
    )
    index_p.add_argument(
        "--checkpoint-every-chunks", type=int, metavar="M",
        help="Save a checkpoint after every M chunks (default: from config)."
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
    sync_p.add_argument("--checkpoint-every-docs", type=int, metavar="N")
    sync_p.add_argument("--checkpoint-every-chunks", type=int, metavar="M")

    # benchmark
    benchmark_p = subparsers.add_parser("benchmark", help="Run testing and benchmark framework for subprocesses.")
    benchmark_p.add_argument("--type", choices=["converter", "summarizer", "embedder", "agent"], help="Run only a specific category of subprocesses.")
    benchmark_p.add_argument("--backend", help="Run benchmark for a single specific backend (e.g. embedder_hf).")
    benchmark_p.add_argument("--results", help="Path to write the results JSON (default: benchmark_results.json in workspace root).")

    # repl
    subparsers.add_parser("repl", help="Start the interactive REPL (default).")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    setup_logging(verbose=args.verbose)

    from config import load_config
    cfg_path = Path(args.config) if args.config else None
    
    try:
        cfg = load_config(cfg_path)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)


    if args.command == "config":
        from config import print_config
        print_config(cfg)
    
    elif args.command == "sync-calibre":
        from calibre.sync import run_sync
        from repl.commands import init_git_state, GIT_STATE
        init_git_state(cfg)
        if not args.dry_run and not args.force and not GIT_STATE["knrs_data_safe_local"]:
            print(f"Error: {cfg.knrs_data} is a git repo but not up-to-date. Use --force to override.", file=sys.stderr)
            sys.exit(1)
        run_sync(cfg, dry_run=args.dry_run, concurrency=args.concurrency)
    
    elif args.command == "sync-summaries":
        from summarizer.sync import run_summary_sync
        from repl.commands import init_git_state, GIT_STATE
        init_git_state(cfg)
        if not args.dry_run and not args.force and not GIT_STATE["knrs_data_safe_local"]:
            print(f"Error: {cfg.knrs_data} is a git repo but not up-to-date. Use --force to override.", file=sys.stderr)
            sys.exit(1)
        run_summary_sync(cfg, dry_run=args.dry_run, concurrency=args.concurrency)
    
    elif args.command == "sync-wiki":
        from wiki.sync import run_wiki_sync, inject_frontmatter_in_notes
        from repl.commands import init_git_state, GIT_STATE
        init_git_state(cfg)
        if not args.force and not GIT_STATE["wiki_path_safe_local"]:
            print(f"Error: {cfg.wiki_path} is a git repo but not up-to-date. Use --force to override.", file=sys.stderr)
            sys.exit(1)
        inject_frontmatter_in_notes(cfg.notes_path, cfg.wiki_path)
        run_wiki_sync(cfg)
        
    elif args.command == "timeline":
        from timelines.extractor import run_extraction, show_timeline
        from timelines.indra_time import parse_point
        from repl.commands import init_git_state, GIT_STATE
        init_git_state(cfg)
        if not args.force and not GIT_STATE["knrs_data_safe_local"]:
            print(f"Error: {cfg.knrs_data} is a git repo but not up-to-date. Use --force to override.", file=sys.stderr)
            sys.exit(1)
            
        output_file = cfg.timelines / "timelines.json"
        run_extraction(cfg.notes_path, output_file)
        
        start_year = None
        if args.from_date:
            try:
                start_year = parse_point(args.from_date)
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
        
        end_year = None
        if args.to_date:
            try:
                end_year = parse_point(args.to_date)
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
                
        # If any filter or raw is provided, show the timeline
        if start_year is not None or end_year is not None or args.context or args.keywords or args.raw:
            show_timeline(
                output_file,
                start_year=start_year,
                end_year=end_year,
                context_filters=args.context,
                keywords=args.keywords,
                raw=args.raw
            )
        
    elif args.command == "index":
        from vector.indexer import KnrsIndexer
        from repl.commands import init_git_state, GIT_STATE
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
            checkpoint_every_docs=args.checkpoint_every_docs,
            checkpoint_every_chunks=args.checkpoint_every_chunks,
        )
        
    elif args.command == "search":
        from vector.search import KnrsSearcher, get_context_aware_text, get_significance
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
                from vector.engine import EmbedderSession
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
        from repl.commands import init_git_state, GIT_STATE, cmd_sync
        init_git_state(cfg)
        
        # Prepare mock args list from argparse to pass to cmd_sync
        mock_args = []
        if args.dry_run:
            mock_args.append("--dry-run")
        if args.force:
            mock_args.append("--force")
        if args.checkpoint_every_docs:
            mock_args.extend(["--checkpoint-every-docs", str(args.checkpoint_every_docs)])
        if args.checkpoint_every_chunks:
            mock_args.extend(["--checkpoint-every-chunks", str(args.checkpoint_every_chunks)])
        
        cmd_sync(mock_args, cfg)

    elif args.command == "benchmark":
        from pathlib import Path
        from benchmark.runner import BenchmarkRunner
        from rich.console import Console
        from rich.table import Table

        workspace_root = Path(__file__).parent.resolve()
        results_path = Path(args.results) if args.results else None

        runner = BenchmarkRunner(workspace_root, results_path=results_path)

        console = Console()
        with console.status("[bold green]Running benchmarks... (this may take a few minutes if local models are loaded)") as status:
            record = runner.run_all(filter_type=args.type, filter_backend=args.backend)

        console.print("\n[bold green]Benchmark Completed Successfully![/bold green]")
        console.print(f"Results saved to: [cyan]{runner.results_path}[/cyan]\n")

        table = Table(title="knrs Subprocess Benchmark Results")
        table.add_column("Backend", style="bold cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Task", style="blue")
        table.add_column("Status", style="bold")
        table.add_column("Load Time", justify="right")
        table.add_column("Latency", justify="right")
        table.add_column("Throughput", justify="right")
        table.add_column("Details / Error", style="red")

        for res in record["results"]:
            status_style = "green" if res["pass_fail"] == "pass" else "red"
            status_text = f"[{status_style}]{res['pass_fail'].upper()}[/{status_style}]"

            load_time = f"{res['load_time_sec']:.2f}s" if res['load_time_sec'] > 0 else "-"
            latency = f"{res['latency_sec']:.2f}s"

            if res["pass_fail"] == "pass" and res["throughput"] > 0:
                throughput = f"{res['throughput']:.2f} {res['throughput_units']}"
            else:
                throughput = "-"

            details = res["error"] or "-"

            table.add_row(
                res["backend"],
                res["backend_type"],
                res["task_name"],
                status_text,
                load_time,
                latency,
                throughput,
                details
            )

        console.print(table)

    elif args.command in (None, "repl"):
        from repl.repl import run_repl
        run_repl(cfg)

if __name__ == "__main__":
    main()
