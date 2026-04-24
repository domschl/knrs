"""
knrs.repl.commands — Slash-command dispatcher for the REPL.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from knrs.config import KnrsConfig

logger = logging.getLogger(__name__)
console = Console()

def cmd_help(args: list[str], cfg: KnrsConfig):
    """Show available commands."""
    table = Table(title="Available Slash Commands")
    table.add_column("Command", style="cyan")
    table.add_column("Description")
    
    table.add_row("/sync-calibre", "Sync Calibre library to MarkdownBooks")
    table.add_row("/sync-summaries", "Sync MarkdownBooks to BookSummaries")
    table.add_row("/sync-wiki", "Sync KnrsData to Wiki/AINotes")
    table.add_row("/timeline", "Extract timelines from Wiki/Notes")
    table.add_row("/index", "Update VectorDB index")
    table.add_row("/search <query>", "Search VectorDB")
    table.add_row("/migrate", "Migrate data from legacy Summarizer project (dry-run)")
    table.add_row("/config", "Show current configuration")
    table.add_row("/exit", "Exit the REPL")
    
    console.print(table)

def cmd_sync_calibre(args: list[str], cfg: KnrsConfig):
    from knrs.calibre.sync import run_sync
    dry_run = "--dry-run" in args
    run_sync(cfg, dry_run=dry_run)

def cmd_sync_summaries(args: list[str], cfg: KnrsConfig):
    from knrs.summarizer.sync import run_summary_sync
    dry_run = "--dry-run" in args
    run_summary_sync(cfg, dry_run=dry_run)

def cmd_sync_wiki(args: list[str], cfg: KnrsConfig):
    from knrs.wiki.sync import run_wiki_sync, inject_uuids_in_notes
    logger.info("Injecting UUIDs into Notes...")
    count = inject_uuids_in_notes(cfg.notes_path)
    logger.info("Updated %d files.", count)
    run_wiki_sync(cfg)

def cmd_timeline(args: list[str], cfg: KnrsConfig):
    from knrs.timelines.extractor import run_extraction
    run_extraction(cfg.notes_path, cfg.timelines / "timelines.json")

def cmd_index(args: list[str], cfg: KnrsConfig):
    from knrs.vector.indexer import KnrsIndexer
    indexer = KnrsIndexer(cfg.vector_db)
    indexer.run_indexing(cfg.markdown_books)

def cmd_search(args: list[str], cfg: KnrsConfig):
    if not args:
        console.print("[red]Usage: /search <query>[/red]")
        return
    query = " ".join(args)
    from knrs.vector.search import KnrsSearcher
    searcher = KnrsSearcher(cfg.vector_db)
    try:
        results = searcher.search(query)
        table = Table(title=f"Search Results for: {query}")
        table.add_column("Score", justify="right", style="green")
        table.add_column("Book", style="bold")
        table.add_column("Snippet")
        
        for r in results:
            table.add_row(f"{r.score:.4f}", r.path, r.text[:100].replace("\n", " ") + "...")
        console.print(table)
    except FileNotFoundError:
        console.print("[red]Error: Index not found. Run /index first.[/red]")

def cmd_migrate(args: list[str], cfg: KnrsConfig):
    from knrs.migration.migrate import run_migration
    execute = "--execute" in args
    run_migration(cfg, dry_run=not execute)

def cmd_config(args: list[str], cfg: KnrsConfig):
    from knrs.config import print_config
    print_config(cfg)

COMMANDS = {
    "/help": cmd_help,
    "/sync-calibre": cmd_sync_calibre,
    "/sync-summaries": cmd_sync_summaries,
    "/sync-wiki": cmd_sync_wiki,
    "/timeline": cmd_timeline,
    "/index": cmd_index,
    "/search": cmd_search,
    "/migrate": cmd_migrate,
    "/config": cmd_config,
}
