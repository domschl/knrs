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
    
    table.add_row("/sync-calibre [--dry-run]", "Sync Calibre library to MarkdownBooks")
    table.add_row("/sync-summaries [--dry-run]", "Sync MarkdownBooks to BookSummaries")
    table.add_row("/sync-wiki", "Sync KnrsData to Wiki/AINotes")
    table.add_row("/sync-external-lib [--dry-run]", "Sync Calibre EPUB/PDF to External Library")
    table.add_row("/check-wiki [--dry-run] [--broken-links-to-italics]", "Check and fix metadata consistency in Wiki")
    table.add_row("/timeline", "Extract timelines from Wiki/Notes")
    table.add_row("/index [--force]", "Update VectorDB index (MarkdownBooks + Wiki/Notes); --force re-embeds everything")
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

def cmd_sync_external_lib(args: list[str], cfg: KnrsConfig):
    from knrs.external_lib.sync import run_external_sync
    dry_run = "--dry-run" in args
    run_external_sync(cfg, dry_run=dry_run)

def cmd_wiki_check(args: list[str], cfg: KnrsConfig):
    from knrs.wiki.checker import run_wiki_check
    dry_run = "--dry-run" in args
    fix_broken_links = "--broken-links-to-italics" in args
    run_wiki_check(cfg, dry_run=dry_run, fix_broken_links=fix_broken_links)

def cmd_timeline(args: list[str], cfg: KnrsConfig):
    from knrs.timelines.extractor import run_extraction
    run_extraction(cfg.notes_path, cfg.timelines / "timelines.json")

def cmd_index(args: list[str], cfg: KnrsConfig):
    from knrs.vector.indexer import KnrsIndexer
    force = "--force" in args
    KnrsIndexer(cfg).run_indexing(cfg.markdown_books, cfg.wiki_path, force=force)

def cmd_search(args: list[str], cfg: KnrsConfig):
    if not args:
        console.print("[red]Usage: /search <query>[/red]")
        return
    query = " ".join(args)
    from knrs.vector.search import KnrsSearcher
    searcher = KnrsSearcher(cfg)
    try:
        results = searcher.search(query)
        table = Table(title=f"Search Results for: {query}")
        table.add_column("Score", justify="right", style="green")
        table.add_column("Source", style="dim")
        table.add_column("Path", style="bold")
        table.add_column("Snippet")
        
        for r in results:
            table.add_row(
                f"{r.score:.4f}",
                r.source_label,
                r.bare_path,
                r.text[:100].replace("\n", " ") + "...",
            )
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
    "/sync-external-lib": cmd_sync_external_lib,
    "/check-wiki": cmd_wiki_check,
    "/timeline": cmd_timeline,
    "/index": cmd_index,
    "/search": cmd_search,
    "/migrate": cmd_migrate,
    "/config": cmd_config,
}
