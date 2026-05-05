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

GIT_STATE = {
    "checked": False,
    "knrs_data_safe_local": True,
    "wiki_path_safe_local": True,
    "knrs_data_safe_remote": True,
    "wiki_path_safe_remote": True,
}

def init_git_state(cfg: KnrsConfig):
    if GIT_STATE["checked"]:
        return
    from knrs.paths import ensure_git_safety
    logger.info("Performing initial git safety checks...")
    
    GIT_STATE["knrs_data_safe_local"] = ensure_git_safety(cfg.knrs_data, check_remote=False)
    if not GIT_STATE["knrs_data_safe_local"]:
        console.print(f"[yellow]Warning: {cfg.knrs_data} is not up-to-date (local changes). State-changing commands will be blocked.[/yellow]")

    GIT_STATE["wiki_path_safe_local"] = ensure_git_safety(cfg.wiki_path, check_remote=False)
    if not GIT_STATE["wiki_path_safe_local"]:
        console.print(f"[yellow]Warning: {cfg.wiki_path} is not up-to-date (local changes). State-changing commands will be blocked.[/yellow]")

    GIT_STATE["knrs_data_safe_remote"] = ensure_git_safety(cfg.knrs_data, check_remote=True)
    if not GIT_STATE["knrs_data_safe_remote"] and GIT_STATE["knrs_data_safe_local"]:
        console.print(f"[yellow]Warning: {cfg.knrs_data} remote is not up-to-date. Vector index updates will be blocked.[/yellow]")
        
    GIT_STATE["wiki_path_safe_remote"] = ensure_git_safety(cfg.wiki_path, check_remote=True)
    if not GIT_STATE["wiki_path_safe_remote"] and GIT_STATE["wiki_path_safe_local"]:
        console.print(f"[yellow]Warning: {cfg.wiki_path} remote is not up-to-date. Vector index updates will be blocked.[/yellow]")

    GIT_STATE["checked"] = True

def cmd_help(args: list[str], cfg: KnrsConfig):
    """Show available commands."""
    table = Table(title="Available Slash Commands")
    table.add_column("Command", style="cyan")
    table.add_column("Description")
    
    table.add_row("/sync [--force] [--dry-run]", "Run full sync pipeline (calibre -> summaries -> wiki -> timeline -> index -> external-lib -> check-wiki)")
    table.add_row("/sync-calibre [--dry-run] [--force]", "Sync Calibre library to MarkdownBooks")
    table.add_row("/sync-summaries [--dry-run] [--force]", "Sync MarkdownBooks to BookSummaries")
    table.add_row("/sync-wiki [--force]", "Sync KnrsData to Wiki/AINotes")
    table.add_row("/sync-external-lib [--dry-run]", "Sync Calibre EPUB/PDF to External Library")
    table.add_row("/check-wiki [--dry-run] [--broken-links-to-italics] [--force]", "Check and fix metadata consistency in Wiki")
    table.add_row("/timeline [--force]", "Extract timelines from Wiki/Notes")
    table.add_row("/index [--force] [--checkpoint-every N]", "Update VectorDB index (MarkdownBooks + Wiki/Notes); --force re-embeds everything AND overrides git safety; default checkpoint: 50 files")
    table.add_row("/search <query>", "Search VectorDB")
    table.add_row("/config", "Show current configuration")
    table.add_row("/exit", "Exit the REPL")
    
    console.print(table)

def cmd_sync_calibre(args: list[str], cfg: KnrsConfig):
    from knrs.calibre.sync import run_sync
    dry_run = "--dry-run" in args
    force = "--force" in args
    if not dry_run and not force and not GIT_STATE["knrs_data_safe_local"]:
        console.print(f"[red]Safety check blocked: {cfg.knrs_data} is a git repo but not up-to-date.[/red]")
        console.print("[yellow]Use --force to override.[/yellow]")
        return
    run_sync(cfg, dry_run=dry_run)

def cmd_sync_summaries(args: list[str], cfg: KnrsConfig):
    from knrs.summarizer.sync import run_summary_sync
    dry_run = "--dry-run" in args
    force = "--force" in args
    if not dry_run and not force and not GIT_STATE["knrs_data_safe_local"]:
        console.print(f"[red]Safety check blocked: {cfg.knrs_data} is a git repo but not up-to-date.[/red]")
        console.print("[yellow]Use --force to override.[/yellow]")
        return
    run_summary_sync(cfg, dry_run=dry_run)

def cmd_sync_wiki(args: list[str], cfg: KnrsConfig):
    from knrs.wiki.sync import run_wiki_sync, inject_frontmatter_in_notes
    force = "--force" in args
    if not force and not GIT_STATE["wiki_path_safe_local"]:
        console.print(f"[red]Safety check blocked: {cfg.wiki_path} is a git repo but not up-to-date.[/red]")
        console.print("[yellow]Use --force to override.[/yellow]")
        return
    logger.info("Injecting missing frontmatter into Notes...")
    count = inject_frontmatter_in_notes(cfg.notes_path, cfg.wiki_path)
    logger.info("Updated %d files.", count)
    run_wiki_sync(cfg)

def cmd_sync_external_lib(args: list[str], cfg: KnrsConfig):
    from knrs.external_lib.sync import run_external_sync
    dry_run = "--dry-run" in args
    run_external_sync(cfg, dry_run=dry_run)

def cmd_wiki_check(args: list[str], cfg: KnrsConfig):
    from knrs.wiki.checker import run_wiki_check
    dry_run = "--dry-run" in args
    force = "--force" in args
    if not dry_run and not force and not GIT_STATE["wiki_path_safe_local"]:
        console.print(f"[red]Safety check blocked: {cfg.wiki_path} is a git repo but not up-to-date.[/red]")
        console.print("[yellow]Use --force to override.[/yellow]")
        return
    fix_broken_links = "--broken-links-to-italics" in args
    run_wiki_check(cfg, dry_run=dry_run, fix_broken_links=fix_broken_links)

def cmd_timeline(args: list[str], cfg: KnrsConfig):
    from knrs.timelines.extractor import run_extraction
    force = "--force" in args
    if not force and not GIT_STATE["knrs_data_safe_local"]:
        console.print(f"[red]Safety check blocked: {cfg.knrs_data} is a git repo but not up-to-date.[/red]")
        console.print("[yellow]Use --force to override.[/yellow]")
        return
    run_extraction(cfg.notes_path, cfg.timelines / "timelines.json")

def cmd_index(args: list[str], cfg: KnrsConfig):
    from knrs.vector.indexer import KnrsIndexer
    force = "--force" in args
    
    if not force and not GIT_STATE["knrs_data_safe_remote"]:
        console.print(f"[red]Safety check blocked: {cfg.knrs_data} is a git repo but not up-to-date (remote check enabled).[/red]")
        console.print("[yellow]Use --force to override.[/yellow]")
        return
    if not force and not GIT_STATE["wiki_path_safe_remote"]:
        console.print(f"[red]Safety check blocked: {cfg.wiki_path} is a git repo but not up-to-date (remote check enabled).[/red]")
        console.print("[yellow]Use --force to override.[/yellow]")
        return

    checkpoint_every = 50
    if "--checkpoint-every" in args:
        idx = args.index("--checkpoint-every")
        try:
            checkpoint_every = int(args[idx + 1])
        except (IndexError, ValueError):
            console.print("[red]--checkpoint-every requires an integer argument[/red]")
            return
    KnrsIndexer(cfg).run_indexing(
        cfg.markdown_books, cfg.wiki_path,
        force=force, checkpoint_every=checkpoint_every,
    )

def cmd_search(args: list[str], cfg: KnrsConfig):
    if not args:
        console.print("[red]Usage: /search <query> [--raw] [--highlight][/red]")
        return
        
    raw = "--raw" in args
    highlight = "--highlight" in args
    query_args = [a for a in args if a not in ("--raw", "--highlight")]
    
    if not query_args:
        console.print("[red]Usage: /search <query> [--raw] [--highlight][/red]")
        return
        
    query = " ".join(query_args)
    from knrs.vector.search import KnrsSearcher, get_context_aware_text, get_significance
    from rich.markdown import Markdown
    from rich.rule import Rule
    
    searcher = KnrsSearcher(cfg)
    try:
        results = searcher.search(query)
        
        # Pre-process all context-aware texts and headers
        processed_results = []
        for i, r in enumerate(results, 1):
            doc_name = Path(r.bare_path).name
            header = f"### {i}. {doc_name}\n"
            header += f"**Path:** `{r.bare_path}` | **Source:** `{r.source_label}` | **Score:** `{r.score:.4f}`\n\n"
            chunk_text = get_context_aware_text(searcher, r)
            processed_results.append([header, chunk_text, r])
            
        if highlight and results:
            from knrs.vector.engine import EmbedderSession
            with EmbedderSession(cfg) as session:
                for idx, (header, chunk_text, r) in enumerate(processed_results):
                    if r.query_embedding is not None:
                        chunk_text = get_significance(chunk_text, r.query_embedding, searcher, raw=raw, session=session)
                        processed_results[idx][1] = chunk_text
                        
        console.print(f"[bold blue]Search Results for: {query}[/bold blue]\n")
        
        for header, chunk_text, r in processed_results:
            if raw:
                console.print(header + chunk_text, markup=False)
                console.print("\n" + "-" * 40 + "\n")
            else:
                console.print(Markdown(header))
                console.print(Markdown(chunk_text))
                console.print(Rule(style="dim"))
                console.print()
            
    except FileNotFoundError:
        console.print("[red]Error: Index not found. Run /index first.[/red]")

def cmd_sync(args: list[str], cfg: KnrsConfig):
    console.print("[bold blue]Executing full sync pipeline...[/bold blue]")
    
    console.print("\n[bold cyan]1/7: Running /sync-calibre[/bold cyan]")
    cmd_sync_calibre(args, cfg)
    
    console.print("\n[bold cyan]2/7: Running /sync-summaries[/bold cyan]")
    cmd_sync_summaries(args, cfg)
    
    console.print("\n[bold cyan]3/7: Running /sync-wiki[/bold cyan]")
    cmd_sync_wiki(args, cfg)
    
    console.print("\n[bold cyan]4/7: Running /timeline[/bold cyan]")
    cmd_timeline(args, cfg)
    
    console.print("\n[bold cyan]5/7: Running /index[/bold cyan]")
    cmd_index(args, cfg)
    
    console.print("\n[bold cyan]6/7: Running /sync-external-lib[/bold cyan]")
    cmd_sync_external_lib(args, cfg)
    
    console.print("\n[bold cyan]7/7: Running /check-wiki[/bold cyan]")
    cmd_wiki_check(args, cfg)
    
    console.print("\n[bold green]Full sync pipeline completed![/bold green]")

def cmd_config(args: list[str], cfg: KnrsConfig):
    from knrs.config import print_config
    print_config(cfg)

COMMANDS = {
    "/help": cmd_help,
    "/sync": cmd_sync,
    "/sync-calibre": cmd_sync_calibre,
    "/sync-summaries": cmd_sync_summaries,
    "/sync-wiki": cmd_sync_wiki,
    "/sync-external-lib": cmd_sync_external_lib,
    "/check-wiki": cmd_wiki_check,
    "/timeline": cmd_timeline,
    "/index": cmd_index,
    "/search": cmd_search,
    "/config": cmd_config,
}
