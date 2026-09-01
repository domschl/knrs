"""
knrs.repl.commands — Slash-command dispatcher for the REPL.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, TYPE_CHECKING

from rich.console import Console
from rich.table import Table
from rich.markup import escape

from config import KnrsConfig
from utils.syncthing import get_syncthing_status

if TYPE_CHECKING:
    from repl.backends import BackendManager

logger = logging.getLogger(__name__)
console = Console()

GIT_STATE = {
    "checked": False,
    "knrs_data_safe_local": True,
    "wiki_path_safe_local": True,
    "knrs_data_safe_remote": True,
    "wiki_path_safe_remote": True,
}

def init_git_state(cfg: KnrsConfig) -> None:
    if GIT_STATE["checked"]:
        return
    from paths import ensure_git_safety
    logger.info("Performing initial git safety checks...")
    
    if getattr(cfg, "auto_git_sync", True):
        from paths import is_git_repo, is_git_uptodate
        sync_needed = False
        for path in [cfg.knrs_data, cfg.wiki_path]:
            if is_git_repo(str(path)) and not is_git_uptodate(str(path), check_remote=True):
                sync_needed = True
                break
        
        if sync_needed:
            console.print("[bold yellow]Git repos not in sync on startup. Attempting auto_git_sync...[/bold yellow]")
            cmd_sync_git(["Auto-sync on startup"], cfg)
            
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

    # Syncthing check
    console.print("[dim]Checking Syncthing status...[/dim]")
    for label, path in [("Calibre", cfg.calibre_path), ("KnrsData", cfg.knrs_data), ("Wiki", cfg.wiki_path), ("VectorDB", cfg.vector_db)]:
        status = get_syncthing_status(path)
        if status:
            if status.get("error"):
                console.print(f"[yellow]Syncthing ({label}): {status['error']}[/yellow]")
            elif not status.get("in_sync"):
                console.print(f"[bold yellow]Warning: Syncthing folder '{label}' is not in sync ({status.get('state')}).[/bold yellow]")
                if status.get("need_bytes", 0) > 0:
                    console.print(f"  [dim]Needs {status['need_bytes']} bytes to reach sync.[/dim]")

    GIT_STATE["checked"] = True

def cmd_help(args: list[str], cfg: KnrsConfig) -> None:
    """Show available commands."""
    console.print("[bold]knrs REPL[/bold] — Type your message to chat with the research agent.")
    console.print("The agent can search, read files, write research, and more.\n")
    
    table = Table(title="Slash Commands")
    table.add_column("Command", style="cyan")
    table.add_column("Description")
    
    table.add_row("/reset", "Clear conversation history and start a new session")
    table.add_row(r"/save-session \[name]", "Save current conversation to a checkpoint")
    table.add_row(r"/load-session \[name]", "Load a saved conversation checkpoint")
    table.add_row("/research-list", "List past research files in AINotes/Research")
    table.add_row(r"/search <query> \[--raw] \[--highlight] \[--summarize]", "Direct VectorDB search (bypasses agent)")
    table.add_row(r"/sync \[--force] \[--dry-run]", "Run full sync pipeline")
    table.add_row(r"/sync-git \[commit-message]", "Git add, commit, pull, push in wiki and data repos")
    table.add_row(r"/sync-calibre \[--dry-run] \[--force]", "Sync Calibre library to MarkdownBooks")
    table.add_row(r"/sync-summaries \[--dry-run] \[--force]", "Sync MarkdownBooks to BookSummaries")
    table.add_row(r"/sync-wiki \[--force]", "Sync KnrsData to Wiki/AINotes")
    table.add_row(r"/sync-external-lib \[--dry-run]", "Sync Calibre EPUB/PDF to External Library")
    table.add_row(r"/check-wiki \[--dry-run] \[--broken-links-to-italics] \[--force]", "Check and fix Wiki metadata consistency")
    table.add_row(r"/organize \[--dry-run] \[--force]", "Restructure the Research directory hierarchically using LLM")
    table.add_row(r"/timeline \[--force] \[--from YYYY] \[--to YYYY] \[--context PAT] \[--raw] \[keywords]", "Extract and filter timelines")
    table.add_row(r"/index \[--force]", "Update VectorDB index")
    table.add_row("/config", "Show current configuration")
    table.add_row("/sync-status", "Check Syncthing synchronization status")
    table.add_row(r"/unload \[backend] \[--force]", "Unload model from server VRAM")
    table.add_row("/backends", "List available backends")
    table.add_row("/models <backend>", "List models for a backend")
    table.add_row("/set-backend <type> <backend>", "Set the active backend")
    table.add_row("/set-param <backend|global> <key> <value>", "Set a configuration parameter")
    table.add_row("/exit", "Exit the REPL")
    
    console.print(table)

def cmd_sync_calibre(args: list[str], cfg: KnrsConfig) -> dict[str, Any] | None:
    from calibre.sync import run_sync
    dry_run = "--dry-run" in args
    force = "--force" in args
    if not dry_run and not force and not GIT_STATE["knrs_data_safe_local"]:
        console.print(f"[red]Safety check blocked: {cfg.knrs_data} is a git repo but not up-to-date.[/red]")
        console.print("[yellow]Use --force to override.[/yellow]")
        return None
    return run_sync(cfg, dry_run=dry_run)

def cmd_sync_summaries(args: list[str], cfg: KnrsConfig) -> dict[str, Any] | None:
    from summarizer.sync import run_summary_sync
    dry_run = "--dry-run" in args
    force = "--force" in args
    if not dry_run and not force and not GIT_STATE["knrs_data_safe_local"]:
        console.print(f"[red]Safety check blocked: {cfg.knrs_data} is a git repo but not up-to-date.[/red]")
        console.print("[yellow]Use --force to override.[/yellow]")
        return None
    return run_summary_sync(cfg, dry_run=dry_run)

def cmd_sync_wiki(args: list[str], cfg: KnrsConfig) -> dict[str, Any] | None:
    from wiki.sync import run_wiki_sync, inject_frontmatter_in_notes
    force = "--force" in args
    if not force and not GIT_STATE["wiki_path_safe_local"]:
        console.print(f"[red]Safety check blocked: {cfg.wiki_path} is a git repo but not up-to-date.[/red]")
        console.print("[yellow]Use --force to override.[/yellow]")
        return None
    logger.info("Injecting missing frontmatter into Notes...")
    count = inject_frontmatter_in_notes(cfg.notes_path, cfg.wiki_path)
    logger.info("Updated %d files.", count)
    res = run_wiki_sync(cfg)
    if res:
        res["frontmatter_updated"] = count
    return res

def cmd_sync_external_lib(args: list[str], cfg: KnrsConfig) -> dict[str, Any]:
    from external_lib.sync import run_external_sync
    dry_run = "--dry-run" in args
    return run_external_sync(cfg, dry_run=dry_run)

def cmd_wiki_check(args: list[str], cfg: KnrsConfig) -> dict[str, Any] | None:
    from wiki.checker import run_wiki_check
    dry_run = "--dry-run" in args
    force = "--force" in args
    if not dry_run and not force and not GIT_STATE["wiki_path_safe_local"]:
        console.print(f"[red]Safety check blocked: {cfg.wiki_path} is a git repo but not up-to-date.[/red]")
        console.print("[yellow]Use --force to override.[/yellow]")
        return None
    fix_broken_links = "--broken-links-to-italics" in args
    return run_wiki_check(cfg, dry_run=dry_run, fix_broken_links=fix_broken_links)

def cmd_organize(args: list[str], cfg: KnrsConfig) -> None:
    """Restructure the AINotes/Research/ directory hierarchically by calling LLM classification."""
    from wiki.organizer import organize_research_directory
    
    dry_run = "--dry-run" in args
    apply = not dry_run
    force = "--force" in args
    
    if apply and not force and not GIT_STATE["wiki_path_safe_local"]:
        console.print(f"[red]Safety check blocked: {cfg.wiki_path} is a git repo but not up-to-date.[/red]")
        console.print("[yellow]Use --force to override.[/yellow]")
        return
        
    console.print("[bold blue]Starting LLM-Guided Research Organizer...[/bold blue]")
    if dry_run:
        console.print("[yellow]DRY-RUN MODE (No changes will be written)[/yellow]\n")
    else:
        console.print("[bold green]APPLYING CHANGES...[/bold green]\n")
        
    try:
        log_lines = organize_research_directory(cfg, dry_run=dry_run)
        if not log_lines:
            console.print("[green]No organization changes proposed or executed.[/green]")
        else:
            for line in log_lines:
                console.print(line)
    except Exception as e:
        console.print(f"[red]Error organizing research directory: {e}[/red]")

def cmd_timeline(args: list[str], cfg: KnrsConfig) -> None:
    from timelines.extractor import run_extraction
    from utils.search import SearchTools
    from timelines.indra_time import parse_point
    from rich.table import Table
    import json

    force = "--force" in args
    raw = "--raw" in args
    
    # Filter args
    start_year = None
    end_year = None
    context_filters = []
    keywords = []
    
    i = 0
    clean_args = [a for a in args if a not in ["--force", "--raw"]]
    while i < len(clean_args):
        arg = clean_args[i]
        if arg == "--from" and i + 1 < len(clean_args):
            try:
                start_year = parse_point(clean_args[i+1])
                i += 2
            except ValueError:
                console.print(f"[red]Invalid --from date: {clean_args[i+1]}[/red]")
                return
        elif arg == "--to" and i + 1 < len(clean_args):
            try:
                end_year = parse_point(clean_args[i+1])
                i += 2
            except ValueError:
                console.print(f"[red]Invalid --to date: {clean_args[i+1]}[/red]")
                return
        elif arg == "--context" and i + 1 < len(clean_args):
            context_filters.append(clean_args[i+1])
            i += 2
        else:
            keywords.append(arg)
            i += 1

    if not force and not GIT_STATE["knrs_data_safe_local"]:
        console.print(f"[red]Safety check blocked: {cfg.knrs_data} is a git repo but not up-to-date.[/red]")
        console.print("[yellow]Use --force to override.[/yellow]")
        return None
        
    output_file = cfg.timelines / "timelines.json"
    res = run_extraction(cfg.notes_path, output_file)
    
    # If any filter or raw is provided, show the timeline
    if start_year is not None or end_year is not None or context_filters or keywords or raw:
        from timelines.extractor import show_timeline
        show_timeline(
            output_file,
            start_year=start_year,
            end_year=end_year,
            context_filters=context_filters,
            keywords=keywords,
            raw=raw
        )
    return res

def cmd_index(args: list[str], cfg: KnrsConfig) -> dict[str, Any] | None:
    from vector.indexer import KnrsIndexer
    force = "--force" in args
    
    if not force and not GIT_STATE["knrs_data_safe_remote"]:
        console.print(f"[red]Safety check blocked: {cfg.knrs_data} is a git repo but not up-to-date (remote check enabled).[/red]")
        console.print("[yellow]Use --force to override.[/yellow]")
        return None
    if not force and not GIT_STATE["wiki_path_safe_remote"]:
        console.print(f"[red]Safety check blocked: {cfg.wiki_path} is a git repo but not up-to-date (remote check enabled).[/red]")
        console.print("[yellow]Use --force to override.[/yellow]")
        return None

    checkpoint_every_docs = None
    if "--checkpoint-every-docs" in args:
        idx = args.index("--checkpoint-every-docs")
        try:
            checkpoint_every_docs = int(args[idx + 1])
        except (IndexError, ValueError):
            console.print("[red]--checkpoint-every-docs requires an integer argument[/red]")
            return None

    checkpoint_every_chunks = None
    if "--checkpoint-every-chunks" in args:
        idx = args.index("--checkpoint-every-chunks")
        try:
            checkpoint_every_chunks = int(args[idx + 1])
        except (IndexError, ValueError):
            console.print("[red]--checkpoint-every-chunks requires an integer argument[/red]")
            return None

    return KnrsIndexer(cfg).run_indexing(
        cfg.markdown_books, cfg.wiki_path,
        force=force, 
        checkpoint_every_docs=checkpoint_every_docs,
        checkpoint_every_chunks=checkpoint_every_chunks,
    )

def cmd_search(args: list[str], cfg: KnrsConfig) -> None:
    if not args:
        console.print("[red]Usage: /search <query> [--raw] [--highlight] [--summarize][/red]")
        return
        
    raw = "--raw" in args
    highlight = "--highlight" in args
    summarize = "--summarize" in args
    query_args = [a for a in args if a not in ("--raw", "--highlight", "--summarize")]
    
    if not query_args:
        console.print("[red]Usage: /search <query> [--raw] [--highlight] [--summarize][/red]")
        return
        
    query = " ".join(query_args)
    from vector.search import KnrsSearcher, get_context_aware_text, get_significance
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
            chunk_text, _, _ = get_context_aware_text(searcher, r)
            processed_results.append([header, chunk_text, r])
            
        if highlight and results:
            from vector.engine import EmbedderSession
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
                
        if summarize and results:
            import tempfile
            from summarizer.engine import answer_query
            from calibre.converter import _split_frontmatter
            import yaml
            
            console.print("[bold yellow]Generating summary from search results...[/bold yellow]")
            
            snippets_for_summary = []
            for _, text, r in processed_results:
                title = "Unknown Title"
                authors = "Unknown Author"
                
                if r.source_label == "books":
                    file_path = searcher.config.markdown_books / r.bare_path
                elif r.source_label == "wiki":
                    file_path = searcher.config.wiki_path / r.bare_path
                else:
                    file_path = None
                    
                if file_path and file_path.exists():
                    try:
                        content = file_path.read_text(encoding="utf-8")
                        fm, _ = _split_frontmatter(content)
                        if fm:
                            meta = yaml.safe_load(fm) or {}
                            title = meta.get("title", title)
                            authors = meta.get("authors", authors)
                            if isinstance(authors, list):
                                authors = ", ".join(authors)
                    except Exception:
                        pass
                
                doc_meta = f"Document: {Path(r.bare_path).name}\nTitle: {title}\nAuthor(s): {authors}\n\n{text}"
                snippets_for_summary.append(doc_meta)
                
            combined_text = "\n\n---\n\n".join(snippets_for_summary)
            
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as src_fd:
                src_fd.write(combined_text)
                src_path = Path(src_fd.name)
                
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as dst_fd:
                dst_path = Path(dst_fd.name)
                
            try:
                enhanced_query = f"{query}\n\nIMPORTANT: When generating your answer, you MUST include references (Title and Author) to the specific sources used for each part of your summary."
                success = answer_query(enhanced_query, src_path, dst_path, cfg.summarizer_name, summary_max_tokens=2500)
                if success and dst_path.exists():
                    with open(dst_path, "r", encoding="utf-8") as f:
                        answer_text = f.read()
                    
                    console.print("\n[bold green]AI Summary of Results:[/bold green]")
                    if raw:
                        console.print(answer_text, markup=False)
                    else:
                        from rich.panel import Panel
                        console.print(Panel(Markdown(answer_text), title="Query Answer", border_style="green"))
                else:
                    console.print("[red]Failed to generate summary.[/red]")
            finally:
                if src_path.exists():
                    src_path.unlink()
                if dst_path.exists():
                    dst_path.unlink()
            
    except FileNotFoundError:
        console.print("[red]Error: Index not found. Run /index first.[/red]")

def cmd_sync_git(args: list[str], cfg: KnrsConfig) -> dict[str, Any]:
    from paths import is_git_repo
    import subprocess
    
    commit_msg = " ".join(args) if args else "Automated sync via knrs /sync-git"
    success_repos = 0
    failed_repos = 0
    errors: list[str] = []
    
    for name, path in [("wiki_path", cfg.wiki_path), ("knrs_data", cfg.knrs_data)]:
        path_str = str(path)
        if not is_git_repo(path_str):
            continue
            
        console.print(f"[bold blue]Syncing git repo at {path_str}...[/bold blue]")
        
        try:
            # 1. git add
            subprocess.run(["git", "add", "-A"], cwd=path_str, check=True)
            
            # 2. git commit
            status = subprocess.run(["git", "status", "--porcelain"], cwd=path_str, capture_output=True, text=True).stdout
            if status:
                final_msg = commit_msg
                if not args and getattr(cfg, "auto_git_sync", True):
                    from summarizer.engine import answer_query
                    import tempfile
                    from pathlib import Path
                    
                    diff = subprocess.run(["git", "diff", "--cached"], cwd=path_str, capture_output=True, text=True).stdout
                    if diff:
                        console.print(f"[dim]Generating commit message for {name}...[/dim]")
                        diff_text = diff[:8000] if len(diff) > 8000 else diff
                        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as src_fd:
                            src_fd.write(f"Git Diff:\n{diff_text}")
                            src_path = Path(src_fd.name)
                        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as dst_fd:
                            dst_path = Path(dst_fd.name)
                        try:
                            q = "Generate a concise git commit message (1 line) for these changes. Respond ONLY with the message, no quotes or prefix."
                            success = answer_query(q, src_path, dst_path, cfg.summarizer_name, summary_max_tokens=80)
                            if success and dst_path.exists():
                                with open(dst_path, "r", encoding="utf-8") as f:
                                    msg = f.read().strip()
                                    if msg:
                                        if msg.startswith('"') and msg.endswith('"'):
                                            msg = msg[1:-1]
                                        final_msg = msg
                        except Exception as e:
                            console.print(f"[yellow]Failed to generate commit message: {e}[/yellow]")
                        finally:
                            if src_path.exists(): src_path.unlink()
                            if dst_path.exists(): dst_path.unlink()

                subprocess.run(["git", "commit", "-m", final_msg], cwd=path_str, check=True)
                console.print(f"[green]Committed changes in {name}.[/green]")
            else:
                console.print(f"[dim]No local changes to commit in {name}.[/dim]")
                
            # 3. git pull
            console.print(f"[dim]Pulling changes from remote for {name}...[/dim]")
            pull_result = subprocess.run(["git", "pull", "--no-rebase", "--no-edit"], cwd=path_str, capture_output=True, text=True)
                
            if pull_result.returncode != 0:
                err = f"Failed to pull changes for {name}: {pull_result.stderr.strip()}"
                console.print(f"[red]{err}[/red]")
                failed_repos += 1
                errors.append(err)
                continue
                
            # 4. git push
            console.print(f"[dim]Pushing changes to remote for {name}...[/dim]")
            push_result = subprocess.run(["git", "push"], cwd=path_str, capture_output=True, text=True)
            if push_result.returncode != 0:
                err = f"Failed to push changes for {name}: {push_result.stderr.strip()}"
                console.print(f"[red]{err}[/red]")
                failed_repos += 1
                errors.append(err)
                continue
                
            console.print(f"[bold green]Successfully synced {name}![/bold green]")
            success_repos += 1
            
            # Unblock in GIT_STATE
            GIT_STATE[f"{name}_safe_local"] = True
            GIT_STATE[f"{name}_safe_remote"] = True
            
        except subprocess.CalledProcessError as e:
            err = f"Error executing git command in {name}: {e}"
            console.print(f"[red]{err}[/red]")
            failed_repos += 1
            errors.append(err)
        except Exception as e:
            err = f"Error syncing {name}: {e}"
            console.print(f"[red]{err}[/red]")
            failed_repos += 1
            errors.append(err)

    return {
        "repos_synced": success_repos,
        "repos_failed": failed_repos,
        "errors": errors,
    }

def cmd_unload(args: list[str], cfg: KnrsConfig) -> None:
    """Explicitly unload the active summarizer model from VRAM."""
    from summarizer.engine import unload_model
    backend_name = args[0] if args and not args[0].startswith("--") else cfg.summarizer_name
    force = "--force" in args or True  # Explicit invocation unloads model
    console.print(f"[bold blue]Requesting unload of model for backend '{backend_name}'...[/bold blue]")
    ok = unload_model(backend_name, force=force)
    if ok:
        console.print("[bold green]Model unloaded successfully.[/bold green]")
    else:
        console.print("[yellow]Model unload failed or backend does not support unload.[/yellow]")

def cmd_sync(args: list[str], cfg: KnrsConfig) -> dict[str, Any] | None:
    force = "--force" in args
    
    if not force and not GIT_STATE["knrs_data_safe_remote"]:
        console.print(f"[red]Safety check blocked: {cfg.knrs_data} is a git repo but not up-to-date (remote check enabled).[/red]")
        console.print("[yellow]Use --force to override.[/yellow]")
        return None
    if not force and not GIT_STATE["wiki_path_safe_remote"]:
        console.print(f"[red]Safety check blocked: {cfg.wiki_path} is a git repo but not up-to-date (remote check enabled).[/red]")
        console.print("[yellow]Use --force to override.[/yellow]")
        return None

    console.print("[bold blue]Executing full sync pipeline...[/bold blue]")
    
    auto_git = getattr(cfg, "auto_git_sync", True)
    total_steps = 8 if auto_git else 7
    step_records: list[dict[str, Any]] = []

    def _run_step(
        step_num: int,
        name: str,
        cmd_str: str,
        runner_fn,
        parse_result_fn,
    ) -> None:
        console.print(f"\n[bold cyan]{step_num}/{total_steps}: Running {cmd_str}[/bold cyan]")
        try:
            raw_res = runner_fn()
            rec = parse_result_fn(raw_res)
            rec["step_num"] = step_num
            rec["name"] = name
            rec["command"] = cmd_str
            step_records.append(rec)
        except Exception as exc:
            logger.exception(f"Unexpected error in {cmd_str}: {exc}")
            console.print(f"[bold red]Unexpected error in {cmd_str}: {exc}[/bold red]")
            step_records.append({
                "step_num": step_num,
                "name": name,
                "command": cmd_str,
                "status": "Failed",
                "success_count": 0,
                "fault_count": 1,
                "details": f"Exception: {exc}",
                "errors": [str(exc)],
            })

    # Step 1: /sync-calibre
    def _parse_calibre(res: dict[str, Any] | None) -> dict[str, Any]:
        if res is None:
            return {"status": "Blocked", "success_count": 0, "fault_count": 0, "details": "Blocked by safety check", "errors": ["Blocked by git safety check"]}
        f = res.get("failure_count", 0)
        s = res.get("success_count", 0)
        t = res.get("total", 0)
        counts = res.get("actions", {})
        act_str = ", ".join(f"{cnt} {k.lower()}" for k, cnt in counts.items() if k != "SKIP" and cnt > 0)
        details = act_str if act_str else ("All files up to date" if t == 0 else f"{s}/{t} actions applied")
        status = "Failed" if f > 0 else "OK"
        return {"status": status, "success_count": s, "fault_count": f, "details": details, "errors": res.get("failed_items", [])}

    _run_step(1, "Calibre Sync", "/sync-calibre", lambda: cmd_sync_calibre(args, cfg), _parse_calibre)

    # Step 2: /sync-summaries
    def _parse_summaries(res: dict[str, Any] | None) -> dict[str, Any]:
        if res is None:
            return {"status": "Blocked", "success_count": 0, "fault_count": 0, "details": "Blocked by safety check", "errors": ["Blocked by git safety check"]}
        f = res.get("failure_count", 0)
        s = res.get("success_count", 0)
        t = res.get("total", 0)
        counts = res.get("actions", {})
        act_str = ", ".join(f"{cnt} {k.lower()}" for k, cnt in counts.items() if k != "SKIP" and cnt > 0)
        details = act_str if act_str else ("All summaries up to date" if t == 0 else f"{s}/{t} actions applied")
        status = "Failed" if f > 0 else "OK"
        return {"status": status, "success_count": s, "fault_count": f, "details": details, "errors": res.get("failed_items", [])}

    _run_step(2, "Book Summaries", "/sync-summaries", lambda: cmd_sync_summaries(args, cfg), _parse_summaries)

    # Step 3: /sync-wiki
    def _parse_wiki(res: dict[str, Any] | None) -> dict[str, Any]:
        if res is None:
            return {"status": "Blocked", "success_count": 0, "fault_count": 0, "details": "Blocked by safety check", "errors": ["Blocked by git safety check"]}
        f = res.get("failure_count", 0)
        s = res.get("success_count", 0)
        t = res.get("total", 0)
        fm_count = res.get("frontmatter_updated", 0)
        details_parts = []
        if fm_count > 0:
            details_parts.append(f"{fm_count} frontmatter injected")
        if s > 0:
            details_parts.append(f"{s} pages synced")
        if not details_parts:
            details_parts.append("All wiki pages up to date" if t == 0 else f"{s}/{t} actions applied")
        details = ", ".join(details_parts)
        status = "Failed" if f > 0 else "OK"
        return {"status": status, "success_count": s, "fault_count": f, "details": details, "errors": res.get("failed_items", [])}

    _run_step(3, "Wiki Sync", "/sync-wiki", lambda: cmd_sync_wiki(args, cfg), _parse_wiki)

    # Step 4: /check-wiki
    def _parse_wiki_check(res: dict[str, Any] | None) -> dict[str, Any]:
        if res is None:
            return {"status": "Blocked", "success_count": 0, "fault_count": 0, "details": "Blocked by safety check", "errors": ["Blocked by git safety check"]}
        c = res.get("checked", 0)
        u = res.get("updated", 0)
        d = res.get("duplicates", 0)
        b = res.get("broken_links", 0)
        fl = res.get("fixed_links", 0)
        m = res.get("malformed_links", 0)
        e = res.get("errors", 0)
        
        issue_parts = []
        if b > 0:
            issue_parts.append(f"{b} broken links ({fl} fixed)" if fl > 0 else f"{b} broken links")
        if d > 0:
            issue_parts.append(f"{d} duplicates")
        if m > 0:
            issue_parts.append(f"{m} malformed")
        
        details = f"Checked {c} files"
        if u > 0:
            details += f", updated {u}"
        if issue_parts:
            details += f" ({', '.join(issue_parts)})"
            
        unfixed_issues = (b - fl) + d + m + e
        if e > 0:
            status = "Failed"
            fault_count = e
        elif unfixed_issues > 0:
            status = "Warning"
            fault_count = 0
        else:
            status = "OK"
            fault_count = 0
            
        errors = []
        if b - fl > 0:
            errors.append(f"{b - fl} unresolved broken wiki link(s)")
        if d > 0:
            errors.append(f"{d} duplicate document name(s)")
        if m > 0:
            errors.append(f"{m} malformed wiki link(s)")
            
        return {"status": status, "success_count": c, "fault_count": fault_count, "details": details, "errors": errors}

    _run_step(4, "Wiki Check", "/check-wiki", lambda: cmd_wiki_check(args, cfg), _parse_wiki_check)

    # Step 5: /timeline
    def _parse_timeline(res: dict[str, Any] | None) -> dict[str, Any]:
        if res is None:
            return {"status": "Blocked", "success_count": 0, "fault_count": 0, "details": "Blocked by safety check", "errors": ["Blocked by git safety check"]}
        fs = res.get("files_scanned", 0)
        ev = res.get("events_extracted", 0)
        f = res.get("failure_count", 0)
        status = "Failed" if f > 0 else "OK"
        details = f"{ev} events extracted from {fs} files"
        return {"status": status, "success_count": ev, "fault_count": f, "details": details, "errors": res.get("failed_items", [])}

    _run_step(5, "Timelines", "/timeline", lambda: cmd_timeline(args, cfg), _parse_timeline)

    # Step 6: /index
    def _parse_index(res: dict[str, Any] | None) -> dict[str, Any]:
        if res is None:
            return {"status": "Blocked", "success_count": 0, "fault_count": 0, "details": "Blocked by safety check", "errors": ["Blocked by git safety check"]}
        fi = res.get("files_indexed", 0)
        ff = res.get("files_failed", 0)
        ci = res.get("chunks_indexed", 0)
        tc = res.get("total_chunks", 0)
        tf = res.get("total_files", 0)
        if fi == 0 and ff == 0:
            details = f"Index up to date ({tc} chunks across {tf} files)"
        else:
            details = f"{fi} files indexed ({ci} chunks, {tc} total)"
        status = "Failed" if ff > 0 else "OK"
        return {"status": status, "success_count": fi, "fault_count": ff, "details": details, "errors": res.get("failed_items", [])}

    _run_step(6, "Vector Index", "/index", lambda: cmd_index(args, cfg), _parse_index)

    # Step 7: /sync-external-lib
    def _parse_ext_lib(res: dict[str, Any] | None) -> dict[str, Any]:
        if res is None:
            return {"status": "Failed", "success_count": 0, "fault_count": 1, "details": "Failed to run", "errors": ["Failed to run"]}
        f = res.get("failure_count", 0)
        s = res.get("success_count", 0)
        t = res.get("total", 0)
        cp = res.get("copied", 0)
        up = res.get("updated", 0)
        rf = res.get("removed_files", 0)
        if t == 0 and f == 0:
            details = "All external files up to date"
        else:
            parts = []
            if cp > 0: parts.append(f"{cp} copied")
            if up > 0: parts.append(f"{up} updated")
            if rf > 0: parts.append(f"{rf} removed")
            details = ", ".join(parts) if parts else f"{s} actions completed"
        status = "Failed" if f > 0 else "OK"
        return {"status": status, "success_count": s, "fault_count": f, "details": details, "errors": res.get("failed_items", [])}

    _run_step(7, "External Library", "/sync-external-lib", lambda: cmd_sync_external_lib(args, cfg), _parse_ext_lib)

    # Step 8: /sync-git
    if auto_git:
        def _parse_git(res: dict[str, Any] | None) -> dict[str, Any]:
            if res is None:
                return {"status": "Failed", "success_count": 0, "fault_count": 1, "details": "Failed", "errors": ["Git sync failed"]}
            s = res.get("repos_synced", 0)
            f = res.get("repos_failed", 0)
            status = "Failed" if f > 0 else "OK"
            details = f"{s} repo(s) synced" if f == 0 else f"{s} repo(s) synced, {f} failed"
            return {"status": status, "success_count": s, "fault_count": f, "details": details, "errors": res.get("errors", [])}

        _run_step(8, "Git Sync", "/sync-git", lambda: cmd_sync_git([], cfg), _parse_git)

    # Display Summary Table
    console.print("\n")
    table = Table(title="Full Sync Pipeline Summary", show_lines=True)
    table.add_column("Step", style="bold")
    table.add_column("Command", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Success", justify="right")
    table.add_column("Faults", justify="right")
    table.add_column("Details")

    total_faults = 0
    total_warnings = 0
    all_errors: list[tuple[str, str]] = []

    for rec in step_records:
        st = rec["status"]
        if st == "OK":
            st_formatted = "[bold green]OK[/bold green]"
        elif st == "Warning":
            st_formatted = "[bold yellow]Warning[/bold yellow]"
            total_warnings += 1
        elif st == "Blocked":
            st_formatted = "[bold magenta]Blocked[/bold magenta]"
        else:
            st_formatted = "[bold red]FAILED[/bold red]"

        faults = rec["fault_count"]
        total_faults += faults
        faults_str = f"[bold red]{faults}[/bold red]" if faults > 0 else "[green]0[/green]"
        
        table.add_row(
            f"{rec['step_num']}. {rec['name']}",
            rec["command"],
            st_formatted,
            str(rec["success_count"]),
            faults_str,
            rec["details"],
        )
        for err in rec.get("errors", []):
            if st != "OK" or faults > 0:
                all_errors.append((rec["command"], err))

    console.print(table)

    if total_faults > 0:
        console.print(f"\n[bold red]⚠ Pipeline completed with {total_faults} fault(s):[/bold red]")
        for cmd_name, err in all_errors:
            console.print(f"  [red]• \\[{escape(cmd_name)}\\] {escape(str(err))}[/red]")
    elif total_warnings > 0:
        console.print("\n[bold yellow]Pipeline completed with warnings (see details above).[/bold yellow]")
    else:
        console.print("\n[bold green]Full sync pipeline completed successfully with 0 faults![/bold green]")

    return {
        "total_faults": total_faults,
        "total_warnings": total_warnings,
        "steps": step_records,
    }

def cmd_config(args: list[str], cfg: KnrsConfig) -> None:
    from config import print_config
    print_config(cfg)

_backend_manager = None
def _get_backend_manager() -> BackendManager:
    global _backend_manager
    if _backend_manager is None:
        from repl.backends import BackendManager
        from paths import resolve
        import logging
        logging.getLogger("knrs.repl.backends").setLevel(logging.WARNING)
        console.print("[dim]Scanning subprocesses...[/dim]")
        _backend_manager = BackendManager(Path(__file__).parent.parent / "subprocesses")
    return _backend_manager

def cmd_backends(args: list[str], cfg: KnrsConfig) -> None:
    mgr = _get_backend_manager()
    table = Table(title="Available Backends")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Platform", style="green")
    table.add_column("Active", justify="center")
    
    active_backends = [cfg.summarizer_name, cfg.embedder_name, cfg.agent_backend_name, "md_converter"]
    
    for name, cap in mgr.get_backends().items():
        is_active = "✓" if name in active_backends else ""
        table.add_row(name, cap.get("type", "unknown"), cap.get("platform", "any"), is_active)
        
    console.print(table)

def cmd_models(args: list[str], cfg: KnrsConfig) -> None:
    if not args:
        console.print("[red]Usage: /models <backend_name>[/red]")
        return
        
    backend_name = args[0]
    mgr = _get_backend_manager()
    cap = mgr.get_backend(backend_name)
    
    if not cap:
        console.print(f"[red]Unknown backend: {backend_name}[/red]")
        return
        
    validated = set(cap.get("validated_models", []))
    available = set(cap.get("available_models", []))
    
    table = Table(title=f"Models for {backend_name}")
    table.add_column("Model Name", style="cyan")
    table.add_column("Validated", justify="center")
    table.add_column("Available", justify="center")
    
    all_models = sorted(validated.union(available))
    for m in all_models:
        v_mark = "[green]✓[/green]" if m in validated else ""
        a_mark = "[green]✓[/green]" if m in available else ""
        table.add_row(m, v_mark, a_mark)
        
    console.print(table)

def cmd_set_backend(args: list[str], cfg: KnrsConfig) -> None:
    if len(args) != 2:
        console.print("[red]Usage: /set-backend <type> <backend_name>[/red]")
        console.print("Example: /set-backend summarizer summarizer_api")
        return
        
    btype, bname = args[0].lower(), args[1]
    mgr = _get_backend_manager()
    cap = mgr.get_backend(bname)
    
    if not cap:
        console.print(f"[red]Unknown backend: {bname}[/red]")
        return
        
    if cap.get("type") != btype:
        console.print(f"[red]Backend {bname} is of type {cap.get('type')}, not {btype}[/red]")
        return
        
    from config import update_knrs_config
    key = f"{btype}_backend_name" if btype == "agent" else f"{btype}_name"
    
    if update_knrs_config(key, bname):
        setattr(cfg, key, bname)
        console.print(f"[green]Successfully set active {btype} to {bname}[/green]")
    else:
        console.print(f"[red]Failed to update configuration[/red]")

def cmd_set_param(args: list[str], cfg: KnrsConfig) -> None:
    if len(args) < 3:
        console.print("[red]Usage: /set-param <backend|global|llm-server> <key> <value>[/red]")
        console.print("Example: /set-param agent_api default_max_tokens 8000")
        return

    target, key = args[0], args[1]
    value_str = " ".join(args[2:])

    # ── Type coercion ──────────────────────────────────────────────────────────
    def coerce(val_str: str, type_hint: str | None = None) -> object:
        """Parse value_str to the appropriate Python type."""
        if val_str.lower() == "true":
            return True
        if val_str.lower() == "false":
            return False
        if type_hint == "int":
            try:
                return int(val_str)
            except ValueError:
                return val_str
        if type_hint == "float":
            try:
                return float(val_str)
            except ValueError:
                return val_str
        try:
            if "." in val_str:
                return float(val_str)
            return int(val_str)
        except ValueError:
            return val_str

    from config import update_knrs_config, update_platform_config, KNRS_CONFIG_FIELDS

    # ── global: knrs.json ──────────────────────────────────────────────────────
    if target == "global":
        if key not in KNRS_CONFIG_FIELDS:
            console.print(f"[red]Invalid global config key: '{key}'[/red]")
            console.print(f"Valid keys: {', '.join(sorted(KNRS_CONFIG_FIELDS))}")
            return
        value = coerce(value_str)
        if update_knrs_config(key, value):
            if hasattr(cfg, key):
                setattr(cfg, key, value)
            console.print(f"[green]Global config updated: {key} = {value!r}[/green]")
        else:
            console.print("[red]Failed to update global config[/red]")
        return

    # ── llm-server: llm_server.json ────────────────────────────────────────────
    if target == "llm-server":
        LLM_SERVER_KEYS = {"url", "api_type", "api_key"}
        if key not in LLM_SERVER_KEYS:
            console.print(f"[red]Invalid llm-server key: '{key}'[/red]")
            console.print(f"Valid keys: {', '.join(sorted(LLM_SERVER_KEYS))}")
            return
        value = coerce(value_str)
        if update_platform_config("llm_server.json", key, value):
            console.print(f"[green]llm-server config updated: {key} = {value!r}[/green]")
        else:
            console.print("[red]Failed to update llm-server config[/red]")
        return

    # ── backend config ─────────────────────────────────────────────────────────
    mgr = _get_backend_manager()
    cap = mgr.get_backend(target)
    if not cap:
        console.print(f"[red]Unknown target or backend: '{target}'[/red]")
        console.print("Use /backends to see available backends.")
        return

    parameters = cap.get("parameters", {})
    # Handle legacy list format gracefully
    if isinstance(parameters, list):
        parameters = {k: {"type": "str"} for k in parameters}

    if parameters and key not in parameters:
        console.print(f"[red]Invalid parameter '{key}' for backend '{target}'.[/red]")
        writable = [k for k, v in parameters.items() if not v.get("read_only")]
        console.print(f"Configurable parameters: {', '.join(writable) or '(none)'}")
        return

    param_meta = parameters.get(key, {})

    # Block read-only params
    if param_meta.get("read_only"):
        console.print(f"[red]Parameter '{key}' is read-only (managed internally by the backend).[/red]")
        return

    # Coerce and range-check
    type_hint = param_meta.get("type")
    value = coerce(value_str, type_hint)
    if type_hint == "int" and not isinstance(value, int):
        console.print(f"[red]Parameter '{key}' expects an integer, got: {value_str!r}[/red]")
        return
    if type_hint == "float" and not isinstance(value, (int, float)):
        console.print(f"[red]Parameter '{key}' expects a float, got: {value_str!r}[/red]")
        return
    if type_hint == "str" and not isinstance(value, str):
        value = value_str  # Keep as string

    min_val = param_meta.get("min")
    max_val = param_meta.get("max")
    if min_val is not None and value < min_val:
        console.print(f"[red]Value {value} is below minimum {min_val} for '{key}'.[/red]")
        return
    if max_val is not None and value > max_val:
        console.print(f"[red]Value {value} exceeds maximum {max_val} for '{key}'.[/red]")
        return

    # Determine config file — prefer the explicit config_file from capabilities
    config_file = cap.get("config_file")
    if not config_file:
        # Fallback: infer from backend name (legacy backends without config_file)
        btype = cap.get("type", "unknown")
        suffix = target.replace(btype + "_", "")
        config_file = f"{btype}_config_{suffix}.json" if suffix != target else f"{btype}_config_{target}.json"

    if update_platform_config(config_file, key, value):
        console.print(f"[green]{config_file}: {key} = {value!r}[/green]")
    else:
        console.print(f"[red]Failed to update {config_file}[/red]")
        console.print(f"[yellow]Tip: run the backend once to auto-create its config, or it will be created on next use.[/yellow]")

def cmd_save_session(args: list[str], cfg: KnrsConfig) -> None:
    """Save the current conversation session."""
    from agent.context import save_session
    from repl.repl import _get_current_state
    
    state = _get_current_state()
    if state is None:
        console.print("[red]No active agent session to save.[/red]")
        return
    
    name = args[0] if args else "default"
    safe_name = "".join(c for c in name if c.isalnum() or c in (" ", "-", "_")).strip()
    if not safe_name:
        safe_name = "default"
    
    ckpt_dir = cfg.wiki_path / "AINotes" / "Research" / ".sessions"
    ckpt_path = ckpt_dir / f"{safe_name}.json"
    save_session(state, ckpt_path)
    console.print(f"[green]Session saved as '{safe_name}'[/green]")


def cmd_load_session(args: list[str], cfg: KnrsConfig) -> None:
    """Load a saved conversation session."""
    from agent.context import load_session
    from repl.repl import _set_current_state
    
    if not args:
        # List available sessions
        ckpt_dir = cfg.wiki_path / "AINotes" / "Research" / ".sessions"
        if not ckpt_dir.exists():
            console.print("[yellow]No saved sessions found.[/yellow]")
            return
        sessions = sorted(ckpt_dir.glob("*.json"))
        if not sessions:
            console.print("[yellow]No saved sessions found.[/yellow]")
            return
        console.print("[bold cyan]Saved sessions:[/bold cyan]")
        for s in sessions:
            console.print(f"  {s.stem}")
        console.print("\n[dim]Use /load-session <name> to load one.[/dim]")
        return
    
    name = args[0]
    ckpt_dir = cfg.wiki_path / "AINotes" / "Research" / ".sessions"
    ckpt_path = ckpt_dir / f"{name}.json"
    
    if not ckpt_path.exists():
        console.print(f"[red]Session '{name}' not found.[/red]")
        return
    
    state = load_session(ckpt_path)
    _set_current_state(state)
    console.print(f"[green]Session '{name}' loaded ({len(state.history)} messages).[/green]")

def cmd_syncthing_status(args: list[str], cfg: KnrsConfig):
    """Check Syncthing status for all relevant paths."""
    table = Table(title="Syncthing Sync Status")
    table.add_column("Folder", style="cyan")
    table.add_column("ID", style="dim")
    table.add_column("State", justify="center")
    table.add_column("In Sync", justify="center")
    table.add_column("Details")

    paths = [
        ("Calibre Library", cfg.calibre_path),
        ("KnrsData", cfg.knrs_data),
        ("Wiki/Notes", cfg.notes_path),
        ("Wiki Root", cfg.wiki_path),
        ("VectorDB", cfg.vector_db),
    ]

    found_any = False
    for label, path in paths:
        status = get_syncthing_status(path)
        if status:
            found_any = True
            f_id = status.get("folder_id", "Unknown")
            state = status.get("state", "Unknown")
            in_sync = "[green]✓[/green]" if status.get("in_sync") else "[red]✗[/red]"
            
            error = status.get("error", "")
            if error:
                table.add_row(label, f_id, "[red]Error[/red]", "✗", error)
            else:
                details = ""
                if status.get("need_bytes", 0) > 0:
                    details = f"{status['need_bytes']} bytes needed"
                table.add_row(label, f_id, state, in_sync, details)

    if found_any:
        console.print(table)
    else:
        console.print("[yellow]No Syncthing folders detected for the current configuration.[/yellow]")

def cmd_research_list(args: list[str], cfg: KnrsConfig):
    research_dir = cfg.wiki_path / "AINotes" / "Research"
    if not research_dir.exists():
        console.print("[yellow]No research directory found.[/yellow]")
        return
        
    console.print("[bold cyan]Past Research Sessions:[/bold cyan]")
    from rich.tree import Tree
    tree = Tree("Research")
    
    for item in sorted(research_dir.iterdir()):
        if item.name.startswith("."): continue
        if item.is_dir():
            branch = tree.add(f"[bold]{item.name}[/bold]")
            for sub in sorted(item.iterdir()):
                if sub.is_file() and sub.suffix == ".md":
                    branch.add(sub.name)
        elif item.is_file() and item.suffix == ".md":
            tree.add(item.name)
            
    console.print(tree)

COMMANDS = {
    "/help": cmd_help,
    "/sync": cmd_sync,
    "/sync-git": cmd_sync_git,
    "/sync-calibre": cmd_sync_calibre,
    "/sync-summaries": cmd_sync_summaries,
    "/sync-wiki": cmd_sync_wiki,
    "/sync-external-lib": cmd_sync_external_lib,
    "/check-wiki": cmd_wiki_check,
    "/organize": cmd_organize,
    "/timeline": cmd_timeline,
    "/index": cmd_index,
    "/search": cmd_search,
    "/config": cmd_config,
    "/backends": cmd_backends,
    "/models": cmd_models,
    "/set-backend": cmd_set_backend,
    "/set-param": cmd_set_param,
    "/save-session": cmd_save_session,
    "/load-session": cmd_load_session,
    "/research-list": cmd_research_list,
    "/sync-status": cmd_syncthing_status,
    "/unload": cmd_unload,
}
