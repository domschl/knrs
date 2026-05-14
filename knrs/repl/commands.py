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
    
    table.add_row(r"/sync \[--force] \[--dry-run]", "Run full sync pipeline (calibre -> summaries -> wiki -> timeline -> index -> external-lib -> check-wiki)")
    table.add_row(r"/sync-calibre \[--dry-run] \[--force]", "Sync Calibre library to MarkdownBooks")
    table.add_row(r"/sync-summaries \[--dry-run] \[--force]", "Sync MarkdownBooks to BookSummaries")
    table.add_row(r"/sync-wiki \[--force]", "Sync KnrsData to Wiki/AINotes")
    table.add_row(r"/sync-external-lib \[--dry-run]", "Sync Calibre EPUB/PDF to External Library")
    table.add_row(r"/check-wiki \[--dry-run] \[--broken-links-to-italics] \[--force]", "Check and fix metadata consistency in Wiki")
    table.add_row(r"/timeline \[--force] \[--from YYYY-MM-DD] \[--to YYYY-MM-DD] \[--context PAT] \[--raw] \[keywords]", "Extract and filter timelines; supports date ranges, context, and keyword search")
    table.add_row(r"/index \[--force] \[--checkpoint-every-docs N] \[--checkpoint-every-chunks M]", "Update VectorDB index (MarkdownBooks + Wiki/Notes); --force re-embeds everything; default checkpoint: 50 files or 5000 chunks")
    table.add_row(r"/search <query> \[--raw] \[--highlight] \[--summarize]", "Search VectorDB; --raw: output text without markdown formatting; --highlight: semantic significance highlighting; --summarize: generate AI summary answering the query")
    table.add_row(r"/research <topic> \[--resume]", "Run research agent on the given topic. Use --resume to continue the last session.")
    table.add_row("/research-list", "List past research sessions")
    table.add_row("/config", "Show current configuration")
    table.add_row("/backends", "List available backends for the current platform")
    table.add_row("/models <backend>", "List available and validated models for a specific backend")
    table.add_row("/set-backend <type> <backend>", "Set the active backend for a given type (e.g., summarizer, embedder)")
    table.add_row("/set-param <backend|global> <key> <value>", "Set a configuration parameter for a specific backend, or global/shared config")
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
    from knrs.utils.search import SearchTools
    from knrs.timelines.indra_time import parse_point
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
        return
        
    output_file = cfg.timelines / "timelines.json"
    run_extraction(cfg.notes_path, output_file)
    
    # If any filter or raw is provided, show the timeline
    if start_year is not None or end_year is not None or context_filters or keywords or raw:
        from knrs.timelines.extractor import show_timeline
        show_timeline(
            output_file,
            start_year=start_year,
            end_year=end_year,
            context_filters=context_filters,
            keywords=keywords,
            raw=raw
        )

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

    checkpoint_every_docs = None
    if "--checkpoint-every-docs" in args:
        idx = args.index("--checkpoint-every-docs")
        try:
            checkpoint_every_docs = int(args[idx + 1])
        except (IndexError, ValueError):
            console.print("[red]--checkpoint-every-docs requires an integer argument[/red]")
            return

    checkpoint_every_chunks = None
    if "--checkpoint-every-chunks" in args:
        idx = args.index("--checkpoint-every-chunks")
        try:
            checkpoint_every_chunks = int(args[idx + 1])
        except (IndexError, ValueError):
            console.print("[red]--checkpoint-every-chunks requires an integer argument[/red]")
            return

    KnrsIndexer(cfg).run_indexing(
        cfg.markdown_books, cfg.wiki_path,
        force=force, 
        checkpoint_every_docs=checkpoint_every_docs,
        checkpoint_every_chunks=checkpoint_every_chunks,
    )

def cmd_search(args: list[str], cfg: KnrsConfig):
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
                
        if summarize and results:
            import tempfile
            from knrs.summarizer.engine import answer_query
            from knrs.calibre.converter import _split_frontmatter
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

def cmd_sync(args: list[str], cfg: KnrsConfig):
    force = "--force" in args
    
    if not force and not GIT_STATE["knrs_data_safe_remote"]:
        console.print(f"[red]Safety check blocked: {cfg.knrs_data} is a git repo but not up-to-date (remote check enabled).[/red]")
        console.print("[yellow]Use --force to override.[/yellow]")
        return
    if not force and not GIT_STATE["wiki_path_safe_remote"]:
        console.print(f"[red]Safety check blocked: {cfg.wiki_path} is a git repo but not up-to-date (remote check enabled).[/red]")
        console.print("[yellow]Use --force to override.[/yellow]")
        return

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

_backend_manager = None
def _get_backend_manager():
    global _backend_manager
    if _backend_manager is None:
        from knrs.repl.backends import BackendManager
        from knrs.paths import resolve
        import logging
        logging.getLogger("knrs.repl.backends").setLevel(logging.WARNING)
        console.print("[dim]Scanning subprocesses...[/dim]")
        _backend_manager = BackendManager(resolve("~/Codeberg/knrs/knrs/subprocesses"))
    return _backend_manager

def cmd_backends(args: list[str], cfg: KnrsConfig):
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

def cmd_models(args: list[str], cfg: KnrsConfig):
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

def cmd_set_backend(args: list[str], cfg: KnrsConfig):
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
        
    from knrs.config import update_knrs_config
    key = f"{btype}_name"
    
    if update_knrs_config(key, bname):
        setattr(cfg, key, bname)
        console.print(f"[green]Successfully set active {btype} to {bname}[/green]")
    else:
        console.print(f"[red]Failed to update configuration[/red]")

def cmd_set_param(args: list[str], cfg: KnrsConfig):
    if len(args) < 3:
        console.print("[red]Usage: /set-param <backend|global|llm-server> <key> <value>[/red]")
        console.print("Example: /set-param summarizer_api model_name gemma-4-26b")
        return
        
    target, key = args[0], args[1]
    value_str = " ".join(args[2:])
    
    # Try to parse value as int/float/bool if possible
    if value_str.lower() == "true": value = True
    elif value_str.lower() == "false": value = False
    else:
        try:
            if "." in value_str: value = float(value_str)
            else: value = int(value_str)
        except ValueError:
            value = value_str

    from knrs.config import update_knrs_config, update_platform_config
    
    if target == "global":
        if update_knrs_config(key, value):
            if hasattr(cfg, key):
                setattr(cfg, key, value)
            console.print(f"[green]Successfully updated global config: {key} = {value}[/green]")
        else:
            console.print("[red]Failed to update global config[/red]")
    elif target == "llm-server":
        if update_platform_config("llm_server.json", key, value):
            console.print(f"[green]Successfully updated llm-server config: {key} = {value}[/green]")
        else:
            console.print("[red]Failed to update llm-server config[/red]")
    else:
        # Assume it's a backend name
        mgr = _get_backend_manager()
        cap = mgr.get_backend(target)
        if not cap:
            console.print(f"[red]Unknown target or backend: {target}[/red]")
            return
            
        btype = cap.get("type", "unknown")
        # Convention: platform configs are named like summarizer_config_api.json, or embedder_config_hf.json
        # Wait, the naming convention isn't perfectly uniform.
        # It's usually <type>_config_<suffix>.json. Let's just try to infer it.
        # summarizer_api -> summarizer_config_api.json
        # summarizer_macos -> summarizer_config_macos.json
        suffix = target.replace(btype + "_", "")
        if not suffix or suffix == target:
            filename = f"{btype}_config_{target}.json"
        else:
            filename = f"{btype}_config_{suffix}.json"
            
        if update_platform_config(filename, key, value):
            console.print(f"[green]Successfully updated {filename}: {key} = {value}[/green]")
        else:
            console.print(f"[yellow]Attempted to update {filename} but it may not exist yet or failed.[/yellow]")

def cmd_research(args: list[str], cfg: KnrsConfig):
    if not args:
        console.print("[red]Usage: /research <topic> [--resume][/red]")
        return
        
    resume = "--resume" in args
    
    # Filter out flags to get the topic
    topic_parts = [a for a in args if not a.startswith("--")]
    if not topic_parts and not resume:
        console.print("[red]Please specify a research topic.[/red]")
        return
        
    topic = " ".join(topic_parts)
    
    from knrs.agent.agent import ResearchAgent
    from knrs.agent.engine import AgentSession
    from pathlib import Path
    
    # Checkpoint path
    safe_topic = "".join(c for c in topic if c.isalnum() or c in (" ", "-", "_")).strip()
    if not safe_topic:
        safe_topic = "ResumeSession"
    ckpt_path = cfg.wiki_path / "AINotes" / "Research" / ".checkpoints" / f"{safe_topic}.json"
    
    try:
        with AgentSession(cfg) as session:
            agent = ResearchAgent(cfg, session)
            
            if resume:
                if ckpt_path.exists():
                    agent.load_checkpoint(ckpt_path)
                    console.print(f"[green]Resumed session from {ckpt_path.name}[/green]")
                    # If we don't have a new topic, just use what we have in history.
                    if topic:
                        agent.history.append({"role": "user", "content": f"New instruction: {topic}"})
                else:
                    console.print(f"[red]No checkpoint found for topic '{topic}' to resume from.[/red]")
                    return
            else:
                # Initial prompt
                init_prompt = f"Please research the following topic: '{topic}'.\n\nDevelop a plan and use the available tools to find relevant information. Then synthesize your findings into a comprehensive research document."
                agent.history.append({"role": "user", "content": init_prompt})
                
            # Execute loop
            step_count = 0
            max_steps = 30
            
            while step_count < max_steps:
                console.print(f"[dim]Agent thinking (Step {step_count+1})...[/dim]")
                
                try:
                    is_done, msg, tool_calls = agent.step()
                except Exception as e:
                    console.print(f"[red]Agent Error: {e}[/red]")
                    break
                    
                agent.save_checkpoint(ckpt_path)
                
                # Display the agent's thought/message via markdown
                from rich.markdown import Markdown
                console.print(Markdown(msg))
                
                if is_done:
                    console.print("[bold green]Research Task Complete![/bold green]")
                    break
                    
                if tool_calls:
                    stop_session = False
                    for tool_call in tool_calls:
                        tool_name = tool_call.get("tool")
                        tool_args = tool_call.get("args", {})
                        
                        console.print(f"[bold cyan]Agent proposes to use tool:[/bold cyan] {tool_name}")
                        
                        console.print(f"[dim]Executing {tool_name}...[/dim]")
                        try:
                            result = agent.execute_tool(tool_call)
                            # print a short preview of the result
                            preview = result[:200].replace("\n", " ") + "..." if len(result) > 200 else result
                            console.print(f"[dim]Tool result preview: {preview}[/dim]")
                        except Exception as e:
                            console.print(f"[red]Tool execution error: {e}[/red]")
                            agent.history.append({"role": "user", "content": f"Tool execution failed with error: {e}"})
                            
                    agent.save_checkpoint(ckpt_path)
                    if stop_session:
                        break
                else:
                    # If neither done nor tool call, the agent just talked to us. Provide it with a nudge.
                    if any(word in msg.lower() for word in ["finished", "completed", "done", "synthesis", "synthesized", "wrote", "written"]):
                        nudge_msg = "[SYSTEM NUDGE]: It looks like you might be finished. If so, please output 'TASK_COMPLETE' to end the session. If not, you MUST execute a tool call using the JSON format."
                    else:
                        nudge_msg = "[SYSTEM NUDGE]: You are procrastinating. You MUST execute your plan immediately by outputting exactly ONE tool call using the required JSON code block format. Do not just describe your plan."
                    
                    agent.history.append({"role": "user", "content": nudge_msg})
                        
                step_count += 1
                
            if step_count >= max_steps:
                console.print("[yellow]Agent reached maximum steps limit.[/yellow]")
    except FileNotFoundError as e:
        console.print(f"[red]Agent backend error: {e}[/red]")
        console.print("[yellow]Check that the agent backend is installed. Use /backends to see available backends and /set-backend agent <name> to switch.[/yellow]")
    except RuntimeError as e:
        console.print(f"[red]Agent backend error: {e}[/red]")

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
    "/sync-calibre": cmd_sync_calibre,
    "/sync-summaries": cmd_sync_summaries,
    "/sync-wiki": cmd_sync_wiki,
    "/sync-external-lib": cmd_sync_external_lib,
    "/check-wiki": cmd_wiki_check,
    "/timeline": cmd_timeline,
    "/index": cmd_index,
    "/search": cmd_search,
    "/config": cmd_config,
    "/backends": cmd_backends,
    "/models": cmd_models,
    "/set-backend": cmd_set_backend,
    "/set-param": cmd_set_param,
    "/research": cmd_research,
    "/research-list": cmd_research_list,
}
