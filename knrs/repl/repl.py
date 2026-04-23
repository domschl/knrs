"""
knrs.repl.repl — The interactive REPL loop.
"""

from __future__ import annotations

import logging
import sys

from rich.console import Console
from rich.prompt import Prompt

from knrs.config import KnrsConfig
from knrs.repl.commands import COMMANDS

logger = logging.getLogger(__name__)
console = Console()

def run_repl(cfg: KnrsConfig):
    """Start the interactive REPL."""
    console.print("[bold blue]Welcome to knrs REPL![/bold blue]")
    console.print("Type [cyan]/help[/cyan] for available commands, or [cyan]/exit[/cyan] to quit.")
    
    while True:
        try:
            line = Prompt.ask("[bold green]knrs[/bold green]")
            line = line.strip()
            
            if not line:
                continue
                
            if line.lower() in ("/exit", "/quit"):
                break
                
            if line.startswith("/"):
                parts = line.split()
                cmd_name = parts[0].lower()
                args = parts[1:]
                
                if cmd_name in COMMANDS:
                    try:
                        COMMANDS[cmd_name](args, cfg)
                    except Exception as e:
                        console.print(f"[red]Error executing {cmd_name}: {e}[/red]")
                        logger.exception("REPL command error")
                else:
                    console.print(f"[yellow]Unknown command: {cmd_name}[/yellow]")
            else:
                # Default behavior for non-slash lines? 
                # Maybe semantic search by default?
                if line:
                    console.print(f"Searching for: [italic]{line}[/italic]...")
                    COMMANDS["/search"](line.split(), cfg)
                    
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type /exit to quit.[/yellow]")
        except EOFError:
            break
            
    console.print("[bold blue]Goodbye![/bold blue]")
