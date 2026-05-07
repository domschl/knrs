"""
knrs.repl.repl — The interactive REPL loop.
"""

from __future__ import annotations

import atexit
import logging
import readline
import sys

from rich.console import Console

from knrs.config import KnrsConfig
from knrs.paths import knrs_history_file
from knrs.repl.commands import COMMANDS

logger = logging.getLogger(__name__)
console = Console()

# Readline-safe colored prompt: \001 and \002 markers tell readline 
# which characters are non-printing escape codes.
REPL_PROMPT = "\001\x1b[1;32m\002knrs\001\x1b[0m\002 "

def _setup_readline():
    """Configure readline for history and tab completion."""
    hist = knrs_history_file()
    if hist.exists():
        try:
            readline.read_history_file(str(hist))
        except Exception as e:
            logger.warning("Could not read history file %s: %s", hist, e)

    readline.set_history_length(1000)
    
    def clean_and_save_history():
        try:
            readline.write_history_file(str(hist))
            # Post-process the file to safely remove EOF characters without triggering libedit escaping bugs
            if hist.exists():
                with open(hist, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                
                with open(hist, "w", encoding="utf-8") as f:
                    for line in lines:
                        if '\x04' not in line and line.strip():
                            f.write(line)
        except Exception as e:
            logger.warning("Could not save history file %s: %s", hist, e)
            
    atexit.register(clean_and_save_history)

    def completer(text: str, state: int) -> str | None:
        # Only complete if the text starts with /
        if not text.startswith("/"):
            return None
        options = [c for c in COMMANDS if c.startswith(text)]
        if state < len(options):
            return options[state]
        return None

    readline.set_completer(completer)
    if "libedit" in readline.__doc__:
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")

def run_repl(cfg: KnrsConfig):
    """Start the interactive REPL."""
    _setup_readline()
    
    from knrs.repl.commands import init_git_state
    init_git_state(cfg)
    
    console.print("[bold blue]Welcome to knrs REPL![/bold blue]")
    console.print("Type [cyan]/help[/cyan] for available commands, or [cyan]/exit[/cyan] to quit.")
    
    while True:
        try:
            line = input(REPL_PROMPT)
            line = line.strip()
            
            if not line:
                continue
                
            if line.lower() in ("/exit", "/quit"):
                break
                
            if line.startswith("/"):
                import shlex
                try:
                    parts = shlex.split(line)
                except ValueError as e:
                    console.print(f"[red]Error parsing command: {e}[/red]")
                    continue
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
