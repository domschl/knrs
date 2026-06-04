"""
repl.repl — The interactive REPL loop with persistent agent session.
"""

from __future__ import annotations

import atexit
import logging
import readline
import sys

from rich.console import Console
from rich.markdown import Markdown

from config import KnrsConfig
from paths import knrs_history_file
from repl.commands import COMMANDS

logger = logging.getLogger(__name__)
console = Console()

# Readline-safe colored prompt: \001 and \002 markers tell readline 
# which characters are non-printing escape codes.
REPL_PROMPT = "\001\x1b[1;32m\002knrs\001\x1b[0m\002 "

# Module-level reference so /save-session and /load-session can access the state.
_current_agent: ResearchAgent | None = None  # type: ignore[name-defined]  # noqa: F821


def _get_current_state():
    """Return the current ConversationState, or None."""
    if _current_agent is not None:
        return _current_agent.state
    return None


def _set_current_state(state):
    """Replace the current agent's ConversationState."""
    if _current_agent is not None:
        _current_agent.state = state

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
    """Start the interactive REPL with a persistent agent session."""
    _setup_readline()
    
    from repl.commands import init_git_state
    init_git_state(cfg)
    
    # ── Start persistent agent session ─────────────────────────────
    from agent.engine import AgentSession
    from agent.agent import ResearchAgent
    from agent.context import ConversationState

    console.print("[dim]Starting agent backend…[/dim]")
    try:
        agent_session = AgentSession(cfg)
        agent_session.__enter__()
    except (FileNotFoundError, RuntimeError) as e:
        console.print(f"[bold red]Error: Could not start agent backend: {e}[/bold red]")
        console.print(
            "[yellow]Check that the agent backend is installed. "
            "Use 'uv run knrs config' to see the current agent_backend_name, "
            "and '/backends' to list available backends.[/yellow]"
        )
        return

    state = ConversationState()
    agent = ResearchAgent(cfg, agent_session, state)

    global _current_agent
    _current_agent = agent

    # Ensure clean shutdown of the agent subprocess.
    def _cleanup_agent():
        try:
            agent_session.__exit__(None, None, None)
        except Exception:
            pass

    atexit.register(_cleanup_agent)

    console.print("[bold blue]Welcome to knrs REPL![/bold blue]")
    console.print(
        "Type your message to chat with the research agent, "
        "or use [cyan]/help[/cyan] for slash commands."
    )
    
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
                
                if cmd_name == "/reset":
                    from agent.prompts import build_system_prompt
                    research_root = cfg.wiki_path / "AINotes" / "Research"
                    prompt = build_system_prompt(
                        include_tools=True,
                        include_syntax=True,
                        research_root=research_root,
                    )
                    state.reset(prompt)
                    console.print("[green]Conversation reset.[/green]")
                    continue

                if cmd_name in COMMANDS:
                    try:
                        COMMANDS[cmd_name](args, cfg)
                    except Exception as e:
                        console.print(f"[red]Error executing {cmd_name}: {e}[/red]")
                        logger.exception("REPL command error")
                else:
                    console.print(f"[yellow]Unknown command: {cmd_name}[/yellow]")
            else:
                # ── Agent conversational input ─────────────────────
                _run_agent_turn(agent, line)
                    
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type /exit to quit.[/yellow]")
        except EOFError:
            break
            
    _cleanup_agent()
    console.print("[bold blue]Goodbye![/bold blue]")


def _run_agent_turn(agent: ResearchAgent, user_message: str) -> None:
    """Execute one agent turn with live output."""

    def on_step(step_num: int, msg: str, tool_calls: list) -> None:
        ctx = agent.state.context_size()
        ctx_str = f"{ctx} chars" if ctx < 1024 else f"{ctx/1024:.1f} KB"

        if tool_calls:
            console.print(
                f"[dim]Agent thinking (Step {step_num + 1}, "
                f"Context: {ctx_str})…[/dim]"
            )
        else:
            # Final response — display it
            console.print(Markdown(msg))

    try:
        agent.respond(
            user_message,
            max_steps=30,
            on_step=on_step,
        )
    except Exception as e:
        console.print(f"[red]Agent error: {e}[/red]")
        logger.exception("Agent turn error")
