"""
agent.agent — Conversational research agent for the knrs REPL.

The agent operates in multi-turn conversational mode: it receives user
messages, optionally invokes tools, and returns text responses.  There is
no autonomous "TASK_COMPLETE" signal — the agent simply yields control
back to the user after each response.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from config import KnrsConfig
from agent.tools import AgentTools
from agent.prompts import SYSTEM_PROMPT
from agent.context import ConversationState, trim_history

if TYPE_CHECKING:
    from agent.engine import AgentSession

logger = logging.getLogger(__name__)


class ResearchAgent:
    """Multi-turn conversational research agent.

    Maintains a persistent conversation via *state* and delegates
    generation to an *AgentSession* backend.
    """

    def __init__(
        self,
        config: KnrsConfig,
        session: AgentSession,
        state: ConversationState | None = None,
    ) -> None:
        self.config = config
        self.session = session
        self.tools = AgentTools(config)

        if state is not None:
            self.state = state
        else:
            self.state = ConversationState()
            self.state.history.append({"role": "system", "content": SYSTEM_PROMPT})

    # ── tool-call extraction (unchanged) ─────────────────────────────

    def _extract_tool_call(self, text: str) -> list[dict[str, Any]]:
        """Find and repair JSON blocks containing 'tool' and 'args'."""
        calls: List[Dict[str, Any]] = []

        # 1. Native format check (high confidence)
        gemma_matches = re.finditer(r'<\|tool_call\>call:(\w+)(\{.*?\})<tool_call\|\>', text)
        for gemma_match in gemma_matches:
            tool_name = gemma_match.group(1)
            args_str = gemma_match.group(2)
            import yaml
            try:
                args = yaml.safe_load(args_str)
                if isinstance(args, dict):
                    calls.append({"tool": tool_name, "args": args})
            except Exception:
                pass

        # 2. Aggressive search for JSON-like blocks
        potential_blocks = []

        # Look for markdown code blocks
        blocks = re.finditer(r'```(?:json)?\s*(.*?)(?:```|$)', text, re.DOTALL)
        for b in blocks:
            content = b.group(1).strip()
            if content.startswith('{') and '"tool"' in content:
                potential_blocks.append(content)

        # Also look for raw JSON outside blocks if nothing found yet
        if not potential_blocks:
            matches = re.finditer(r'\{\s*"tool"\s*:\s*"[^"]+?".*?\}', text, re.DOTALL)
            for m in matches:
                potential_blocks.append(m.group(0))

            if not potential_blocks:
                start = text.find('{"tool"')
                if start == -1:
                    start = text.find('{\n  "tool"')
                if start != -1:
                    potential_blocks.append(text[start:])

        def try_parse(raw: str) -> Optional[Dict[str, Any]]:
            # Try direct JSON
            try:
                p = json.loads(raw)
                if isinstance(p, dict) and "tool" in p: return p
            except Exception: pass

            # Try to repair trailing braces
            for i in range(1, 5):
                try:
                    p = json.loads(raw + ("}" * i))
                    if isinstance(p, dict) and "tool" in p: return p
                except Exception: pass

            # Try to repair trailing quote + braces
            for i in range(1, 5):
                try:
                    p = json.loads(raw + '"' + ("}" * i))
                    if isinstance(p, dict) and "tool" in p: return p
                except Exception: pass

            # Try YAML fallback (much more forgiving with strings/newlines)
            import yaml
            try:
                p = yaml.safe_load(raw)
                if isinstance(p, dict) and "tool" in p: return p
            except Exception: pass

            # Last ditch: try to escape unescaped internal quotes in "content"
            # This is specifically for file_write
            if '"content"' in raw:
                try:
                    c_start = raw.find('"content"')
                    v_start = raw.find('"', c_start + 9)
                    if v_start != -1:
                        v_end = raw.rfind('}', v_start)
                        v_end = raw.rfind('"', v_start, v_end)
                        if v_end != -1:
                            prefix = raw[:v_start+1]
                            middle = raw[v_start+1:v_end]
                            suffix = raw[v_end:]
                            repaired = prefix + middle.replace('"', '\\"') + suffix
                            p = json.loads(repaired)
                            if isinstance(p, dict) and "tool" in p: return p
                except Exception: pass

            return None

        for block in potential_blocks:
            parsed = try_parse(block)
            if parsed and parsed not in calls:
                calls.append(parsed)

        return calls

    # ── single generation step ──────────────────────────────────────

    def step(self) -> tuple[str, list[dict[str, Any]]]:
        """Run one generation step.

        Returns:
            (agent_message, tool_calls)
        """
        response_text = self.session.generate(
            self.state.history, max_tokens=10000
        )
        self.state.append_assistant(response_text)

        tool_calls = self._extract_tool_call(response_text)
        return response_text, tool_calls

    # ── tool execution ──────────────────────────────────────────────

    def execute_tool(self, tool_call: dict[str, Any]) -> str:
        """Execute a tool and append result to history."""
        tool_name = tool_call.get("tool")
        args: Dict[str, Any] = tool_call.get("args", {})
        
        # Fallback if the agent mistakenly put arguments at the top level instead of in 'args'
        if not args or not isinstance(args, dict):
            args = {k: v for k, v in tool_call.items() if k not in ("tool", "args")}

        is_blocked = False
        error_msg = ""

        # Check for exact repetition
        if tool_name != "file_list" and tool_call in self.state.call_history:
            is_blocked = True
            error_msg = f"You have already executed `{tool_name}` with these exact arguments."

        # Check for highly similar vector searches
        elif tool_name == "vector_search":
            query: str = args.get("query", "")
            words1 = set(re.findall(r'\w+', query.lower()))
            if words1:
                for past_call in self.state.call_history:
                    if past_call.get("tool") == "vector_search":
                        past_query: str = past_call.get("args", {}).get("query", "")
                        words2 = set(re.findall(r'\w+', past_query.lower()))
                        if words2:
                            overlap = len(words1.intersection(words2))
                            smaller_len = min(len(words1), len(words2))
                            if smaller_len > 0 and (overlap / smaller_len) >= 0.8 and abs(len(words1) - len(words2)) <= 1:
                                is_blocked = True
                                error_msg = f"Search query '{query}' is too similar to past query '{past_query}'."
                                break

        if is_blocked:
            self.state.consecutive_blocks += 1
            block_msg = f"[SYSTEM]: {error_msg} Please use different keywords or proceed with the information you already have."
            self.state.append_tool_result(tool_name, block_msg)
            return block_msg

        self.state.consecutive_blocks = 0
        self.state.call_history.append(tool_call)

        # Print concise 1-line tool execution log
        from rich.console import Console
        c = Console()
        summary = ""
        if tool_name == "vector_search":
            summary = f"query: '{args.get('query', '')}'"
        elif tool_name == "file_read":
            summary = f"file: '{args.get('path', '')}' (lines {args.get('start_line', 1)} to {args.get('end_line', -1)})"
        elif tool_name == "file_list":
            summary = f"dir: '{args.get('directory', '')}'"
        elif tool_name == "timeline_query":
            summary = f"filter: {args}"
        elif tool_name == "file_write":
            summary = f"write: '{args.get('path', '')}'"
        elif tool_name == "file_append":
            summary = f"append: '{args.get('path', '')}'"
        elif tool_name == "create_directory":
            summary = f"mkdir: '{args.get('path', '')}'"
        elif tool_name == "file_move":
            summary = f"move: '{args.get('src', '')}' to '{args.get('dst', '')}'"
        elif tool_name == "wikipedia_search":
            summary = f"wiki search: '{args.get('query', '')}'"
        elif tool_name == "wikipedia_fetch":
            summary = f"wiki fetch: '{args.get('title', '')}'"
        elif tool_name == "wikilink_search":
            summary = f"wikilink search: '{args.get('query', '')}'"
        elif tool_name == "check_wiki":
            summary = "checking AINotes/Research/ metadata"
        elif tool_name == "update_index":
            summary = "updating vector index"
        elif tool_name == "extract_timeline":
            summary = f"extract timeline: '{args.get('path', '')}'"
        else:
            summary = str(args)

        c.print(f"[bold cyan]  → {tool_name}[/bold cyan] [dim]({summary})…[/dim]")
        logger.info(f"Executing tool {tool_name} with args: {summary}")

        try:
            result = self.tools.dispatch(tool_name, args)
        except TypeError as e:
            result = f"Error: Invalid arguments for {tool_name}. Please check the required parameters. ({e})"
        except Exception as e:
            result = f"Error executing {tool_name}: {e}"

        # Track written files
        if tool_name in ("file_write", "file_append") and not str(result).startswith("Error"):
            path_str = args.get("path", "")
            if path_str and path_str not in self.state.written_files:
                self.state.written_files.append(path_str)

        self.state.append_tool_result(tool_name, result)
        return result

    # ── multi-turn respond (main REPL entry point) ──────────────────

    def respond(
        self,
        user_message: str,
        *,
        max_steps: int = 30,
        on_step: Any = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Process one user message through the full agent loop.

        Appends the user message, then runs step() in a loop until the
        agent produces a response with no tool calls or the step limit
        is hit.

        If the agent performed research (used search/read tools) but wrote
        a long chat response without saving a file, it gets a one-time
        reminder that file_write is mandatory before the response is
        accepted as final.

        Args:
            user_message: The user's free-text input.
            max_steps:    Maximum number of generation steps.
            on_step:      Optional callback ``(step_num, msg, tool_calls) -> None``
                          for live display in the REPL.

        Returns:
            (final_text_response, all_tool_actions)
        """
        # Trim if needed before adding the new message
        trim_history(self.state)

        self.state.append_user(user_message)

        all_tool_actions: List[Dict[str, Any]] = []
        final_text = ""
        files_written_before = set(self.state.written_files)
        _write_nudge_sent = False  # only nudge once per turn

        # Classify which tools were used this turn
        _RESEARCH_TOOLS = {
            "vector_search", "file_read", "wikipedia_search",
            "wikipedia_fetch", "timeline_query",
        }

        for step_num in range(max_steps):
            msg, tool_calls = self.step()

            if on_step:
                on_step(step_num, msg, tool_calls)

            if not tool_calls:
                # Agent produced a text-only response.
                # Check: did it do research this turn but skip file_write?
                if not _write_nudge_sent:
                    used_research = any(
                        tc.get("tool") in _RESEARCH_TOOLS
                        for tc in all_tool_actions
                    )
                    new_files = set(self.state.written_files) - files_written_before
                    
                    if len(msg) > 500 and not new_files:
                        _write_nudge_sent = True
                        if used_research:
                            nudge = (
                                "[SYSTEM]: You have conducted research and produced a detailed "
                                "response, but you have NOT saved it to a file. "
                                "This is REQUIRED. Please use file_write to save your findings "
                                "to AINotes/Research/ NOW, then follow with check_wiki() and "
                                "update_index(). After saving, you may write a brief chat "
                                "message confirming what was saved and where."
                            )
                        else:
                            nudge = (
                                "[SYSTEM]: You have written a detailed response relying entirely on "
                                "your pre-trained knowledge WITHOUT using any local research tools. "
                                "This is unacceptable. You MUST use tools (like vector_search, file_read) "
                                "to research the local knowledge base, and then save your findings using "
                                "file_write. Please begin your research now."
                            )
                        self.state.append_user(nudge)
                        continue  # loop again — don't accept this as the final response

                final_text = msg
                break

            # Execute tool calls
            for tc in tool_calls:
                all_tool_actions.append(tc)
                self.execute_tool(tc)
        else:
            # Reached max_steps
            final_text = msg  # type: ignore[possibly-undefined]

        return final_text, all_tool_actions

