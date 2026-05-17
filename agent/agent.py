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

        result = self.tools.dispatch(tool_name, args)

        # Track written files
        if tool_name in ("file_write", "file_append"):
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

        for step_num in range(max_steps):
            msg, tool_calls = self.step()

            if on_step:
                on_step(step_num, msg, tool_calls)

            if not tool_calls:
                # Agent is done — this is the conversational response
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
