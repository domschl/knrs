from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

from knrs.config import KnrsConfig
from knrs.agent.tools import AgentTools
from knrs.agent.prompts import SYSTEM_PROMPT

if TYPE_CHECKING:
    from knrs.agent.engine import AgentSession

logger = logging.getLogger(__name__)

class ResearchAgent:
    def __init__(self, config: KnrsConfig, session: AgentSession) -> None:
        """Initialize the research agent.

        Args:
            config:  Resolved KnrsConfig.
            session: An object with a generate(messages, max_tokens, temperature) method
                     (typically an AgentSession from knrs.agent.engine).
        """
        self.config = config
        self.session = session
        self.tools = AgentTools(config)
        self.history: List[Dict[str, str]] = []
        self.call_history: List[Dict[str, Any]] = []
        self.consecutive_blocks: int = 0
        
        self.history.append({"role": "system", "content": SYSTEM_PROMPT})
        
    def load_checkpoint(self, path: Path) -> None:
        """Load conversation history from a JSON file."""
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                self.history = json.load(f)
                
    def save_checkpoint(self, path: Path) -> None:
        """Save conversation history to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    def get_context_size(self) -> int:
        """Return the approximate context size in characters."""
        return len(json.dumps(self.history))

    def _extract_tool_call(self, text: str) -> list[dict[str, Any]]:
        """Find all JSON blocks containing 'tool' and 'args'."""
        calls: List[Dict[str, Any]] = []
        # First check for Gemma native tool call format
        gemma_matches = re.finditer(r'<\|tool_call>call:(\w+)(\{.*?\})<tool_call\|>', text)
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

        if calls:
            return calls

        # Look for code blocks first
        matches = re.finditer(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match.group(1))
                if isinstance(parsed, dict) and "tool" in parsed and "args" in parsed:
                    calls.append(parsed)
            except Exception:
                pass
                
        if calls:
            return calls

        # Fallback: scan for '{' and try to parse JSON by matching braces
        start_idx = 0
        while True:
            start_idx = text.find('{', start_idx)
            if start_idx == -1:
                break
            
            brace_count = 0
            in_string = False
            escape = False
            end_idx = -1
            
            for i in range(start_idx, len(text)):
                c = text[i]
                if escape:
                    escape = False
                elif c == '\\':
                    escape = True
                elif c == '"':
                    in_string = not in_string
                elif not in_string:
                    if c == '{':
                        brace_count += 1
                    elif c == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i
                            break
                            
            if end_idx != -1:
                try:
                    parsed = json.loads(text[start_idx:end_idx+1])
                    if isinstance(parsed, dict) and "tool" in parsed and "args" in parsed:
                        if parsed not in calls:
                            calls.append(parsed)
                except Exception:
                    pass
            
            start_idx += 1
            
        return calls

    def step(self) -> tuple[bool, str, list[dict[str, Any]]]:
        """
        Run one step of the agent loop.
        Returns: (is_done, agent_message, tool_calls)
        """
        response_text = self.session.generate(self.history, max_tokens=10000)
        self.history.append({"role": "assistant", "content": response_text})
        
        tool_calls = self._extract_tool_call(response_text)
        
        # Only mark as done if there's no tool call to execute
        is_done = ("TASK_COMPLETE" in response_text.upper() or "TASK COMPLETE" in response_text.upper()) and not tool_calls
        
        return is_done, response_text, tool_calls
        
    def execute_tool(self, tool_call: dict[str, Any]) -> str:
        """Execute the tool and append result to history."""
        tool_name = tool_call.get("tool")
        args: Dict[str, Any] = tool_call.get("args", {})
        
        is_blocked = False
        error_msg = ""
        
        # Check for exact repetition
        if tool_call in self.call_history:
            is_blocked = True
            error_msg = f"You have already executed `{tool_name}` with these exact arguments."

        # Check for highly similar vector searches
        elif tool_name == "vector_search":
            query: str = args.get("query", "")
            words1 = set(re.findall(r'\w+', query.lower()))
            if words1:
                for past_call in self.call_history:
                    if past_call.get("tool") == "vector_search":
                        past_query: str = past_call.get("args", {}).get("query", "")
                        words2 = set(re.findall(r'\w+', past_query.lower()))
                        if words2:
                            overlap = len(words1.intersection(words2))
                            smaller_len = min(len(words1), len(words2))
                            # If 80% of words overlap and length difference is small
                            if smaller_len > 0 and (overlap / smaller_len) >= 0.8 and abs(len(words1) - len(words2)) <= 2:
                                is_blocked = True
                                error_msg = f"Search blocked. Query '{query}' is too similar to past query '{past_query}'."
                                break

        if is_blocked:
            self.consecutive_blocks += 1
            if self.consecutive_blocks >= 3:
                fatal_msg = f"[SYSTEM FATAL]: {error_msg} You have repeatedly ignored system blocks. Your search capabilities are now DISABLED. You MUST immediately write your final synthesis based on the information you have and output 'TASK_COMPLETE'."
                self.history.append({"role": "user", "content": fatal_msg})
                return fatal_msg
            else:
                block_msg = f"[SYSTEM BLOCK {self.consecutive_blocks}/3]: {error_msg} You MUST use fundamentally different keywords, now please synthesize your findings and output 'TASK_COMPLETE' to finish the session."
                self.history.append({"role": "user", "content": block_msg})
                return block_msg
                                
        self.consecutive_blocks = 0
        self.call_history.append(tool_call)
        
        result = self.tools.dispatch(tool_name, args)
        
        tool_msg = f"Tool result for {tool_name}:\n{result}"
        self.history.append({"role": "user", "content": tool_msg})
        return result
