from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

from config import KnrsConfig
from agent.tools import AgentTools
from agent.prompts import SYSTEM_PROMPT

if TYPE_CHECKING:
    from agent.engine import AgentSession

logger = logging.getLogger(__name__)

class ResearchAgent:
    def __init__(self, config: KnrsConfig, session: AgentSession) -> None:
        """Initialize the research agent.

        Args:
            config:  Resolved KnrsConfig.
            session: An object with a generate(messages, max_tokens, temperature) method
                     (typically an AgentSession from agent.engine).
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
        """Find and repair JSON blocks containing 'tool' and 'args'."""
        calls: List[Dict[str, Any]] = []
        
        # 1. Native format check (high confidence)
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

        # 2. Aggressive search for JSON-like blocks
        # We look for anything that starts with {"tool" or similar
        potential_blocks = []
        
        # Look for markdown code blocks
        blocks = re.finditer(r'```(?:json)?\s*(.*?)(?:```|$)', text, re.DOTALL)
        for b in blocks:
            content = b.group(1).strip()
            if content.startswith('{') and '"tool"' in content:
                potential_blocks.append(content)
        
        # Also look for raw JSON outside blocks if nothing found yet
        if not potential_blocks:
            # Match from the first '{' that seems to be a tool call to the end of the text
            # This handles cases where the agent forgets the closing ``` or trailing braces
            matches = re.finditer(r'\{\s*"tool"\s*:\s*"[^"]+".*?\}', text, re.DOTALL)
            for m in matches:
                potential_blocks.append(m.group(0))
            
            # If still nothing, look for start of JSON and take until end
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
                    # Find content field start
                    c_start = raw.find('"content"')
                    v_start = raw.find('"', c_start + 9)
                    if v_start != -1:
                        # Find the last " before a closing brace
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

    def step(self) -> tuple[bool, str, list[dict[str, Any]]]:
        """
        Run one step of the agent loop.
        Returns: (is_done, agent_message, tool_calls)
        """
        response_text = self.session.generate(self.history, max_tokens=10000)
        self.history.append({"role": "assistant", "content": response_text})
        
        tool_calls = self._extract_tool_call(response_text)
        
        # TASK_COMPLETE check
        is_done_signal = "TASK_COMPLETE" in response_text.upper() or "TASK COMPLETE" in response_text.upper()
        
        # If the agent says it's done but we found NO valid tool calls, 
        # check if it *tried* to use a tool but we failed to parse it.
        if is_done_signal and not tool_calls:
            # Heuristic for failed tool call: contains {"tool": but no successful parse
            if '{"tool"' in response_text or '{\n  "tool"' in response_text:
                error_msg = "[SYSTEM ERROR]: It looks like you tried to use a tool, but the JSON format was invalid (e.g., missing closing braces or unescaped quotes). I could not execute the tool. Please REPEAT the tool call with correct, valid JSON, and DO NOT output 'TASK_COMPLETE' until the tool has successfully executed."
                self.history.append({"role": "user", "content": error_msg})
                # Prevent exit
                return False, response_text + f"\n\n{error_msg}", []
        
        is_done = is_done_signal and not tool_calls
        return is_done, response_text, tool_calls
        
    def execute_tool(self, tool_call: dict[str, Any]) -> str:
        """Execute the tool and append result to history."""
        tool_name = tool_call.get("tool")
        args: Dict[str, Any] = tool_call.get("args", {})
        
        is_blocked = False
        error_msg = ""
        
        # Check for exact repetition
        if tool_name != "file_list" and tool_call in self.call_history:
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
                            if smaller_len > 0 and (overlap / smaller_len) >= 0.8 and abs(len(words1) - len(words2)) <= 1:
                                if self.consecutive_blocks < 3:
                                    self.consecutive_blocks += 1
                                    break
                                else:
                                    is_blocked = True
                                    error_msg = f"Search blocked. Query '{query}' is too similar to past query '{past_query}'."
                                    break

        if is_blocked:
            self.consecutive_blocks += 1
            if self.consecutive_blocks >= 6:
                fatal_msg = f"[SYSTEM FATAL]: {error_msg} You have repeatedly ignored system blocks. Your search capabilities are now DISABLED. You MUST immediately use the 'file_write' tool to save your final synthesized research to a markdown document, and only then output 'TASK_COMPLETE'."
                self.history.append({"role": "user", "content": fatal_msg})
                return fatal_msg
            elif self.consecutive_blocks >= 3:
                block_msg = f"[SYSTEM BLOCK {self.consecutive_blocks}/3]: {error_msg} You MUST use fundamentally different keywords for new searches, please continue research with the previous search results, or if you have enough information, use the 'file_write' tool to save your research to a markdown document before outputting 'TASK_COMPLETE'."
                self.history.append({"role": "user", "content": block_msg})
                return block_msg
                                
        self.consecutive_blocks = 0
        self.call_history.append(tool_call)
        
        result = self.tools.dispatch(tool_name, args)
        
        tool_msg = f"Tool result for {tool_name}:\n{result}"
        self.history.append({"role": "user", "content": tool_msg})
        return result
