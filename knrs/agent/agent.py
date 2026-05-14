import json
import logging
import re
from pathlib import Path

from knrs.config import KnrsConfig
from knrs.agent.tools import AgentTools
from knrs.agent.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class ResearchAgent:
    def __init__(self, config: KnrsConfig, session):
        """Initialize the research agent.

        Args:
            config:  Resolved KnrsConfig.
            session: An object with a generate(messages, max_tokens, temperature) method
                     (typically an AgentSession from knrs.agent.engine).
        """
        self.config = config
        self.session = session
        self.tools = AgentTools(config)
        self.history = []
        self.call_history = []
        
        self.history.append({"role": "system", "content": SYSTEM_PROMPT})
        
    def load_checkpoint(self, path: Path):
        """Load conversation history from a JSON file."""
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                self.history = json.load(f)
                
    def save_checkpoint(self, path: Path):
        """Save conversation history to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    def _extract_tool_call(self, text: str) -> list[dict]:
        """Find all JSON blocks containing 'tool' and 'args'."""
        calls = []
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
                if "tool" in parsed and "args" in parsed:
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

    def step(self) -> tuple[bool, str, list[dict]]:
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
        
    def execute_tool(self, tool_call: dict) -> str:
        """Execute the tool and append result to history."""
        tool_name = tool_call.get("tool")
        args = tool_call.get("args", {})
        
        result = self.tools.dispatch(tool_name, args)
        
        # Check for repetition
        warning = ""
        if tool_call in self.call_history:
            warning = "\n\n[SYSTEM WARNING]: You have called this tool with identical arguments before. Repeating actions will not change the results. Please refine your query, read a different file/range, or move on to synthesis if you have enough info."
        self.call_history.append(tool_call)
        
        tool_msg = f"Tool result for {tool_name}:\n{result}{warning}"
        self.history.append({"role": "user", "content": tool_msg})
        return result
