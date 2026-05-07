import json
import logging
import re
from pathlib import Path

from knrs.config import KnrsConfig
from knrs.agent.llm_client import LLMClient
from knrs.agent.tools import AgentTools
from knrs.agent.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class ResearchAgent:
    def __init__(self, config: KnrsConfig, model_name: str):
        self.config = config
        self.client = LLMClient(model_name)
        self.tools = AgentTools(config)
        self.history = []
        
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

        # Fallback to finding anything that looks like JSON
        matches = re.finditer(r'(\{[\s\S]*?"tool"[\s\S]*?\})', text)
        for match in matches:
            try:
                parsed = json.loads(match.group(1))
                if "tool" in parsed and "args" in parsed:
                    if parsed not in calls:
                        calls.append(parsed)
            except Exception:
                pass
                
        return calls

    def step(self) -> tuple[bool, str, list[dict]]:
        """
        Run one step of the agent loop.
        Returns: (is_done, agent_message, tool_calls)
        """
        response_text = self.client.generate(self.history, max_tokens=2500)
        self.history.append({"role": "assistant", "content": response_text})
        
        tool_calls = self._extract_tool_call(response_text)
        
        # Only mark as done if there's no tool call to execute
        is_done = "TASK_COMPLETE" in response_text and not tool_calls
        
        return is_done, response_text, tool_calls
        
    def execute_tool(self, tool_call: dict) -> str:
        """Execute the tool and append result to history."""
        tool_name = tool_call.get("tool")
        args = tool_call.get("args", {})
        
        result = self.tools.dispatch(tool_name, args)
        
        tool_msg = f"Tool result for {tool_name}:\n{result}"
        self.history.append({"role": "user", "content": tool_msg})
        return result
