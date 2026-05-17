"""
agent.context — Conversation state management for the agentic REPL.

Provides ConversationState for multi-turn conversation tracking,
history trimming for long contexts, and session persistence.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Default context budget in characters.  Targets 256K-token models;
# 200K chars ≈ 50-60K tokens leaves headroom for the response.
DEFAULT_MAX_CONTEXT_CHARS = 200_000


@dataclass
class ConversationState:
    """Mutable state for one REPL conversation session."""

    history: List[Dict[str, str]] = field(default_factory=list)
    call_history: List[Dict[str, Any]] = field(default_factory=list)
    consecutive_blocks: int = 0
    written_files: List[str] = field(default_factory=list)

    # ── helpers ────────────────────────────────────────────────────────

    def context_size(self) -> int:
        """Approximate context size in characters."""
        return sum(len(m.get("content", "")) for m in self.history)

    def append_user(self, text: str) -> None:
        self.history.append({"role": "user", "content": text})

    def append_assistant(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text})

    def append_tool_result(self, tool_name: str, result: str) -> None:
        self.history.append({
            "role": "user",
            "content": f"Tool result for {tool_name}:\n{result}",
        })

    def reset(self, system_prompt: str) -> None:
        """Clear history and start fresh with the system prompt."""
        self.history.clear()
        self.history.append({"role": "system", "content": system_prompt})
        self.call_history.clear()
        self.consecutive_blocks = 0
        self.written_files.clear()


# ── History trimming ──────────────────────────────────────────────────────────

def trim_history(
    state: ConversationState,
    max_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> None:
    """Compress older messages when the context exceeds *max_chars*.

    Strategy:
        1. Always keep the system prompt (index 0).
        2. Always keep the last *tail_count* messages verbatim.
        3. Summarize everything in between into a single compressed message.
    """
    if state.context_size() <= max_chars:
        return

    tail_count = 20  # keep last N messages verbatim

    if len(state.history) <= tail_count + 1:
        # Not enough messages to compress meaningfully.
        return

    system_msg = state.history[0] if state.history[0]["role"] == "system" else None
    tail = state.history[-tail_count:]
    middle = state.history[1:-tail_count] if system_msg else state.history[:-tail_count]

    # Build a compact summary of the middle section.
    summary_parts: List[str] = []
    for msg in middle:
        role = msg["role"]
        content = msg.get("content", "")
        # Keep a short excerpt of each message.
        excerpt = content[:300]
        if len(content) > 300:
            excerpt += "…"
        summary_parts.append(f"[{role}] {excerpt}")

    summary_text = (
        "[CONTEXT SUMMARY — older messages compressed]\n"
        + "\n".join(summary_parts)
    )

    new_history: List[Dict[str, str]] = []
    if system_msg:
        new_history.append(system_msg)
    new_history.append({"role": "user", "content": summary_text})
    new_history.extend(tail)

    state.history = new_history
    logger.info(
        "Trimmed conversation history: %d → %d messages (%d chars)",
        len(middle) + len(tail) + (1 if system_msg else 0),
        len(new_history),
        state.context_size(),
    )


# ── Session persistence ──────────────────────────────────────────────────────

def save_session(state: ConversationState, path: Path) -> None:
    """Serialize conversation state to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "history": state.history,
        "call_history": state.call_history,
        "written_files": state.written_files,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Session saved to %s", path)


def load_session(path: Path) -> ConversationState:
    """Deserialize conversation state from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    state = ConversationState(
        history=payload.get("history", []),
        call_history=payload.get("call_history", []),
        written_files=payload.get("written_files", []),
    )
    logger.info(
        "Session loaded from %s (%d messages)", path, len(state.history)
    )
    return state
