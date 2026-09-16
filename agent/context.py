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
from typing import Any

logger = logging.getLogger(__name__)

# Default context budget in characters. Targets 256K-token models;
# 200K chars ≈ 50-60K tokens leaves headroom for the response.
DEFAULT_MAX_CONTEXT_CHARS = 200_000
DEFAULT_MODEL_CONTEXT_TOKENS = 262_144
DEFAULT_CHARS_PER_TOKEN = 4
DEFAULT_MODEL_CONTEXT_CHARS = DEFAULT_MODEL_CONTEXT_TOKENS * DEFAULT_CHARS_PER_TOKEN


@dataclass
class ConversationState:
    """Mutable state for one REPL conversation session."""

    history: list[dict[str, str]] = field(default_factory=list)
    call_history: list[dict[str, Any]] = field(default_factory=list)
    consecutive_blocks: int = 0
    written_files: list[str] = field(default_factory=list)

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


# ── Context Compact Trigger Parsing ──────────────────────────────────────────


def parse_compact_trigger(
    trigger: str | int | float,
    model_context_chars: int = DEFAULT_MODEL_CONTEXT_CHARS,
) -> int:
    """Calculate the context character threshold that triggers compaction.

    Args:
        trigger: Either a percentage string (e.g. "90%", "85.5%"), a float ratio
                 (e.g. 0.9), or an absolute context size as an int or string
                 (e.g. 180000, "180k", "200000 chars", "50000 tokens").
        model_context_chars: Total capacity of the model's context in characters.

    Returns:
        Integer threshold in characters (minimum 1,000).
    """
    import re

    if isinstance(trigger, str):
        s = trigger.strip()
        if s.endswith("%"):
            try:
                pct = float(s[:-1].strip())
                return max(1000, int((pct / 100.0) * model_context_chars))
            except ValueError:
                logger.warning("Invalid percentage trigger %r; falling back to 90%%", trigger)
                return max(1000, int(0.9 * model_context_chars))

        s_lower = s.lower()
        match = re.match(r"^([\d.]+)\s*([kmg]?)\s*(tokens?|t|chars?|c)?$", s_lower)
        if match:
            try:
                val = float(match.group(1))
                mult = match.group(2)
                unit = match.group(3) or ""
                if mult == "k":
                    val *= 1_000
                elif mult == "m":
                    val *= 1_000_000
                elif mult == "g":
                    val *= 1_000_000_000

                if unit.startswith("t"):
                    val *= DEFAULT_CHARS_PER_TOKEN
                return max(1000, int(val))
            except (ValueError, OverflowError):
                pass

        try:
            return max(1000, int(float(s)))
        except ValueError:
            logger.warning("Could not parse compact trigger %r; falling back to 90%%", trigger)
            return max(1000, int(0.9 * model_context_chars))


    if isinstance(trigger, (int, float)):
        if 0.0 < trigger <= 1.0:
            return max(1000, int(trigger * model_context_chars))
        return max(1000, int(trigger))

    return max(1000, int(0.9 * model_context_chars))


# ── Structured History Compaction ───────────────────────────────────────────

def _extract_citations(text: str) -> list[str]:
    """Extract source paths, wiki links, and citations from text."""
    import re
    citations: set[str] = set()
    # Match patterns like books:Path/To/Doc.md, wiki:Notes/X.md, AINotes/Research/X.md
    patterns = [
        r"(?:books|wiki):[A-Za-z0-9_\-./]+\.md(?:#L\d+(?:-L\d+)?)?",
        r"AINotes/[A-Za-z0-9_\-./]+\.md",
        r"https?://[^\s\)\],]+",
        r"\[\[([^\]]+)\]\]",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text):
            citations.add(m.group(0))
    return sorted(citations)


def _distill_tool_result(content: str, max_excerpt_len: int = 500) -> str:
    """Condense a large tool output while retaining tool name, paths, citations, and status."""
    lines = content.splitlines()
    first_line = lines[0] if lines else "Tool result:"
    
    # Check for failure or errors
    has_error = any("error" in line.lower() or "fail" in line.lower() for line in lines[:5])
    citations = _extract_citations(content)
    
    # If already small enough, keep as is
    if len(content) <= max_excerpt_len:
        return content

    distilled_parts = [first_line]
    if has_error:
        error_lines = [l for l in lines if "error" in l.lower() or "fail" in l.lower()][:3]
        distilled_parts.append("Status / Error: " + "; ".join(error_lines))
        
    if citations:
        distilled_parts.append("Sources & Citations: " + ", ".join(citations[:10]))
        
    # Take representative head and tail
    head_snippet = "\n".join(lines[1:6]).strip()
    if head_snippet:
        distilled_parts.append("Excerpt:\n" + head_snippet[:300])
        
    distilled_parts.append(f"[Output pruned: {len(content)} chars → retained key citations and status]")
    return "\n".join(distilled_parts)


def compact_history(
    state: ConversationState,
    max_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    *,
    preserve_tail: int = 6,
    force: bool = False,
) -> dict[str, Any]:
    """Compact older messages to bring total context within *max_chars*.

    Strategy:
        1. Always preserve the system prompt (index 0) verbatim.
        2. Always preserve the most recent *preserve_tail* messages verbatim.
        3. Distill bulky older tool results (observation pruning).
        4. If still exceeding *max_chars* or forced, synthesize older dialog into
           a structured research state (goals, cited sources, created files, findings).
        5. Maintain proper conversational turn alternation for chat templates.

    Returns:
        Dict with metrics: compacted (bool), before_chars, after_chars, saved_chars, percent_reduction.
    """
    before_chars = state.context_size()
    before_msgs = len(state.history)

    if not force and before_chars <= max_chars:
        return {
            "compacted": False,
            "before_chars": before_chars,
            "after_chars": before_chars,
            "saved_chars": 0,
            "percent_reduction": 0.0,
        }

    if before_msgs <= 2:
        return {
            "compacted": False,
            "before_chars": before_chars,
            "after_chars": before_chars,
            "saved_chars": 0,
            "percent_reduction": 0.0,
        }

    system_msg = state.history[0] if state.history[0]["role"] == "system" else None
    msgs = state.history[1:] if system_msg else state.history[:]

    # Calculate actual tail count to preserve
    actual_tail_count = min(preserve_tail, len(msgs))
    if actual_tail_count >= len(msgs) and len(msgs) > 2:
        actual_tail_count = max(2, len(msgs) // 2)

    tail = msgs[-actual_tail_count:] if actual_tail_count > 0 else []
    middle = msgs[:-actual_tail_count] if actual_tail_count > 0 else msgs[:]

    if not middle:
        return {
            "compacted": False,
            "before_chars": before_chars,
            "after_chars": before_chars,
            "saved_chars": 0,
            "percent_reduction": 0.0,
        }

    # Phase 1: Distill bulky tool results in middle
    distilled_middle: list[dict[str, str]] = []
    for msg in middle:
        role = msg["role"]
        content = msg.get("content", "")
        if role == "user" and content.startswith("Tool result for "):
            distilled_middle.append({"role": role, "content": _distill_tool_result(content)})
        else:
            distilled_middle.append(msg)

    # Check if Phase 1 brought context under budget (unless force=True)
    temp_size = (len(system_msg["content"]) if system_msg else 0) + \
                sum(len(m.get("content", "")) for m in distilled_middle) + \
                sum(len(m.get("content", "")) for m in tail)

    if not force and temp_size <= max_chars:
        new_history: list[dict[str, str]] = []
        if system_msg:
            new_history.append(system_msg)
        new_history.extend(distilled_middle)
        new_history.extend(tail)
        state.history = new_history
        after_chars = state.context_size()
        saved = max(0, before_chars - after_chars)
        logger.info(
            "Pruned middle tool results: %d → %d chars (%d msgs)",
            before_chars, after_chars, len(new_history)
        )
        return {
            "compacted": True,
            "before_chars": before_chars,
            "after_chars": after_chars,
            "saved_chars": saved,
            "percent_reduction": (saved / before_chars * 100.0) if before_chars > 0 else 0.0,
        }

    # Phase 2: Structured Research State Synthesis
    all_citations: set[str] = set()
    user_queries: list[str] = []
    findings: list[str] = []

    for msg in middle:
        role = msg["role"]
        content = msg.get("content", "")
        citations = _extract_citations(content)
        all_citations.update(citations)

        if role == "user":
            if not content.startswith("Tool result for ") and not content.startswith("[SYSTEM]"):
                clean_query = content.strip().replace("\n", " ")
                if clean_query and len(clean_query) > 5:
                    user_queries.append(clean_query[:200])
        elif role == "assistant":
            # Extract key insights or plan steps
            lines = [l.strip() for l in content.splitlines() if l.strip()]
            for line in lines:
                if line.startswith(("-", "*", "•", "1.", "2.", "3.", "4.", "5.", "#")):
                    findings.append(line[:250])

    summary_sections: list[str] = ["[COMPACTED CONTEXT STATE — Prior research & dialogue distilled]"]

    if user_queries:
        summary_sections.append("### User Goals & Directives:")
        for q in user_queries[-5:]:
            summary_sections.append(f"- {q}")

    if all_citations:
        summary_sections.append("### Verified Sources & Citations Consulted:")
        for c in sorted(all_citations)[:25]:
            summary_sections.append(f"- `{c}`")

    if state.written_files:
        summary_sections.append("### Research Files Created / Modified:")
        for wf in state.written_files:
            summary_sections.append(f"- `{wf}`")

    if findings:
        summary_sections.append("### Key Findings & Working Hypotheses:")
        for f in findings[:15]:
            summary_sections.append(f"- {f}")

    summary_text = "\n".join(summary_sections)

    new_history = []
    if system_msg:
        new_history.append(system_msg)
    new_history.append({"role": "user", "content": summary_text})
    new_history.append({
        "role": "assistant",
        "content": "Acknowledged. I retain the distilled research state, citations, and verified findings in memory. Continuing with the conversation.",
    })
    new_history.extend(tail)

    state.history = new_history
    after_chars = state.context_size()
    saved = max(0, before_chars - after_chars)

    logger.info(
        "Compacted conversation history: %d → %d messages (%d → %d chars, %.1f%% reduction)",
        before_msgs, len(new_history), before_chars, after_chars,
        (saved / before_chars * 100.0) if before_chars > 0 else 0.0,
    )

    return {
        "compacted": True,
        "before_chars": before_chars,
        "after_chars": after_chars,
        "saved_chars": saved,
        "percent_reduction": (saved / before_chars * 100.0) if before_chars > 0 else 0.0,
    }


def trim_history(
    state: ConversationState,
    max_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> None:
    """Backward-compatible wrapper for compact_history."""
    compact_history(state, max_chars=max_chars, force=False)



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
