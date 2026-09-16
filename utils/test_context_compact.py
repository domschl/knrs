from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent.agent import ResearchAgent
from agent.context import (
    ConversationState,
    compact_history,
    parse_compact_trigger,
    trim_history,
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_MODEL_CONTEXT_CHARS,
)
from config import KnrsConfig, load_config
from repl.commands import cmd_compact


def test_parse_compact_trigger_percentage() -> None:
    # 90% of 1,000,000 chars = 900,000
    model_chars = 1_000_000
    assert parse_compact_trigger("90%", model_chars) == 900_000
    assert parse_compact_trigger("80 %", model_chars) == 800_000
    assert parse_compact_trigger("50%", model_chars) == 500_000
    assert parse_compact_trigger(0.75, model_chars) == 750_000


def test_parse_compact_trigger_absolute() -> None:
    model_chars = 1_000_000
    assert parse_compact_trigger(150_000, model_chars) == 150_000
    assert parse_compact_trigger("150000", model_chars) == 150_000
    assert parse_compact_trigger("150k", model_chars) == 150_000
    assert parse_compact_trigger("200K", model_chars) == 200_000


def test_parse_compact_trigger_tokens() -> None:
    model_chars = 1_000_000
    # 50,000 tokens * 4 chars/token = 200,000 chars
    assert parse_compact_trigger("50000 tokens", model_chars) == 200_000
    assert parse_compact_trigger("50k tokens", model_chars) == 200_000


def test_parse_compact_trigger_fallback() -> None:
    model_chars = 100_000
    assert parse_compact_trigger("invalid_spec", model_chars) == 90_000
    assert parse_compact_trigger("", model_chars) == 90_000


def test_compact_history_preserves_system_and_tail() -> None:
    state = ConversationState()
    state.history.append({"role": "system", "content": "You are a research agent."})

    # Add 12 turns of user/assistant/tool messages
    for i in range(12):
        state.history.append({"role": "user", "content": f"User query {i} regarding topic {i}"})
        state.history.append({
            "role": "user",
            "content": f"Tool result for vector_search:\nFound books:History/Rome.md#L{i*10} with score 0.9.\n" + ("x" * 2000),
        })
        state.history.append({
            "role": "assistant",
            "content": f"- Finding {i}: Rome was built gradually, see [books:History/Rome.md#L{i*10}].",
        })

    before_chars = state.context_size()
    assert before_chars > 20_000

    # Compact down to 5000 chars
    res = compact_history(state, max_chars=5000, preserve_tail=4, force=True)
    assert res["compacted"] is True
    assert res["after_chars"] < before_chars
    assert res["saved_chars"] > 0

    # Check that system prompt is still intact at index 0
    assert state.history[0]["role"] == "system"
    assert state.history[0]["content"] == "You are a research agent."

    # Check that the tail messages (last 4) are preserved verbatim
    assert state.history[-1]["role"] == "assistant"
    assert "Finding 11" in state.history[-1]["content"]

    # Check that middle contains the synthesized research state
    assert any("[COMPACTED CONTEXT STATE" in m["content"] for m in state.history)
    # Check that citations were preserved in the distilled summary
    assert any("books:History/Rome.md" in m["content"] for m in state.history)


def test_compact_history_tool_distillation() -> None:
    state = ConversationState()
    state.history.append({"role": "system", "content": "System prompt"})
    state.history.append({"role": "user", "content": "Initial question"})
    state.history.append({
        "role": "user",
        "content": "Tool result for file_read:\nLine 1\nLine 2\n" + ("Long text " * 500) + "\nbooks:Philosophy/Kant.md#L45",
    })
    state.history.append({"role": "assistant", "content": "Analyzed Kant's work."})
    state.history.append({"role": "user", "content": "Follow-up question"})
    state.history.append({"role": "assistant", "content": "Final answer"})

    before_chars = state.context_size()
    # Prune with target that triggers tool distillation
    res = compact_history(state, max_chars=before_chars - 100, preserve_tail=2, force=False)
    assert res["compacted"] is True
    assert res["after_chars"] < before_chars
    # Citations in tool output must be preserved
    assert any("books:Philosophy/Kant.md" in m["content"] for m in state.history)


def test_cmd_compact_minimal_context(capsys: pytest.CaptureFixture[str]) -> None:
    state = ConversationState()
    state.history.append({"role": "system", "content": "System prompt"})
    state.history.append({"role": "user", "content": "Hi"})

    mock_cfg = MagicMock(spec=KnrsConfig)
    with patch("repl.repl._get_current_state", return_value=state):
        cmd_compact([], mock_cfg)

    captured = capsys.readouterr()
    assert "already minimal" in captured.out


def test_cmd_compact_execution(capsys: pytest.CaptureFixture[str]) -> None:
    state = ConversationState()
    state.history.append({"role": "system", "content": "System prompt"})
    for i in range(10):
        state.history.append({"role": "user", "content": f"Query {i}"})
        state.history.append({"role": "user", "content": f"Tool result for search:\n" + ("Result data " * 100)})
        state.history.append({"role": "assistant", "content": f"Answer {i}"})

    mock_cfg = MagicMock(spec=KnrsConfig)
    with patch("repl.repl._get_current_state", return_value=state):
        with patch("repl.repl._current_agent", None):
            cmd_compact(["5000"], mock_cfg)

    captured = capsys.readouterr()
    assert "Context compacted successfully" in captured.out
    assert "Messages:" in captured.out
    assert "Saved:" in captured.out


def test_config_context_compact_trigger(tmp_path: Path) -> None:
    cfg_file = tmp_path / "knrs.json"
    cfg_data = {
        "calibre_path": str(tmp_path / "calibre"),
        "notes_path": str(tmp_path / "notes"),
        "knrs_data": str(tmp_path / "data"),
        "wiki_path": str(tmp_path / "wiki"),
        "vector_db_path": str(tmp_path / "vdb"),
        "context_compact_trigger": "85%",
        "model_context_size": 131072,
    }
    cfg_file.write_text(json.dumps(cfg_data), encoding="utf-8")

    cfg = load_config(cfg_file)
    assert cfg.context_compact_trigger == "85%"
    assert cfg.model_context_size == 131072


def test_config_context_compact_trigger_default(tmp_path: Path) -> None:
    cfg_file = tmp_path / "knrs.json"
    cfg_data = {
        "calibre_path": str(tmp_path / "calibre"),
        "notes_path": str(tmp_path / "notes"),
        "knrs_data": str(tmp_path / "data"),
        "wiki_path": str(tmp_path / "wiki"),
        "vector_db_path": str(tmp_path / "vdb"),
    }
    cfg_file.write_text(json.dumps(cfg_data), encoding="utf-8")

    cfg = load_config(cfg_file)
    assert cfg.context_compact_trigger == "90%"
    assert cfg.model_context_size is None


def test_config_model_context_size_string_units(tmp_path: Path) -> None:
    cfg_file = tmp_path / "knrs.json"
    cfg_data = {
        "calibre_path": str(tmp_path / "calibre"),
        "notes_path": str(tmp_path / "notes"),
        "knrs_data": str(tmp_path / "data"),
        "wiki_path": str(tmp_path / "wiki"),
        "vector_db_path": str(tmp_path / "vdb"),
        "model_context_size": "128k",
    }
    cfg_file.write_text(json.dumps(cfg_data), encoding="utf-8")

    cfg = load_config(cfg_file)
    assert cfg.model_context_size == 128_000


def test_agent_respond_no_fixed_step_limit(tmp_path: Path) -> None:
    cfg = MagicMock(spec=KnrsConfig)
    cfg.context_compact_trigger = "90%"
    cfg.model_context_size = None
    cfg.wiki_path = tmp_path / "wiki"
    cfg.notes_path = tmp_path / "notes"
    cfg.calibre_path = tmp_path / "calibre"
    cfg.knrs_data = tmp_path / "data"
    cfg.vector_db_path = tmp_path / "vdb"
    session = MagicMock()
    session.get_context_window_chars.return_value = 1_000_000

    agent = ResearchAgent(cfg, session)

    # Simulate an agent doing 35 steps of tools and then finishing on step 36
    step_count = 0

    def mock_step() -> tuple[str, list[dict[str, Any]]]:
        nonlocal step_count
        step_count += 1
        if step_count <= 35:
            return f"Step {step_count}", [{"tool": "vector_search", "args": {"query": "test"}}]
        return "Final answer after extensive research.", []

    with patch.object(agent, "step", side_effect=mock_step), \
         patch.object(agent, "execute_tool", return_value="tool output"):
        final_text, tool_actions = agent.respond("Research deep history")

    assert step_count == 36
    assert len(tool_actions) == 35
    assert "Final answer" in final_text


def test_agent_respond_explicit_max_steps(tmp_path: Path) -> None:
    cfg = MagicMock(spec=KnrsConfig)
    cfg.context_compact_trigger = "90%"
    cfg.model_context_size = None
    cfg.wiki_path = tmp_path / "wiki"
    cfg.notes_path = tmp_path / "notes"
    cfg.calibre_path = tmp_path / "calibre"
    cfg.knrs_data = tmp_path / "data"
    cfg.vector_db_path = tmp_path / "vdb"
    session = MagicMock()
    session.get_context_window_chars.return_value = 1_000_000

    agent = ResearchAgent(cfg, session)
    step_count = 0

    def mock_step() -> tuple[str, list[dict[str, Any]]]:
        nonlocal step_count
        step_count += 1
        return f"Step {step_count}", [{"tool": "vector_search", "args": {"query": "test"}}]

    with patch.object(agent, "step", side_effect=mock_step), \
         patch.object(agent, "execute_tool", return_value="tool output"):
        final_text, tool_actions = agent.respond("Research deep history", max_steps=5)

    assert step_count == 5
    assert len(tool_actions) == 5
    assert final_text == "Step 5"

