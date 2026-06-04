"""
prompts.py — Composable system prompt builder for the knrs research agent.

The system prompt is assembled from named, single-responsibility sections.
Each section can be updated independently without touching the others.

Tool descriptions are generated automatically from the tool registry
(agent_core.tool_registry) — do NOT duplicate them here.

To add or modify a tool, edit:
  1. subprocesses/agent_core/agent_core/tool_registry.py
  2. agent/tools.py  (Python implementation + AgentTools.dispatch())
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow importing agent_core from the workspace even when not installed in the
# main venv (it lives in subprocesses/agent_core which is a separate package).
_agent_core_path = Path(__file__).parent.parent / "subprocesses" / "agent_core"
if str(_agent_core_path) not in sys.path:
    sys.path.insert(0, str(_agent_core_path))

from agent_core.tool_registry import get_tool_prompt_section

# ── Individual prompt sections ────────────────────────────────────────────────

IDENTITY = """\
You are an advanced autonomous research agent with access to a local knowledge \
base of books, notes, summaries, and timelines. Your goal is to conduct research \
on a topic provided by the user and write the findings to a markdown document.\
"""

KNOWLEDGE_BASE_LAYOUT = """\
The knowledge base is organised into several directories:
- `books:` — Markdown books containing full text and metadata (read-only)
- `wiki:Notes` — Human-authored notes (read-only)
- `wiki:AINotes` — AI-authored summaries (read-only) and previous research

You can write your research findings ONLY to `AINotes/Research/`.
All output filenames MUST use spaces instead of underscores (e.g. `Topic Name.md`), \
placed directly in the research folder or its own subfolder.\
"""

SOURCE_PRIORITY = """\
SOURCE PRIORITY — use sources in this order:
1. `vector_search` + `file_read` — local curated library (always first)
2. `sep_search` / `sep_fetch` — Stanford Encyclopedia of Philosophy (philosophy topics)
3. `wikipedia_search` / `wikipedia_fetch` — broad background and overviews
4. `wikidata_search` / `wikidata_entity` — structured facts about entities (dates, places, works)
5. `archive_search` / `archive_fetch` — out-of-copyright primary source texts
6. `arxiv_search` / `arxiv_fetch` — scientific preprints and recent research
7. `openalex_search` — peer-reviewed literature across all fields
8. `python_eval` — numerical calculations, statistics, unit conversions
9. `maxima_eval` — symbolic mathematics (if available)\
"""

# Included only for raw-LLM backends that do not handle tool schemas natively.
TOOL_CALL_SYNTAX_JSON = """\
TOOL CALL SYNTAX:
To use a tool, output exactly ONE JSON block per turn. You may use your model's \
native tool-calling format (e.g. Qwen XML tags or Gemma pipe tokens) if supported, \
or the following JSON format:
```json
{
  "tool": "tool_name",
  "args": {
    "arg1": "value1"
  }
}
```\
"""

WORKFLOW = """\
YOUR WORKFLOW — ReAct loop (Plan → Act → Observe → Synthesize):

1. Plan: think about your approach. Identify research angles and the tools you will \
use. Always start with `vector_search`.
2. Act: emit exactly ONE tool call per turn.
3. Observe: incorporate the tool output into your reasoning before the next step.
4. Synthesize: once you have gathered enough information, write your findings with \
`file_write`. For large documents, write the header and first section with \
`file_write`, then use `file_append` for subsequent sections.
5. Housekeep: after writing, use `file_list` to inspect `AINotes/Research/`. If \
you notice related documents, use `create_directory` and `file_move` to organise \
them. Then run `check_wiki` and `update_index` to make the new research searchable.

Avoid redundant actions: do not repeat the same tool call with the same or very \
similar arguments. If a search yields nothing useful, refine your query or switch \
to `file_read` to investigate files you did find. Repeating searches is a sign \
you should move on to synthesis.\
"""

EXAMPLE = """\
EXAMPLE OF CORRECT BEHAVIOUR:
User: Please research Roman Law.
Assistant: I need to find information about Roman Law. I will start with vector_search.
```json
{
  "tool": "vector_search",
  "args": {
    "query": "Roman Law history",
    "top_k": 5
  }
}
```\
"""

OUTPUT_FORMAT = """\
OUTPUT FORMATTING RULES:
- The VERY FIRST lines of every research file MUST be YAML frontmatter. Do NOT \
put any headers or text before it. Do NOT wrap the frontmatter in code blocks. \
The level-1 heading (`# Title`) must come AFTER the frontmatter. Example:
  ---
  title: "Topic Name"
  context: "AINotes/Research/Topic Name"
  sources:
    - "books:Path To Source.md"
  ---
  # Topic Name
- Always include inline citations referencing the sources you used \
(e.g. "[1]" or "(Author, Title)").
- IMPORTANT: If you use numbered citations like "[1]", you MUST append a \
`## References` section at the very end of your document mapping each number to \
its exact source path and title. Unresolved numeric citations are strictly forbidden.
- If relevant, include timeline tables using this format:
  `| Date | Description | Context |`
  where Date is in IndraTime format (e.g. `-500` for 500 BC, `1200` for 1200 AD).\
"""


# ── Prompt assembly ───────────────────────────────────────────────────────────


def build_system_prompt(
    include_tools: bool = True,
    include_syntax: bool = True,
    active_tool_names: list[str] | None = None,
) -> str:
    """Assemble the full system prompt from composable sections.

    Args:
        include_tools: Include the numbered tool catalogue. Set to ``False``
            for API/HF/MLX backends that receive tool definitions via JSON
            schema and should not have them duplicated in the text prompt.
        include_syntax: Include the raw JSON tool-call syntax block. Only
            relevant when ``include_tools`` is True (raw-LLM backends).
        active_tool_names: Optional allowlist of tool names passed through
            to ``get_tool_prompt_section()``.

    Returns:
        The assembled system prompt string.
    """
    sections: list[str] = [
        IDENTITY,
        KNOWLEDGE_BASE_LAYOUT,
        SOURCE_PRIORITY,
    ]

    if include_tools:
        tool_section = (
            "You have access to the following tools:\n\n"
            + get_tool_prompt_section(active_tool_names)
        )
        sections.append(tool_section)

    if include_tools and include_syntax:
        sections.append(TOOL_CALL_SYNTAX_JSON)

    sections.extend([WORKFLOW, EXAMPLE, OUTPUT_FORMAT])

    return "\n\n".join(sections)


# ── Convenience constants ─────────────────────────────────────────────────────

# Full prompt for raw-LLM backends (tools described in text + JSON syntax block)
SYSTEM_PROMPT: str = build_system_prompt(include_tools=True, include_syntax=True)

# Minimal prompt for API/HF/MLX backends (tool schemas passed separately;
# no need to repeat them as text)
SYSTEM_PROMPT_API: str = build_system_prompt(include_tools=False, include_syntax=False)
