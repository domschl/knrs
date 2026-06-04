"""
tool_registry.py — Single Source of Truth for all agent tool definitions.

Adding or modifying a tool requires changes in exactly TWO places:
  1. This file (tool definition + dispatch)
  2. agent/tools.py (Python implementation + AgentTools.dispatch())

All JSON schemas and natural-language prompt sections are generated
automatically from this registry.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolParam:
    """Describes a single parameter of a tool."""

    name: str
    type: str  # "string", "integer", "number", "boolean", "array"
    description: str
    required: bool = True
    default: Any = None
    items_type: str | None = None  # element type for "array" params


@dataclass
class ToolDef:
    """Canonical definition of one agent tool.

    Generates both the OpenAI-compatible JSON schema (for API/HF backends)
    and the numbered natural-language prompt entry (for raw-LLM backends).
    """

    name: str
    description: str
    params: list[ToolParam] = field(default_factory=list)
    example_args: dict[str, Any] = field(default_factory=dict)
    # Logical grouping — used for documentation only
    category: str = "general"
    # If set, this tool is silently omitted when the binary is not on PATH
    requires_binary: str | None = None

    def to_json_schema(self) -> dict[str, Any]:
        """Generate an OpenAI-compatible function tool schema dict."""
        properties: dict[str, Any] = {}
        required_params: list[str] = []

        for p in self.params:
            prop: dict[str, Any] = {"type": p.type, "description": p.description}
            if p.type == "array" and p.items_type:
                prop["items"] = {"type": p.items_type}
            if p.default is not None:
                prop["default"] = p.default
            properties[p.name] = prop
            if p.required:
                required_params.append(p.name)

        schema: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                },
            },
        }
        if required_params:
            schema["function"]["parameters"]["required"] = required_params
        return schema

    def to_prompt_text(self, number: int) -> str:
        """Generate a numbered natural-language tool entry for raw-LLM prompts."""
        param_sig = ", ".join(p.name for p in self.params)
        lines = [f"{number}. `{self.name}({param_sig})`"]
        lines.append(f"   {self.description}")
        for p in self.params:
            req_marker = "" if p.required else " *(optional)*"
            default_str = f" (default: {p.default})" if p.default is not None else ""
            lines.append(f"   - `{p.name}` ({p.type}){req_marker}{default_str}: {p.description}")
        if self.example_args:
            ex = json.dumps({"tool": self.name, "args": self.example_args})
            lines.append(f"   Example: {ex}")
        return "\n".join(lines)

    def get_param_types(self) -> dict[str, str]:
        """Return a {param_name: type} mapping — used by coerce_tool_args."""
        return {p.name: p.type for p in self.params}


# ── Canonical tool registry ───────────────────────────────────────────────────
#
# Add new tools HERE. Everything else (schemas, prompts, coercion) is derived.
#
TOOL_REGISTRY: list[ToolDef] = [
    # ── Local knowledge-base tools ────────────────────────────────────────────
    ToolDef(
        name="vector_search",
        description=(
            "Semantic search across all indexed files of the local knowledge base. "
            "Primary source for research — always start here."
        ),
        params=[
            ToolParam("query", "string", "The search phrase."),
            ToolParam(
                "top_k",
                "integer",
                "Number of results to return.",
                required=False,
                default=5,
            ),
        ],
        example_args={"query": "Roman Law history", "top_k": 5},
        category="local",
    ),
    ToolDef(
        name="file_read",
        description=(
            "Read lines from a specific file. Useful to read more context around a "
            "search result snippet or a wikilink target. Supports absolute paths, "
            "prefixed paths (e.g. 'books:History Of Rome.md'), bare stems "
            "(e.g. 'History Of Rome'), or bracketed wiki-links (e.g. '[[History Of Rome]]'). "
            "Use -1 for end_line to read to the end of the file."
        ),
        params=[
            ToolParam(
                "path",
                "string",
                "The path to read. Supports prefixed paths, bare stems, or bracketed links.",
            ),
            ToolParam("start_line", "integer", "The starting line number (1-indexed)."),
            ToolParam("end_line", "integer", "The ending line number (use -1 to read to the end)."),
        ],
        example_args={"path": "[[History Of Rome]]", "start_line": 1, "end_line": 100},
        category="local",
    ),
    ToolDef(
        name="file_list",
        description="List files in a given directory prefix.",
        params=[
            ToolParam(
                "directory",
                "string",
                "The directory prefix (e.g. 'wiki:Notes', 'books:', 'wiki:AINotes/Research').",
            )
        ],
        example_args={"directory": "wiki:Notes"},
        category="local",
    ),
    ToolDef(
        name="timeline_query",
        description=(
            "Query the local parsed timeline events database. All arguments are optional. "
            "Returns a formatted markdown table of events."
        ),
        params=[
            ToolParam(
                "start_year",
                "integer",
                "Start year filter (negative for BC).",
                required=False,
            ),
            ToolParam(
                "end_year",
                "integer",
                "End year filter (negative for BC).",
                required=False,
            ),
            ToolParam(
                "context_filters",
                "array",
                "List of context strings to filter by.",
                required=False,
                items_type="string",
            ),
            ToolParam(
                "keywords",
                "array",
                "List of keywords to filter by.",
                required=False,
                items_type="string",
            ),
        ],
        example_args={"start_year": -500, "end_year": 500, "keywords": ["rome", "law"]},
        category="local",
    ),
    ToolDef(
        name="wikilink_search",
        description=(
            "Search local wiki file index for documents whose title matches a query. "
            "Returns stems usable as [[wikilink]] targets."
        ),
        params=[ToolParam("query", "string", "Search query.")],
        example_args={"query": "rome"},
        category="local",
    ),
    # ── Research output tools ─────────────────────────────────────────────────
    ToolDef(
        name="file_write",
        description=(
            "Write (overwrite) content to a file in AINotes/Research/. "
            "Filenames must use spaces not underscores (e.g. 'Roman Law/General Principles.md')."
        ),
        params=[
            ToolParam(
                "path",
                "string",
                "The relative destination path within AINotes/Research/ "
                "(e.g. 'Roman Law/General Principles.md').",
            ),
            ToolParam("content", "string", "The complete markdown file content."),
        ],
        example_args={
            "path": "Roman Law/General Principles.md",
            "content": "---\ntitle: ...\n---\n# Roman Law\n...",
        },
        category="output",
    ),
    ToolDef(
        name="file_append",
        description=(
            "Append content to an existing file in AINotes/Research/. "
            "Use this for large documents to build them section by section and avoid token limits."
        ),
        params=[
            ToolParam(
                "path",
                "string",
                "The relative destination path within AINotes/Research/.",
            ),
            ToolParam("content", "string", "The markdown content to append."),
        ],
        example_args={
            "path": "Roman Law/General Principles.md",
            "content": "\n## Section 2\n...",
        },
        category="output",
    ),
    ToolDef(
        name="create_directory",
        description="Create a subdirectory in AINotes/Research/.",
        params=[
            ToolParam(
                "path",
                "string",
                "The relative directory path to create within AINotes/Research/ (e.g. 'Roman Law').",
            )
        ],
        example_args={"path": "Roman Law"},
        category="output",
    ),
    ToolDef(
        name="file_move",
        description=(
            "Move or rename a file or directory strictly within AINotes/Research/. "
            "Both src and dst must be within AINotes/Research/."
        ),
        params=[
            ToolParam("src", "string", "Source path relative to AINotes/Research/."),
            ToolParam("dst", "string", "Destination path relative to AINotes/Research/."),
        ],
        example_args={
            "src": "General Principles.md",
            "dst": "Roman Law/General Principles.md",
        },
        category="output",
    ),
    # ── External information sources ──────────────────────────────────────────
    ToolDef(
        name="wikipedia_search",
        description=(
            "Search Wikipedia for an article title. Returns the top 10 matching "
            "article titles and a brief snippet. Use after local vector_search."
        ),
        params=[ToolParam("query", "string", "The search term query.")],
        example_args={"query": "Bavarian Illuminati"},
        category="external",
    ),
    ToolDef(
        name="wikipedia_fetch",
        description=(
            "Download a full Wikipedia article in plain text and automatically save "
            "it to AINotes/Research/Wikipedia/. Returns a preview and the local "
            "file path so you can read it in detail using file_read."
        ),
        params=[ToolParam("title", "string", "The exact Wikipedia article title.")],
        example_args={"title": "Illuminati"},
        category="external",
    ),
    # ── Stanford Encyclopedia of Philosophy ──────────────────────────────────
    ToolDef(
        name="sep_search",
        description=(
            "Search the Stanford Encyclopedia of Philosophy for article titles. "
            "Returns matching entry slugs and titles. Use sep_fetch to download a full article."
        ),
        params=[ToolParam("query", "string", "The philosophical topic to search for.")],
        example_args={"query": "Plato theory of forms"},
        category="external",
    ),
    ToolDef(
        name="sep_fetch",
        description=(
            "Download a full Stanford Encyclopedia of Philosophy article by its slug "
            "(e.g. 'plato', 'kant-reason'). Saves to AINotes/Research/SEP/ and returns "
            "a preview. Use file_read to read the full article. "
            "Prefer SEP over Wikipedia for philosophy topics."
        ),
        params=[
            ToolParam(
                "entry",
                "string",
                "The SEP article slug (e.g. 'plato', 'aristotle', 'kant-reason').",
            )
        ],
        example_args={"entry": "plato"},
        category="external",
    ),
    # ── arXiv ─────────────────────────────────────────────────────────────────
    ToolDef(
        name="arxiv_search",
        description=(
            "Search arXiv for academic preprints by keyword. Returns titles, IDs, "
            "authors, and abstract previews. Use arxiv_fetch to download a full entry."
        ),
        params=[
            ToolParam("query", "string", "The search query (keywords, author, title)."),
            ToolParam(
                "max_results",
                "integer",
                "Number of results to return (max 10).",
                required=False,
                default=5,
            ),
        ],
        example_args={"query": "quantum gravity holography", "max_results": 5},
        category="external",
    ),
    ToolDef(
        name="arxiv_fetch",
        description=(
            "Download an arXiv paper's abstract and metadata by arXiv ID. "
            "Saves to AINotes/Research/arXiv/ and returns a preview."
        ),
        params=[
            ToolParam(
                "arxiv_id",
                "string",
                "The arXiv paper ID (e.g. '2301.07041', 'hep-th/9802150').",
            )
        ],
        example_args={"arxiv_id": "2301.07041"},
        category="external",
    ),
    # ── OpenAlex ──────────────────────────────────────────────────────────────
    ToolDef(
        name="openalex_search",
        description=(
            "Search OpenAlex for peer-reviewed academic works. Returns titles, authors, "
            "year, DOI, and abstract previews. Good complement to arXiv for published literature."
        ),
        params=[
            ToolParam("query", "string", "The search query."),
            ToolParam(
                "max_results",
                "integer",
                "Number of results to return (max 10).",
                required=False,
                default=5,
            ),
        ],
        example_args={"query": "Roman law property rights", "max_results": 5},
        category="external",
    ),
    # ── Wikidata ──────────────────────────────────────────────────────────────
    ToolDef(
        name="wikidata_search",
        description=(
            "Search Wikidata for entities (people, places, concepts, works) by name. "
            "Returns Q-identifiers and descriptions. Use wikidata_entity to get full structured data."
        ),
        params=[ToolParam("query", "string", "The entity name to search for.")],
        example_args={"query": "Immanuel Kant"},
        category="external",
    ),
    ToolDef(
        name="wikidata_entity",
        description=(
            "Fetch structured data for a Wikidata entity by Q-identifier. "
            "Returns key facts: birth/death dates, occupation, nationality, notable works, etc."
        ),
        params=[
            ToolParam(
                "entity_id",
                "string",
                "The Wikidata Q-identifier (e.g. 'Q9312' for Immanuel Kant).",
            )
        ],
        example_args={"entity_id": "Q9312"},
        category="external",
    ),
    # ── Internet Archive ──────────────────────────────────────────────────────
    ToolDef(
        name="archive_search",
        description=(
            "Search the Internet Archive for freely available out-of-copyright texts. "
            "Returns item identifiers, titles, creators, and dates. "
            "Use archive_fetch to download the full text."
        ),
        params=[
            ToolParam("query", "string", "The search query."),
            ToolParam(
                "max_results",
                "integer",
                "Number of results to return (max 10).",
                required=False,
                default=5,
            ),
        ],
        example_args={"query": "Cicero De Republica", "max_results": 5},
        category="external",
    ),
    ToolDef(
        name="archive_fetch",
        description=(
            "Download the text of an Internet Archive item by its identifier. "
            "Saves to AINotes/Research/InternetArchive/ and returns a preview. "
            "Ideal for out-of-copyright primary source texts."
        ),
        params=[
            ToolParam(
                "identifier",
                "string",
                "The Internet Archive item identifier (e.g. 'cicero-de-republica').",
            )
        ],
        example_args={"identifier": "cicero-de-republica"},
        category="external",
    ),
    # ── Computational tools ───────────────────────────────────────────────────
    ToolDef(
        name="python_eval",
        description=(
            "Execute a Python snippet in the local virtual environment and return its stdout/stderr. "
            "Full standard library and installed packages are available (use import statements as needed). "
            "Use for numerical computations, data parsing, unit conversions, statistics, and complex algorithms."
        ),
        params=[
            ToolParam(
                "code",
                "string",
                "Python code to execute. Use print() to produce output. Standard import statements are supported.",
            )
        ],
        example_args={"code": "import math\nprint(round(math.pi * 5**2, 4))"},
        category="compute",
    ),
    ToolDef(
        name="maxima_eval",
        description=(
            "Evaluate a Maxima computer algebra system expression. "
            "Use for symbolic mathematics: integration, differentiation, solving equations, "
            "simplification, and linear algebra. "
            "Expressions use Maxima syntax (e.g. 'integrate(sin(x), x)')."
        ),
        params=[
            ToolParam(
                "expression",
                "string",
                "A Maxima expression or command (without the trailing semicolon).",
            )
        ],
        example_args={"expression": "integrate(sin(x)^2, x)"},
        category="compute",
        requires_binary="maxima",
    ),
    # ── Housekeeping tools ────────────────────────────────────────────────────
    ToolDef(
        name="check_wiki",
        description=(
            "Ensure all files in AINotes/Research/ have proper metadata "
            "(uuid, context, creation_date). Call this after writing research files."
        ),
        params=[],
        example_args={},
        category="housekeeping",
    ),
    ToolDef(
        name="update_index",
        description=(
            "Run the full vector index update so newly written research becomes "
            "searchable. Call this after writing research files and running check_wiki."
        ),
        params=[],
        example_args={},
        category="housekeeping",
    ),
    ToolDef(
        name="extract_timeline",
        description=(
            "Extract timeline tables from a research file and merge into the "
            "timeline database."
        ),
        params=[
            ToolParam(
                "path",
                "string",
                "The relative path within AINotes/Research/.",
            )
        ],
        example_args={"path": "Roman Law/Timeline.md"},
        category="housekeeping",
    ),
]


# ── Public API ────────────────────────────────────────────────────────────────


def is_python_eval_enabled() -> bool:
    """Helper to check if python_eval is enabled in the configuration file."""
    import json
    from pathlib import Path
    try:
        cfg_path = Path("~/.config/knrs/knrs.json").expanduser().resolve()
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                return cfg.get("enable_python_eval", True)
    except Exception:
        pass
    return True


def get_available_tools(active_names: list[str] | None = None) -> list[ToolDef]:
    """Return the filtered list of tools that should be active.

    Args:
        active_names: Optional explicit allowlist of tool names (e.g. from
            config). When ``None``, all tools that pass binary availability
            checks are returned.

    Binary availability is checked automatically for tools with
    ``requires_binary`` set.
    """
    python_enabled = is_python_eval_enabled()
    result: list[ToolDef] = []
    for t in TOOL_REGISTRY:
        if t.name == "python_eval" and not python_enabled:
            continue
        if active_names is not None and t.name not in active_names:
            continue
        if t.requires_binary and not shutil.which(t.requires_binary):
            continue
        result.append(t)
    return result


def get_json_schemas(active_names: list[str] | None = None) -> list[dict[str, Any]]:
    """Generate the JSON schema array for API/HF/MLX backends.

    Pass directly to ``tools=`` in the OpenAI payload or
    ``apply_chat_template(tools=...)``.
    """
    return [t.to_json_schema() for t in get_available_tools(active_names)]


def get_tool_prompt_section(active_names: list[str] | None = None) -> str:
    """Generate a numbered tool catalogue for raw-LLM system prompts."""
    tools = get_available_tools(active_names)
    return "\n\n".join(t.to_prompt_text(i + 1) for i, t in enumerate(tools))


def coerce_tool_args(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Coerce argument values to the types declared in the registry.

    Replaces the previously duplicated ``coerce_tool_args`` functions in
    agent_hf.py and agent_macos.py.
    """
    # Find the tool definition
    tool_def: ToolDef | None = None
    for t in TOOL_REGISTRY:
        if t.name == name:
            tool_def = t
            break

    if tool_def is None:
        return args

    param_types = tool_def.get_param_types()
    coerced: dict[str, Any] = {}

    for k, v in args.items():
        prop_type = param_types.get(k)
        if prop_type == "integer":
            try:
                coerced[k] = int(v)
            except Exception:
                coerced[k] = v
        elif prop_type == "number":
            try:
                coerced[k] = float(v)
            except Exception:
                coerced[k] = v
        elif prop_type == "boolean":
            if isinstance(v, str):
                if v.lower() in ("true", "1", "yes"):
                    coerced[k] = True
                elif v.lower() in ("false", "0", "no"):
                    coerced[k] = False
                else:
                    coerced[k] = v
            else:
                coerced[k] = bool(v)
        else:
            coerced[k] = v

    return coerced
