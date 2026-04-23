# knrs — Implementation Plan

## Background

`knrs` synthesises three predecessor projects (EbookTools, LocalResearch, Summarizer) into a
single, cohesive LLM-assisted knowledge-base wiki.  
The plan is structured in **seven sequential phases**. Each phase has concrete, testable
deliverables and explicitly states what it reuses from predecessor code.

---

## Resolved Decisions

| # | Question | Decision |
|---|----------|----------|
| 1 | VectorDB tooling | Own implementation using **`embedding-gemma-300m`**; no `qmd`. |
| 2 | Wiki/Notes UUID convention | `knrs` **writes a UUID** into Notes frontmatter when one is absent. This is the only case where a source-of-truth file (Notes) may be modified by `knrs`. Pattern: same as `previous/EbookTools/ebook_tools.py` `notes` command. |
| 3 | systemd service | **Out of scope** for all current phases. |
| 4 | MCP / VectorDB access interface | **Out of scope** for all current phases. |
| 5 | Migration scope | Source paths are read from `~/.config/summarizer/summarizer_config.json`: field `"markdown_path"` (MarkdownBooks) and `"summaries_path"` (BookSummaries). |

---

## Proposed Changes

---

### Phase 1 — Project Scaffolding

Stand up a clean `uv`-managed Python project with configuration loading, logging, and
the skeletal package layout.

#### [NEW] `pyproject.toml`
- `uv`-managed, Python >= 3.13
- Initial dependencies: `pyyaml`, `rich`, `pillow`
- Internal packages declared as `uv` workspace members

#### [NEW] `knrs/` — top-level package
```
knrs/
├── __init__.py
├── config.py          # Load / validate ~/.config/knrs/*.json
├── logging_setup.py   # Structured logging via `logging` + `rich`
└── paths.py           # Central path-resolution helpers (expand ~, resolve)
```

#### [NEW] `~/.config/knrs/knrs.json` (schema)
```json
{
  "calibre_path": "~/ReferenceLibrary/Calibre Library",
  "notes_path":   "~/Wiki/Notes",
  "knrs_data":    "~/KnrsData",
  "wiki_path":    "~/Wiki",
  "target_series": [],
  "summarizer_name": "summarizer_macos"
}
```

**Deliverables**
- `uv sync` succeeds; `uv run python -m knrs --version` prints a version string.
- Config loader raises a clear error if required keys are missing.

**Reuse**: none from predecessors (clean slate).

---

### Phase 2 — Calibre Sync Pipeline

Port and integrate the Summarizer's two-phase sync into `knrs`, producing
`MarkdownBooks`, `BookCoverIcons`, `BookLibrary`, and `BookSummaries`.

#### [NEW] `knrs/calibre/`
```
knrs/calibre/
├── __init__.py
├── library.py        # Scan Calibre dir, parse metadata.opf -> CalibreBook dataclass
├── converter.py      # Pandoc (EPUB) / Docling (PDF) conversion; atomic writes
├── cover.py          # Resize cover.jpg -> BookCoverIcon UUID.jpg
└── sync.py           # Two-phase plan-then-execute Calibre->MarkdownBook sync
```

**Key behaviours (all ported from `previous/Summarizer/calibre_sync.py`)**

| Behaviour | Detail |
|-----------|--------|
| Format priority | MD > EPUB > PDF per book |
| Frontmatter | UUID, source SHA-256, source format, converter version |
| Naming | `title_to_filename()` from `naming.py` (80-char cap, collision = fatal error) |
| Sync phases | Phase 1: build action list (add / update / rename / delete); Phase 2: execute |
| Orphan cleanup | Remove `MarkdownBook` + `BookCoverIcon` if source UUID no longer exists |
| BookLibrary | Copy EPUB + PDF organised by `series` (from `previous/EbookTools`) |
| **Series casing** | **Preserve Calibre metadata casing exactly** for all `series`-based directory paths. The old Summarizer project normalised series to all-lowercase; this project does **not**. |

#### [NEW] `knrs/summarizer/`
```
knrs/summarizer/
├── __init__.py
├── engine.py         # Abstract SummarizerEngine; dispatch to platform backends
└── sync.py           # Two-phase MarkdownBook->BookSummary sync
```

- Reuse `previous/Summarizer/summarizer_core` as a dependency or copy
- Platform backends (`summarizer_macos`, `summarizer_linux`, `summarizer_gc_gemma4_31b`)
  remain as subprocess workers (same pattern as predecessor)
- Frontmatter: source MarkdownBook SHA-256, summary model name + version

#### [NEW] `knrs/naming.py`
- Direct port of `previous/Summarizer/naming.py` (no functional changes in Phase 2)

**Deliverables**
- `uv run python -m knrs sync --dry-run` prints a full action list without writing files.
- `uv run python -m knrs sync` produces `MarkdownBooks/`, `BookCoverIcons/`, `BookSummaries/`.
- Re-running is idempotent (no unnecessary conversions).
- Rename / delete propagation works end-to-end.

**Reuse**: `calibre_sync.py`, `naming.py`, `summarizer_core/`, summarizer backends from
`previous/Summarizer`.

---

### Phase 3 — Timeline Extraction

Extract IndraTime timeline events from `Wiki/Notes` Markdown tables and write
`KnrsData/Timelines/timelines.json`.

#### [NEW] `knrs/timelines/`
```
knrs/timelines/
├── __init__.py
├── indra_time.py     # IndraTime parser (port of LocalResearch time_lines.py subset)
└── extractor.py      # Scan Wiki/Notes; extract tables; emit timelines.json
```

**Key behaviours**

| Behaviour | Detail |
|-----------|--------|
| Source scan | All `Wiki/Notes/**/*.md` that have `uuid` in frontmatter |
| Table detection | Markdown table rows with at least one cell parseable as IndraTime |
| Output | `timelines.json`: array of `{ uuid, note_file, event_label, start, end? }` |
| Incremental | SHA-256 per note file; re-extract only on change |

**Deliverables**
- `uv run python -m knrs timelines` writes `KnrsData/Timelines/timelines.json`.
- IndraTime round-trip unit tests pass (BC, BP, kya, Ma, Ga scales).

**Reuse**: `previous/LocalResearch/time_lines.py`, `previous/LocalResearch/IndraTimeFormat.md`.

---

### Phase 4 — Wiki Assembly

Assemble `Wiki/AINotes/Books/<series>/` from the intermediate data.

#### [NEW] `knrs/wiki/`
```
knrs/wiki/
├── __init__.py
├── assembler.py      # Merge metadata + cover + summary -> Wiki/AINotes entry
└── sync.py           # Two-phase diff between KnrsData and Wiki/AINotes
```

**Structure of each `Wiki/AINotes/Books/<series>/<filename>.md`**

```markdown
---
uuid: <UUID>
title: <title>
authors: [...]
series: <series>
tags: [...]
...
---

# <Title>

![Cover](../../../KnrsData/BookCoverIcons/<UUID>.jpg)

[Open in Calibre](<calibre_path>/<series>/<title>)

## Summary

<contents of BookSummary, body only (frontmatter stripped)>

## Description

<Calibre book description, converted from HTML — see rule below>
```

**Key behaviours**

| Behaviour | Detail |
|-----------|--------|
| Change detection | Regenerate Wiki entry if summary SHA-256 or Calibre metadata changes |
| Orphan cleanup | Remove `Wiki/AINotes` entry if corresponding `BookSummary` no longer exists |
| Wiki-links | Both `Wiki/Notes` and `Wiki/AINotes` use `[[filename]]` links |
| Cover path | Relative path from the Wiki entry to `BookCoverIcons/` |
| **HTML description** | The `<description>` field in `metadata.opf` is raw HTML. Convert to plain Markdown: strip all HTML tags, **do not** add blockquote `> ` prefixes, and collapse runs of more than 2 consecutive newlines to a single blank line. |
| **Notes UUID generation** | When scanning `Wiki/Notes`, any `.md` file with no `uuid` in its frontmatter has a UUID generated and written back. This is the **only** write operation permitted on source-of-truth files. Pattern from `previous/EbookTools/ebook_tools.py` `notes` command. |

**Deliverables**
- `Wiki/AINotes/Books/` is fully populated and browsable in Obsidian.
- `[[wiki-links]]` between `Notes` and `AINotes` resolve correctly.
- Notes files without UUID receive one; Notes files with UUID are not modified.
- Re-running is idempotent.

**Reuse**: `previous/EbookTools/ebook_tools.py` (notes generation logic, cover embedding, UUID injection).

---

### Phase 5 — VectorDB & Semantic Search

Own-implementation vector index using a single embedding model. No external vector DB
tooling (`qmd` etc.) in this phase.

#### [NEW] `knrs/vector/`
```
knrs/vector/
├── __init__.py
├── indexer.py        # Chunk MarkdownBooks; compute embeddings; persist VectorDB
└── search.py         # Query interface (returns ranked list of book/chunk UUIDs)
```

**Key behaviours**

| Behaviour | Detail |
|-----------|--------|
| Embedding model | **`embedding-gemma-300m`** — single model, no others in Phase 5 |
| Chunking | Split MarkdownBooks into sections; store UUID + source filename per chunk |
| Incremental | Reindex only changed MarkdownBooks (SHA-256 comparison) |
| Interface | Python API only; MCP adapter is out of scope |

**Deliverables**
- `uv run python -m knrs index` populates `KnrsData/VectorDB/`.
- `uv run python -m knrs search "query string"` returns ranked results with titles.

**Reuse**: `previous/LocalResearch/vector_store.py`, `previous/LocalResearch/document_store.py`.

---

### Phase 6 — REPL & Orchestration

Tie all phases together into the command-line REPL with slash-commands and progress display.

#### [NEW] `knrs/__main__.py`
- Entry point; `uv run python -m knrs` starts the REPL

#### [NEW] `knrs/repl/`
```
knrs/repl/
├── __init__.py
├── repl.py           # Main REPL loop (readline / prompt_toolkit)
└── commands.py       # Slash-command dispatcher
```

**Slash-commands (initial set)**

| Command | Description |
|---------|-------------|
| `/sync [--dry-run]` | Full two-phase sync: Calibre -> MarkdownBooks -> Summaries -> Wiki |
| `/sync-books [--dry-run]` | Calibre -> MarkdownBooks + covers only |
| `/sync-summaries [--dry-run]` | MarkdownBooks -> BookSummaries only |
| `/sync-wiki [--dry-run]` | Assemble / refresh `Wiki/AINotes` |
| `/timelines [--dry-run]` | Re-extract timelines from `Wiki/Notes` |
| `/index [--dry-run]` | Rebuild VectorDB |
| `/search <query>` | Semantic search over VectorDB |
| `/status` | Show counts and staleness of each derived data type |
| `/config` | Print resolved configuration |
| `/help` | List commands |
| `/exit` | Quit |

**Key behaviours**

| Behaviour | Detail |
|-----------|--------|
| Progress | `rich` progress bar; `2/200 converted` style |
| Dry-run | Every mutating command supports `--dry-run`; prints planned actions |
| Concurrency | `--concurrency N` flag passed through to converters and summarisers |
| Error reporting | Collision errors, missing config, conversion failures surfaced clearly |

**Deliverables**
- REPL starts and accepts commands.
- `/sync --dry-run` on a full Calibre library completes without errors.
- `/status` accurately reflects current state of all derived data.

**Reuse**: `previous/LocalResearch/research_console.py` (REPL patterns).

---

### Phase 7 — Migration

One-time migration of existing `previous/Summarizer` output into the new `KnrsData/`
structure.

#### Source path discovery

Source paths are read from **`~/.config/summarizer/summarizer_config.json`**:

| Config field | Contains |
|---|---|
| `"markdown_path"` | Root of existing MarkdownBooks (to be migrated to `KnrsData/MarkdownBooks/`) |
| `"summaries_path"` | Root of existing BookSummaries (to be migrated to `KnrsData/BookSummaries/`) |

#### [NEW] `knrs/migration/`
```
knrs/migration/
├── __init__.py
└── migrate.py        # Scan old MarkdownBooks + BookSummaries; rename/move to KnrsData
```

**Steps**

1. Load `~/.config/summarizer/summarizer_config.json`; resolve `markdown_path` and `summaries_path`.
2. Read all existing `MarkdownBook` files; extract UUID from frontmatter.
3. Look up UUID in Calibre; compute new filename and **series directory** (preserving Calibre casing).
4. Rename / move file (`git mv` to preserve history).
5. Repeat for `BookSummaries` using the same UUID-to-new-path mapping.
6. Remove orphaned summaries whose source UUID is no longer in Calibre.
7. Log all actions; abort on any collision.

> [!CAUTION]
> Migration is a one-time operation. Run with `--dry-run` first and review carefully.
> All moves should be committed to Git in a single dedicated commit before running `/sync`.

**Deliverables**
- `uv run python -m knrs migrate --dry-run` lists all planned moves/deletes.
- `uv run python -m knrs migrate` executes moves; produces a migration log.
- Post-migration `/sync --dry-run` shows zero actions needed (idempotency check).

**Reuse**: `previous/Summarizer/migration/` scripts.

---

## Verification Plan

### Per-Phase Automated Tests

- Phase 1: `pytest knrs/tests/test_config.py` — config loading, path expansion.
- Phase 2: `pytest knrs/tests/test_naming.py` — filename generation, collision detection.
- Phase 2: `pytest knrs/tests/test_calibre.py` — `metadata.opf` parsing, format priority.
- Phase 3: `pytest knrs/tests/test_indra_time.py` — IndraTime round-trips for all scales.
- Phase 4: `pytest knrs/tests/test_wiki_assembler.py` — Wiki entry structure.
- Phase 6: Manual REPL walkthrough against a small test Calibre library.
- Phase 7: Dry-run migration report reviewed against actual file system before execution.

### Manual Verification

- After Phase 4: open `Wiki/` in Obsidian; verify wiki-links, cover images, and summaries render correctly.
- After Phase 5: run `/search` queries and verify relevance of returned results.
- After Phase 7: confirm Git history shows clean `git mv` operations.
