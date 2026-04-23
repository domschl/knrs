# LLM-Enabled Knowledge-Base Wiki — Vision Plan

## Purpose

`knrs` synthesises three preceding projects into a unified, LLM-assisted knowledge-base wiki.
The core idea is to compile provided data (books, personal notes — the *sources of truth*)
into a two-pronged Markdown wiki that is maintained:

- **`Wiki/Notes`** — by the human user (read-only for `knrs`), and  
- **`Wiki/AINotes`** — automatically, by LLM models or algorithmic conversion pipelines.

---

## General Remarks

- **Implementation language**: Python ≥ 3.13
- **Package management**: `uv`
- **Configuration**: JSON files stored in `~/.config/knrs/`
- **Version control**: Git (manual human responsibility; not automated)
- **Design inspiration**: Andrej Karpathy's
  [llm-wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)
  ideas may be incorporated where fitting.

---

## Predecessor Projects

All three projects are included as Git submodules under `previous/` for reference and
selective code reuse.

| Project | Location | Core Role |
|---------|----------|-----------|
| **EbookTools** | `previous/EbookTools` | Calibre metadata export, cover icons, BookLibrary copy organised by `series`, timeline extraction from Markdown tables using IndraTime format |
| **LocalResearch** | `previous/LocalResearch` | Vector-embedding index over TXT/PDF, 3D embedding visualisation, research console with IndraTime timeline compilation |
| **Summarizer** | `previous/Summarizer` | Calibre → Markdown conversion (Docling / Pandoc), Gemma 4 summarisation, two-phase sync, wiki-compatible naming (80-char cap, UUID tracking, SHA-256 hashing) |

---

## Definitions & Terminology

| Term | Definition |
|------|-----------|
| **Calibre Library** | A Calibre library directory; read-only source of truth for book data. Interaction is filesystem-only: `metadata.opf`, `cover.jpg`, and book files in MD (priority 1), EPUB (priority 2), PDF (priority 3). |
| **UUID** | The stable Calibre book identifier, stored in `metadata.opf`. Also used as primary key in frontmatter of human-Wiki notes. Notes without UUID are ignored. |
| **IndraTime** | A string-based deep-time format used in human wiki timeline tables, covering cosmological to historical dates. Documented in `previous/LocalResearch/IndraTimeFormat.md`. |
| **Wiki-link** | An Obsidian/Emacs `[[filename]]` style cross-reference, usable in both `Wiki/Notes` and `Wiki/AINotes`. |
| **MarkdownBook** | A Markdown document generated from a Calibre book; the canonical intermediate text representation. |
| **BookSummary** | An LLM-generated summary Markdown document, derived from a MarkdownBook. |
| **BookCoverIcon** | A resized cover thumbnail stored as `<UUID>.jpg`. |
| **Sync** | A two-phase *plan-then-execute* operation that propagates changes from a source of truth to all derived data, including deletion of orphaned derived files. |
| **KnrsData** | Root directory for all intermediate data (configurable path). |
| **Wiki** | Root directory for the resulting wiki (configurable path). |

---

## Sources of Truth

Sources of truth are **read-only** for `knrs`; they are only modified by the human user.

### Calibre Library

- Documents in EPUB, PDF, or MD format.
- Each book identified by UUID (`metadata.opf`).
- Interaction: filesystem-only (no Calibre API / DB).
- Format priority for conversion: MD > EPUB > PDF.

### Human Wiki (`Wiki/Notes`)

- A human-maintained collection of Markdown documents.
- May contain YAML frontmatter with a `uuid` field; entries without `uuid` are **ignored** by `knrs`.
- May contain Markdown tables with timestamps in **IndraTime** format (used for timeline extraction).
- Contains wiki-links (`[[…]]`) to `Wiki/AINotes` entries.

### Change Propagation

Changes in any source of truth must propagate to **all derived data**, including deletion of orphaned derived files when a source entry is removed.

---

## Derived Data

### Intermediate Data (stored in `KnrsData/`)

| Artifact | Location | Description | Source |
|----------|----------|-------------|--------|
| **MarkdownBooks** | `KnrsData/MarkdownBooks/<series>/` | Markdown text of each Calibre book. Generated via Pandoc (EPUB) or Docling (PDF/OCR). Frontmatter contains UUID, SHA-256 hash of source file, source format, and converter version. | Calibre Library |
| **BookCoverIcons** | `KnrsData/BookCoverIcons/` | Cover thumbnails as `<UUID>.jpg`. | Calibre Library (`cover.jpg`) |
| **BookSummaries** | `KnrsData/BookSummaries/<series>/` | LLM-generated summaries, prefixed `Summary of …`. Frontmatter contains source MarkdownBook SHA-256 hash and summary model version. | MarkdownBooks |
| **BookLibrary** | `KnrsData/BookLibrary/<series>/` | Backup copy of EPUB and PDF files from Calibre, organised by `series`. | Calibre Library |
| **Timelines** | `KnrsData/Timelines/timelines.json` | Compiled timeline events in IndraTime format, extracted from `Wiki/Notes` tables. | Human Wiki |
| **VectorDB** | `KnrsData/VectorDB/` | Embedding vectors for semantic search. Initial model: `gemma-embedding-300`. Scope: MarkdownBooks (and optionally `Wiki/Notes`). | MarkdownBooks (+ Notes) |

#### Example `KnrsData/` layout

```
KnrsData/
├── BookCoverIcons/
│   ├── aa34d602-c543-43b5-8624-4a484b973b40.jpg
│   └── aa3e3690-ab4b-4721-aacd-6b98417b4e69.jpg
├── BookLibrary/
│   ├── Archaeology/
│   │   ├── History of Troja.epub
│   │   └── History of Troja.pdf
│   ├── Physics/
│   └── Science Fiction/
├── BookSummaries/
│   ├── Archaeology/
│   │   └── Summary of History of Troja - AuthorName.md
│   ├── Physics/
│   └── Science Fiction/
├── MarkdownBooks/
│   ├── Archaeology/
│   │   └── History of Troja - AuthorName.md
│   ├── Physics/
│   └── Science Fiction/
├── Timelines/
│   └── timelines.json
└── VectorDB/
```

### Resulting Data (stored in `Wiki/AINotes/`)

`Wiki/AINotes/Books/<series>/` contains one Markdown document per Calibre book. Each document:

1. **YAML frontmatter** — full Calibre metadata (title, author, series, tags, UUID, …)
2. **Cover icon** — embedded thumbnail linking back to the Calibre library location
3. **Book summary** — rendered content of the corresponding `BookSummary`
4. **Description** — the book description from Calibre metadata

The full book text is **not** included in the Wiki (too large). Access to book content is via the VectorDB (MCP or similar interface).

#### Example `Wiki/` layout

```
Wiki/
├── AINotes/
│   └── Books/
│       ├── Archaeology/
│       │   └── History of Troja - AuthorName.md
│       ├── Physics/
│       └── Science Fiction/
└── Notes/                  # read-only — maintained by human
```

---

## Naming Convention

- Filenames derived from Calibre metadata: `<Title> - <Author>.md`
- Maximum base length: **80 characters** (before `.md` extension)
- Trailing series numbering (arabic or roman) is preserved during truncation
- Unsafe filesystem characters are sanitised (`:` → ` —`, `/` → `-`, etc.)
- Summary files are prefixed: `Summary of <base-filename>.md`
- Collision detection is **case-insensitive**; any collision is a **fatal error** requiring user intervention in Calibre metadata
- UUID is used as the stable identifier across renames and moves

---

## User Interface

### REPL

The primary interface is a **command-line REPL** (Read–Eval–Print Loop).

Design goals:

- Slash-commands (e.g. `/sync`, `/status`, `/dry-run`) to initiate data transformations
- Progress display (e.g. `2/200 converted`)
- Differential / incremental updates: only changed sources trigger re-processing
- Designed for future extension into a full LLM-connected agentic tool (no LLM connection in Phase 1)

### Third-Party Tool Compatibility

- **Obsidian / Emacs** — both `Wiki/AINotes` and `Wiki/Notes` are standard wiki Markdown; `[[wiki-links]]` work out of the box
- **Agentic coding tools** (`gemini-cli`, etc.) — can read and alter `Wiki` parts directly

---

## Migration & Administration

- The `previous/Summarizer` project has already produced a significant corpus of `MarkdownBooks` and `BookSummaries`; these must be migrated (renamed / moved) to the new directory structure.
- All configuration (paths for sources of truth, intermediate data, wiki) is stored in JSON files under `~/.config/knrs/`.
- Backup and version management of all Markdown and image files is via **Git** (manual, human responsibility).
- Future option: conversion pipeline as a **Linux systemd** service triggered by filesystem watches on the sources of truth.
