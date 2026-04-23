## LLM enabled knowledge base Wiki

`knrs` is a project that is supposed to synthesize three preciding projects that have overlapping functionality.

Idea is to compile provided data (books, notes, see below, sources of truth) into a two-pronged Markdown wiki that is maintained in part by human (`Wiki/Notes`) and in part by LLM models or algorithmic conversions (`Wiki/AINotes`)

## General remarks

- Implementation language is Python
- Project and package management via `uv`

## Previous projects

- EbookTools (in `previous/EbookTools`): use a Calibre library as source and creates Markdown files for each book containing the book's metadata and a link to the Calibre library. It creates a copy of Epub and PDF files organized by Calibre's `series` metadata. This is the oldest project.
- LocalResearch (in `previous/LocalResearch`): uses Calibre library as source for TXT and PDF files that are indexed using vector embeddings. Extracts timeline information from tables in a Markdown notes collection. The time-format used is described in `previous/LocalResearch/IndraTimeFormat.md`. Access is provided via console (`previous/LocalResearch/research_console.py`) and a simple web client that provides 3D visualization of the embedding spaces.
- Summarizer (in `previous/Summarizer`): uses Calibre library as source and converts both epub and pdf documents into markdown text (`previous/Summarizer/calibre_sync.py`). In a second step, GEMMA 4 models are used to generate summaries for all texts (`previous/Summarizer/summarizer_sync`). This is the latest project of the three previous projects.

## Synthesis architecture

### Sources of truth

The data-sources are read-only for `knrs`, and are only modified by Human

- Calibre: A Calibre libre containing documents in EPUB, PDF, or MD format. Interaction with Calibre is file-system only, for each book we look at `metadata.opf`, `cover.jpg`, and documents in MD (prio 1), EPUB (prio 2), or PDF (prio 3) format.
- Notes: (`Wiki/Notes`) A human-maintained Wiki collection of Markdown documents that include tables with timelines that use the IndraTime format.

Changes in the sources of truth need to propagate to all derived data. This includes deleting derived data, if the source has been deleted.

Main-key is the UUID metadata field in Calibre's metadata, or the UUID field in the frontmatter of human Wiki. Wiki entries without UUID are ignored. 

### Derived data

#### Intermediate data

- MarkdownBooks: For each Calibre book, a Markdown representation is generated (using the strategy from `previous/Summarizer/calibre_sync.py`). This involves OCR (docling) or file-conversion (pandoc), if no Markdown document is in the Calibre library
- BookCoverIcons: For each book, a cover icon jpg is generated (UUID.jpg)
- Summary: For each book, a summary is generated, using the Markdown representation as source (`previous/Summarizer/
- Timelines: The timeline information is compiled from human's wiki (see `previous/LocalResearch/research_console.py`)
- BookLibrary: A copy of PDF and EPUB documents from the Calibre library arranged by `series` metadata as directory structure (functionality of `previous/EbookTools`) as backup.
- VectorDB: Embeddings calculated from MarkdownBooks (and `Wiki/Notes`?). First step: use embeddings in similar fashion as `previous/LocalResearch`, but restrict to one embeddings model (embeddings gemma 300 for start). Evaluate the use of tool `qmd` instead?

Example file-structure for the intermediate data:

```
KnrsData
├── BookCoverIcons
│   ├── aa34d602-c543-43b5-8624-4a484b973b40.jpg
│   └── aa3e3690-ab4b-4721-aacd-6b98417b4e69.jpg
├── BookLibrary
│   ├── Archaeology
│   │   ├── Summary of History of Troja.epub
│   │   └── Summary of History of Troja.pdf
│   ├── Physics
│   └── Science Fiction
├── BookSummaries
│   ├── Archaeology
│   │   └── Summary of History of Troja.md
│   ├── Physics
│   └── Science Fiction
├── MarkdownBooks
│   ├── Archaeology
│   │   └── History of Troja.md
│   ├── Physics
│   └── Science Fiction
├── Timelines
│   └── timelines.json
└── VectorDB
```

#### Resulting data

The result will be a Markdown-based wiki that is maintained in half by humans (`Wiki/Notes`) and in half by this program (`Wiki/AINotes`). Both Wikis can use `[[wiki-links]]` to reference content.

Example file-structure:

```
Wiki
├── AINotes
│   └── Books
│       ├── Archaeology
│       │   └── History of Troja.md
│       ├── Physics
│       └── Science Fiction
└── Notes    # read-only! maintained by human
```

Automatically and algorithmically created will the the `Wiki/AINotes/Books` folder structure. It contains, organized by `series`, Markdown documents for each calibre book. Each markdown document contains 1. frontmatter that contains the Calibre metadata, and in the Markdown body: 2. Icon of Cover and link to Calibre library, 3. the summary of the book content (from `BookSummaries`), and 4. the description from the Calibre metadata. (1., 2., 4.) are generated in a manner similar to the 'notes' functionality of `previous/EbookTools/ebook_tools.py`. The full book text is not part of the Wiki structure (far too large). Access to book content will be via VectorDB (via MCP or similar)

## User interfaces

### Repl implementation

The main interface is a command line repl. Der repl will be implemented with the idea in mind to extend it into an agentic tool connected to an llm. In the firsts phase, no LLM connection is used (we rely on existing third party agentic tools for starting, s.b.), and command necessary to initiate data transformations are entered as slash-command just as with similar third party agentic tools. A suitable set of command that initates the transformation of the documents from source of truth into the different intermediate and resulting data formats needs to be defined. Differential and successive changes to the sources of truth must be propagated efficiently.

### Compatibility with third party tools

- Markdown tools such 'Obsidian' or 'Emacs' can be used to work with the `Wiki` knowledge base. Both parts: `Wiki/AINotes` and `Wiki/Notes` are standard wiki markdown files and may be interlinked with `[[wiki-links]]`.
- Agentic coding tools such as `gemini-cli` or similar can be used to create and alter `Wiki` parts.

## Administration

The project `previous/Summarizer` has already produced a large number of `MarkdownBooks` and corresponding `BookSummaries`. The existing data needs to be migrated into this project.

Configuration data (e.g. location of sources of truth, intermediate and resulting data) shall be stored in JSON files.

At a later point, the conversion pipeline might run as a (linux systemd) service, initiated on file-changes at the sources of truth.

Backup and version management for all markdown and image files is done via Git and is not automated (manual human responsibility)
