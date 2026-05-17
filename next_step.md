The knrs project so far implements a number of tools for local research on local books, documents and notes as prepared by the research pipeline (`/sync`).
The next step shall be to implement a truely agentic repl that integrates research and markdown-file creation into the dialog with the agent.

## General repl agent functionality

- Users shall be able to trigger research without explicit `/research` commands, but through context of the dialog.
- Users should be able to extend and refine existing research with additional directions.
- The agent should be able to organize its findings within the <wiki_path>/AINotes/Research/ folder structure. This involves creating new folder structures and moving files as needed.
- The agent should be able to interconnect research findings using wikilinks, both within the research notes and the 'human' wiki part at <notes_path>.
- The agent should be able to update timeline, markdown metadata (via /check-wiki type of functionality), and vector index so that newly generated research is immediately available for retrieval and timeline integration.

## Restrictions

- modification of content by the agent is only allowed on files within the <wiki_path>/AINotes/Research/ folder structure.
- Read access is allowed on all markdown files in <wiki_path> and <knrs_data> or via the vector index (when available).

## Priorisation of sources

- Local information either via file or vector index is prioritized over external information sources like wikipedia search.

### Point-of-view for the research project

- The local data sources reflect a wide range of different viewpoints: ranging between established science, philosophy, religion, fiction, esotericism, and hypothetical or speculative investigations. The agent should adapt that diversity by characterizing viewpoints and their interrelations or conflicts without going into dogmatic statements or labels like "pseudo", "fringe", "mainstream", "heretical", etc. Logical contradictions should be pointed out through arguments, not through labels, or ridiculing. (Probably requires some system-prompt instructions)
