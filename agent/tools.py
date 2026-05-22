# =========================================================================
# DEVELOPER WARNING: SINGLE SOURCE OF TRUTH (SST) FOR AGENT TOOLS
#
# If you add, modify, or remove any agent tools, you MUST update:
# 1. agent/tools.py (The dynamic dispatch & implementation)
# 2. agent/prompts.py (The text-based instructions for raw LLMs)
# 3. subprocesses/agent_api/agent_api.py (The JSON schema array)
# 4. subprocesses/agent_macos/agent_macos.py (The JSON schema array)
# 5. subprocesses/agent_hf/agent_hf.py (The JSON schema array)
# =========================================================================

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any
from config import KnrsConfig

logger = logging.getLogger(__name__)

class AgentTools:
    def __init__(self, config: KnrsConfig) -> None:
        self.config: KnrsConfig = config
        self.research_root: Path = self.config.wiki_path / "AINotes" / "Research"
        self.research_root.mkdir(parents=True, exist_ok=True)
        
    def _is_safe_read_path(self, path: Path) -> bool:
        """Check if a path is within the allowed read roots."""
        roots: list[Path] = [
            self.config.markdown_books,
            self.config.book_summaries,
            self.config.wiki_path
        ]
        try:
            resolved = path.resolve()
            return any(resolved.is_relative_to(r.resolve()) for r in roots)
        except Exception:
            return False

    def _is_safe_write_path(self, path: Path) -> bool:
        """Check if a path is strictly within the AINotes/Research root."""
        try:
            resolved = path.resolve()
            return resolved.is_relative_to(self.research_root.resolve())
        except Exception:
            return False
            
    def _resolve_read_path(self, path_str: str) -> Path | None:
        """Resolve a string path (absolute, relative, or prefixed) to a safe Path object.
        Supports:
          - Bracketed links: "[[Some Page]]" -> "Some Page"
          - Direct relative/absolute paths
          - Missing extensions: "wiki:Notes/Some Page" -> "wiki:Notes/Some Page.md"
          - Stems/Wiki-links: "Some Page" -> resolved to full path in wiki/books/summaries
        """
        import unicodedata

        # 1. Clean bracketed links and whitespace
        path_str = path_str.strip()
        if path_str.startswith("[[") and path_str.endswith("]]"):
            path_str = path_str[2:-2].strip()

        # 2. Extract logical prefixes
        prefix = None
        if path_str.startswith("books:"):
            prefix = "books"
            sub_path = path_str.split(":", 1)[1]
        elif path_str.startswith("wiki:"):
            prefix = "wiki"
            sub_path = path_str.split(":", 1)[1]
        elif path_str == "books":
            return self.config.markdown_books
        elif path_str == "wiki":
            return self.config.wiki_path
        else:
            sub_path = path_str

        # 3. Direct checks (with and without appending .md/.markdown extensions)
        def find_direct(base_dir: Path, rel_path: str) -> Path | None:
            # Try exact path
            p = base_dir / rel_path
            if p.exists():
                return p
            # Try common extensions
            for ext in (".md", ".markdown"):
                p_ext = base_dir / f"{rel_path}{ext}"
                if p_ext.exists():
                    return p_ext
            return None

        # Resolve based on prefix
        if prefix == "books":
            resolved = find_direct(self.config.markdown_books, sub_path)
            if resolved: return resolved
        elif prefix == "wiki":
            resolved = find_direct(self.config.wiki_path, sub_path)
            if resolved: return resolved
        else:
            p = Path(path_str)
            if p.is_absolute() and p.exists():
                return p
            
            # Try direct relative path under wiki first, then books
            resolved = find_direct(self.config.wiki_path, sub_path)
            if resolved: return resolved
            resolved = find_direct(self.config.markdown_books, sub_path)
            if resolved: return resolved

        # 4. Search by stem (Wiki-link style)
        # Normalize the target stem to match case-insensitively
        target_stem = Path(sub_path).stem
        target_norm = unicodedata.normalize("NFC", target_stem).strip().lower()

        # Determine search roots based on prefix
        roots = []
        if prefix == "wiki":
            roots = [self.config.wiki_path]
        elif prefix == "books":
            roots = [self.config.markdown_books]
        else:
            roots = [
                self.config.wiki_path,
                self.config.markdown_books,
                self.config.book_summaries,
            ]

        for root in roots:
            for md_path in root.rglob("*.md"):
                if ".stfolder" in md_path.parts or ".git" in md_path.parts:
                    continue
                stem_norm = unicodedata.normalize("NFC", md_path.stem).strip().lower()
                if stem_norm == target_norm:
                    return md_path

        return None

    def vector_search(self, query: str, top_k: int = 5) -> str:
        """Semantic search across indexed files."""
        try:
            try:
                top_k = int(top_k)
            except Exception as e:
                logger.warning(f"Failed to cast top_k parameter '{top_k}' (type {type(top_k).__name__}) to int: {e}")
                top_k = 5
            from vector.search import KnrsSearcher, get_context_aware_text
            from calibre.converter import _split_frontmatter
            import yaml
            
            searcher = KnrsSearcher(self.config)
            results = searcher.search(query, top_k=min(top_k, 10))
            
            if not results:
                return "No semantic search results found."
                
            output: list[str] = []
            for i, r in enumerate(results, 1):
                text, start_line, end_line = get_context_aware_text(searcher, r)
                title = "Unknown"
                
                # Try to get title
                file_path: Path | None = None
                if r.source_label == "books":
                    file_path = self.config.markdown_books / r.bare_path
                elif r.source_label == "wiki":
                    file_path = self.config.wiki_path / r.bare_path
                    
                if file_path and file_path.exists():
                    try:
                        content = file_path.read_text(encoding="utf-8")
                        fm, _ = _split_frontmatter(content)
                        if fm:
                            meta: dict[str, Any] = yaml.safe_load(fm) or {}
                            title = meta.get("title", title)
                    except Exception:
                        pass
                
                line_info = f"Lines {start_line}-{end_line}" if start_line > 0 else "Unknown lines"
                output.append(f"Result {i} (Score: {r.score:.3f})\nSource: {r.path}\nTitle: {title}\nLocation: {line_info}\nContent snippet:\n{text}")
                
            return "\n\n---\n\n".join(output)
        except Exception as e:
            return f"Error during vector search: {e}"

    def file_read(self, path: str, start_line: int = 1, end_line: int = -1) -> str:
        """Read lines from a markdown file."""
        try:
            try:
                start_line = int(start_line)
            except Exception as e:
                logger.warning(f"Failed to cast start_line parameter '{start_line}' (type {type(start_line).__name__}) to int: {e}")
                start_line = 1
            try:
                end_line = int(end_line)
            except Exception as e:
                logger.warning(f"Failed to cast end_line parameter '{end_line}' (type {type(end_line).__name__}) to int: {e}")
                end_line = -1
            p = self._resolve_read_path(path)
            if not p or not p.exists():
                return f"Error: File not found: {path}"
                
            if not self._is_safe_read_path(p):
                return f"Error: Path is outside allowed read directories: {path}"
                
            content = p.read_text(encoding="utf-8")
            lines = content.splitlines()
            
            total_lines = len(lines)
            if start_line < 1: start_line = 1
            if end_line == -1 or end_line > total_lines: end_line = total_lines
            
            # 1-indexed
            selected = lines[start_line - 1 : end_line]
            return "\n".join(selected)
        except Exception as e:
            return f"Error reading file {path}: {e}"

    def wikipedia_search(self, query: str) -> str:
        """Search Wikipedia for an article title."""
        try:
            import urllib.request
            import urllib.parse
            import json
            import re
            
            params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "utf8": "",
                "format": "json"
            }
            url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
            req = urllib.request.Request(url, headers={"User-Agent": "knrs/0.1.0 (https://github.com/domschl/knrs) AgentBot"})
            
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
                
            search_results = data.get("query", {}).get("search", [])
            if not search_results:
                return f"No Wikipedia articles found for '{query}'."
            
            output = [f"Search results for '{query}':"]
            for i, res in enumerate(search_results[:10], 1):
                title = res["title"]
                snippet = re.sub(r'<[^>]+>', '', res["snippet"]) # Remove HTML tags from snippet
                output.append(f"{i}. {title} - {snippet}...")
            return "\n".join(output)
        except Exception as e:
            return f"Error searching Wikipedia: {e}"

    def wikipedia_fetch(self, title: str) -> str:
        """Download a Wikipedia article in plain text and save it to AINotes/Research/Wikipedia/."""
        try:
            # Normalize title: isolate original Wikipedia query and format local wiki title
            postfix = " (Wikipedia)"
            if title.endswith(postfix):
                original_title = title[:-len(postfix)].strip()
            else:
                original_title = title.strip()
                
            wiki_title = original_title + postfix

            # Clean filename using wiki_title, allowing parenthesis
            safe_title = "".join([c if c.isalnum() or c in " -_()" else "_" for c in wiki_title])
            wiki_dir = self.research_root / "Wikipedia"
            file_path = wiki_dir / f"{safe_title}.md"

            # Check local cache first (case-insensitive)
            cached_file = None
            if wiki_dir.exists():
                target_file_name = f"{safe_title}.md".lower()
                for item in wiki_dir.iterdir():
                    if item.is_file() and item.name.lower() == target_file_name:
                        cached_file = item
                        break

            if cached_file:
                try:
                    content = cached_file.read_text(encoding="utf-8")
                    from calibre.converter import _split_frontmatter
                    _, body = _split_frontmatter(content)
                    body_clean = body.strip()
                    if body_clean:
                        # Clean up duplicate title header at the start of the body
                        if body_clean.startswith(f"# {title}"):
                            body_clean = body_clean[len(f"# {title}"):].strip()
                        elif body_clean.startswith(f"# {wiki_title}"):
                            body_clean = body_clean[len(f"# {wiki_title}"):].strip()
                        elif body_clean.startswith(f"# {original_title}"):
                            body_clean = body_clean[len(f"# {original_title}"):].strip()
                        elif body_clean.startswith("#"):
                            lines = body_clean.splitlines()
                            if lines and lines[0].strip().startswith("#"):
                                body_clean = "\n".join(lines[1:]).strip()

                        preview = body_clean[:500] + "..." if len(body_clean) > 500 else body_clean
                        return f"Successfully loaded cached article '{wiki_title}' from {cached_file.relative_to(self.config.wiki_path)}\n\nPreview:\n{preview}\n\nUse file_read with this path to read the full article."
                except Exception as e:
                    logger.warning("Error reading cached file %s: %s. Falling back to download.", cached_file, e)

            import urllib.request
            import urllib.parse
            import json
            
            params = {
                "action": "query",
                "prop": "extracts",
                "explaintext": "1",
                "titles": original_title,
                "format": "json"
            }
            url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
            req = urllib.request.Request(url, headers={"User-Agent": "knrs/0.1.0 (https://github.com/domschl/knrs) AgentBot"})
            
            with urllib.request.urlopen(req, timeout=20) as response:
                data = json.loads(response.read().decode('utf-8'))
            
            pages = data.get("query", {}).get("pages", {})
            page = next(iter(pages.values()))
            if "missing" in page:
                return f"Error: Wikipedia article '{original_title}' not found."
                
            content = page.get("extract", "")
            if not content:
                return f"Error: No content found for '{original_title}'."
                
            # Save it to AINotes/Research/Wikipedia/
            wiki_dir.mkdir(parents=True, exist_ok=True)
            
            # Write frontmatter and content
            from calibre.converter import atomic_write
            md_content = f"---\ntitle: \"{wiki_title}\"\nsource: \"Wikipedia\"\n---\n\n# {wiki_title}\n\n{content}"
            atomic_write(file_path, md_content)
            
            preview = content[:500] + "..." if len(content) > 500 else content
            return f"Successfully downloaded '{wiki_title}' to {file_path.relative_to(self.config.wiki_path)}\n\nPreview:\n{preview}\n\nUse file_read with this path to read the full article."
        except Exception as e:
            return f"Error fetching Wikipedia article: {e}"

    def file_list(self, directory: str) -> str:
        """List files in a directory."""
        try:
            p = self._resolve_read_path(directory)
            if not p or not p.exists():
                return f"Error: Directory not found: {directory}"
                
            if not self._is_safe_read_path(p):
                return f"Error: Path is outside allowed read directories: {directory}"
                
            files: list[str] = []
            for item in p.iterdir():
                if item.name.startswith("."): continue
                type_str = "DIR" if item.is_dir() else "FILE"
                files.append(f"{type_str}\t{item.name}")
                
            return "\n".join(sorted(files))
        except Exception as e:
            return f"Error listing directory {directory}: {e}"

    def timeline_query(self, start_year: float | None = None, end_year: float | None = None, context_filters: list[str] | None = None, keywords: list[str] | None = None) -> str:
        """Query timeline database and return formatted table."""
        try:
            if start_year is not None:
                try:
                    start_year = float(start_year)
                except Exception as e:
                    logger.warning(f"Failed to cast start_year parameter '{start_year}' (type {type(start_year).__name__}) to float: {e}")
                    start_year = None
            if end_year is not None:
                try:
                    end_year = float(end_year)
                except Exception as e:
                    logger.warning(f"Failed to cast end_year parameter '{end_year}' (type {type(end_year).__name__}) to float: {e}")
                    end_year = None
            from timelines.extractor import query_timeline_data, format_timeline_as_markdown_table
            timeline_file = self.config.timelines / "timelines.json"
            
            if isinstance(keywords, str):
                keywords = [keywords]
                
            events = query_timeline_data(timeline_file, start_year, end_year, context_filters, keywords)
            if not events:
                return "No timeline events matched the criteria."
                
            # Limit events to prevent context overflow
            if len(events) > 50:
                events = events[:50]
                truncated = "\n\n*(Note: Results truncated to first 50 events)*"
            else:
                truncated = ""
                
            return format_timeline_as_markdown_table(events) + truncated
        except Exception as e:
            return f"Error querying timelines: {e}"

    def _sanitize_write_path(self, path: str) -> Path:
        """Sanitize a path string to ensure it safely resolves within research_root."""
        # Remove redundant prefix
        if path.startswith("AINotes/Research/"):
            path = path[len("AINotes/Research/"):]
            
        # Strip logical prefixes the agent might mistakenly include
        if path.startswith("books:"):
            path = path[len("books:"):]
        elif path.startswith("wiki:"):
            path = path[len("wiki:"):]
            
        # Strip absolute prefix if agent passes e.g., /home/.../AINotes/Research/...
        # By removing the root slash, Path(path) will treat it as relative.
        while path.startswith("/"):
            path = path[1:]
            
        # Replace colon which might cause issues
        path = path.replace(":", "_")
        
        p = Path(path)
        # If it tries to navigate up, flatten to just its name
        if ".." in p.parts:
            p = Path(p.name)
            
        if not p.name or p.name == ".":
            p = Path("unnamed_research_file.md")
            
        return self.research_root / p

    def _check_filename_uniqueness(self, target_path: Path, exclude_paths: list[Path] | None = None) -> str | None:
        """Check if the stem of target_path is already used by an existing md file in wiki_path.
        Returns an error message if duplicate found, else None.
        """
        import unicodedata
        
        if target_path.suffix.lower() != ".md":
            return None
            
        target_stem = unicodedata.normalize("NFC", target_path.stem).strip().lower()
        exclude_set = {p.resolve() for p in (exclude_paths or []) if p.exists()}
        
        for md_path in self.config.wiki_path.rglob("*.md"):
            if ".stfolder" in md_path.parts or ".git" in md_path.parts:
                continue
            
            try:
                resolved_md = md_path.resolve()
            except Exception:
                resolved_md = md_path
                
            if resolved_md in exclude_set:
                continue
                
            norm_stem = unicodedata.normalize("NFC", md_path.stem).strip().lower()
            if norm_stem == target_stem:
                try:
                    rel_path = md_path.relative_to(self.config.wiki_path)
                except ValueError:
                    rel_path = md_path
                return (
                    f"Error: The filename stem '{md_path.stem}' is already in use at '{rel_path}'. "
                    f"Filenames must be globally unique within the wiki tree to ensure wikilinks function correctly. "
                    f"Please select a different, unique filename."
                )
        return None

    def _check_filename_uniqueness_for_move(self, p_src: Path, p_dst: Path) -> str | None:
        """Verify that moving p_src to p_dst will not introduce any duplicate stems.
        Returns error string if duplicate is found, else None.
        """
        files_to_check: list[tuple[Path, Path]] = []
        
        if p_src.is_file():
            if p_dst.is_dir():
                expected_dest = p_dst / p_src.name
            else:
                expected_dest = p_dst
            files_to_check.append((p_src, expected_dest))
        elif p_src.is_dir():
            if p_dst.is_dir():
                base_dest = p_dst / p_src.name
            else:
                base_dest = p_dst
                
            for p in p_src.rglob("*.md"):
                if p.is_file():
                    try:
                        rel = p.relative_to(p_src)
                        expected_dest = base_dest / rel
                        files_to_check.append((p, expected_dest))
                    except ValueError:
                        pass
                        
        exclude_paths = [src for src, _ in files_to_check]
        
        for _, dest in files_to_check:
            dup_error = self._check_filename_uniqueness(dest, exclude_paths=exclude_paths)
            if dup_error:
                return dup_error
                
        return None

    def file_write(self, path: str, content: str) -> str:
        """Write content to a file in AINotes/Research/."""
        try:
            from calibre.converter import atomic_write
            
            p = self._sanitize_write_path(path)
                
            if not self._is_safe_write_path(p):
                return f"Error: Cannot write outside {self.research_root}"
                
            # Check for duplicate filename stem across the entire wiki tree
            dup_error = self._check_filename_uniqueness(p, exclude_paths=[p])
            if dup_error:
                return dup_error

            atomic_write(p, content)
            return f"Successfully wrote to {p}."
        except Exception as e:
            return f"Error writing to file {path}: {e}"

    def file_append(self, path: str, content: str) -> str:
        """Append content to a file in AINotes/Research/."""
        try:
            p = self._sanitize_write_path(path)
                
            if not self._is_safe_write_path(p):
                return f"Error: Cannot write outside {self.research_root}"
                
            if not p.exists():
                return f"Error: File {path} does not exist. Use file_write for the initial write."
                
            with open(p, "a", encoding="utf-8") as f:
                f.write(content)
            return f"Successfully appended to {p}."
        except Exception as e:
            return f"Error appending to file {path}: {e}"

    def create_directory(self, path: str) -> str:
        """Create a directory in AINotes/Research/."""
        try:
            p = self._sanitize_write_path(path)
                
            if not self._is_safe_write_path(p):
                return f"Error: Cannot create directory outside {self.research_root}"
                
            p.mkdir(parents=True, exist_ok=True)
            return f"Successfully created directory {p}"
        except Exception as e:
            return f"Error creating directory {path}: {e}"

    def file_move(self, src: str, dst: str) -> str:
        """Move or rename a file or directory within AINotes/Research/."""
        try:
            import shutil

            p_src = self._sanitize_write_path(src)
            p_dst = self._sanitize_write_path(dst)
                
            if not self._is_safe_write_path(p_src):
                return f"Error: Cannot move source from outside {self.research_root}"
            if not self._is_safe_write_path(p_dst):
                return f"Error: Cannot move destination outside {self.research_root}"
                
            if not p_src.exists():
                return f"Error: Source does not exist: {src}"
                
            # Check for duplicate filename stems that would result from this move
            dup_error = self._check_filename_uniqueness_for_move(p_src, p_dst)
            if dup_error:
                return dup_error

            # Create destination directory if it doesn't exist
            p_dst.parent.mkdir(parents=True, exist_ok=True)
                
            shutil.move(str(p_src), str(p_dst))
            return f"Successfully moved {src} to {dst}"
        except Exception as e:
            return f"Error moving {src} to {dst}: {e}"

    def wikilink_search(self, query: str) -> str:
        """Search wiki file index for documents whose stem matches a query.

        Returns matching stems that can be used as ``[[wikilink]]`` targets.
        """
        import unicodedata
        try:
            query_lower = unicodedata.normalize("NFC", query).strip().lower()
            matches: list[str] = []

            for md_path in self.config.wiki_path.rglob("*.md"):
                if ".stfolder" in md_path.parts or ".git" in md_path.parts:
                    continue
                stem = md_path.stem
                norm_stem = unicodedata.normalize("NFC", stem).strip().lower()
                if query_lower in norm_stem:
                    # Deduplicate by stem
                    if stem not in matches:
                        matches.append(stem)

            if not matches:
                return f"No wiki documents found matching '{query}'."

            matches.sort()
            if len(matches) > 50:
                matches = matches[:50]
                truncated = "\n\n*(Showing first 50 matches)*"
            else:
                truncated = ""

            return "Matching documents (usable as [[wikilink]] targets):\n" + \
                   "\n".join(f"  [[{m}]]" for m in matches) + truncated
        except Exception as e:
            return f"Error searching wikilinks: {e}"

    def check_wiki(self) -> str:
        """Run metadata checks on all files in AINotes/Research/."""
        try:
            from wiki.checker import ensure_minimal_frontmatter
            from calibre.converter import _split_frontmatter, atomic_write
            import yaml

            checked = 0
            updated = 0

            for md_path in self.research_root.rglob("*.md"):
                if md_path.name.startswith("."):
                    continue
                checked += 1
                try:
                    content = md_path.read_text(encoding="utf-8")
                    fm_raw, body = _split_frontmatter(content)
                    meta = yaml.safe_load(fm_raw) if fm_raw else {}
                    if not isinstance(meta, dict):
                        meta = {}

                    if ensure_minimal_frontmatter(md_path, self.config.wiki_path, meta):
                        new_fm = yaml.dump(meta, default_flow_style=False, allow_unicode=True, indent=2)
                        new_content = f"---\n{new_fm}---\n{body}"
                        atomic_write(md_path, new_content)
                        updated += 1
                except Exception as e:
                    logger.warning("check_wiki: skipping %s: %s", md_path.name, e)

            return f"Checked {checked} files in AINotes/Research/. Updated metadata in {updated} files."
        except Exception as e:
            return f"Error running check_wiki: {e}"

    def update_index(self) -> str:
        """Run the full differential vector index update."""
        try:
            from vector.indexer import KnrsIndexer

            indexer = KnrsIndexer(self.config)
            indexer.run_indexing(
                self.config.markdown_books,
                self.config.wiki_path,
            )
            return "Vector index update complete."
        except Exception as e:
            return f"Error updating index: {e}"

    def extract_timeline(self, path: str) -> str:
        """Extract timeline tables from a research file and merge into timelines.json."""
        try:
            from timelines.extractor import extract_from_file

            p = self._sanitize_write_path(path)
            if not p.exists():
                return f"Error: File not found: {path}"

            # Extract events using wiki_path as root (research files live under wiki_path)
            events = extract_from_file(p, self.config.wiki_path)

            if not events:
                return f"No timeline tables found in {path}."

            # Load existing timeline data
            timeline_file = self.config.timelines / "timelines.json"
            existing: list[Any] = []
            if timeline_file.exists():
                import json as _json
                with timeline_file.open("r", encoding="utf-8") as f:
                    existing = _json.load(f)

            # Remove any existing events from this file, then add new ones
            rel_path = str(p.relative_to(self.config.wiki_path))
            existing = [e for e in existing if e.get("source_file") != rel_path]
            existing.extend(e.to_dict() for e in events)

            # Sort by start_year
            existing.sort(key=lambda x: (x.get("start_year", 0), x.get("end_year", 0)))

            timeline_file.parent.mkdir(parents=True, exist_ok=True)
            import json as _json
            with timeline_file.open("w", encoding="utf-8") as f:
                _json.dump(existing, f, indent=2)

            return f"Extracted {len(events)} timeline events from {path} and merged into timelines.json."
        except Exception as e:
            return f"Error extracting timeline from {path}: {e}"

    def dispatch(self, tool_name: str, args: dict[str, Any]) -> str:
        """Execute a tool dynamically."""
        logger.debug(f"Agent tool call: {tool_name}({args})")
        if tool_name == "vector_search":
            return self.vector_search(**args)
        elif tool_name == "file_read":
            return self.file_read(**args)
        elif tool_name == "file_list":
            return self.file_list(**args)
        elif tool_name == "timeline_query":
            return self.timeline_query(**args)
        elif tool_name == "file_write":
            return self.file_write(**args)
        elif tool_name == "file_append":
            return self.file_append(**args)
        elif tool_name == "create_directory":
            return self.create_directory(**args)
        elif tool_name == "file_move":
            return self.file_move(**args)
        elif tool_name == "wikipedia_search":
            return self.wikipedia_search(**args)
        elif tool_name == "wikipedia_fetch":
            return self.wikipedia_fetch(**args)
        elif tool_name == "wikilink_search":
            return self.wikilink_search(**args)
        elif tool_name == "check_wiki":
            return self.check_wiki()
        elif tool_name == "update_index":
            return self.update_index()
        elif tool_name == "extract_timeline":
            return self.extract_timeline(**args)
        else:
            return f"Error: Unknown tool {tool_name}"

