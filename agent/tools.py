# To add or modify a tool, edit:
#   1. subprocesses/agent_core/agent_core/tool_registry.py  (definition, schema, prompt text)
#   2. This file — AgentTools implementation + dispatch()


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

    def _get_search_cache_file(self) -> Path:
        if "test_temp_sandbox" in str(self.config.knrs_data):
            return self.config.knrs_data / "wikipedia_search_cache.json"
        from paths import knrs_config_dir
        return knrs_config_dir() / "wikipedia_search_cache.json"

    def _load_search_cache(self) -> dict[str, str]:
        cache_file = self._get_search_cache_file()
        if cache_file.exists():
            try:
                with cache_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
            except Exception as e:
                logger.warning("Error loading wikipedia search cache: %s", e)
        return {}

    def _save_search_cache(self, cache: dict[str, str]) -> None:
        cache_file = self._get_search_cache_file()
        try:
            from calibre.converter import atomic_write
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(cache_file, json.dumps(cache, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.warning("Error saving wikipedia search cache: %s", e)

    def wikipedia_search(self, query: str) -> str:
        """Search Wikipedia for an article title."""
        normalized_query = query.strip().lower()
        cache = self._load_search_cache()
        if normalized_query in cache:
            logger.info("Using cached Wikipedia search results for query: '%s'", query)
            return cache[normalized_query]

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
                result_str = f"No Wikipedia articles found for '{query}'."
                cache[normalized_query] = result_str
                self._save_search_cache(cache)
                return result_str
            
            output = [f"Search results for '{query}':"]
            for i, res in enumerate(search_results[:10], 1):
                title = res["title"]
                snippet = re.sub(r'<[^>]+>', '', res["snippet"]) # Remove HTML tags from snippet
                output.append(f"{i}. {title} - {snippet}...")
            result_str = "\n".join(output)
            
            cache[normalized_query] = result_str
            self._save_search_cache(cache)
            return result_str
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

    # ── Stanford Encyclopedia of Philosophy ──────────────────────────────────

    _SEP_UA = "knrs/0.1.0 AgentBot (research assistant; contact: research@agent.local)"

    def sep_search(self, query: str) -> str:
        """Search the Stanford Encyclopedia of Philosophy."""
        import urllib.request
        import urllib.parse
        from bs4 import BeautifulSoup

        params = urllib.parse.urlencode({"query": query})
        url = f"https://plato.stanford.edu/search/search?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": self._SEP_UA})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                html = resp.read().decode("utf-8", errors="replace")
            soup = BeautifulSoup(html, "html.parser")
            results: list[str] = []
            seen: set[str] = set()
            for a in soup.find_all("a", href=True):
                href: str = a["href"]
                if "/entries/" in href:
                    # Normalise href to extract the slug
                    slug = href.split("/entries/")[-1].strip("/")
                    if not slug or slug in seen:
                        continue
                    seen.add(slug)
                    title = a.get_text(strip=True) or slug
                    results.append(f"- {title}  (slug: `{slug}`)")
            if not results:
                return (
                    f"No SEP results for '{query}'. "
                    "Try sep_fetch with a known slug, e.g. sep_fetch(entry='plato')."
                )
            return f"SEP search results for '{query}':\n" + "\n".join(results[:15])
        except Exception as e:
            return f"Error searching SEP: {e}"

    def sep_fetch(self, entry: str) -> str:
        """Download a Stanford Encyclopedia of Philosophy article by slug."""
        import re
        import urllib.request
        from bs4 import BeautifulSoup
        from calibre.converter import atomic_write

        entry = entry.strip().strip("/")
        url = f"https://plato.stanford.edu/entries/{entry}/"
        req = urllib.request.Request(url, headers={"User-Agent": self._SEP_UA})

        # Cache path
        sep_dir = self.research_root / "SEP"
        safe_name = re.sub(r"[^\w\s-]", "_", entry)
        file_path = sep_dir / f"{safe_name} (SEP).md"

        if file_path.exists():
            text = file_path.read_text(encoding="utf-8")
            preview = text[text.find("\n\n") + 2:][:500].strip()
            rel = file_path.relative_to(self.config.wiki_path)
            return (
                f"Loaded cached SEP article '{entry}' from {rel}\n\nPreview:\n{preview}...\n\n"
                f"Use file_read to read the full article."
            )

        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                html = resp.read().decode("utf-8", errors="replace")
            soup = BeautifulSoup(html, "html.parser")

            title_el = soup.find("h1")
            title = title_el.get_text(strip=True) if title_el else entry.title()

            # Remove navigation, bibliography, related-entries sidebars
            for tag in soup.select("nav, header, footer, script, style, "
                                   "#bibliography, #other-internet-resources, "
                                   "#related-entries, #toc, .toc"):
                tag.decompose()

            main = (
                soup.find("div", id="main-text")
                or soup.find("div", id="article-content")
                or soup.find("article")
                or soup
            )
            text = main.get_text(separator="\n", strip=True)
            text = re.sub(r"\n{3,}", "\n\n", text)

            sep_dir.mkdir(parents=True, exist_ok=True)
            md = (
                f'---\ntitle: "{title} (SEP)"\n'
                f'source: "Stanford Encyclopedia of Philosophy"\n'
                f'source_url: "{url}"\n---\n\n# {title}\n\n{text}'
            )
            atomic_write(file_path, md)
            rel = file_path.relative_to(self.config.wiki_path)
            preview = text[:500].strip()
            return (
                f"Downloaded SEP article '{title}' to {rel}\n\nPreview:\n{preview}...\n\n"
                f"Use file_read to read the full article."
            )
        except Exception as e:
            return f"Error fetching SEP entry '{entry}': {e}"

    # ── arXiv ─────────────────────────────────────────────────────────────────

    def arxiv_search(self, query: str, max_results: int = 5) -> str:
        """Search arXiv for academic papers."""
        import urllib.request
        import urllib.parse
        import xml.etree.ElementTree as ET

        try:
            max_results = max(1, min(int(max_results), 10))
        except Exception:
            max_results = 5

        params = urllib.parse.urlencode({
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        })
        url = f"http://export.arxiv.org/api/query?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "knrs/0.1.0 AgentBot"})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                xml_data = resp.read().decode("utf-8")
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            root = ET.fromstring(xml_data)
            entries = root.findall("atom:entry", ns)
            if not entries:
                return f"No arXiv papers found for '{query}'."
            lines = [f"arXiv search results for '{query}':"]
            for e in entries:
                raw_id = e.findtext("atom:id", "", ns)
                arxiv_id = raw_id.split("/abs/")[-1] if "/abs/" in raw_id else raw_id
                title = e.findtext("atom:title", "", ns).strip().replace("\n", " ")
                authors = [a.findtext("atom:name", "", ns) for a in e.findall("atom:author", ns)]
                author_str = ", ".join(authors[:3]) + ("..." if len(authors) > 3 else "")
                pub = (e.findtext("atom:published", "", ns) or "")[:10]
                abstract = e.findtext("atom:summary", "", ns).strip().replace("\n", " ")[:200]
                lines.append(
                    f"\n- {title}\n"
                    f"  ID: {arxiv_id} | Published: {pub}\n"
                    f"  Authors: {author_str}\n"
                    f"  Abstract: {abstract}..."
                )
            return "\n".join(lines)
        except Exception as e:
            return f"Error searching arXiv: {e}"

    def arxiv_fetch(self, arxiv_id: str) -> str:
        """Download an arXiv paper abstract and metadata by ID."""
        import urllib.request
        import xml.etree.ElementTree as ET
        from calibre.converter import atomic_write

        arxiv_id = arxiv_id.strip()
        cache_id = arxiv_id.replace("/", "_").replace(".", "_")
        arxiv_dir = self.research_root / "arXiv"
        file_path = arxiv_dir / f"{cache_id} (arXiv).md"

        if file_path.exists():
            text = file_path.read_text(encoding="utf-8")
            preview = text[text.find("\n\n") + 2:][:500].strip()
            rel = file_path.relative_to(self.config.wiki_path)
            return (
                f"Loaded cached arXiv paper '{arxiv_id}' from {rel}\n\nPreview:\n{preview}...\n\n"
                f"Use file_read to read the full entry."
            )

        url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        req = urllib.request.Request(url, headers={"User-Agent": "knrs/0.1.0 AgentBot"})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                xml_data = resp.read().decode("utf-8")
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            root = ET.fromstring(xml_data)
            entry = root.find("atom:entry", ns)
            if entry is None:
                return f"Error: arXiv paper '{arxiv_id}' not found."

            title = entry.findtext("atom:title", "", ns).strip().replace("\n", " ")
            authors = [a.findtext("atom:name", "", ns) for a in entry.findall("atom:author", ns)]
            abstract = entry.findtext("atom:summary", "", ns).strip()
            pub = (entry.findtext("atom:published", "", ns) or "")[:10]
            links = [
                f"{lk.get('title', lk.get('type', ''))}: {lk.get('href')}"
                for lk in entry.findall("atom:link", ns)
                if lk.get("type") in ("text/html", "application/pdf")
            ]

            md = (
                f'---\ntitle: "{title} (arXiv:{arxiv_id})"\n'
                f'source: "arXiv"\nsource_url: "https://arxiv.org/abs/{arxiv_id}"\n'
                f'published: "{pub}"\n---\n\n# {title}\n\n'
                f'**Authors:** {", ".join(authors)}\n'
                f'**Published:** {pub}\n**arXiv ID:** {arxiv_id}\n\n'
                f'## Abstract\n\n{abstract}\n\n'
                f'## Links\n\n' + "\n".join(f"- {l}" for l in links)
            )
            arxiv_dir.mkdir(parents=True, exist_ok=True)
            atomic_write(file_path, md)
            rel = file_path.relative_to(self.config.wiki_path)
            preview = abstract[:500].strip()
            return (
                f"Downloaded arXiv paper '{title}' to {rel}\n\nPreview:\n{preview}...\n\n"
                f"Use file_read to read the full entry."
            )
        except Exception as e:
            return f"Error fetching arXiv paper '{arxiv_id}': {e}"

    # ── OpenAlex ──────────────────────────────────────────────────────────────

    def openalex_search(self, query: str, max_results: int = 5) -> str:
        """Search OpenAlex for peer-reviewed academic works."""
        import json
        import urllib.request
        import urllib.parse

        try:
            max_results = max(1, min(int(max_results), 10))
        except Exception:
            max_results = 5

        params = urllib.parse.urlencode({
            "search": query,
            "per-page": max_results,
            "select": "id,title,authorships,publication_year,doi,open_access,abstract_inverted_index",
            "mailto": "knrs@research.agent",
        })
        url = f"https://api.openalex.org/works?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "knrs/0.1.0 AgentBot"})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            works = data.get("results", [])
            if not works:
                return f"No OpenAlex results for '{query}'."
            lines = [f"OpenAlex results for '{query}':"]
            for work in works:
                title = work.get("title") or "Unknown title"
                year = work.get("publication_year", "")
                doi = work.get("doi") or ""
                authors = [
                    a["author"]["display_name"]
                    for a in work.get("authorships", [])[:3]
                    if a.get("author")
                ]
                author_str = ", ".join(authors) + (
                    "..." if len(work.get("authorships", [])) > 3 else ""
                )
                # Reconstruct abstract from inverted index
                abstract = ""
                aii = work.get("abstract_inverted_index")
                if aii:
                    try:
                        max_pos = max(pos for positions in aii.values() for pos in positions)
                        words: list[str] = [""] * (max_pos + 1)
                        for word, positions in aii.items():
                            for pos in positions:
                                if pos <= max_pos:
                                    words[pos] = word
                        abstract = " ".join(words)[:250]
                    except Exception:
                        pass
                lines.append(
                    f"\n- {title} ({year})\n"
                    f"  Authors: {author_str}\n"
                    f"  DOI: {doi}\n"
                    f"  Abstract: {abstract}..."
                )
            return "\n".join(lines)
        except Exception as e:
            return f"Error searching OpenAlex: {e}"

    # ── Wikidata ──────────────────────────────────────────────────────────────

    def wikidata_search(self, query: str) -> str:
        """Search Wikidata for entities by name."""
        import json
        import urllib.request
        import urllib.parse

        params = urllib.parse.urlencode({
            "action": "wbsearchentities",
            "search": query,
            "language": "en",
            "limit": 10,
            "format": "json",
        })
        url = f"https://www.wikidata.org/w/api.php?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "knrs/0.1.0 AgentBot"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            entities = data.get("search", [])
            if not entities:
                return f"No Wikidata entities found for '{query}'."
            lines = [f"Wikidata search results for '{query}':"]
            for e in entities:
                qid = e.get("id", "")
                label = e.get("label", "")
                desc = e.get("description", "")
                lines.append(f"- {label} ({qid}): {desc}")
            return "\n".join(lines)
        except Exception as e:
            return f"Error searching Wikidata: {e}"

    # Property labels for wikidata_entity output
    _WD_PROPS: dict[str, str] = {
        "P569": "date of birth",
        "P570": "date of death",
        "P19": "place of birth",
        "P20": "place of death",
        "P21": "sex or gender",
        "P27": "country of citizenship",
        "P106": "occupation",
        "P31": "instance of",
        "P571": "inception",
        "P576": "dissolved",
        "P577": "publication date",
        "P585": "point in time",
        "P800": "notable works",
        "P50": "author",
        "P123": "publisher",
        "P136": "genre",
        "P364": "original language",
        "P495": "country of origin",
    }

    def wikidata_entity(self, entity_id: str) -> str:
        """Fetch structured data for a Wikidata entity by Q-identifier."""
        import json
        import urllib.request

        entity_id = entity_id.strip().upper()
        if not entity_id.startswith("Q"):
            return "Error: entity_id must be a Q-identifier (e.g. 'Q9312' for Kant)."

        url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
        req = urllib.request.Request(url, headers={"User-Agent": "knrs/0.1.0 AgentBot"})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            entity = data.get("entities", {}).get(entity_id, {})
            if not entity:
                return f"Entity {entity_id} not found."

            label = entity.get("labels", {}).get("en", {}).get("value", entity_id)
            description = entity.get("descriptions", {}).get("en", {}).get("value", "")
            claims = entity.get("claims", {})

            facts: list[str] = []
            for prop, prop_label in self._WD_PROPS.items():
                if prop not in claims:
                    continue
                snak = claims[prop][0].get("mainsnak", {})
                dv = snak.get("datavalue", {})
                dtype = dv.get("type", "")
                value = dv.get("value", "")
                if dtype == "string":
                    facts.append(f"- {prop_label}: {value}")
                elif dtype == "monolingualtext":
                    facts.append(f"- {prop_label}: {value.get('text', '') if isinstance(value, dict) else value}")
                elif dtype == "time" and isinstance(value, dict):
                    facts.append(f"- {prop_label}: {value.get('time', '')}")
                elif dtype == "wikibase-entityid" and isinstance(value, dict):
                    sub_id = value.get("id", "")
                    facts.append(f"- {prop_label}: {sub_id}")

            result = f"**{label}** ({entity_id})\n{description}\n"
            if facts:
                result += "\nKey facts:\n" + "\n".join(facts)
            result += f"\n\nFull data: https://www.wikidata.org/wiki/{entity_id}"
            return result
        except Exception as e:
            return f"Error fetching Wikidata entity '{entity_id}': {e}"

    # ── Internet Archive ──────────────────────────────────────────────────────

    _IA_UA = "knrs/0.1.0 AgentBot (research assistant)"

    def archive_search(self, query: str, max_results: int = 5) -> str:
        """Search the Internet Archive for texts."""
        import json
        import urllib.request
        import urllib.parse

        try:
            max_results = max(1, min(int(max_results), 10))
        except Exception:
            max_results = 5

        params = urllib.parse.urlencode({
            "q": f"({query}) AND mediatype:texts",
            "output": "json",
            "rows": max_results,
            "sort": "downloads desc",
        }) + "&fl[]=identifier&fl[]=title&fl[]=creator&fl[]=date"
        url = f"https://archive.org/advancedsearch.php?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": self._IA_UA})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            docs = data.get("response", {}).get("docs", [])
            if not docs:
                return f"No Internet Archive texts found for '{query}'."
            lines = [f"Internet Archive search results for '{query}':"]
            for doc in docs:
                ident = doc.get("identifier", "")
                title = doc.get("title", "Unknown title")
                creator = doc.get("creator", "")
                date = doc.get("date", "")
                if isinstance(creator, list):
                    creator = ", ".join(creator)
                if isinstance(date, list):
                    date = date[0]
                lines.append(
                    f"- {title}\n"
                    f"  Creator: {creator} | Date: {date}\n"
                    f"  Identifier: {ident}\n"
                    f"  URL: https://archive.org/details/{ident}"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"Error searching Internet Archive: {e}"

    def archive_fetch(self, identifier: str) -> str:
        """Download a text from the Internet Archive by identifier."""
        import json
        import urllib.request
        from calibre.converter import atomic_write

        identifier = identifier.strip()
        ia_dir = self.research_root / "InternetArchive"
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in identifier)
        file_path = ia_dir / f"{safe_id} (Internet Archive).md"

        if file_path.exists():
            text = file_path.read_text(encoding="utf-8")
            preview = text[text.find("\n\n") + 2:][:500].strip()
            rel = file_path.relative_to(self.config.wiki_path)
            return (
                f"Loaded cached Internet Archive text '{identifier}' from {rel}\n\nPreview:\n{preview}...\n\n"
                f"Use file_read to read the full text."
            )

        meta_url = f"https://archive.org/metadata/{identifier}"
        req = urllib.request.Request(meta_url, headers={"User-Agent": self._IA_UA})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                meta = json.loads(resp.read().decode("utf-8"))
            if not meta:
                return f"Error: Internet Archive item '{identifier}' not found."

            ia_meta = meta.get("metadata", {})

            def _first(v: object) -> str:
                if isinstance(v, list):
                    return v[0] if v else ""
                return str(v) if v else ""

            title = _first(ia_meta.get("title", identifier))
            creator = _first(ia_meta.get("creator", ""))
            date = _first(ia_meta.get("date", ""))
            description = _first(ia_meta.get("description", ""))

            # Find a suitable text file
            files = meta.get("files", [])
            text_file: str | None = None
            for preferred_ext in ("_djvu.txt", "_full_text.txt", ".txt"):
                for f in files:
                    fname = f.get("name", "")
                    if fname.endswith(preferred_ext) and not fname.endswith("_meta.txt"):
                        text_file = fname
                        break
                if text_file:
                    break

            text_content = ""
            if text_file:
                text_url = f"https://archive.org/download/{identifier}/{text_file}"
                text_req = urllib.request.Request(text_url, headers={"User-Agent": self._IA_UA})
                try:
                    with urllib.request.urlopen(text_req, timeout=30) as resp:
                        raw = resp.read()
                    text_content = raw.decode("utf-8", errors="replace")
                    if len(text_content) > 50_000:
                        text_content = (
                            text_content[:50_000]
                            + "\n\n[... text truncated at 50 000 characters ...]"
                        )
                except Exception as e:
                    text_content = f"(Could not download text: {e})"
            else:
                text_content = (
                    f"(No plain-text file found for this item. "
                    f"Browse manually: https://archive.org/details/{identifier})"
                )

            ia_dir.mkdir(parents=True, exist_ok=True)
            md = (
                f'---\ntitle: "{title} (Internet Archive)"\n'
                f'source: "Internet Archive"\n'
                f'source_url: "https://archive.org/details/{identifier}"\n'
                f'creator: "{creator}"\ndate: "{date}"\n---\n\n'
                f'# {title}\n\n'
                f'**Creator:** {creator}\n**Date:** {date}\n**Identifier:** {identifier}\n\n'
                f'## Description\n\n{description}\n\n'
                f'## Text\n\n{text_content}'
            )
            atomic_write(file_path, md)
            rel = file_path.relative_to(self.config.wiki_path)
            preview = text_content[:500].strip()
            return (
                f"Downloaded Internet Archive text '{title}' to {rel}\n\nPreview:\n{preview}...\n\n"
                f"Use file_read to read the full text."
            )
        except Exception as e:
            return f"Error fetching Internet Archive item '{identifier}': {e}"

    # ── Computational tools ───────────────────────────────────────────────────

    def python_eval(self, code: str) -> str:
        """Execute a sandboxed Python snippet via RestrictedPython."""
        import io
        import math
        import statistics

        try:
            from RestrictedPython import (
                compile_restricted,
                safe_globals,
                safe_builtins,
                PrintCollector,
            )
            from RestrictedPython.Guards import safer_getattr, guarded_unpack_sequence
        except ImportError:
            return "Error: RestrictedPython is not installed. Run: uv add RestrictedPython"

        allowed_builtins: dict[str, object] = {
            **safe_builtins,
            "range": range,
            "len": len,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "sorted": sorted,
            "reversed": reversed,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "isinstance": isinstance,
            "type": type,
            "repr": repr,
        }
        globs: dict[str, object] = {
            **safe_globals,
            "__builtins__": allowed_builtins,
            "math": math,
            "statistics": statistics,
            "_getattr_": safer_getattr,
            "_getitem_": lambda obj, key: obj[key],  # default item access
            "_getiter_": iter,
            "_write_": lambda x: x,
            "_inplacevar_": lambda op, x, y: x + y if op == "+=" else x - y,
            "_print_": PrintCollector,  # RestrictedPython print() guard
            "_unpack_sequence_": guarded_unpack_sequence,
        }

        try:
            byte_code = compile_restricted(code, "<python_eval>", "exec")
            exec(byte_code, globs)  # noqa: S102
            # PrintCollector: calling the instance returns collected text
            printer = globs.get("_print")
            output = printer() if callable(printer) else ""
            return output if output.strip() else "(executed — no output produced)"
        except SyntaxError as e:
            return f"Syntax error: {e}"
        except Exception as e:
            return f"Runtime error: {e}"




    def maxima_eval(self, expression: str) -> str:
        """Evaluate a Maxima CAS expression."""
        import shutil
        import subprocess

        if not shutil.which("maxima"):
            return "Error: 'maxima' is not installed. Install it via your package manager."

        batch = f"display2d: false$ {expression.rstrip(';')};"
        try:
            result = subprocess.run(
                ["maxima", "--quiet", "--batch-string", batch],
                capture_output=True,
                text=True,
                timeout=30,
            )
            # Keep only output lines (those with (%o or raw values)
            raw = result.stdout + result.stderr
            lines = [
                l for l in raw.splitlines()
                if l.strip()
                and not l.startswith("Maxima")
                and "(%i" not in l
                and not l.startswith(";;")
            ]
            output = "\n".join(lines).strip()
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: Maxima evaluation timed out (30 s)."
        except Exception as e:
            return f"Error running Maxima: {e}"

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
        elif tool_name == "sep_search":
            return self.sep_search(**args)
        elif tool_name == "sep_fetch":
            return self.sep_fetch(**args)
        elif tool_name == "arxiv_search":
            return self.arxiv_search(**args)
        elif tool_name == "arxiv_fetch":
            return self.arxiv_fetch(**args)
        elif tool_name == "openalex_search":
            return self.openalex_search(**args)
        elif tool_name == "wikidata_search":
            return self.wikidata_search(**args)
        elif tool_name == "wikidata_entity":
            return self.wikidata_entity(**args)
        elif tool_name == "archive_search":
            return self.archive_search(**args)
        elif tool_name == "archive_fetch":
            return self.archive_fetch(**args)
        elif tool_name == "python_eval":
            return self.python_eval(**args)
        elif tool_name == "maxima_eval":
            return self.maxima_eval(**args)
        elif tool_name == "check_wiki":
            return self.check_wiki()
        elif tool_name == "update_index":
            return self.update_index()
        elif tool_name == "extract_timeline":
            return self.extract_timeline(**args)
        else:
            return f"Error: Unknown tool '{tool_name}'"


