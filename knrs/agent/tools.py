from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from knrs.config import KnrsConfig

logger = logging.getLogger(__name__)

class AgentTools:
    def __init__(self, config: KnrsConfig) -> None:
        self.config: KnrsConfig = config
        self.research_root: Path = self.config.wiki_path / "AINotes" / "Research"
        self.research_root.mkdir(parents=True, exist_ok=True)
        
    def _is_safe_read_path(self, path: Path) -> bool:
        """Check if a path is within the allowed read roots."""
        roots: List[Path] = [
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
            
    def _resolve_read_path(self, path_str: str) -> Optional[Path]:
        """Resolve a string path (absolute, relative, or prefixed) to a safe Path object."""
        if path_str.startswith("books:"):
            return self.config.markdown_books / path_str.split(":", 1)[1]
        elif path_str.startswith("wiki:"):
            return self.config.wiki_path / path_str.split(":", 1)[1]
        elif path_str == "books":
            return self.config.markdown_books
        elif path_str == "wiki":
            return self.config.wiki_path
            
        p = Path(path_str)
        if not p.is_absolute():
            # Try against wiki first, then books
            test_wiki = self.config.wiki_path / p
            if test_wiki.exists(): return test_wiki
            test_books = self.config.markdown_books / p
            if test_books.exists(): return test_books
            return p # Fallback to returning the relative path as-is, though it might fail safety checks
            
        return p

    def vector_search(self, query: str, top_k: int = 5) -> str:
        """Semantic search across indexed files."""
        try:
            from knrs.vector.search import KnrsSearcher, get_context_aware_text
            from knrs.calibre.converter import _split_frontmatter
            import yaml
            
            searcher = KnrsSearcher(self.config)
            results = searcher.search(query, top_k=min(top_k, 10))
            
            if not results:
                return "No semantic search results found."
                
            output: List[str] = []
            for i, r in enumerate(results, 1):
                text, start_line, end_line = get_context_aware_text(searcher, r)
                title = "Unknown"
                
                # Try to get title
                file_path: Optional[Path] = None
                if r.source_label == "books":
                    file_path = self.config.markdown_books / r.bare_path
                elif r.source_label == "wiki":
                    file_path = self.config.wiki_path / r.bare_path
                    
                if file_path and file_path.exists():
                    try:
                        content = file_path.read_text(encoding="utf-8")
                        fm, _ = _split_frontmatter(content)
                        if fm:
                            meta: Dict[str, Any] = yaml.safe_load(fm) or {}
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
            import urllib.request
            import urllib.parse
            import json
            
            params = {
                "action": "query",
                "prop": "extracts",
                "explaintext": "1",
                "titles": title,
                "format": "json"
            }
            url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
            req = urllib.request.Request(url, headers={"User-Agent": "knrs/0.1.0 (https://github.com/domschl/knrs) AgentBot"})
            
            with urllib.request.urlopen(req, timeout=20) as response:
                data = json.loads(response.read().decode('utf-8'))
            
            pages = data.get("query", {}).get("pages", {})
            page = next(iter(pages.values()))
            if "missing" in page:
                return f"Error: Wikipedia article '{title}' not found."
                
            content = page.get("extract", "")
            if not content:
                return f"Error: No content found for '{title}'."
                
            # Save it to AINotes/Research/Wikipedia/
            wiki_dir = self.research_root / "Wikipedia"
            wiki_dir.mkdir(parents=True, exist_ok=True)
            
            # Clean filename
            safe_title = "".join([c if c.isalnum() or c in " -_" else "_" for c in title])
            file_path = wiki_dir / f"{safe_title}.md"
            
            # Write frontmatter and content
            from knrs.calibre.converter import atomic_write
            md_content = f"---\ntitle: \"{title}\"\nsource: \"Wikipedia\"\n---\n\n# {title}\n\n{content}"
            atomic_write(file_path, md_content)
            
            preview = content[:500] + "..." if len(content) > 500 else content
            return f"Successfully downloaded '{title}' to {file_path.relative_to(self.config.wiki_path)}\n\nPreview:\n{preview}\n\nUse file_read with this path to read the full article."
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
                
            files: List[str] = []
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
            from knrs.timelines.extractor import query_timeline_data, format_timeline_as_markdown_table
            timeline_file = self.config.knrs_data / "timelines.json"
            
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

    def file_write(self, path: str, content: str) -> str:
        """Write content to a file in AINotes/Research/."""
        try:
            from knrs.calibre.converter import atomic_write
            
            p = self._sanitize_write_path(path)
                
            if not self._is_safe_write_path(p):
                return f"Error: Cannot write outside {self.research_root}"
                
            atomic_write(p, content)
            return f"Successfully wrote to {p}. If you are finished, output TASK_COMPLETE."
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
            return f"Successfully appended to {p}. If you are finished, output TASK_COMPLETE."
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
                
            # Create destination directory if it doesn't exist
            p_dst.parent.mkdir(parents=True, exist_ok=True)
                
            shutil.move(str(p_src), str(p_dst))
            return f"Successfully moved {src} to {dst}"
        except Exception as e:
            return f"Error moving {src} to {dst}: {e}"

    def dispatch(self, tool_name: str, args: dict[str, Any]) -> str:
        """Execute a tool dynamically."""
        logger.info(f"Agent tool call: {tool_name}({args})")
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
        else:
            return f"Error: Unknown tool {tool_name}"
