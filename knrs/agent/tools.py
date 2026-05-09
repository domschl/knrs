import json
import logging
import os
from pathlib import Path
from knrs.config import KnrsConfig

logger = logging.getLogger(__name__)

class AgentTools:
    def __init__(self, config: KnrsConfig):
        self.config = config
        self.research_root = self.config.wiki_path / "AINotes" / "Research"
        self.research_root.mkdir(parents=True, exist_ok=True)
        
    def _is_safe_read_path(self, path: Path) -> bool:
        """Check if a path is within the allowed read roots."""
        roots = [
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
        """Resolve a string path (absolute, relative, or prefixed) to a safe Path object."""
        if path_str.startswith("books:"):
            return self.config.markdown_books / path_str.split(":", 1)[1]
        elif path_str.startswith("wiki:"):
            return self.config.wiki_path / path_str.split(":", 1)[1]
            
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
                
            output = []
            for i, r in enumerate(results, 1):
                text = get_context_aware_text(searcher, r)
                title = "Unknown"
                
                # Try to get title
                if r.source_label == "books":
                    file_path = self.config.markdown_books / r.bare_path
                elif r.source_label == "wiki":
                    file_path = self.config.wiki_path / r.bare_path
                else:
                    file_path = None
                    
                if file_path and file_path.exists():
                    try:
                        content = file_path.read_text(encoding="utf-8")
                        fm, _ = _split_frontmatter(content)
                        if fm:
                            meta = yaml.safe_load(fm) or {}
                            title = meta.get("title", title)
                    except Exception:
                        pass
                
                output.append(f"Result {i} (Score: {r.score:.3f})\nSource: {r.path}\nTitle: {title}\nContent snippet:\n{text[:1500]}...")
                
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

    def file_list(self, directory: str) -> str:
        """List files in a directory."""
        try:
            p = self._resolve_read_path(directory)
            if not p or not p.exists():
                return f"Error: Directory not found: {directory}"
                
            if not self._is_safe_read_path(p):
                return f"Error: Path is outside allowed read directories: {directory}"
                
            files = []
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

    def file_write(self, path: str, content: str) -> str:
        """Write content to a file in AINotes/Research/."""
        try:
            from knrs.calibre.converter import atomic_write
            
            # Remove redundant AINotes/Research/ prefix if agent included it
            if path.startswith("AINotes/Research/"):
                path = path[len("AINotes/Research/"):]
            
            p = Path(path)
            if not p.is_absolute():
                p = self.research_root / p
                
            if not self._is_safe_write_path(p):
                return f"Error: Cannot write outside {self.research_root}"
                
            atomic_write(p, content)
            return f"Successfully wrote to {p}. If you are finished, output TASK_COMPLETE."
        except Exception as e:
            return f"Error writing to file {path}: {e}"

    def file_append(self, path: str, content: str) -> str:
        """Append content to a file in AINotes/Research/."""
        try:
            # Remove redundant AINotes/Research/ prefix if agent included it
            if path.startswith("AINotes/Research/"):
                path = path[len("AINotes/Research/"):]
            
            p = Path(path)
            if not p.is_absolute():
                p = self.research_root / p
                
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
            # Remove redundant AINotes/Research/ prefix if agent included it
            if path.startswith("AINotes/Research/"):
                path = path[len("AINotes/Research/"):]
                
            p = Path(path)
            if not p.is_absolute():
                p = self.research_root / p
                
            if not self._is_safe_write_path(p):
                return f"Error: Cannot create directory outside {self.research_root}"
                
            p.mkdir(parents=True, exist_ok=True)
            return f"Successfully created directory {p}"
        except Exception as e:
            return f"Error creating directory {path}: {e}"

    def dispatch(self, tool_name: str, args: dict) -> str:
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
        else:
            return f"Error: Unknown tool {tool_name}"
