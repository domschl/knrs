"""
knrs.wiki.assembler — Merge KnrsData into unified Wiki/AINotes pages.

Handles:
- HTML -> Markdown description conversion.
- Merging Book metadata, Covers, and Summaries.
- Wiki-link generation.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

def html_to_md_description(html: str) -> str:
    """
    Convert raw Calibre HTML description to clean Markdown.
    
    Rules:
    - Strip all tags.
    - No blockquote prefixes.
    - Collapse >2 newlines.
    """
    if not html:
        return ""
    
    # Simple tag stripping
    text = re.sub(r'<[^>]+>', '', html)
    
    # Unescape common entities (minimal)
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    
    # Collapse newlines: more than 2 becomes exactly 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def assemble_wiki_page(
    metadata: dict[str, Any],
    summary_md: str,
    cover_icon_path: Path | None = None,
) -> str:
    """
    Assemble the content for an AINotes wiki page.
    
    Args:
        metadata:        Book metadata (from MarkdownBook frontmatter).
        summary_md:     The summary content (excluding its own frontmatter).
        cover_icon_path: Path to the UUID.jpg icon (relative to wiki root).
        
    Returns:
        Full Markdown content for the wiki page.
    """
    title = metadata.get('title', 'Untitled')
    authors = metadata.get('authors', [])
    author_str = ", ".join(authors) if authors else "Unknown Author"
    series = metadata.get('series', '')
    
    # Clean description
    desc_html = metadata.get('description', '')
    description = html_to_md_description(desc_html)
    
    lines = []
    lines.append(f"# {title}")
    lines.append(f"**{author_str}**")
    if series:
        lines.append(f"Series: {series}")
    lines.append("")
    
    if cover_icon_path:
        # Wiki link to the icon
        lines.append(f"![Cover Icon]({cover_icon_path})")
        lines.append("")
        
    if description:
        lines.append("## Description")
        lines.append(description)
        lines.append("")
        
    lines.append("## Summary")
    lines.append(summary_md.strip())
    lines.append("")
    
    lines.append("---")
    lines.append(f"UUID: {metadata.get('uuid', '')}")
    
    return "\n".join(lines)
