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

import yaml
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

def html_to_md_description(html_text: str) -> str:
    """
    Convert raw Calibre HTML description to Markdown.
    """
    if not html_text:
        return ""
        
    md_tokens = [
        ("<h3>", "### "),
        ("<h4>", "#### "),
        ("</h1>", "\n\n"),
        ("</h2>", "\n\n"),
        ("</h3>", "\n\n"),
        ("</h4>", "\n\n"),
        ("<em>", " *"),
        ("</em>", "* "),
        ("<strong>", "**"),
        ("</strong>", "** "),
        ("<p>", ""),
        ("</p>", "\n\n"),
        ("<br>", "\n\n"),
        ("<br/>", "\n\n"),
        ("<br />", "\n\n"),
        ("<li>", "- "),
        ("</li>", "\n"),
        ("  ", " "),
        ("  ", " "),
    ]
    for token in md_tokens:
        html_text = html_text.replace(token[0], token[1])
        
    try:
        text = BeautifulSoup(html_text, features="lxml").get_text()
    except Exception:
        text = BeautifulSoup(html_text, features="html.parser").get_text()
        
    # Remove whitespace (including non-breaking spaces) from lines that contain only whitespace
    text = re.sub(r'^[^\S\r\n]+$', '', text, flags=re.MULTILINE)
    # Collapse 3 or more newlines into exactly 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip() + "\n"

def assemble_wiki_page(
    metadata: dict[str, Any],
    summary_md: str,
    cover_icon_path: Path | None = None,
    calibre_hex_name: str | None = None,
) -> str:
    """
    Assemble the content for an AINotes wiki page.
    
    Args:
        metadata:        Book metadata (from MarkdownBook frontmatter).
        summary_md:     The summary content (excluding its own frontmatter).
        cover_icon_path: Path to the UUID.jpg icon (relative to wiki root).
        calibre_hex_name: The hex-encoded calibre library name for calibre:// links.
        
    Returns:
        Full Markdown content for the wiki page.
    """
    
    lines = []
    
    # 1. YAML frontmatter containing the calibre metadata information
    # Exclude 'description' as it is appended in the text
    fm_metadata = metadata.copy()
    fm_metadata.pop('description', None)
    fm = yaml.dump(fm_metadata, default_flow_style=False, allow_unicode=True)
    lines.append("---")
    lines.append(fm.strip())
    lines.append("---")
    lines.append("")
    
    title = metadata.get('title', 'Untitled')
    lines.append(f"## {title}")
    lines.append("")
    
    authors = metadata.get('authors', [])
    author_str = ", ".join(authors) if authors else "Unknown Author"
    author_label = "Author" if len(authors) == 1 else "Authors"
    lines.append(f"{author_label}: {author_str}")
    lines.append("")
    
    if cover_icon_path:
        lines.append(f"![]({cover_icon_path})")
        lines.append("")
        
    identifiers = metadata.get('identifiers', [])
    calibre_id = None
    if isinstance(identifiers, dict):
        calibre_id = identifiers.get('calibre_id') or identifiers.get('calibre')
    elif isinstance(identifiers, list):
        for id_item in identifiers:
            if isinstance(id_item, str) and id_item.startswith('calibre_id/'):
                calibre_id = id_item.split('/', 1)[1]
                break
            elif isinstance(id_item, dict):
                calibre_id = id_item.get('calibre_id') or id_item.get('calibre')
                if calibre_id:
                    break
    
    if calibre_id and calibre_hex_name:
        lines.append(f"[Calibre link](calibre://show-book/_hex_-{calibre_hex_name}/{calibre_id})")
        lines.append("")
    
    # 2.a Header '## Summary' followed by the book summary
    lines.append("## Summary")
    lines.append(summary_md.strip())
    lines.append("")
    
    # 2.b '## Description' followed by calibre metadata dc:description text
    desc_html = metadata.get('description', '')
    description = html_to_md_description(desc_html)
    if description:
        lines.append("## Description")
        lines.append(description.strip())
        lines.append("")
        
    return "\n".join(lines)
