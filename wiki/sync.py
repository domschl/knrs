"""
knrs.wiki.sync — Two-phase KnrsData → Wiki/AINotes synchronisation.

Also handles UUID injection for Wiki/Notes documents.
"""

from __future__ import annotations

import logging
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Any

import yaml

from calibre.converter import _split_frontmatter, atomic_write
from config import KnrsConfig
from wiki.assembler import assemble_wiki_page

logger = logging.getLogger(__name__)

@dataclass
class WikiAction:
    action: Literal["ADD", "UPDATE", "REMOVE", "SKIP"]
    uuid: str
    title: str
    target_path: Path
    source_book: Path | None = None
    source_summary: Path | None = None

def inject_frontmatter_in_notes(notes_path: Path, wiki_path: Path) -> int:
    """
    Scan Wiki/Notes and inject uuid, context, and creation_date into frontmatter if missing.
    Returns the number of files updated.
    """
    from wiki.checker import ensure_minimal_frontmatter
    updated_count = 0
    for md_path in notes_path.rglob("*.md"):
        if "Templates" in md_path.parts:
            continue
        try:
            content = md_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error("Could not read %s: %s", md_path, e)
            continue
            
        fm_raw, body = _split_frontmatter(content)
        try:
            meta = yaml.safe_load(fm_raw) if fm_raw else {}
        except Exception:
            meta = {}
            
        if not isinstance(meta, dict):
            meta = {}
        
        if ensure_minimal_frontmatter(md_path, wiki_path, meta):
            new_fm = yaml.dump(meta, default_flow_style=False, allow_unicode=True, indent=2)
            new_content = f"---\n{new_fm}---\n{body}"
            atomic_write(md_path, new_content)
            logger.info("Updated frontmatter in %s", md_path.name)
            updated_count += 1
            
    return updated_count

def update_wikilinks(wiki_path: Path, rename_map: dict[str, str], dry_run: bool = False) -> None:
    """Scan all .md files in wiki_path and update links according to rename_map."""
    import re
    
    if not rename_map:
        return

    # Normalize keys for safe matching
    normalized_map = {
        unicodedata.normalize("NFC", k).strip().lower(): v
        for k, v in rename_map.items()
    }
    
    updated_files = 0
    updated_links = 0
    
    for md_path in wiki_path.rglob("*.md"):
        if ".stfolder" in md_path.parts or ".git" in md_path.parts:
            continue
            
        try:
            content = md_path.read_text(encoding="utf-8")
        except OSError:
            continue
            
        def link_replacer(match):
            nonlocal updated_links
            target = match.group(1)
            rest = match.group(2)
            
            norm_target = unicodedata.normalize("NFC", target).strip().lower()
            if norm_target in normalized_map:
                new_target = normalized_map[norm_target]
                updated_links += 1
                return f"[[{new_target}{rest}]]"
            return match.group(0)
            
        # Match [[Target]] or [[Target|Alias]] or [[Target#Anchor]]
        new_content = re.sub(r"\[\[([^|\]#]+)([^\]]*)\]\]", link_replacer, content)
        
        if new_content != content:
            if not dry_run:
                atomic_write(md_path, new_content)
                logger.debug("Updated wikilinks in %s", md_path)
            updated_files += 1
            
    if dry_run:
        logger.info("[dry-run] Would update %d wikilinks across %d files.", updated_links, updated_files)
    elif updated_files > 0:
        logger.info("Updated %d wikilinks across %d files.", updated_links, updated_files)

def plan_wiki_sync(
    cfg: KnrsConfig,
    markdown_index: dict[str, dict],
    summary_index: dict[str, dict],
    existing_wiki_index: dict[str, Path],
) -> list[WikiAction]:
    """Plan the KnrsData -> Wiki/AINotes sync."""
    actions = []
    
    # We only generate AINotes for books that have BOTH a markdown book and a summary
    ready_uuids = set(markdown_index.keys()) & set(summary_index.keys())
    
    for uid in sorted(ready_uuids):
        mi = markdown_index[uid]
        si = summary_index[uid]
        
        # Target path in Wiki/AINotes/Books/Series/Filename.md
        # Filename is same as MarkdownBook filename
        target = cfg.ai_notes_books / mi['series'] / mi['filename']
        
        if uid not in existing_wiki_index:
            actions.append(WikiAction(
                action="ADD", uuid=uid, title=mi['title'],
                target_path=target, source_book=mi['path'],
                source_summary=si['path']
            ))
        else:
            # Check for staleness: compare mtimes or just always update for now?
            # Implementation plan says "two-phase sync", usually implies change detection.
            # For Wiki, if source_hash in Book or source_md_hash in Summary changed,
            # it should have triggered a RECONVERT/RESUMMARISE.
            # We can check the target's mtime against sources.
            existing_path = existing_wiki_index[uid]
            norm_target = unicodedata.normalize("NFC", str(target))
            norm_existing = unicodedata.normalize("NFC", str(existing_path))
            
            if norm_target != norm_existing:
                # Rename/Move needed
                actions.append(WikiAction(
                    action="UPDATE", uuid=uid, title=mi['title'],
                    target_path=target, source_book=mi['path'],
                    source_summary=si['path']
                ))
            else:
                # Simple update check: use the actual filesystem path for stat()
                # to avoid NFD/NFC mismatch on macOS (target is NFC, filesystem is NFD).
                existing_mtime = existing_path.stat().st_mtime
                if existing_mtime < mi['path'].stat().st_mtime or \
                   existing_mtime < si['path'].stat().st_mtime:
                    actions.append(WikiAction(
                        action="UPDATE", uuid=uid, title=mi['title'],
                        target_path=target, source_book=mi['path'],
                        source_summary=si['path']
                    ))
                else:
                    actions.append(WikiAction(
                        action="SKIP", uuid=uid, title=mi['title'],
                        target_path=target
                    ))
                    
    # Remove orphaned AINotes
    for uid, path in existing_wiki_index.items():
        if uid not in ready_uuids:
            actions.append(WikiAction(
                action="REMOVE", uuid=uid, title="Orphaned", target_path=path
            ))
            
    return actions

def run_wiki_sync(cfg: KnrsConfig, dry_run: bool = False) -> dict[str, Any]:
    """Full KnrsData -> Wiki/AINotes sync orchestrator."""
    from calibre.library import scan_existing_markdowns
    from summarizer.sync import scan_existing_summaries
    
    logger.info("Scanning KnrsData...")
    markdown_index = scan_existing_markdowns(cfg.markdown_books)
    summary_index = scan_existing_summaries(cfg.book_summaries)
    
    logger.info("Scanning existing AINotes...")
    existing_wiki = {}
    for md_path in cfg.ai_notes_books.rglob("*.md"):
        try:
            content = md_path.read_text(encoding="utf-8")
            fm_raw, _ = _split_frontmatter(content)
            meta = yaml.safe_load(fm_raw)
            if isinstance(meta, dict):
                uid = meta.get('uuid')
                if uid: existing_wiki[uid] = md_path
        except Exception:
            continue
            
    actions = plan_wiki_sync(cfg, markdown_index, summary_index, existing_wiki)
    
    # Summary
    counts = {"ADD": 0, "UPDATE": 0, "REMOVE": 0, "SKIP": 0}
    for a in actions: counts[a.action] += 1
    
    logger.info("Wiki Sync Plan: ADD=%d, UPDATE=%d, REMOVE=%d (SKIP=%d)",
                counts['ADD'], counts['UPDATE'], counts['REMOVE'], counts['SKIP'])
    
    executable = [a for a in actions if a.action != "SKIP"]
    if not executable:
        return {
            "success_count": 0,
            "failure_count": 0,
            "total": 0,
            "actions": counts,
            "failed_items": [],
        }

    if dry_run:
        return {
            "success_count": len(executable),
            "failure_count": 0,
            "total": len(executable),
            "actions": counts,
            "failed_items": [],
        }
        
    rename_map: dict[str, str] = {}
    success_count = 0
    failure_count = 0
    failed_items: list[str] = []
        
    for a in actions:
        if a.action == "SKIP": continue
        
        try:
            if a.action == "REMOVE":
                a.target_path.unlink(missing_ok=True)
                logger.info("Removed %s", a.target_path.name)
                success_count += 1
                continue
                
            # ADD or UPDATE
            mi_fm, _ = _split_frontmatter(a.source_book.read_text(encoding="utf-8"))
            si_fm, si_body = _split_frontmatter(a.source_summary.read_text(encoding="utf-8"))
            
            metadata = yaml.safe_load(mi_fm)
            if not isinstance(metadata, dict):
                logger.error("Failed to parse metadata for %s", a.source_book)
                failure_count += 1
                failed_items.append(f"{a.action}: {a.title} (invalid frontmatter)")
                continue
                
            # Copy cover icon to AINotes/Books/Covers and calculate relative path
            src_icon = cfg.book_cover_icons / f"{a.uuid}.jpg"
            if src_icon.exists():
                dst_icon = cfg.ai_notes_books / "Covers" / f"{a.uuid}.jpg"
                dst_icon.parent.mkdir(parents=True, exist_ok=True)
                if not dst_icon.exists() or dst_icon.stat().st_mtime < src_icon.stat().st_mtime:
                    import shutil
                    shutil.copy2(src_icon, dst_icon)
                
                import os
                try:
                    icon_rel = Path(os.path.relpath(dst_icon, a.target_path.parent))
                except ValueError:
                    icon_rel = dst_icon
            else:
                icon_rel = None
                
            calibre_hex_name = "".join([hex(ord(c))[2:] for c in cfg.calibre_library_name])
            content = assemble_wiki_page(metadata, si_body, icon_rel, calibre_hex_name)
            
            # If we moved/renamed, remove old
            if a.action == "UPDATE":
                existing_path = existing_wiki.get(a.uuid)
                if existing_path and existing_path != a.target_path:
                    rename_map[existing_path.stem] = a.target_path.stem
                    existing_path.unlink(missing_ok=True)
                    
            atomic_write(a.target_path, content)
            logger.info("%s %s", a.action, a.target_path.name)
            success_count += 1
        except Exception as exc:
            logger.error("Error executing wiki action %s for %s: %s", a.action, a.title, exc)
            failure_count += 1
            failed_items.append(f"{a.action}: {a.title} ({exc})")
        
    if rename_map and not dry_run:
        update_wikilinks(cfg.wiki_path, rename_map, dry_run=False)

    return {
        "success_count": success_count,
        "failure_count": failure_count,
        "total": len(executable),
        "actions": counts,
        "failed_items": failed_items,
    }
