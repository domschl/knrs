"""
knrs.wiki.checker — Validates and fixes metadata consistency across the Wiki.
"""

from __future__ import annotations

import datetime
import logging
import subprocess
import uuid
from pathlib import Path

import yaml

from calibre.converter import _split_frontmatter, atomic_write
from config import KnrsConfig

logger = logging.getLogger(__name__)

def _get_git_creation_date(filepath: Path, repo_dir: Path) -> str | None:
    """Retrieve file creation date via git log if available."""
    try:
        creation_date = subprocess.check_output(
            args=[
                "git",
                "-C",
                str(repo_dir),
                "--no-pager",
                "log",
                "--follow",
                "--format=%aI",
                "--reverse",
                str(filepath),
            ],
            stderr=subprocess.DEVNULL,
        )
        cr_date = creation_date.decode("utf-8").strip().split("\n")[0]
        if cr_date:
            try:
                # Validate the date format
                dt = datetime.datetime.fromisoformat(cr_date)
                return dt.isoformat()
            except ValueError as e:
                logger.debug("Error parsing date %s for %s: %s", cr_date, filepath, e)
    except Exception:
        pass
    return None

def _get_fs_creation_date(filepath: Path) -> str | None:
    """Fallback to filesystem modification date if git is unavailable."""
    try:
        stat = filepath.stat()
        ts = getattr(stat, "st_birthtime", stat.st_mtime)
        dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
        return dt.isoformat()
    except Exception as e:
        logger.warning("Error getting file modification date %s: %s", filepath, e)
        return None

def _ensure_creation_date(filepath: Path, repo_dir: Path, meta: dict) -> bool:
    changed = False
    if "creation" in meta:
        meta["creation_date"] = meta.pop("creation")
        changed = True

    if not meta.get("creation_date"):
        c_date = _get_git_creation_date(filepath, repo_dir)
        if not c_date:
            c_date = _get_fs_creation_date(filepath)
        if not c_date:
            c_date = datetime.datetime.now(datetime.timezone.utc).isoformat()
            
        meta["creation_date"] = c_date
        changed = True

    return changed

def ensure_minimal_frontmatter(md_path: Path, wiki_path: Path, meta: dict) -> bool:
    """Ensure uuid, context, and creation_date exist in meta. Returns True if changed."""
    needs_update = False
    
    # 1. UUID
    if not meta.get("uuid"):
        meta["uuid"] = str(uuid.uuid4())
        needs_update = True
        
    # 2. Context
    try:
        rel_path = md_path.parent.relative_to(wiki_path)
        context_str = str(rel_path).replace("\\", "/") # Normalize slashes
        if context_str == ".":
            context_str = ""
    except ValueError:
        context_str = ""
        
    if meta.get("context") != context_str:
        meta["context"] = context_str
        needs_update = True
        
    # 3. Creation Date
    if _ensure_creation_date(md_path, wiki_path, meta):
        needs_update = True
        
    return needs_update

def run_wiki_check(cfg: KnrsConfig, *, dry_run: bool = False, fix_broken_links: bool = False) -> None:
    """
    Check and enforce metadata consistency across the Wiki.
    Ensures 'creation_date', 'context', and 'uuid' are correctly set.
    Also detects duplicate document names and broken wiki links.
    Can automatically fix broken links by replacing them with italicized text.
    """
    import re
    import unicodedata
    from rich.console import Console
    from rich.table import Table

    logger.info("Starting Wiki Check on %s", cfg.wiki_path)
    
    total_checked = 0
    total_updated = 0
    
    templates_dir = cfg.notes_path / "Templates"
    
    file_index: dict[str, list[Path]] = {}
    extracted_links: dict[Path, set[str]] = {}
    malformed_links: dict[Path, list[str]] = {}
    
    for md_path in cfg.wiki_path.rglob("*.md"):
        if ".stfolder" in md_path.parts or ".git" in md_path.parts:
            continue
            
        try:
            if md_path.is_relative_to(templates_dir):
                continue
        except AttributeError:
            pass # Ignore if older python version
            
        total_checked += 1
        
        # Build index for link checking (NFC normalized, lowercase)
        norm_stem = unicodedata.normalize("NFC", md_path.stem).strip().lower()
        if norm_stem not in file_index:
            file_index[norm_stem] = []
        file_index[norm_stem].append(md_path)
        
        try:
            content = md_path.read_text(encoding="utf-8")
        except OSError as e:
            logger.error("Could not read %s: %s", md_path, e)
            continue
            
        fm_raw, body = _split_frontmatter(content)
        
        # Extract links from body: [[Target]] or [[Target|Alias]] or [[Target#Anchor]]
        # We capture the target in group 1, and ensure the link is properly closed with ]]
        raw_links = [m.group(1) for m in re.finditer(r"\[\[([^|\]#]+)[^\]]*\]\]", body)]
        # Normalize extracted links
        extracted_links[md_path] = {
            unicodedata.normalize("NFC", lnk).strip().lower() 
            for lnk in raw_links if lnk.strip()
        }
        
        # Check for malformed links (e.g. missing closing brackets)
        for match in re.finditer(r"\[\[(?:(?!\[\[).)*", body):
            link_text = match.group(0).strip()
            # If it does not contain a fully closed [[...]], it's malformed
            if not re.search(r"\[\[.*?\]\]", link_text):
                if "\n" in link_text:
                    link_text = link_text.split("\n")[0] # Only take the first line to keep it clean
                if md_path not in malformed_links:
                    malformed_links[md_path] = []
                malformed_links[md_path].append(link_text[:80])
        
        try:
            meta = yaml.safe_load(fm_raw) if fm_raw else {}
        except Exception as e:
            logger.error("Failed parsing YAML in %s: %s", md_path, e)
            meta = {}
            
        if not isinstance(meta, dict):
            meta = {}
            
        needs_update = ensure_minimal_frontmatter(md_path, cfg.wiki_path, meta)
        
        if needs_update:
            if dry_run:
                logger.info("[dry-run] Would update metadata in %s", md_path)
            else:
                new_fm = yaml.dump(meta, default_flow_style=False, allow_unicode=True, indent=2)
                new_content = f"---\n{new_fm}---\n{body}"
                atomic_write(md_path, new_content)
                logger.debug("Updated metadata in %s", md_path)
            total_updated += 1
            
    if dry_run:
        logger.info("[dry-run] Wiki Check Complete. Checked %d files. Would update %d files.", total_checked, total_updated)
    else:
        logger.info("Wiki Check Complete. Checked %d files. Updated %d files.", total_checked, total_updated)

    console = Console()
    from rich.markup import escape
    
    # Check for duplicates
    duplicate_table = Table(title="Duplicate Document Names (Ambiguous Links)")
    duplicate_table.add_column("Document Name", style="cyan")
    duplicate_table.add_column("Paths", style="red")
    
    duplicate_count = 0
    for stem, paths in file_index.items():
        if len(paths) > 1:
            duplicate_count += 1
            paths_str = "\n".join(str(p.relative_to(cfg.wiki_path)) for p in paths)
            duplicate_table.add_row(stem, paths_str)
            
    if duplicate_count > 0:
        console.print(duplicate_table)
        
    # Check for broken links
    broken_links_table = Table(title="Broken Wiki Links")
    broken_links_table.add_column("Source Document", style="cyan")
    broken_links_table.add_column("Missing Target Link", style="red")
    
    broken_link_count = 0
    fixed_link_count = 0
    
    # Sort files for deterministic output
    for md_path in sorted(extracted_links.keys()):
        broken_links_in_file = [lnk for lnk in extracted_links[md_path] if lnk not in file_index]
        
        for link in sorted(broken_links_in_file):
            broken_link_count += 1
            broken_links_table.add_row(str(md_path.relative_to(cfg.wiki_path)), escape(f"[[{link}]]"))
            
        if broken_links_in_file and fix_broken_links and not dry_run:
            try:
                content = md_path.read_text(encoding="utf-8")
                
                def replacer(match):
                    nonlocal fixed_link_count
                    target = match.group(1)
                    norm = unicodedata.normalize("NFC", target).strip().lower()
                    if norm not in file_index:
                        fixed_link_count += 1
                        inner = target + match.group(2)
                        return f"_{inner}_"
                    return match.group(0)
                    
                new_content = re.sub(r"\[\[([^|\]#]+)([^\]]*)\]\]", replacer, content)
                if new_content != content:
                    atomic_write(md_path, new_content)
                    logger.debug("Fixed broken links in %s", md_path)
            except Exception as e:
                logger.error("Failed fixing links in %s: %s", md_path, e)
                
    if broken_link_count > 0:
        console.print(broken_links_table)
        
    # Check for malformed links
    malformed_links_table = Table(title="Malformed Wiki Links (Missing Closing Brackets)")
    malformed_links_table.add_column("Source Document", style="cyan")
    malformed_links_table.add_column("Malformed Text", style="red")
    
    malformed_count = 0
    for md_path in sorted(malformed_links.keys()):
        for malformed_text in malformed_links[md_path]:
            malformed_count += 1
            malformed_links_table.add_row(str(md_path.relative_to(cfg.wiki_path)), escape(malformed_text))
            
    if malformed_count > 0:
        console.print(malformed_links_table)
        
    logger.info(
        "Link Check Summary: %d duplicate document names, %d broken links, %d malformed links.", 
        duplicate_count, broken_link_count, malformed_count
    )
    if fix_broken_links and not dry_run:
        logger.info("Fixed %d broken link instances.", fixed_link_count)

    return {
        "checked": total_checked,
        "updated": total_updated,
        "duplicates": duplicate_count,
        "broken_links": broken_link_count,
        "fixed_links": fixed_link_count,
        "malformed_links": malformed_count,
        "errors": 0,
    }

