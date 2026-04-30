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

from knrs.calibre.converter import _split_frontmatter, atomic_write
from knrs.config import KnrsConfig

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

def run_wiki_check(cfg: KnrsConfig, *, dry_run: bool = False) -> None:
    """
    Check and enforce metadata consistency across the Wiki.
    Ensures 'creation_date', 'context', and 'uuid' are correctly set.
    """
    logger.info("Starting Wiki Check on %s", cfg.wiki_path)
    
    total_checked = 0
    total_updated = 0
    
    templates_dir = cfg.notes_path / "Templates"
    
    for md_path in cfg.wiki_path.rglob("*.md"):
        if ".stfolder" in md_path.parts or ".git" in md_path.parts:
            continue
            
        try:
            if md_path.is_relative_to(templates_dir):
                continue
        except AttributeError:
            pass # Ignore if older python version
            
        total_checked += 1
        
        try:
            content = md_path.read_text(encoding="utf-8")
        except OSError as e:
            logger.error("Could not read %s: %s", md_path, e)
            continue
            
        fm_raw, body = _split_frontmatter(content)
        
        try:
            meta = yaml.safe_load(fm_raw) if fm_raw else {}
        except Exception as e:
            logger.error("Failed parsing YAML in %s: %s", md_path, e)
            meta = {}
            
        if not isinstance(meta, dict):
            meta = {}
            
        needs_update = False
        
        # 1. UUID
        if not meta.get("uuid"):
            meta["uuid"] = str(uuid.uuid4())
            needs_update = True
            
        # 2. Context
        try:
            rel_path = md_path.parent.relative_to(cfg.wiki_path)
            context_str = str(rel_path).replace("\\", "/") # Normalize slashes
            if context_str == ".":
                context_str = ""
        except ValueError:
            context_str = ""
            
        if meta.get("context") != context_str:
            meta["context"] = context_str
            needs_update = True
            
        # 3. Creation Date
        if _ensure_creation_date(md_path, cfg.wiki_path, meta):
            needs_update = True
            
        if needs_update:
            if dry_run:
                logger.info("[dry-run] Would update metadata in %s", md_path)
            else:
                new_fm = yaml.dump(meta, default_flow_style=False, allow_unicode=True, indent=2)
                # Drop null values if we want, but simple dump is fine here for created values.
                new_content = f"---\n{new_fm}---\n{body}"
                atomic_write(md_path, new_content)
                logger.debug("Updated metadata in %s", md_path)
            total_updated += 1
            
    if dry_run:
        logger.info("[dry-run] Wiki Check Complete. Checked %d files. Would update %d files.", total_checked, total_updated)
    else:
        logger.info("Wiki Check Complete. Checked %d files. Updated %d files.", total_checked, total_updated)
