"""
knrs.calibre.sync — Two-phase Calibre → MarkdownBooks synchronisation.

Phase 1: Scan both Calibre library and existing MarkdownBooks directory,
         then compute a list of Actions (plan).
Phase 2: Execute the actions (convert, rename, move, delete, update).

Series directory names preserve Calibre metadata casing exactly.
"""

from __future__ import annotations

import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from calibre.converter import (
    atomic_write,
    build_markdown_with_frontmatter,
    convert_book,
    update_frontmatter_inplace,
)
from calibre.cover import generate_cover_icon, icon_is_current
from calibre.library import (
    CalibreBook,
    scan_calibre_library,
    scan_existing_markdowns,
)
from config import KnrsConfig
from naming import check_collisions

logger = logging.getLogger(__name__)

ActionType = Literal[
    "ADD", "REMOVE", "RECONVERT", "RENAME", "MOVE", "UPDATE_METADATA", "SKIP"
]


@dataclass
class SyncAction:
    action: ActionType
    uuid: str
    title: str
    # Fields populated depending on action type:
    source_file: Path | None = None
    source_format: str = ""
    source_hash: str = ""
    target_path: Path | None = None
    old_path: Path | None = None
    new_path: Path | None = None
    opf_path: Path | None = None
    book_dir: Path | None = None
    book: CalibreBook | None = None


# ─── Phase 1: Plan ────────────────────────────────────────────────────────────

def plan_sync(
    calibre_index: dict[str, CalibreBook],
    markdown_index: dict[str, dict],
    markdown_root: Path,
    icon_root: Path,
) -> list[SyncAction]:
    """
    Compare Calibre and existing Markdown state. Return a list of SyncActions.
    """
    actions: list[SyncAction] = []
    calibre_uuids = set(calibre_index)
    markdown_uuids = set(markdown_index)

    # ── Books in Calibre but not yet converted → ADD ──────────────────
    for uuid in sorted(calibre_uuids - markdown_uuids):
        book = calibre_index[uuid]
        target = markdown_root / book.series_dir / book.expected_filename
        actions.append(SyncAction(
            action="ADD", uuid=uuid, title=book.title,
            source_file=book.source_file, source_format=book.source_format,
            source_hash=book.source_hash, target_path=target,
            opf_path=book.opf_path, book_dir=book.book_dir, book=book,
        ))

    # ── Markdowns whose source UUID is gone from Calibre → REMOVE ────
    for uuid in sorted(markdown_uuids - calibre_uuids):
        mi = markdown_index[uuid]
        actions.append(SyncAction(
            action="REMOVE", uuid=uuid, title=mi["title"],
            old_path=mi["path"],
        ))

    # ── Books in both → check for changes ────────────────────────────
    for uuid in sorted(calibre_uuids & markdown_uuids):
        book = calibre_index[uuid]
        mi   = markdown_index[uuid]

        # Source content changed → RECONVERT (takes priority over rename/move)
        if mi["source_hash"] and mi["source_hash"] != book.source_hash:
            target = markdown_root / book.series_dir / book.expected_filename
            actions.append(SyncAction(
                action="RECONVERT", uuid=uuid, title=book.title,
                source_file=book.source_file, source_format=book.source_format,
                source_hash=book.source_hash, target_path=target,
                old_path=mi["path"],
                opf_path=book.opf_path, book_dir=book.book_dir, book=book,
            ))
            continue

        expected_fn     = book.expected_filename
        expected_series = book.series_dir

        needs_rename = mi["filename"] != expected_fn
        needs_move   = mi["series"] != expected_series

        if needs_rename or needs_move:
            target = markdown_root / expected_series / expected_fn
            action_type: ActionType = "RENAME" if (needs_rename and not needs_move) else "MOVE"
            actions.append(SyncAction(
                action=action_type, uuid=uuid, title=book.title,
                old_path=mi["path"], new_path=target, book=book,
            ))
            continue

        # Only metadata changed (tags, description, etc.)
        _METADATA_KEYS = ["tags", "description", "publisher", "publication_date",
                          "title_sort", "series", "authors"]
        meta_changed = any(
            book.__dict__.get(k, "") != mi["metadata"].get(k, "")
            for k in _METADATA_KEYS
        )
        # Check for missing icon or legacy 'icon' field in frontmatter
        icon_path = icon_root / f"{uuid}.jpg"
        has_legacy_icon = "icon" in mi["metadata"]
        
        # Check if context is correct (to fix legacy files)
        expected_context = f"AINotes/Books/{book.series_dir}"
        context_mismatch = mi["metadata"].get("context") != expected_context

        # Also update if source_hash was missing from the frontmatter
        if not mi["source_hash"] or meta_changed or context_mismatch or not icon_path.exists() or has_legacy_icon:
            actions.append(SyncAction(
                action="UPDATE_METADATA", uuid=uuid, title=book.title,
                old_path=mi["path"],
                source_hash=book.source_hash,
                source_format=book.source_format,
                book=book,
            ))
            continue

        actions.append(SyncAction(action="SKIP", uuid=uuid, title=book.title))

    return actions


def _check_planned_collisions(
    actions: list[SyncAction],
    markdown_index: dict[str, dict],
) -> list[dict]:
    """Return collision groups for all planned filenames (empty = no collisions)."""
    entries = []
    for a in actions:
        if a.action == "SKIP":
            mi = markdown_index.get(a.uuid, {})
            fname = mi.get("filename", "")
        elif a.action in ("ADD", "RECONVERT"):
            fname = a.target_path.name if a.target_path else ""
        elif a.action in ("RENAME", "MOVE"):
            fname = a.new_path.name if a.new_path else ""
        elif a.action == "UPDATE_METADATA":
            mi = markdown_index.get(a.uuid, {})
            fname = mi.get("filename", "")
        else:
            continue
        if fname:
            entries.append({"filename": fname, "uuid": a.uuid, "title": a.title})
    return check_collisions(entries)


# ─── Phase 2: Execute ─────────────────────────────────────────────────────────

def _execute_action(
    action: SyncAction,
    cfg: KnrsConfig,
    *,
    dry_run: bool,
    idx: int,
    total: int,
) -> None:
    prefix = f"[{idx}/{total}]"

    if action.action == "SKIP":
        return

    elif action.action in ("ADD", "RECONVERT"):
        verb = "ADD" if action.action == "ADD" else "RECONVERT"
        logger.info("%s %s: %s", prefix, verb, action.title)
        if action.action == "RECONVERT" and action.old_path and not dry_run:
            action.old_path.unlink(missing_ok=True)
        if action.book:
            ok = convert_book(
                action.book, action.target_path, dry_run=dry_run
            )
            if ok and not dry_run and action.book.cover_path:
                generate_cover_icon(
                    action.book.cover_path, cfg.book_cover_icons,
                    action.book.uuid, dry_run=dry_run,
                )

    elif action.action == "REMOVE":
        logger.info("%s REMOVE: %s", prefix, action.title)
        if not dry_run:
            if action.old_path:
                action.old_path.unlink(missing_ok=True)
                logger.debug("Removed %s", action.old_path)
            # Also remove the icon
            icon_path = cfg.book_cover_icons / f"{action.uuid}.jpg"
            if icon_path.exists():
                icon_path.unlink(missing_ok=True)
                logger.debug("Removed icon %s", icon_path)

    elif action.action in ("RENAME", "MOVE"):
        logger.info(
            "%s %s: %s -> %s",
            prefix, action.action,
            action.old_path.name if action.old_path else "?",
            action.new_path.name if action.new_path else "?",
        )
        if not dry_run and action.old_path and action.new_path:
            action.new_path.parent.mkdir(parents=True, exist_ok=True)
            action.old_path.rename(action.new_path)
            if action.book:
                _update_meta_fields(action.new_path, action.book)

    elif action.action == "UPDATE_METADATA":
        logger.info(
            "%s UPDATE_METADATA: %s", prefix,
            action.old_path.name if action.old_path else action.title,
        )
        if not dry_run and action.old_path and action.book:
            _update_meta_fields(action.old_path, action.book,
                                extra={"source_hash": action.source_hash,
                                       "source_format": action.source_format})
            # Ensure icon exists
            icon_path = cfg.book_cover_icons / f"{action.uuid}.jpg"
            if not icon_path.exists() and action.book.cover_path:
                generate_cover_icon(
                    action.book.cover_path, cfg.book_cover_icons,
                    action.book.uuid, dry_run=False
                )


def _update_meta_fields(md_path: Path, book: CalibreBook, extra: dict | None = None) -> None:
    """Write changed Calibre metadata fields back into a MarkdownBook frontmatter."""
    updates = {
        "title":            book.title,
        "title_sort":       book.title_sort,
        "authors":          book.authors,
        "series":           book.series,
        "tags":             book.tags,
        "languages":        book.languages,
        "identifiers":      book.identifiers,
        "publisher":        book.publisher,
        "publication_date": book.publication_date,
        "context":          f"AINotes/Books/{book.series_dir}",
        "description":      book.description,
    }
    if extra:
        updates.update(extra)
    # Always try to remove legacy 'icon' field
    update_frontmatter_inplace(
        md_path, 
        {k: v for k, v in updates.items() if v not in ("", [], None)},
        remove_fields=["icon"]
    )


def _cleanup_icon_debris(icon_root: Path, active_uuids: set[str], dry_run: bool):
    """Remove icons that don't belong to any active book UUID."""
    if not icon_root.exists():
        return
    for icon_path in icon_root.glob("*.jpg"):
        uuid = icon_path.stem
        if uuid not in active_uuids:
            if dry_run:
                logger.info("[dry-run] REMOVE DEBRIS ICON: %s", icon_path.name)
            else:
                logger.info("Removing debris icon: %s", icon_path.name)
                icon_path.unlink(missing_ok=True)


# ─── Orchestrator ─────────────────────────────────────────────────────────────

def run_sync(
    cfg: KnrsConfig,
    *,
    dry_run: bool = False,
    concurrency: int = 1,
) -> None:
    """
    Full two-phase Calibre → MarkdownBooks sync.

    Args:
        cfg:                   Resolved KnrsConfig.
        dry_run:               Print the plan but do not write any files.
        concurrency:           Number of parallel conversion workers.
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

    # ── Phase 1: Scan & Plan ──────────────────────────────────────────
    logger.info("Phase 1: Scanning Calibre library at %s", cfg.calibre_path)
    calibre_index = scan_calibre_library(cfg.calibre_path, cfg.target_series)

    logger.info("Phase 1: Scanning existing MarkdownBooks at %s", cfg.markdown_books)
    markdown_index = scan_existing_markdowns(cfg.markdown_books)

    logger.info("Phase 1: Building sync plan…")
    actions = plan_sync(calibre_index, markdown_index, cfg.markdown_books, cfg.book_cover_icons)

    collisions = _check_planned_collisions(actions, markdown_index)
    if collisions:
        logger.error("ABORTING — filename collision(s) detected:")
        for grp in collisions:
            logger.error("  %s", grp["filename"])
            for e in grp["entries"]:
                logger.error("    UUID=%s  Title=%s", e["uuid"], e["title"])
        logger.error("Fix conflicting titles in Calibre and re-run.")
        sys.exit(1)

    # Summary
    counts: dict[str, int] = {}
    for a in actions:
        counts[a.action] = counts.get(a.action, 0) + 1
    for verb in ("ADD", "RECONVERT", "RENAME", "MOVE", "UPDATE_METADATA", "REMOVE", "SKIP"):
        if counts.get(verb, 0):
            logger.info("  %s: %d", verb, counts[verb])

    executable = [a for a in actions if a.action != "SKIP"]
    if not executable:
        logger.info("Nothing to do — all files are up to date.")
        return

    if dry_run:
        logger.info("[dry-run] Would execute %d action(s). No files written.", len(executable))
        for a in executable:
            logger.info("  %s  %s", a.action, a.title)
        return

    # ── Phase 2: Execute ──────────────────────────────────────────────
    logger.info("Phase 2: Executing %d action(s) (concurrency=%d)…",
                len(executable), concurrency)

    sequential = [a for a in executable if a.action not in ("ADD", "RECONVERT")]
    parallel   = [a for a in executable if a.action in ("ADD", "RECONVERT")]
    total      = len(executable)
    done       = 0

    # Sequential (fast) actions first
    for a in sequential:
        done += 1
        _execute_action(a, cfg, dry_run=False, idx=done, total=total)

    # Parallel conversions
    if parallel:
        if concurrency > 1:
            with ProcessPoolExecutor(max_workers=concurrency) as exe:
                futs = {
                    exe.submit(
                        _execute_action, a, cfg,
                        dry_run=False, idx=done + i + 1, total=total
                    ): a
                    for i, a in enumerate(parallel)
                }
                for fut in as_completed(futs):
                    exc = fut.exception()
                    if exc:
                        logger.error("Worker error: %s", exc)
                    done += 1
        else:
            for a in parallel:
                done += 1
                _execute_action(a, cfg, dry_run=False, idx=done, total=total)

    # Cleanup debris icons
    _cleanup_icon_debris(cfg.book_cover_icons, set(calibre_index.keys()), dry_run=dry_run)

    logger.info("Sync complete.")
