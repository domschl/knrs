"""
knrs.summarizer.sync — Two-phase MarkdownBook → BookSummary synchronisation.

Mirrors the structure of knrs.calibre.sync but with MarkdownBooks as the
source of truth (instead of Calibre) and BookSummaries as the target.
"""

from __future__ import annotations

import logging
import sys
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from naming import (
    check_collisions,
    compute_file_hash,
    compute_markdown_content_hash,
    generate_summary_filename,
)
from summarizer.engine import summarize_file

logger = logging.getLogger(__name__)

SummaryActionType = Literal[
    "ADD", "REMOVE", "RESUMMARISE", "RENAME", "MOVE", "UPDATE_HASH", "UPDATE_METADATA", "SKIP"
]


@dataclass
class SummaryAction:
    action: SummaryActionType
    uuid: str
    title: str
    source_path: Path | None = None
    content_hash: str = ""
    target_path: Path | None = None
    old_path: Path | None = None
    new_path: Path | None = None
    metadata: dict | None = None


# ─── Scanning helpers ─────────────────────────────────────────────────────────

def _read_frontmatter(md_path: Path) -> dict | None:
    """Return YAML frontmatter dict from a Markdown file, or None."""
    try:
        text = md_path.read_text(encoding="utf-8")
    except OSError:
        return None
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n", 4)
    if end == -1:
        return None
    try:
        return yaml.safe_load(text[4:end + 1]) or {}
    except Exception:
        return {}


def scan_existing_summaries(summaries_root: Path) -> dict[str, dict]:
    """Scan BookSummaries dir. Returns {uuid: info_dict}."""
    index: dict[str, dict] = {}
    for md_path in sorted(summaries_root.rglob("*.md")):
        meta = _read_frontmatter(md_path)
        if not meta or not meta.get("uuid"):
            continue
        uuid = meta["uuid"]
        series_rel = str(md_path.parent.relative_to(summaries_root))
        index[uuid] = {
            "path":           md_path,
            "filename":       unicodedata.normalize("NFC", md_path.name),
            "series":         unicodedata.normalize("NFC", series_rel),
            "title":          meta.get("title", ""),
            "authors":        meta.get("authors", []),
            "source_md_hash": meta.get("source_md_hash", ""),
            "summary_version": meta.get("summary_version", ""),
            "metadata":       meta,
        }
    logger.info("Summary scan: found %d existing summaries with UUIDs", len(index))
    return index


def scan_markdown_sources(
    markdown_root: Path, target_series: list[str]
) -> dict[str, dict]:
    """Scan MarkdownBooks dir. Returns {uuid: info_dict}."""
    index: dict[str, dict] = {}
    filter_lower = {s.lower() for s in target_series}

    for md_path in sorted(markdown_root.rglob("*.md")):
        meta = _read_frontmatter(md_path)
        if not meta or not meta.get("uuid"):
            continue
        uuid = meta["uuid"]
        series_rel = str(md_path.parent.relative_to(markdown_root))

        if filter_lower:
            if series_rel.lower() not in filter_lower and series_rel != ".":
                continue

        title   = meta.get("title", "")
        authors = meta.get("authors", [])
        first   = authors[0] if authors else ""

        try:
            content_hash = compute_markdown_content_hash(md_path)
        except OSError:
            continue

        index[uuid] = {
            "path":                     md_path,
            "filename":                 unicodedata.normalize("NFC", md_path.name),
            "series":                   unicodedata.normalize("NFC", series_rel),
            "title":                    title,
            "authors":                  authors,
            "first_author":             first,
            "content_hash":             content_hash,
            "expected_summary_filename": generate_summary_filename(title, first),
            "metadata":                 meta,
        }
    logger.info("Markdown scan: found %d source files (series filter: %s)",
                len(index), ", ".join(filter_lower) if filter_lower else "all")
    return index


# ─── Phase 1: Plan ────────────────────────────────────────────────────────────

def plan_summary_sync(
    markdown_index: dict[str, dict],
    summary_index: dict[str, dict],
    summaries_root: Path,
) -> list[SummaryAction]:
    actions: list[SummaryAction] = []
    md_uuids  = set(markdown_index)
    sum_uuids = set(summary_index)

    # New MarkdownBook, no summary yet → ADD
    for uuid in sorted(md_uuids - sum_uuids):
        mi = markdown_index[uuid]
        target = summaries_root / mi["series"] / mi["expected_summary_filename"]
        actions.append(SummaryAction(
            action="ADD", uuid=uuid, title=mi["title"],
            source_path=mi["path"], content_hash=mi["content_hash"],
            target_path=target,
        ))

    # Summary with no MarkdownBook source → REMOVE
    for uuid in sorted(sum_uuids - md_uuids):
        si = summary_index[uuid]
        actions.append(SummaryAction(
            action="REMOVE", uuid=uuid, title=si["title"], old_path=si["path"],
        ))

    # Both exist → check for changes
    for uuid in sorted(md_uuids & sum_uuids):
        mi = markdown_index[uuid]
        si = summary_index[uuid]

        # Source content changed → RESUMMARISE
        if si["source_md_hash"] and si["source_md_hash"] != mi["content_hash"]:
            target = summaries_root / mi["series"] / mi["expected_summary_filename"]
            actions.append(SummaryAction(
                action="RESUMMARISE", uuid=uuid, title=mi["title"],
                source_path=mi["path"], content_hash=mi["content_hash"],
                target_path=target, old_path=si["path"],
            ))
            continue

        # Filename or series changed → RENAME / MOVE
        expected_fn     = mi["expected_summary_filename"]
        expected_series = mi["series"]
        needs_rename = si["filename"] != expected_fn
        needs_move   = si["series"]   != expected_series
        if needs_rename or needs_move:
            target = summaries_root / expected_series / expected_fn
            actions.append(SummaryAction(
                action="RENAME" if (needs_rename and not needs_move) else "MOVE",
                uuid=uuid, title=mi["title"],
                old_path=si["path"], new_path=target,
            ))
            continue

        # Metadata changed (tags)? → UPDATE_METADATA
        if mi["metadata"].get("tags") != si["metadata"].get("tags"):
            actions.append(SummaryAction(
                action="UPDATE_METADATA", uuid=uuid, title=mi["title"],
                old_path=si["path"], metadata={"tags": mi["metadata"].get("tags")}
            ))
            continue

        # source_md_hash missing → backfill
        if not si["source_md_hash"]:
            actions.append(SummaryAction(
                action="UPDATE_HASH", uuid=uuid, title=mi["title"],
                old_path=si["path"], content_hash=mi["content_hash"],
            ))
            continue

        actions.append(SummaryAction(action="SKIP", uuid=uuid, title=mi["title"]))

    return actions


# ─── Phase 2: Execute ─────────────────────────────────────────────────────────

def _execute_summary_action(
    action: SummaryAction,
    cfg: KnrsConfig,
    *,
    dry_run: bool,
    idx: int,
    total: int,
) -> bool:
    prefix = f"[{idx}/{total}]"

    if action.action == "SKIP":
        return True

    elif action.action in ("ADD", "RESUMMARISE"):
        verb = action.action
        logger.info("%s %s: %s", prefix, verb, action.title)
        if action.action == "RESUMMARISE" and action.old_path and not dry_run:
            action.old_path.unlink(missing_ok=True)
        return summarize_file(
            action.source_path, action.target_path, action.content_hash,
            cfg.summarizer_name, dry_run=dry_run,
        )

    elif action.action == "REMOVE":
        logger.info("%s REMOVE: %s", prefix, action.title)
        if not dry_run and action.old_path:
            action.old_path.unlink(missing_ok=True)
        return True

    elif action.action in ("RENAME", "MOVE"):
        logger.info("%s %s: %s -> %s", prefix, action.action,
                    action.old_path.name if action.old_path else "?",
                    action.new_path.name if action.new_path else "?")
        if not dry_run and action.old_path and action.new_path:
            action.new_path.parent.mkdir(parents=True, exist_ok=True)
            action.old_path.rename(action.new_path)
        return True

    elif action.action == "UPDATE_HASH":
        logger.info("%s UPDATE_HASH: %s", prefix,
                    action.old_path.name if action.old_path else action.title)
        if not dry_run and action.old_path and action.content_hash:
            from calibre.converter import update_frontmatter_inplace
            update_frontmatter_inplace(
                action.old_path, {"source_md_hash": action.content_hash}
            )
        return True

    elif action.action == "UPDATE_METADATA":
        logger.info("%s UPDATE_METADATA: %s", prefix,
                    action.old_path.name if action.old_path else action.title)
        if not dry_run and action.old_path and action.metadata:
            from calibre.converter import update_frontmatter_inplace
            update_frontmatter_inplace(action.old_path, action.metadata)
        return True

    return True


# ─── Orchestrator ─────────────────────────────────────────────────────────────

def run_summary_sync(
    cfg: KnrsConfig,
    *,
    dry_run: bool = False,
    concurrency: int = 1,
) -> None:
    """
    Full two-phase MarkdownBook → BookSummary sync.

    Args:
        cfg:              Resolved KnrsConfig.
        dry_run:          Log the plan but do not write anything.
        concurrency:      Number of parallel summariser workers.
    """
    # ── Phase 1: Scan & Plan ──────────────────────────────────────────
    logger.info("Phase 1: Scanning existing summaries at %s", cfg.book_summaries)
    summary_index = scan_existing_summaries(cfg.book_summaries)

    logger.info("Phase 1: Scanning MarkdownBooks at %s", cfg.markdown_books)
    markdown_index = scan_markdown_sources(cfg.markdown_books, cfg.target_series)

    logger.info("Phase 1: Building summary sync plan…")
    actions = plan_summary_sync(markdown_index, summary_index, cfg.book_summaries)

    counts: dict[str, int] = {}
    for a in actions:
        counts[a.action] = counts.get(a.action, 0) + 1
    for verb in ("ADD", "RESUMMARISE", "RENAME", "MOVE", "UPDATE_HASH", "UPDATE_METADATA", "REMOVE", "SKIP"):
        if counts.get(verb, 0):
            logger.info("  %s: %d", verb, counts[verb])

    executable = [a for a in actions if a.action != "SKIP"]
    if not executable:
        logger.info("Nothing to do — all summaries are up to date.")
        return

    if dry_run:
        logger.info("[dry-run] Would execute %d action(s). No files written.", len(executable))
        for a in executable:
            logger.info("  %s  %s", a.action, a.title)
        return

    # ── Phase 2: Execute ──────────────────────────────────────────────
    logger.info("Phase 2: Executing %d action(s) (concurrency=%d)…",
                len(executable), concurrency)

    sequential = [a for a in executable if a.action not in ("ADD", "RESUMMARISE")]
    parallel   = [a for a in executable if a.action in ("ADD", "RESUMMARISE")]
    total      = len(executable)
    done       = 0

    for a in sequential:
        done += 1
        _execute_summary_action(a, cfg, dry_run=False, idx=done, total=total)

    if parallel:
        if concurrency > 1:
            with ProcessPoolExecutor(max_workers=concurrency) as exe:
                futs = {
                    exe.submit(
                        _execute_summary_action, a, cfg,
                        dry_run=False, idx=done + i + 1, total=total,
                    ): a
                    for i, a in enumerate(parallel)
                }
                for fut in as_completed(futs):
                    exc = fut.exception()
                    if exc:
                        logger.error("Summariser worker error: %s", exc)
                    done += 1
        else:
            for a in parallel:
                done += 1
                _execute_summary_action(a, cfg, dry_run=False, idx=done, total=total)

    logger.info("Summary sync complete.")
