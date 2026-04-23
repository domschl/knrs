"""
knrs.summarizer.engine — Summarizer subprocess dispatch.

Each platform summarizer (summarizer_linux, summarizer_macos,
summarizer_gc_gemma4_31b) lives as a self-contained subdirectory with its
own venv under the previous/Summarizer submodule.  This module resolves
the right script and Python executable and fires it as a subprocess,
identical to the pattern in previous/Summarizer/summarizer_sync.py.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from knrs.calibre.converter import update_frontmatter_inplace

logger = logging.getLogger(__name__)


def _summarizer_script(summarizer_root: Path, summarizer_name: str) -> Path:
    """Return the path to the summarizer script."""
    return summarizer_root / summarizer_name / f"{summarizer_name}.py"


def _summarizer_python(script: Path) -> str:
    """Return the Python executable for the summarizer subprocess."""
    venv = script.parent / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def summarize_file(
    source_md: Path,
    target_path: Path,
    source_md_hash: str,
    summarizer_root: Path,
    summarizer_name: str,
    *,
    dry_run: bool = False,
) -> bool:
    """
    Summarise *source_md* and write the result to *target_path*.

    After a successful summarisation the source_md_hash is backfilled into
    the summary's frontmatter so future syncs can detect staleness.

    Args:
        source_md:       MarkdownBook file to summarise.
        target_path:     Where to write the BookSummary.
        source_md_hash:  SHA-256 of source_md at the time of dispatch.
        summarizer_root: Directory containing summarizer sub-directories
                         (typically previous/Summarizer/).
        summarizer_name: One of summarizer_linux / summarizer_macos /
                         summarizer_gc_gemma4_31b.
        dry_run:         Log but do not execute.

    Returns:
        True on success, False on failure.
    """
    script = _summarizer_script(summarizer_root, summarizer_name)
    if not script.exists():
        logger.error("Summarizer script not found: %s", script)
        return False

    if target_path.exists():
        logger.info("Summary already exists, skipping: %s", target_path.name)
        return True

    if dry_run:
        logger.info(
            "[dry-run] SUMMARISE %s -> %s  (via %s)",
            source_md.name, target_path.name, summarizer_name,
        )
        return True

    target_path.parent.mkdir(parents=True, exist_ok=True)
    python_exe = _summarizer_python(script)

    logger.info("Summarising: %s", source_md.name)
    try:
        p = subprocess.Popen(
            [python_exe, str(script), str(source_md), str(target_path)]
        )
        try:
            p.wait()
        except BaseException:
            p.kill()
            p.wait()
            raise

        if p.returncode != 0:
            logger.error(
                "Summariser exited %d for %s", p.returncode, source_md.name
            )
            return False

        # Backfill the source hash so future syncs can detect staleness.
        if target_path.exists() and source_md_hash:
            try:
                update_frontmatter_inplace(
                    target_path, {"source_md_hash": source_md_hash}
                )
            except Exception as exc:
                logger.error("Failed to write source_md_hash: %s", exc)

        logger.info("Summary written: %s", target_path.name)
        return True

    except Exception as exc:
        logger.error("Summariser subprocess failed for %s: %s", source_md.name, exc)
        return False
