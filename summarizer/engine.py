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

from calibre.converter import update_frontmatter_inplace

logger = logging.getLogger(__name__)


def _summarizer_script(summarizer_name: str) -> Path:
    """Return the path to the summarizer script."""
    base = Path(__file__).parent.parent / "subprocesses"
    return base / summarizer_name / f"{summarizer_name}.py"


def _summarizer_python(script: Path) -> str:
    """Return the Python executable for the summarizer subprocess."""
    venv = script.parent / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def summarize_file(
    source_md: Path,
    target_path: Path,
    source_md_hash: str,
    summarizer_name: str,
    *,
    dry_run: bool = False,
    summary_max_tokens: int = 1500,
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
    script = _summarizer_script(summarizer_name)
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
    import os
    env = os.environ.copy()
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        env["KNRS_VERBOSE"] = "1"
    try:
        p = subprocess.Popen(
            [python_exe, str(script), str(source_md), str(target_path), "--summary_max_tokens", str(summary_max_tokens)],
            env=env
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

def answer_query(
    query: str,
    source_md: Path,
    target_path: Path,
    summarizer_name: str,
    summary_max_tokens: int = 1500,
) -> bool:
    """
    Answer a query based on the contents of *source_md* and write the result to *target_path*.

    Args:
        query:           The question to answer.
        source_md:       Markdown file containing context.
        target_path:     Where to write the answer.
        summarizer_name: One of summarizer_linux / summarizer_macos /
                         summarizer_gc_gemma4_31b.

    Returns:
        True on success, False on failure.
    """
    script = _summarizer_script(summarizer_name)
    if not script.exists():
        logger.error("Summarizer script not found: %s", script)
        return False

    target_path.parent.mkdir(parents=True, exist_ok=True)
    python_exe = _summarizer_python(script)

    logger.info("Answering query: %s", query)
    import os
    env = os.environ.copy()
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        env["KNRS_VERBOSE"] = "1"
    try:
        p = subprocess.Popen(
            [python_exe, str(script), str(source_md), str(target_path), "--query", query, "--summary_max_tokens", str(summary_max_tokens)],
            env=env
        )
        try:
            p.wait()
        except BaseException:
            p.kill()
            p.wait()
            raise

        if p.returncode != 0:
            logger.error("Summariser exited %d for query", p.returncode)
            return False

        return True

    except Exception as exc:
        logger.error("Summariser subprocess failed for query: %s", exc)
        return False


def unload_model(summarizer_name: str) -> bool:
    """
    Actively unload the model currently in use.

    Returns:
        True on success, False on failure.
    """
    script = _summarizer_script(summarizer_name)
    if not script.exists():
        logger.error("Summarizer script not found: %s", script)
        return False

    python_exe = _summarizer_python(script)
    logger.info("Requesting unload of summarizer model via %s", summarizer_name)
    import os
    env = os.environ.copy()
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        env["KNRS_VERBOSE"] = "1"
    try:
        p = subprocess.Popen(
            [python_exe, str(script), "--unload"],
            env=env
        )
        p.wait()
        return p.returncode == 0
    except Exception as exc:
        logger.error("Summariser unload subprocess failed: %s", exc)
        return False
