"""
knrs.calibre.converter — Document conversion to Markdown.

Dispatches to:
  - Direct copy  : for source_format == "markdown"
  - Pandoc        : for source_format == "epub"
  - Docling (CLI) : for source_format == "pdf" (and docx/pptx/xlsx)

The converter sub-processes live in the same directories as the old
Summarizer project:  converter_linux/  and  converter_macos/ .
knrs resolves them relative to the previous/Summarizer submodule path
that is stored in config.

Atomic writes ensure files are never left in a corrupted state.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from knrs.calibre.library import CalibreBook

logger = logging.getLogger(__name__)

CONVERTER_VERSION = "0.1"


# ─── Converter version string ─────────────────────────────────────────────────

def _pandoc_version() -> str:
    try:
        result = subprocess.run(
            ["pandoc", "--version"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.split("\n")[0].split()[-1]
    except Exception:
        pass
    return "unknown"


def _docling_version() -> str:
    # docling is now in a subprocess venv, so we can't get its version directly here.
    return "external"


def get_converter_version(source_format: str) -> str:
    """Build the converter_version metadata string for the given source format."""
    base = f"knrs-calibre-sync {CONVERTER_VERSION}"
    if source_format == "epub":
        return f"pandoc-{_pandoc_version()} {base}"
    elif source_format == "pdf":
        return f"docling-{_docling_version()} {base}"
    return base


# ─── Frontmatter helpers ──────────────────────────────────────────────────────

def _split_frontmatter(text: str) -> tuple[str, str]:
    """Split a Markdown document into (frontmatter_body, content). Returns ('', text) if none."""
    sep = "---\n"
    if not text.startswith(sep):
        return "", text
    end = text.find("\n---\n", len(sep))
    if end == -1:
        return "", text
    return text[len(sep):end + 1], text[end + len("\n---\n"):]


def build_markdown_with_frontmatter(book: CalibreBook, md_body: str) -> str:
    """
    Prepend YAML frontmatter derived from *book* metadata to *md_body*.
    Any pre-existing frontmatter in *md_body* is replaced.
    """
    _, content = _split_frontmatter(md_body)

    meta: dict = {
        "uuid":              book.uuid,
        "title":             book.title,
        "title_sort":        book.title_sort,
        "authors":           book.authors,
        "series":            book.series,
        "tags":              book.tags,
        "languages":         book.languages,
        "identifiers":       book.identifiers,
        "publisher":         book.publisher,
        "publication_date":  book.publication_date,
        "creation_date":     book.creation_date,
        "description":       book.description,
        "source_hash":       book.source_hash,
        "source_format":     book.source_format,
        "converter_version": get_converter_version(book.source_format),
    }

    # Drop empty / null values to keep frontmatter tidy.
    filtered = {
        k: v for k, v in meta.items()
        if v not in (None, "", [], {})
    }

    header = yaml.dump(filtered, default_flow_style=False, allow_unicode=True, indent=2)
    return f"---\n{header}---\n{content}"


def update_frontmatter_inplace(md_path: Path, updates: dict, remove_fields: list[str] | None = None) -> None:
    """Update specific fields in a file's YAML frontmatter without reconverting."""
    text = md_path.read_text(encoding="utf-8")
    fm_raw, content = _split_frontmatter(text)
    try:
        meta = yaml.safe_load(fm_raw) or {}
    except Exception:
        meta = {}
    meta.update({k: v for k, v in updates.items() if v not in (None, "", [], {})})
    if remove_fields:
        for field in remove_fields:
            meta.pop(field, None)
    header = yaml.dump(meta, default_flow_style=False, allow_unicode=True, indent=2)
    atomic_write(md_path, f"---\n{header}---\n{content}")


# ─── Atomic write ─────────────────────────────────────────────────────────────

def atomic_write(dest: Path, content: str | bytes) -> None:
    """Write *content* to *dest* atomically (write-to-tmp then os.replace)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if isinstance(content, str) else "wb"
    enc  = "utf-8" if isinstance(content, str) else None
    fd, tmp = tempfile.mkstemp(
        dir=dest.parent, prefix=f".tmp_{dest.name}_"
    )
    try:
        with os.fdopen(fd, mode, encoding=enc) as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, dest)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


# ─── Conversion dispatch ──────────────────────────────────────────────────────

def _converter_script() -> Path:
    """Resolve the platform-appropriate converter script path."""
    base = Path(__file__).parent.parent / "subprocesses"
    if sys.platform == "darwin":
        script = base / "converter_macos" / "converter_macos.py"
    else:
        script = base / "converter_linux" / "converter_linux.py"
    return script


def _converter_python(script: Path) -> str:
    """Return the Python executable to use for the converter subprocess."""
    venv_python = script.parent / ".venv" / "bin" / "python"
    return str(venv_python) if venv_python.exists() else sys.executable


def convert_book(
    book: CalibreBook,
    target_path: Path,
    *,
    dry_run: bool = False,
) -> bool:
    """
    Convert *book* to Markdown and write to *target_path*.

    Args:
        book:                  The CalibreBook to convert.
        target_path:           Where to write the resulting .md file.
        summarizer_submodule:  Path to previous/Summarizer (for converter scripts).
        dry_run:               Log actions without writing anything.

    Returns:
        True on success, False on failure.
    """
    if dry_run:
        logger.info(
            "[dry-run] CONVERT [%s] %s -> %s",
            book.source_format, book.source_file.name, target_path,
        )
        return True

    if book.source_format == "markdown":
        logger.info("Markdown source (copy+frontmatter): %s", book.source_file.name)
        try:
            md_body = book.source_file.read_text(encoding="utf-8")
            final   = build_markdown_with_frontmatter(book, md_body)
            atomic_write(target_path, final)
            return True
        except Exception as exc:
            logger.error("Failed to process markdown source %s: %s", book.source_file, exc)
            return False

    # epub / pdf  → subprocess converter
    script = _converter_script()
    if not script.exists():
        logger.error("Converter script not found: %s", script)
        return False

    python_exe = _converter_python(script)
    tmp_target  = target_path.with_suffix(".conversion.tmp")

    logger.info(
        "Converting [%s] %s -> %s", book.source_format, book.source_file.name, target_path
    )
    try:
        p = subprocess.Popen(
            [python_exe, str(script), str(book.source_file), str(tmp_target)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            stdout, stderr = p.communicate()
        except BaseException:
            p.kill()
            p.wait()
            raise

        if p.returncode != 0:
            logger.error("Converter failed for %s:\n%s", book.source_file, stderr)
            if tmp_target.exists():
                tmp_target.unlink()
            return False

        md_body = tmp_target.read_text(encoding="utf-8")
        tmp_target.unlink()
        final = build_markdown_with_frontmatter(book, md_body)
        atomic_write(target_path, final)
        logger.info("Converted: %s", target_path.name)
        return True

    except Exception as exc:
        logger.error("Conversion subprocess failed for %s: %s", book.source_file, exc)
        if tmp_target.exists():
            tmp_target.unlink()
        return False
