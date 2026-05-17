"""
knrs.calibre.library — Calibre library scanning and metadata parsing.

Reads metadata.opf files (filesystem-only, no Calibre API/DB) and exposes
the result as a CalibreBook dataclass.

Series directory names preserve the casing from Calibre metadata exactly.
The old Summarizer project normalised series to lowercase; knrs does not.
"""

from __future__ import annotations

import logging
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from naming import capitalize_series, compute_file_hash, generate_filename

logger = logging.getLogger(__name__)

_OPF_NS = {
    "opf": "http://www.idpf.org/2007/opf",
    "dc":  "http://purl.org/dc/elements/1.1/",
}

# Priority order for source file selection (highest first).
_FORMAT_PRIORITY: list[tuple[str, str]] = [
    (".md",   "markdown"),
    (".epub", "epub"),
    (".pdf",  "pdf"),
    (".docx", "pdf"),   # treated as pdf-class for docling
    (".pptx", "pdf"),
    (".xlsx", "pdf"),
]


@dataclass
class CalibreBook:
    """All information derived from a single Calibre book directory."""

    uuid: str
    calibre_id: str
    title: str
    title_sort: str
    authors: list[str]
    series: str           # original casing from metadata.opf
    tags: list[str]
    languages: list[str]
    identifiers: list[str]
    publisher: str
    publication_date: str
    creation_date: str
    description: str      # raw HTML from metadata.opf

    book_dir: Path
    opf_path: Path
    cover_path: Path | None   # None if no cover.jpg exists

    source_file: Path
    source_format: str        # "markdown" | "epub" | "pdf"
    source_hash: str          # SHA-256 of source_file

    # Derived
    first_author: str = field(init=False)
    series_dir: str = field(init=False)   # series name used as directory component
    expected_filename: str = field(init=False)

    def __post_init__(self) -> None:
        self.first_author = self.authors[0] if self.authors else ""
        self.series_dir = capitalize_series(self.series) if self.series else "Unspecified"
        self.expected_filename = generate_filename(self.title, self.first_author)


# ─── Internal helpers ──────────────────────────────────────────────────────────

def _parse_date(raw: str) -> str:
    """Parse an ISO 8601 date string from OPF and return a UTC ISO string."""
    if not raw:
        return ""
    try:
        fmt = "%Y-%m-%dT%H:%M:%S.%f%z" if '.' in raw else "%Y-%m-%dT%H:%M:%S%z"
        return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc).isoformat()
    except ValueError:
        return raw


def _find_source_file(book_dir: Path) -> tuple[Path, str] | tuple[None, str]:
    """
    Return the best source file in a Calibre book directory and its format name.
    Priority: markdown > epub > pdf (and docx/pptx/xlsx treated as pdf-class).
    Returns (None, "") if no supported file exists.
    """
    files = {f.suffix.lower(): f for f in book_dir.iterdir() if f.is_file()}
    for ext, fmt in _FORMAT_PRIORITY:
        if ext in files:
            return files[ext], fmt
    return None, ""


def parse_opf(opf_path: Path, calibre_path: Path) -> CalibreBook | None:
    """
    Parse a single metadata.opf file and return a CalibreBook, or None on error.

    Args:
        opf_path:     Absolute path to the metadata.opf file.
        calibre_path: Root of the Calibre library (stored in metadata as context).
    """
    try:
        root = ET.parse(opf_path).getroot()
    except ET.ParseError as exc:
        logger.error("Failed to parse OPF %s: %s", opf_path, exc)
        return None

    xml_meta = root.find("opf:metadata", _OPF_NS)
    if xml_meta is None:
        logger.error("No <metadata> in OPF %s", opf_path)
        return None

    def _text(tag: str) -> str:
        el = xml_meta.find(tag, _OPF_NS)
        return str(el.text).strip() if el is not None and el.text else ""

    title        = _text("dc:title")
    description  = _text("dc:description")   # raw HTML
    publisher    = _text("dc:publisher")
    raw_date     = _text("dc:date")

    # Authors (role=aut)
    authors: list[str] = []
    for creator in xml_meta.findall("dc:creator", _OPF_NS):
        role_attr = "{http://www.idpf.org/2007/opf}role"
        if creator.attrib.get(role_attr) == "aut" and creator.text:
            authors.append(creator.text.strip())

    tags      = [str(s.text).strip() for s in xml_meta.findall("dc:subject",  _OPF_NS) if s.text]
    languages = [str(l.text).strip() for l in xml_meta.findall("dc:language", _OPF_NS) if l.text]

    uuid = calibre_id = title_sort = series = timestamp_raw = ""
    for id_el in xml_meta.findall("dc:identifier", _OPF_NS):
        id_attr = id_el.attrib.get("id", "")
        if id_attr == "uuid_id"    and id_el.text: uuid        = id_el.text.strip()
        if id_attr == "calibre_id" and id_el.text: calibre_id  = id_el.text.strip()

    identifiers: list[str] = []
    for id_el in xml_meta.findall("dc:identifier", _OPF_NS):
        scheme_attr = "{http://www.idpf.org/2007/opf}scheme"
        scheme = id_el.attrib.get(scheme_attr, "")
        val    = (id_el.text or "").strip()
        if scheme and scheme not in ("calibre", "uuid") and val:
            identifiers.append(f"{scheme}/{val}")
    if calibre_id:
        identifiers.append(f"calibre_id/{calibre_id}")

    for meta_el in xml_meta.findall("opf:meta", _OPF_NS):
        name = meta_el.attrib.get("name", "")
        content = meta_el.attrib.get("content", "")
        if name == "calibre:series":       series        = content
        if name == "calibre:title_sort":   title_sort    = content
        if name == "calibre:timestamp":    timestamp_raw = content.split(".")[0]

    book_dir  = opf_path.parent
    cover     = book_dir / "cover.jpg"
    cover_path = cover if cover.exists() else None

    source_file, source_format = _find_source_file(book_dir)
    if source_file is None:
        logger.debug("No supported source file in %s — skipping", book_dir)
        return None

    try:
        source_hash = compute_file_hash(source_file)
    except OSError as exc:
        logger.error("Cannot hash source file %s: %s", source_file, exc)
        return None

    return CalibreBook(
        uuid=uuid,
        calibre_id=calibre_id,
        title=title or "Untitled",
        title_sort=title_sort,
        authors=authors,
        series=series,                     # preserve original casing
        tags=tags,
        languages=languages,
        identifiers=identifiers,
        publisher=publisher,
        publication_date=_parse_date(raw_date),
        creation_date=_parse_date(timestamp_raw) if timestamp_raw else "",
        description=description,
        book_dir=book_dir,
        opf_path=opf_path,
        cover_path=cover_path,
        source_file=source_file,
        source_format=source_format,
        source_hash=source_hash,
    )


def scan_calibre_library(
    calibre_path: Path,
    target_series: list[str],
) -> dict[str, CalibreBook]:
    """
    Walk the Calibre library and return {uuid: CalibreBook} for all books.

    Args:
        calibre_path:  Root of the Calibre library.
        target_series: If non-empty, only include books whose series
                       matches one of these strings (case-insensitive).
                       If empty, include all books.

    Returns:
        Dict mapping UUID strings to CalibreBook instances.
    """
    index: dict[str, CalibreBook] = {}
    filter_lower = {s.lower() for s in target_series}

    for opf_path in sorted(calibre_path.rglob("metadata.opf")):
        # Skip Calibre's internal trash directory.
        if ".caltrash" in opf_path.parts:
            continue

        book = parse_opf(opf_path, calibre_path)
        if book is None:
            continue
        if not book.uuid:
            logger.warning("No UUID in %s — skipping", opf_path)
            continue
        if filter_lower and book.series.lower() not in filter_lower:
            continue

        if book.uuid in index:
            logger.error(
                "Duplicate UUID %s: %s and %s",
                book.uuid, index[book.uuid].book_dir, book.book_dir,
            )
        else:
            index[book.uuid] = book

    logger.info("Calibre scan: found %d books (series filter: %s)",
                len(index), ", ".join(filter_lower) if filter_lower else "all")
    return index


def scan_existing_markdowns(markdown_root: Path) -> dict[str, dict]:
    """
    Scan an existing MarkdownBooks directory. Returns {uuid: info_dict}.

    The info_dict mirrors what the old calibre_sync.py produced so that
    calibre.sync.plan_sync() can be kept straightforward.
    """
    import yaml

    index: dict[str, dict] = {}

    for md_path in sorted(markdown_root.rglob("*.md")):
        try:
            text = md_path.read_text(encoding="utf-8")
        except OSError:
            continue

        # Quick frontmatter split
        if not text.startswith("---\n"):
            continue
        end = text.find("\n---\n", 4)
        if end == -1:
            continue
        fm_raw = text[4:end]
        try:
            meta = yaml.safe_load(fm_raw)
        except Exception:
            continue
        if not isinstance(meta, dict) or not meta.get("uuid"):
            continue

        uuid = meta["uuid"]
        series_rel = md_path.parent.relative_to(markdown_root)
        index[uuid] = {
            "path":        md_path,
            "filename":    unicodedata.normalize("NFC", md_path.name),
            "series":      unicodedata.normalize("NFC", str(series_rel)),
            "title":       meta.get("title", ""),
            "authors":     meta.get("authors", []),
            "source_hash": meta.get("source_hash", ""),
            "metadata":    meta,
        }

    logger.info("Markdown scan: found %d existing files with UUIDs", len(index))
    return index
