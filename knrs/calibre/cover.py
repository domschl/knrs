"""
knrs.calibre.cover — BookCoverIcon generation.

Resizes a Calibre book's cover.jpg to a thumbnail and saves it as
KnrsData/BookCoverIcons/<UUID>.jpg.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

_ICON_HEIGHT = 256   # pixels; width is scaled proportionally
_ICON_QUALITY = 85   # JPEG quality


def generate_cover_icon(
    cover_src: Path,
    icons_dir: Path,
    uuid: str,
    *,
    dry_run: bool = False,
) -> Path | None:
    """
    Resize *cover_src* to a thumbnail and write it to *icons_dir/<uuid>.jpg*.

    Args:
        cover_src:  Source cover.jpg from the Calibre book directory.
        icons_dir:  Target directory (KnrsData/BookCoverIcons/).
        uuid:       Calibre book UUID — used as the output filename.
        dry_run:    If True, log what would happen but do not write anything.

    Returns:
        Path to the written icon, or None if the source does not exist or
        an error occurred.
    """
    if not cover_src.exists():
        logger.debug("No cover at %s — skipping icon generation", cover_src)
        return None

    dest = icons_dir / f"{uuid}.jpg"

    if dry_run:
        logger.info("[dry-run] ICON %s -> %s", cover_src.name, dest)
        return dest

    try:
        icons_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(cover_src) as img:
            img = img.convert("RGB")   # ensure no RGBA/palette modes
            h_pct = _ICON_HEIGHT / img.height
            w_new = max(1, int(img.width * h_pct))
            resized = img.resize((w_new, _ICON_HEIGHT), Image.Resampling.LANCZOS)
            resized.save(dest, format="JPEG", quality=_ICON_QUALITY, optimize=True)
        logger.debug("Icon written: %s", dest)
        return dest
    except Exception as exc:
        logger.error("Failed to generate icon for UUID %s: %s", uuid, exc)
        return None


def icon_is_current(icons_dir: Path, uuid: str) -> bool:
    """Return True if a cover icon already exists for the given UUID."""
    return (icons_dir / f"{uuid}.jpg").exists()
