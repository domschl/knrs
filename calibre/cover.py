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

_ICON_HEIGHT = 512   # pixels; width is scaled proportionally (never upscaled)
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

    Images with height <= _ICON_HEIGHT are converted to RGB JPEG without upscaling.

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
            if img.height > _ICON_HEIGHT:
                h_pct = _ICON_HEIGHT / img.height
                w_new = max(1, int(img.width * h_pct))
                out_img = img.resize((w_new, _ICON_HEIGHT), Image.Resampling.LANCZOS)
            else:
                out_img = img
            out_img.save(dest, format="JPEG", quality=_ICON_QUALITY, optimize=True)
        logger.debug("Icon written: %s", dest)
        return dest
    except Exception as exc:
        logger.error("Failed to generate icon for UUID %s: %s", uuid, exc)
        return None


def icon_is_current(
    icons_dir: Path,
    uuid: str,
    cover_src: Path | None = None,
) -> bool:
    """Return True if a cover icon exists and meets the target resolution."""
    dest = icons_dir / f"{uuid}.jpg"
    if not dest.exists():
        return False

    try:
        with Image.open(dest) as img:
            dest_h = img.height
    except Exception:
        return False

    if dest_h >= _ICON_HEIGHT:
        return True

    if cover_src is not None and cover_src.exists():
        try:
            with Image.open(cover_src) as src_img:
                src_h = src_img.height
            return dest_h >= min(_ICON_HEIGHT, src_h)
        except Exception:
            return True

    return True
