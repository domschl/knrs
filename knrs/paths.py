"""
knrs.paths — Central path-resolution helpers.

All paths in knrs are resolved through this module so that ~ expansion,
absolute resolution, and existence checks happen in exactly one place.
"""

from __future__ import annotations

from pathlib import Path


def resolve(path: str | Path) -> Path:
    """Expand ~ and resolve to an absolute Path."""
    return Path(path).expanduser().resolve()


def resolve_dir(path: str | Path, *, create: bool = False) -> Path:
    """
    Resolve *path* to an absolute directory Path.

    Args:
        path:   The path to resolve (may contain ~).
        create: If True, create the directory (and any parents) when missing.

    Returns:
        Resolved absolute Path.

    Raises:
        NotADirectoryError: If the resolved path exists but is a file.
    """
    p = resolve(path)
    if p.exists() and not p.is_dir():
        raise NotADirectoryError(f"Expected a directory, got a file: {p}")
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def knrs_config_dir() -> Path:
    """Return ~/.config/knrs, creating it if necessary."""
    d = resolve("~/.config/knrs")
    d.mkdir(parents=True, exist_ok=True)
    return d


def knrs_config_file() -> Path:
    """Return the path to the main config file ~/.config/knrs/knrs.json."""
    return knrs_config_dir() / "knrs.json"


def knrs_history_file() -> Path:
    """Return the path to the REPL history file ~/.config/knrs/history."""
    return knrs_config_dir() / "history"
