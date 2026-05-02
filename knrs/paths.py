"""
knrs.paths — Central path-resolution helpers.

All paths in knrs are resolved through this module so that ~ expansion,
absolute resolution, and existence checks happen in exactly one place.
"""

from __future__ import annotations

from pathlib import Path
import os
import subprocess


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


def is_git_repo(path: str) -> bool:
    path = os.path.expanduser(path)
    return os.path.exists(os.path.join(path, ".git"))


def is_git_uptodate(path: str, check_remote:bool) -> bool:
    path = os.path.expanduser(path)
    local_changes = subprocess.run(["git", "status", "--porcelain"], cwd=path, capture_output=True, text=True).stdout == ""
    if check_remote:
        _ = subprocess.run(["git", "fetch"], cwd=path, capture_output=True)
        remote_changes = subprocess.run(["git", "status", "--porcelain", "--branch"], cwd=path, capture_output=True, text=True).stdout == ""
        return local_changes and remote_changes
    else:
        return local_changes


def ensure_git_safety(path: str | Path, force: bool = False, check_remote: bool = True) -> bool:
    """
    Check if the path is a git repo and if it's up-to-date.
    Returns True if safe to proceed, False otherwise.
    """
    p_str = str(path)
    if is_git_repo(p_str):
        if not is_git_uptodate(p_str, check_remote=check_remote):
            return force
    return True
