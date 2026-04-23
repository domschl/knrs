"""
knrs.config — Load and validate ~/.config/knrs/knrs.json.

Example config file:

    {
        "calibre_path":    "~/ReferenceLibrary/Calibre Library",
        "notes_path":      "~/Wiki/Notes",
        "knrs_data":       "~/KnrsData",
        "wiki_path":       "~/Wiki",
        "target_series":   [],
        "summarizer_name": "summarizer_linux"
    }

Required keys: calibre_path, notes_path, knrs_data, wiki_path.
Optional keys: target_series (default []), summarizer_name (default "summarizer_linux").
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from knrs.paths import knrs_config_file, resolve

logger = logging.getLogger(__name__)

# Keys that must be present in the config file.
_REQUIRED_KEYS: frozenset[str] = frozenset(
    {"calibre_path", "notes_path", "knrs_data", "wiki_path"}
)


@dataclass
class KnrsConfig:
    """Validated, resolved configuration for a knrs session."""

    calibre_path: Path
    notes_path: Path
    knrs_data: Path
    wiki_path: Path
    target_series: list[str] = field(default_factory=list)
    summarizer_name: str = "summarizer_linux"

    # ------------------------------------------------------------------ #
    # Derived path helpers                                                 #
    # ------------------------------------------------------------------ #

    @property
    def markdown_books(self) -> Path:
        return self.knrs_data / "MarkdownBooks"

    @property
    def book_cover_icons(self) -> Path:
        return self.knrs_data / "BookCoverIcons"

    @property
    def book_summaries(self) -> Path:
        return self.knrs_data / "BookSummaries"

    @property
    def book_library(self) -> Path:
        return self.knrs_data / "BookLibrary"

    @property
    def timelines(self) -> Path:
        return self.knrs_data / "Timelines"

    @property
    def vector_db(self) -> Path:
        return self.knrs_data / "VectorDB"

    @property
    def ai_notes(self) -> Path:
        return self.wiki_path / "AINotes"

    @property
    def ai_notes_books(self) -> Path:
        return self.ai_notes / "Books"


def load_config(config_path: Path | None = None) -> KnrsConfig:
    """
    Load and validate the knrs configuration file.

    Args:
        config_path: Override path; defaults to ~/.config/knrs/knrs.json.

    Returns:
        A validated KnrsConfig instance with all paths resolved.

    Raises:
        FileNotFoundError:  Config file does not exist.
        KeyError:           A required key is missing.
        ValueError:         A value has an unexpected type.
    """
    path = config_path or knrs_config_file()
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            f"Create it with the required keys: {sorted(_REQUIRED_KEYS)}"
        )

    with path.open(encoding="utf-8") as fh:
        raw: dict[str, Any] = json.load(fh)

    # Check required keys.
    missing = _REQUIRED_KEYS - raw.keys()
    if missing:
        raise KeyError(
            f"Config file {path} is missing required key(s): {sorted(missing)}"
        )

    # Validate types for string fields.
    for key in _REQUIRED_KEYS:
        if not isinstance(raw[key], str):
            raise ValueError(
                f"Config key '{key}' must be a string, got {type(raw[key]).__name__!r}"
            )

    target_series = raw.get("target_series", [])
    if not isinstance(target_series, list):
        raise ValueError("Config key 'target_series' must be a list of strings.")

    summarizer_name = raw.get("summarizer_name", "summarizer_linux")
    if not isinstance(summarizer_name, str):
        raise ValueError("Config key 'summarizer_name' must be a string.")

    cfg = KnrsConfig(
        calibre_path=resolve(raw["calibre_path"]),
        notes_path=resolve(raw["notes_path"]),
        knrs_data=resolve(raw["knrs_data"]),
        wiki_path=resolve(raw["wiki_path"]),
        target_series=target_series,
        summarizer_name=summarizer_name,
    )

    logger.debug("Config loaded from %s", path)
    logger.debug("  calibre_path:    %s", cfg.calibre_path)
    logger.debug("  notes_path:      %s", cfg.notes_path)
    logger.debug("  knrs_data:       %s", cfg.knrs_data)
    logger.debug("  wiki_path:       %s", cfg.wiki_path)
    logger.debug("  target_series:   %s", cfg.target_series or "(all)")
    logger.debug("  summarizer_name: %s", cfg.summarizer_name)

    return cfg


def print_config(cfg: KnrsConfig) -> None:
    """Pretty-print the resolved configuration to stdout."""
    from rich import print as rprint
    from rich.panel import Panel
    from rich.table import Table

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold cyan")
    table.add_column("Value", style="white")

    rows = [
        ("calibre_path", str(cfg.calibre_path)),
        ("notes_path", str(cfg.notes_path)),
        ("knrs_data", str(cfg.knrs_data)),
        ("wiki_path", str(cfg.wiki_path)),
        ("target_series", ", ".join(cfg.target_series) or "(all)"),
        ("summarizer_name", cfg.summarizer_name),
    ]
    for key, val in rows:
        table.add_row(key, val)

    rprint(Panel(table, title="[bold]knrs configuration[/bold]", expand=False))
