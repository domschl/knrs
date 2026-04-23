"""
knrs.logging_setup — Structured logging backed by `rich`.

Call `setup_logging()` once at startup. After that, every module uses the
standard `logging.getLogger(__name__)` API; rich handles the presentation.
"""

from __future__ import annotations

import logging

from rich.logging import RichHandler


def setup_logging(verbose: bool = False) -> None:
    """
    Configure the root logger.

    Args:
        verbose: If True, set level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                show_path=False,
                markup=True,
            )
        ],
    )
    # Suppress overly chatty third-party loggers at DEBUG level.
    for noisy in ("PIL", "urllib3", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
