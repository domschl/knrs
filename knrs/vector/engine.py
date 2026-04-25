"""
knrs.vector.engine — Embedding subprocess dispatch.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from knrs.config import KnrsConfig

logger = logging.getLogger(__name__)


def _embedder_script(embedder_name: str) -> Path:
    """Return the path to the embedder script."""
    base = Path(__file__).parent.parent / "subprocesses"
    return base / embedder_name / f"{embedder_name}.py"


def _embedder_python(script: Path) -> str:
    """Return the Python executable for the embedder subprocess."""
    venv = script.parent / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def get_embeddings(texts: list[str], config: KnrsConfig) -> np.ndarray:
    """
    Compute embeddings for a list of strings using the configured embedder subprocess.
    """
    if not texts:
        return np.array([], dtype=np.float32)

    embedder_name = config.embedder_name
    script = _embedder_script(embedder_name)
    if not script.exists():
        raise FileNotFoundError(f"Embedder script not found: {script}")

    python_exe = _embedder_python(script)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        input_json = tmp_path / "input.json"
        output_npy = tmp_path / "output.npy"

        with input_json.open("w", encoding="utf-8") as f:
            json.dump(texts, f)

        logger.info("Running embedder %s for %d texts...", embedder_name, len(texts))
        try:
            subprocess.run(
                [python_exe, str(script), str(input_json), str(output_npy)],
                check=True
            )
        except subprocess.CalledProcessError as e:
            logger.error("Embedder failed with exit code %d", e.returncode)
            raise RuntimeError(f"Embedder subprocess failed with exit code {e.returncode}") from e

        if not output_npy.exists():
            raise RuntimeError("Embedder did not produce output file.")

        embeddings = np.load(output_npy)
        return embeddings
