"""
knrs.vector.engine — Embedding subprocess dispatch.

Two interfaces
--------------
``get_embeddings(texts, config)``
    One-shot call: spawns a fresh subprocess, embeds, exits.
    Kept for backward compatibility / small ad-hoc use (e.g. search queries).

``EmbedderSession(config)``
    Context-manager that keeps a single subprocess alive for the full
    session.  The model is loaded **once** on entry; subsequent ``embed()``
    calls send batches via stdin and receive results without restarting the
    process.  Use this for long indexing runs.

    Example::

        with EmbedderSession(cfg) as session:
            for texts in batches:
                emb = session.embed(texts)
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from types import TracebackType

import numpy as np

from config import KnrsConfig

logger = logging.getLogger(__name__)


# ─── Path helpers ──────────────────────────────────────────────────────────────

def _embedder_script(embedder_name: str) -> Path:
    """Return the path to the embedder script."""
    base = Path(__file__).parent.parent / "subprocesses"
    return base / embedder_name / f"{embedder_name}.py"


def _embedder_python(script: Path) -> str:
    """Return the Python executable for the embedder subprocess."""
    venv = script.parent / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


# ─── One-shot helper (backward compat) ────────────────────────────────────────

def get_embeddings(texts: list[str], config: KnrsConfig, encode_mode: str = "query") -> np.ndarray:
    """
    Compute embeddings for a list of strings using a one-shot subprocess.

    Suitable for small batches (e.g. search queries).  For bulk indexing use
    ``EmbedderSession`` so the model is loaded only once.
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

        logger.debug("Running embedder %s (mode=%s) for %d texts...", embedder_name, encode_mode, len(texts))
        import os
        env = os.environ.copy()
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            env["KNRS_VERBOSE"] = "1"
        try:
            subprocess.run(
                [python_exe, str(script), "--mode", encode_mode, str(input_json), str(output_npy)],
                capture_output=True,
                text=True,
                env=env,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error("Embedder failed with exit code %d", e.returncode)
            if e.stdout:
                logger.error("Embedder stdout:\n%s", e.stdout)
            if e.stderr:
                logger.error("Embedder stderr:\n%s", e.stderr)
            raise RuntimeError(
                f"Embedder subprocess failed with exit code {e.returncode}"
            ) from e

        if not output_npy.exists():
            raise RuntimeError("Embedder did not produce output file.")

        return np.load(str(output_npy))


# ─── Persistent session ────────────────────────────────────────────────────────

class EmbedderSession:
    """
    Persistent embedding subprocess — the model is loaded exactly once.

    Usage::

        with EmbedderSession(cfg) as session:
            embeddings = session.embed(["text one", "text two"])

    The subprocess is shut down cleanly when the context exits.
    """

    def __init__(self, config: KnrsConfig) -> None:
        self.config = config
        self._proc: subprocess.Popen | None = None
        self._tmp:  tempfile.TemporaryDirectory | None = None
        self._input_json: Path | None = None
        self._output_npy: Path | None = None

    # ── Context manager ────────────────────────────────────────────────────

    def __enter__(self) -> "EmbedderSession":
        embedder_name = self.config.embedder_name
        script = _embedder_script(embedder_name)
        if not script.exists():
            raise FileNotFoundError(f"Embedder script not found: {script}")
        python_exe = _embedder_python(script)

        logger.debug(
            "Launching persistent embedder '%s' (model will load once)…",
            embedder_name,
        )
        import os
        env = os.environ.copy()
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            env["KNRS_VERBOSE"] = "1"
        self._proc = subprocess.Popen(
            [python_exe, str(script), "--server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,          # line-buffered text mode
            env=env,
        )

        # Block until the subprocess has finished loading the model.
        ready_line = self._proc.stdout.readline().strip()  # type: ignore[union-attr]
        if ready_line != "READY":
            stderr_content = ""
            if self._proc.stderr:
                try:
                    import select
                    if select.select([self._proc.stderr], [], [], 2.0)[0]:
                        stderr_content = self._proc.stderr.read()
                except Exception:
                    pass
            self._proc.kill()
            raise RuntimeError(
                f"Embedder subprocess did not send READY; got: {ready_line!r}\nStderr:\n{stderr_content}"
            )
        logger.debug("Embedder model loaded and ready.")

        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self._input_json = tmp / "input.json"
        self._output_npy = tmp / "output.npy"
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._proc is not None:
            try:
                self._proc.stdin.close()   # type: ignore[union-attr]
                self._proc.wait(timeout=30)
            except Exception:
                self._proc.kill()
            self._proc = None
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None

    # ── Embedding ──────────────────────────────────────────────────────────

    def embed(self, texts: list[str], encode_mode: str = "document") -> np.ndarray:
        """
        Embed *texts* and return a ``(N, D)`` float32 array.

        The subprocess stays alive; this just writes a batch file, sends the
        paths over stdin, and reads the ``DONE`` acknowledgement.
        """
        if not texts:
            return np.array([], dtype=np.float32)
        if self._proc is None or self._input_json is None:
            raise RuntimeError("EmbedderSession is not active — use as a context manager.")

        with self._input_json.open("w", encoding="utf-8") as f:
            json.dump(texts, f)

        # Send the batch paths to the subprocess.
        self._proc.stdin.write(          # type: ignore[union-attr]
            f"{encode_mode} {self._input_json} {self._output_npy}\n"
        )
        self._proc.stdin.flush()         # type: ignore[union-attr]

        # Block until the subprocess signals completion.
        response = self._proc.stdout.readline().strip()  # type: ignore[union-attr]
        if response.startswith("ERROR"):
            raise RuntimeError(f"Embedder subprocess error: {response}")
        if response != "DONE":
            raise RuntimeError(f"Unexpected embedder response: {response!r}")

        return np.load(str(self._output_npy))
