"""
knrs.agent.engine — Agent subprocess dispatch.

Provides ``AgentSession``, a context manager that keeps a single agent
backend subprocess alive for the full research session.  The model is
loaded **once** on entry; subsequent ``generate()`` calls send the full
conversation history via stdin and receive the response without restarting
the process.

Mirrors the pattern established by ``knrs.vector.engine.EmbedderSession``.

Example::

    with AgentSession(cfg) as session:
        response = session.generate(messages, max_tokens=10000)
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from types import TracebackType

from knrs.config import KnrsConfig

logger = logging.getLogger(__name__)


# ─── Path helpers ──────────────────────────────────────────────────────────────

def _agent_script(agent_name: str) -> Path:
    """Return the path to the agent backend script."""
    base = Path(__file__).parent.parent / "subprocesses"
    return base / agent_name / f"{agent_name}.py"


def _agent_python(script: Path) -> str:
    """Return the Python executable for the agent subprocess."""
    venv = script.parent / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


# ─── Persistent session ────────────────────────────────────────────────────────

class AgentSession:
    """
    Persistent agent subprocess — the model is loaded exactly once.

    Usage::

        with AgentSession(cfg) as session:
            text = session.generate(messages, max_tokens=10000)

    The subprocess is shut down cleanly when the context exits.
    """

    def __init__(self, config: KnrsConfig) -> None:
        self.config = config
        self._proc: subprocess.Popen | None = None

    # ── Context manager ────────────────────────────────────────────────────

    def __enter__(self) -> "AgentSession":
        agent_name = self.config.agent_backend_name
        script = _agent_script(agent_name)
        if not script.exists():
            raise FileNotFoundError(f"Agent script not found: {script}")
        python_exe = _agent_python(script)

        logger.info(
            "Launching persistent agent backend '%s'…",
            agent_name,
        )
        self._proc = subprocess.Popen(
            [python_exe, str(script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,          # line-buffered text mode
        )

        # Block until the subprocess has finished loading the model.
        ready_line = self._proc.stdout.readline().strip()  # type: ignore[union-attr]
        if ready_line != "READY":
            self._proc.kill()
            raise RuntimeError(
                f"Agent subprocess did not send READY; got: {ready_line!r}"
            )
        logger.info("Agent backend loaded and ready.")
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

    # ── Generation ─────────────────────────────────────────────────────────

    def generate(
        self,
        messages: list[dict],
        max_tokens: int = 10000,
        temperature: float = 0.2,
    ) -> str:
        """
        Send the full conversation history to the agent subprocess and
        return the model's response text.

        Raises RuntimeError on protocol errors or subprocess crashes.
        """
        if self._proc is None:
            raise RuntimeError(
                "AgentSession is not active — use as a context manager."
            )

        # Send request as a single JSON line
        request = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        self._proc.stdin.write(json.dumps(request) + "\n")  # type: ignore[union-attr]
        self._proc.stdin.flush()                              # type: ignore[union-attr]

        # Read response (blocks until generation completes)
        response_line = self._proc.stdout.readline()  # type: ignore[union-attr]
        if not response_line:
            raise RuntimeError("Agent subprocess closed stdout unexpectedly")

        response = json.loads(response_line.strip())

        if "error" in response:
            raise RuntimeError(f"Agent backend error: {response['error']}")

        return response["text"]
