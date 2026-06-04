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
from typing import Any, Type

from config import KnrsConfig

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

    _active_session: AgentSession | None = None
    _ref_count: int = 0

    def __init__(self, config: KnrsConfig) -> None:
        self.config: KnrsConfig = config
        self._proc: subprocess.Popen[str] | None = None

    # ── Context manager ────────────────────────────────────────────────────

    def __enter__(self) -> AgentSession:
        if AgentSession._active_session is not None:
            active_name = AgentSession._active_session.config.agent_backend_name
            current_name = self.config.agent_backend_name
            if active_name != current_name:
                raise RuntimeError(
                    f"Cannot start agent backend '{current_name}': "
                    f"backend '{active_name}' is already running and only "
                    f"one agent backend can be active at a time."
                )
            AgentSession._ref_count += 1
            self._proc = AgentSession._active_session._proc
            logger.info(
                "Reusing active persistent agent backend '%s' (ref_count: %d).",
                current_name,
                AgentSession._ref_count,
            )
            return self

        agent_name = self.config.agent_backend_name
        script = _agent_script(agent_name)
        if not script.exists():
            raise FileNotFoundError(f"Agent script not found: {script}")
        python_exe = _agent_python(script)

        logger.info(
            "Launching persistent agent backend '%s'…",
            agent_name,
        )
        import os
        env = os.environ.copy()
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            env["KNRS_VERBOSE"] = "1"
        self._proc = subprocess.Popen(
            [python_exe, str(script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,          # line-buffered text mode
            env=env,
        )

        # Block until the subprocess has finished loading the model.
        if self._proc.stdout is None:
             raise RuntimeError("Subprocess stdout is None")
             
        ready_line = self._proc.stdout.readline().strip()
        if ready_line != "READY":
            self._proc.kill()
            raise RuntimeError(
                f"Agent subprocess did not send READY; got: {ready_line!r}"
            )
        logger.info("Agent backend loaded and ready.")
        AgentSession._active_session = self
        AgentSession._ref_count = 1
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._proc is None:
            return
        AgentSession._ref_count -= 1
        if AgentSession._ref_count > 0:
            logger.info(
                "Releasing active persistent agent backend reference (remaining: %d).",
                AgentSession._ref_count,
            )
            self._proc = None
            return

        proc_to_clean = self._proc or (
            AgentSession._active_session._proc
            if AgentSession._active_session
            else None
        )
        AgentSession._active_session = None
        AgentSession._ref_count = 0
        self._proc = None

        if proc_to_clean is not None:
            try:
                if proc_to_clean.stdin is not None:
                    proc_to_clean.stdin.close()
                proc_to_clean.wait(timeout=30)
            except Exception:
                proc_to_clean.kill()

    # ── Generation ─────────────────────────────────────────────────────────

    def generate(
        self,
        messages: list[dict[str, str]],
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

        if self._proc.stdin is None:
            raise RuntimeError("Subprocess stdin is None")
        if self._proc.stdout is None:
            raise RuntimeError("Subprocess stdout is None")

        # Send request as a single JSON line
        request = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        self._proc.stdin.write(json.dumps(request) + "\n")
        self._proc.stdin.flush()

        # Read response (blocks until generation completes)
        response_line = self._proc.stdout.readline()
        if not response_line:
            raise RuntimeError("Agent subprocess closed stdout unexpectedly")

        response: dict[str, Any] = json.loads(response_line.strip())

        if "error" in response:
            raise RuntimeError(f"Agent backend error: {response['error']}")

        return response.get("text", "")
