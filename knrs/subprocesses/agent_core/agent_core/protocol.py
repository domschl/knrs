"""
agent_core.protocol — JSON-line IPC helpers for persistent agent subprocesses.

The parent process (knrs agent loop) communicates with the subprocess via
stdin/stdout using one JSON object per line:

    Request  (parent → subprocess stdin):
        {"messages": [...], "max_tokens": 10000, "temperature": 0.2}

    Response (subprocess → parent stdout):
        {"text": "..."}

    Error    (subprocess → parent stdout):
        {"error": "description"}
"""

import json
import sys
from typing import TextIO


def read_request(stream: TextIO = None) -> dict | None:
    """Read one JSON-line request from stdin (or given stream).

    Returns None on EOF (parent closed stdin → time to exit).
    """
    if stream is None:
        stream = sys.stdin
    line = stream.readline()
    if not line:
        return None
    return json.loads(line.strip())


def write_response(text: str, stream: TextIO = None) -> None:
    """Write a successful response to stdout (or given stream)."""
    if stream is None:
        stream = sys.stdout
    stream.write(json.dumps({"text": text}) + "\n")
    stream.flush()


def write_error(message: str, stream: TextIO = None) -> None:
    """Write an error response to stdout (or given stream)."""
    if stream is None:
        stream = sys.stdout
    stream.write(json.dumps({"error": message}) + "\n")
    stream.flush()


def send_request(
    messages: list[dict],
    max_tokens: int = 10000,
    temperature: float = 0.2,
    stream: TextIO = None,
) -> None:
    """Send a request to a subprocess's stdin (parent-side helper)."""
    if stream is None:
        stream = sys.stdout
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    stream.write(json.dumps(payload) + "\n")
    stream.flush()


def read_response(stream: TextIO = None) -> dict:
    """Read one JSON-line response from a subprocess's stdout (parent-side helper).

    Returns a dict with either {"text": "..."} or {"error": "..."}.
    Raises RuntimeError on unexpected EOF.
    """
    if stream is None:
        stream = sys.stdin
    line = stream.readline()
    if not line:
        raise RuntimeError("Agent subprocess closed stdout unexpectedly")
    return json.loads(line.strip())
