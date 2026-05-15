"""
agent_api — OpenAI-compatible API agent backend.

Persistent subprocess that communicates via stdin/stdout JSON-line protocol.
Connects to a local llama-server, vLLM, or any OpenAI-compatible endpoint.

Usage:
    python agent_api.py                     # persistent mode (stdin/stdout)
    python agent_api.py --capabilities      # print capabilities JSON and exit
"""

import json
import signal
import sys
import logging
import threading

import requests

# Setup logging (to stderr so stdout stays clean for protocol)
from rich.logging import RichHandler
from rich.console import Console

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            show_path=False,
            markup=False,
            console=Console(stderr=True),
        )
    ],
)
logger = logging.getLogger("agent_api")

# Suppress KeyboardInterrupt globally
signal.signal(signal.SIGINT, signal.SIG_DFL)

from agent_core.protocol import read_request, write_response, write_error
from typing import TypedDict

from summarizer_core.utils import get_platform_config, get_llm_server_config, watchdog, validate_config

# ── Config schema ──────────────────────────────────────────────────────────────

CONFIG_FILE = "agent_config_api.json"

class AgentApiConfig(TypedDict):
    model_name: str
    default_max_tokens: int
    default_temperature: float

CONFIG_SCHEMA: dict[str, str] = {
    "model_name": "str",
    "default_max_tokens": "int",
    "default_temperature": "float",
}

DEFAULT_LOCAL_CONFIG: AgentApiConfig = {
    "model_name": "Qwen3.6-35B-A3B-UD-Q4_K_XL",
    "default_max_tokens": 10000,
    "default_temperature": 0.2,
}


class ApiAgentEngine:
    def __init__(self, server_cfg: dict, local_cfg: dict):
        self.url = server_cfg["url"].rstrip("/")
        self.api_key = server_cfg.get("api_key")
        self.model = local_cfg["model_name"]
        logger.info(f"Agent API backend: {self.url} (Model: {self.model})")

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 10000,
        temperature: float = 0.2,
    ) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Normalize role names for OpenAI API
        formatted = []
        for m in messages:
            role = m["role"]
            if role == "model":
                role = "assistant"
            formatted.append({"role": role, "content": m["content"]})

        payload = {
            "model": self.model,
            "messages": formatted,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            response = requests.post(
                f"{self.url}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=1800,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.HTTPError as e:
            logger.error(f"API request failed: {e}")
            try:
                logger.error(f"Response: {e.response.text}")
            except Exception:
                pass
            raise
        except Exception as e:
            logger.error(f"API request failed: {e}")
            raise


def run_persistent(engine: ApiAgentEngine):
    """Main loop: read JSON requests from stdin, write responses to stdout."""
    # Signal readiness
    sys.stdout.write("READY\n")
    sys.stdout.flush()

    while True:
        req = read_request()
        if req is None:
            # Parent closed stdin → clean exit
            logger.info("Stdin closed, shutting down.")
            break

        messages = req.get("messages", [])
        max_tokens = req.get("max_tokens", 10000)
        temperature = req.get("temperature", 0.2)

        try:
            text = engine.chat(messages, max_tokens, temperature)
            write_response(text)
        except Exception as e:
            write_error(str(e))


def main():
    w = threading.Thread(target=watchdog, daemon=True)
    w.start()

    import argparse

    parser = argparse.ArgumentParser(description="Agent backend using OpenAI-compatible API")
    parser.add_argument("--capabilities", action="store_true", help="Print backend capabilities as JSON")
    args = parser.parse_args()

    server_config = get_llm_server_config()

    if args.capabilities:
        url = server_config.get("url", "http://localhost:8180").rstrip("/")
        api_key = server_config.get("api_key")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        available_models = []
        try:
            response = requests.get(f"{url}/v1/models", headers=headers, timeout=2)
            response.raise_for_status()
            data = response.json()
            available_models = [m["id"] for m in data.get("data", [])]
        except Exception as e:
            logger.error(f"Failed to query {url}/v1/models: {e}")
            # Don't exit with error — capabilities should still work even if server is down
            # Just report empty available_models

        local_cfg = get_platform_config(CONFIG_FILE, DEFAULT_LOCAL_CONFIG)
        cap = {
            "name": "agent_api",
            "type": "agent",
            "config_file": CONFIG_FILE,
            "platform": "any",
            "validated_models": [DEFAULT_LOCAL_CONFIG["model_name"]],
            "available_models": available_models,
            "parameters": {
                "model_name":          {"type": "str"},
                "default_max_tokens":  {"type": "int",   "min": 100, "max": 128000},
                "default_temperature": {"type": "float", "min": 0.0, "max": 2.0},
            },
        }
        print(json.dumps(cap))
        sys.exit(0)

    local_cfg = get_platform_config(CONFIG_FILE, DEFAULT_LOCAL_CONFIG)
    errors = validate_config(local_cfg, CONFIG_SCHEMA)
    if errors:
        for e in errors:
            logger.error("Config error in %s: %s", CONFIG_FILE, e)
        sys.exit(1)
    engine = ApiAgentEngine(server_config, local_cfg)
    run_persistent(engine)


if __name__ == "__main__":
    main()
