from __future__ import annotations

import json
import signal
import sys
import logging
import threading
from typing import Any, Dict, List, Optional, TypedDict

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
logger = logging.getLogger("agent_macos")

# Noise filter for external libraries
class NoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "AFC is enabled" in msg:
            return False
        if "HTTP Request" in msg and "200 OK" in msg:
            return False
        return True

for handler in logging.root.handlers:
    handler.addFilter(NoiseFilter())

# Suppress KeyboardInterrupt globally
signal.signal(signal.SIGINT, signal.SIG_DFL)

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

from agent_core.protocol import read_request, write_response, write_error
from summarizer_core.utils import get_platform_config, watchdog, validate_config

# ── Config schema ──────────────────────────────────────────────────────────────

CONFIG_FILE = "agent_config_macos.json"

class AgentMacosConfig(TypedDict):
    model_id: str
    kv_bits: float
    kv_quant_scheme: str
    default_max_tokens: int
    default_temperature: float

CONFIG_SCHEMA: Dict[str, str] = {
    "model_id": "str",
    "kv_bits": "float",
    "kv_quant_scheme": "str",
    "default_max_tokens": "int",
    "default_temperature": "float",
}

DEFAULT_CONFIG: AgentMacosConfig = {
    "model_id": "mlx-community/gemma-4-26b-a4b-it-4bit",
    "kv_bits": 3.5,
    "kv_quant_scheme": "turboquant",
    "default_max_tokens": 10000,
    "default_temperature": 0.2,
}


class MLXAgentEngine:
    def __init__(self, config: Dict[str, Any]) -> None:
        model_id: str = config.get("model_id", DEFAULT_CONFIG["model_id"])
        self.kv_bits: float = config.get("kv_bits", DEFAULT_CONFIG["kv_bits"])
        self.kv_quant_scheme: str = config.get("kv_quant_scheme", DEFAULT_CONFIG["kv_quant_scheme"])

        logger.info(f"Loading MLX model from {model_id}...")
        self.model, self.processor = load(model_id)
        self.config_data: Dict[str, Any] = load_config(model_id)
        logger.info("Model loaded successfully.")

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 10000,
        temperature: float = 0.2,
    ) -> str:
        # Normalize role names
        normalized: List[Dict[str, str]] = []
        for m in messages:
            role = m["role"]
            if role == "model":
                role = "assistant"
            normalized.append({"role": role, "content": m["content"]})

        # Apply chat template to the full conversation
        prompt: str = apply_chat_template(self.processor, self.config_data, normalized, num_images=0)

        output: Any = generate(
            self.model,
            self.processor,
            prompt,
            [],
            max_tokens=max_tokens,
            temp=temperature,
            repetition_penalty=1.1,
            kv_bits=self.kv_bits,
            kv_quant_scheme=self.kv_quant_scheme,
            verbose=False,
        )

        if hasattr(output, "text"):
            text = str(getattr(output, "text"))
        else:
            text = str(output)
        return text.strip()


def run_persistent(engine: MLXAgentEngine) -> None:
    """Main loop: read JSON requests from stdin, write responses to stdout."""
    sys.stdout.write("READY\n")
    sys.stdout.flush()

    while True:
        req = read_request()
        if req is None:
            logger.info("Stdin closed, shutting down.")
            break

        messages: List[Dict[str, str]] = req.get("messages", [])
        max_tokens: int = req.get("max_tokens", 10000)
        temperature: float = req.get("temperature", 0.2)

        try:
            text = engine.chat(messages, max_tokens, temperature)
            write_response(text)
        except Exception as e:
            logger.exception(f"Generation error: {e}")
            write_error(str(e))


def main() -> None:
    w = threading.Thread(target=watchdog, daemon=True)
    w.start()

    import argparse

    parser = argparse.ArgumentParser(description="Agent backend using MLX (macOS)")
    parser.add_argument("--capabilities", action="store_true", help="Print backend capabilities as JSON")
    args = parser.parse_args()

    if args.capabilities:
        config = get_platform_config(CONFIG_FILE, DEFAULT_CONFIG)
        cap: Dict[str, Any] = {
            "name": "agent_macos",
            "type": "agent",
            "config_file": CONFIG_FILE,
            "platform": "macos",
            "validated_models": [DEFAULT_CONFIG["model_id"]],
            "available_models": [config.get("model_id", DEFAULT_CONFIG["model_id"])],
            "parameters": {
                "model_id":            {"type": "str"},
                "kv_bits":             {"type": "float", "min": 1.0, "max": 8.0},
                "kv_quant_scheme":     {"type": "str"},
                "default_max_tokens":  {"type": "int",   "min": 100, "max": 128000},
                "default_temperature": {"type": "float", "min": 0.0, "max": 2.0},
            },
        }
        print(json.dumps(cap))
        sys.exit(0)

    config = get_platform_config(CONFIG_FILE, DEFAULT_CONFIG)
    errors = validate_config(config, CONFIG_SCHEMA)
    if errors:
        for e in errors:
            logger.error("Config error in %s: %s", CONFIG_FILE, e)
        sys.exit(1)
    engine = MLXAgentEngine(config)
    run_persistent(engine)


if __name__ == "__main__":
    main()
