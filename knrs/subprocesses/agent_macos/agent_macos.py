"""
agent_macos — MLX agent backend for Apple Silicon.

Persistent subprocess that loads a model via mlx-vlm once and
serves multi-turn conversation requests via stdin/stdout JSON-line protocol.

Usage:
    python agent_macos.py                     # persistent mode (stdin/stdout)
    python agent_macos.py --capabilities      # print capabilities JSON and exit
"""

import json
import signal
import sys
import logging
import threading

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
    def filter(self, record):
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
from summarizer_core.utils import get_platform_config, watchdog

# Constants
VERSION = "0.1.0"
DEFAULT_CONFIG = {
    "model_id": "mlx-community/gemma-4-26b-a4b-it-4bit",
    "kv_bits": 3.5,
    "kv_quant_scheme": "turboquant",
    "default_max_tokens": 10000,
    "default_temperature": 0.2,
}


class MLXAgentEngine:
    def __init__(self, config: dict):
        model_id = config.get("model_id", DEFAULT_CONFIG["model_id"])
        self.kv_bits = config.get("kv_bits", DEFAULT_CONFIG["kv_bits"])
        self.kv_quant_scheme = config.get("kv_quant_scheme", DEFAULT_CONFIG["kv_quant_scheme"])

        logger.info(f"Loading MLX model from {model_id}...")
        self.model, self.processor = load(model_id)
        self.config = load_config(model_id)
        logger.info("Model loaded successfully.")

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 10000,
        temperature: float = 0.2,
    ) -> str:
        # Normalize role names
        normalized = []
        for m in messages:
            role = m["role"]
            if role == "model":
                role = "assistant"
            normalized.append({"role": role, "content": m["content"]})

        # Apply chat template to the full conversation
        prompt = apply_chat_template(self.processor, self.config, normalized, num_images=0)

        output = generate(
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


def run_persistent(engine: MLXAgentEngine):
    """Main loop: read JSON requests from stdin, write responses to stdout."""
    sys.stdout.write("READY\n")
    sys.stdout.flush()

    while True:
        req = read_request()
        if req is None:
            logger.info("Stdin closed, shutting down.")
            break

        messages = req.get("messages", [])
        max_tokens = req.get("max_tokens", 10000)
        temperature = req.get("temperature", 0.2)

        try:
            text = engine.chat(messages, max_tokens, temperature)
            write_response(text)
        except Exception as e:
            logger.exception(f"Generation error: {e}")
            write_error(str(e))


def main():
    w = threading.Thread(target=watchdog, daemon=True)
    w.start()

    import argparse

    parser = argparse.ArgumentParser(description="Agent backend using MLX (macOS)")
    parser.add_argument("--capabilities", action="store_true", help="Print backend capabilities as JSON")
    args = parser.parse_args()

    if args.capabilities:
        cap = {
            "name": "agent_macos",
            "type": "agent",
            "platform": "macos",
            "validated_models": [DEFAULT_CONFIG["model_id"]],
            "available_models": [DEFAULT_CONFIG["model_id"]],
            "parameters": ["model_id", "kv_bits", "kv_quant_scheme", "default_max_tokens", "default_temperature"],
        }
        print(json.dumps(cap))
        sys.exit(0)

    config = get_platform_config("agent_config_macos.json", DEFAULT_CONFIG)
    engine = MLXAgentEngine(config)
    run_persistent(engine)


if __name__ == "__main__":
    main()
