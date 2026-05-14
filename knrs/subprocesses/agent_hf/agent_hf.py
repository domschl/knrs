"""
agent_hf — HuggingFace Transformers agent backend.

Persistent subprocess that loads a generative/chat model once and
serves multi-turn conversation requests via stdin/stdout JSON-line protocol.

Usage:
    python agent_hf.py                     # persistent mode (stdin/stdout)
    python agent_hf.py --capabilities      # print capabilities JSON and exit
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
logger = logging.getLogger("agent_hf")

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

from agent_core.protocol import read_request, write_response, write_error
from summarizer_core.utils import get_platform_config, watchdog

# Constants
VERSION = "0.1.0"
DEFAULT_CONFIG = {
    "model_id": "Qwen/Qwen3-32B-AWQ",
    "device": "auto",
    "torch_dtype": "auto",
    "default_max_tokens": 10000,
    "default_temperature": 0.2,
}


class HFAgentEngine:
    def __init__(self, config: dict):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = config.get("model_id", DEFAULT_CONFIG["model_id"])
        device = config.get("device", DEFAULT_CONFIG["device"])
        dtype_str = config.get("torch_dtype", DEFAULT_CONFIG["torch_dtype"])

        # Resolve torch dtype
        if dtype_str == "auto":
            torch_dtype = "auto"
        elif hasattr(torch, dtype_str):
            torch_dtype = getattr(torch, dtype_str)
        else:
            torch_dtype = "auto"

        logger.info(f"Loading model {model_id} (device={device}, dtype={dtype_str})...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device,
        )
        logger.info("Model loaded successfully.")

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 10000,
        temperature: float = 0.2,
    ) -> str:
        import torch

        # Normalize role names
        normalized = []
        for m in messages:
            role = m["role"]
            if role == "model":
                role = "assistant"
            normalized.append({"role": role, "content": m["content"]})

        # Apply chat template
        input_text = self.tokenizer.apply_chat_template(
            normalized,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),  # Avoid zero temperature
                do_sample=temperature > 0,
            )

        # Decode only the new tokens (skip the input)
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()


def run_persistent(engine: HFAgentEngine):
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

    parser = argparse.ArgumentParser(description="Agent backend using HuggingFace Transformers")
    parser.add_argument("--capabilities", action="store_true", help="Print backend capabilities as JSON")
    args = parser.parse_args()

    if args.capabilities:
        config = get_platform_config("agent_config_hf.json", DEFAULT_CONFIG)
        cap = {
            "name": "agent_hf",
            "type": "agent",
            "platform": "any",
            "validated_models": [DEFAULT_CONFIG["model_id"]],
            "available_models": [config.get("model_id", DEFAULT_CONFIG["model_id"])],
            "parameters": ["model_id", "device", "torch_dtype", "default_max_tokens", "default_temperature"],
        }
        print(json.dumps(cap))
        sys.exit(0)

    config = get_platform_config("agent_config_hf.json", DEFAULT_CONFIG)
    engine = HFAgentEngine(config)
    run_persistent(engine)


if __name__ == "__main__":
    main()
