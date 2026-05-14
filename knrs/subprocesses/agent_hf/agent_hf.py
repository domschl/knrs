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
        if "The fast path is not available" in msg:
            return False
        return True

for handler in logging.root.handlers:
    handler.addFilter(NoiseFilter())

import warnings
warnings.filterwarnings("ignore", message=".*The fast path is not available.*")

# Suppress KeyboardInterrupt globally
signal.signal(signal.SIGINT, signal.SIG_DFL)

from agent_core.protocol import read_request, write_response, write_error
from typing import TypedDict

from summarizer_core.utils import get_platform_config, watchdog, validate_config

# ── Config schema ──────────────────────────────────────────────────────────────

CONFIG_FILE = "agent_config_hf.json"

class AgentHfConfig(TypedDict):
    model_id: str
    device: str
    torch_dtype: str
    default_max_tokens: int
    default_temperature: float
    load_in_4bit: bool

CONFIG_SCHEMA: dict[str, str] = {
    "model_id": "str",
    "device": "str",
    "torch_dtype": "str",
    "default_max_tokens": "int",
    "default_temperature": "float",
    "load_in_4bit": "bool",
}

DEFAULT_CONFIG: AgentHfConfig = {
    "model_id": "Qwen/Qwen3-32B-AWQ",
    "device": "auto",
    "torch_dtype": "auto",
    "default_max_tokens": 10000,
    "default_temperature": 0.2,
    "load_in_4bit": False,
}


class HFAgentEngine:
    def _get_device(self, config_device: str = "auto") -> str:
        import torch
        if config_device and config_device != "auto":
            return config_device

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def __init__(self, config: dict):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = config.get("model_id", DEFAULT_CONFIG["model_id"])
        device = self._get_device(config.get("device", DEFAULT_CONFIG["device"]))
        dtype_str = config.get("torch_dtype", DEFAULT_CONFIG["torch_dtype"])

        # Resolve torch dtype
        if dtype_str == "auto":
            torch_dtype = "auto"
        elif hasattr(torch, dtype_str):
            torch_dtype = getattr(torch, dtype_str)
        else:
            torch_dtype = "auto"

        load_in_4bit = config.get("load_in_4bit", DEFAULT_CONFIG["load_in_4bit"])

        quantization_config = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype if torch_dtype != "auto" else torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            logger.info("4-bit quantization enabled via bitsandbytes.")

        logger.info(f"Loading model {model_id} (device={device}, dtype={dtype_str})...")

        if device == "xpu" and hasattr(torch, "xpu"):
            # Monkey-patch mem_get_info to avoid transformers/accelerate crashes on Intel Arc
            def fake_mem_get_info(*args, **kwargs):
                # Return fake 16GB free, 16GB total to satisfy accelerate's caching_allocator_warmup
                return (16 * 1024**3, 16 * 1024**3)
            torch.xpu.mem_get_info = fake_mem_get_info
            if hasattr(torch.xpu, "memory"):
                torch.xpu.memory.mem_get_info = fake_mem_get_info

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device,
            quantization_config=quantization_config,
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
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
            
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
        config = get_platform_config(CONFIG_FILE, DEFAULT_CONFIG)
        cap = {
            "name": "agent_hf",
            "type": "agent",
            "config_file": CONFIG_FILE,
            "platform": "any",
            "validated_models": [DEFAULT_CONFIG["model_id"]],
            "available_models": [config.get("model_id", DEFAULT_CONFIG["model_id"])],
            "parameters": {
                "model_id":            {"type": "str"},
                "device":              {"type": "str"},
                "torch_dtype":         {"type": "str"},
                "default_max_tokens":  {"type": "int",   "min": 100, "max": 128000},
                "default_temperature": {"type": "float", "min": 0.0, "max": 2.0},
                "load_in_4bit":        {"type": "bool"},
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
    engine = HFAgentEngine(config)
    run_persistent(engine)


if __name__ == "__main__":
    main()
