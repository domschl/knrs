from __future__ import annotations

import json
import signal
import sys
import logging
import threading
import os
from typing import Any, Dict, List, Optional, TypedDict, Union

# Prevent fragmentation and XPU allocation crashes during model load
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
    def filter(self, record: logging.LogRecord) -> bool:
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
warnings.filterwarnings("ignore", message=".*_check_is_size.*")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Suppress KeyboardInterrupt globally
signal.signal(signal.SIGINT, signal.SIG_DFL)

from agent_core.protocol import read_request, write_response, write_error
from summarizer_core.utils import get_platform_config, watchdog, validate_config

# new: https://ai.google.dev/gemma/docs/mtp/mtp

# ── Config schema ──────────────────────────────────────────────────────────────

CONFIG_FILE = "agent_config_hf.json"

class AgentHfConfig(TypedDict):
    model_id: str
    device: str
    torch_dtype: str
    default_max_tokens: int
    default_temperature: float
    load_in_4bit: bool

CONFIG_SCHEMA: Dict[str, str] = {
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

    def __init__(self, config: Dict[str, Any]) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id: str = config.get("model_id", DEFAULT_CONFIG["model_id"])
        device = self._get_device(config.get("device", DEFAULT_CONFIG["device"]))
        dtype_str: str = config.get("torch_dtype", DEFAULT_CONFIG["torch_dtype"])

        # Resolve torch dtype
        torch_dtype: Union[str, torch.dtype]
        if dtype_str == "auto":
            torch_dtype = "auto"
        elif hasattr(torch, dtype_str) and isinstance(getattr(torch, dtype_str), torch.dtype):
            torch_dtype = getattr(torch, dtype_str)
        else:
            logger.warning(f"Invalid torch_dtype '{dtype_str}' in config. Falling back to 'auto'.")
            torch_dtype = "auto"

        load_in_4bit: bool = config.get("load_in_4bit", DEFAULT_CONFIG["load_in_4bit"])

        quantization_config = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype if isinstance(torch_dtype, torch.dtype) else torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            logger.info("4-bit quantization enabled via bitsandbytes.")

        logger.info(f"Loading model {model_id} (device={device}, dtype={dtype_str})...")

        if device == "xpu" and hasattr(torch, "xpu"):
            # Monkey-patch to avoid transformers/accelerate crashes on Intel Arc
            # Bypass XPU OOM crash during caching allocator warmup
            import transformers.modeling_utils
            transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None

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
        messages: List[Dict[str, str]],
        max_tokens: int = 10000,
        temperature: float = 0.2,
    ) -> str:
        import torch

        # Normalize role names
        normalized: List[Dict[str, str]] = []
        for m in messages:
            role = m["role"]
            if role == "model":
                role = "assistant"
            normalized.append({"role": role, "content": m["content"]})

        # Apply chat template
        input_text: str = self.tokenizer.apply_chat_template(
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
        response: str = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
            
        return response.strip()


def run_persistent(engine: HFAgentEngine) -> None:
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

    parser = argparse.ArgumentParser(description="Agent backend using HuggingFace Transformers")
    parser.add_argument("--capabilities", action="store_true", help="Print backend capabilities as JSON")
    args = parser.parse_args()

    if args.capabilities:
        config = get_platform_config(CONFIG_FILE, DEFAULT_CONFIG)
        cap: Dict[str, Any] = {
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
