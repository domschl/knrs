# To add or modify a tool, edit:
#   1. subprocesses/agent_core/agent_core/tool_registry.py
#   2. agent/tools.py  (Python implementation + AgentTools.dispatch())

from __future__ import annotations


import json
import re
import signal
import sys
import logging
import threading
from typing import Any, TypedDict

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
    kv_bits: float | None
    kv_quant_scheme: str | None
    default_max_tokens: int
    default_temperature: float

CONFIG_SCHEMA: dict[str, str] = {
    "model_id": "str",
    "kv_bits": "float?",
    "kv_quant_scheme": "str?",
    "default_max_tokens": "int?",
    "default_temperature": "float?",
}

DEFAULT_CONFIG: AgentMacosConfig = {
    "model_id": "mlx-community/gemma-4-26b-a4b-it-4bit",
    "kv_bits": 3.5,
    "kv_quant_scheme": "turboquant",
    "default_max_tokens": 10000,
    "default_temperature": 0.2,
}

DEFAULT_CONFIG_ALT: AgentMacosConfig = {
    "model_id": "mlx-community/Qwen3.6-35B-A3B-4bit",
    "kv_bits": 4.0,
    "kv_quant_scheme": None,
    "default_max_tokens": 10000,
    "default_temperature": 0.2,
}

from agent_core.tool_registry import get_json_schemas as _get_json_schemas, coerce_tool_args

# Tool schemas are generated from the canonical registry.
# To add or modify tools, edit tool_registry.py.
tools: list[dict[str, Any]] = _get_json_schemas()


def parse_gemma_args(args_str: str) -> dict[str, Any] | str:
    """Parse Gemma-style tool call arguments that may contain <|"|> string delimiters and bare keys."""
    args_str = args_str.strip()
    if not (args_str.startswith("{") and args_str.endswith("}")):
        return args_str
    
    content = args_str[1:-1].strip()
    if not content:
        return {}
        
    pattern = r"(\w+)\s*:\s*(<\|\"\|>.*?<\|\"\|>|\[.*?\]|[^,]+)"
    matches = re.findall(pattern, content, re.DOTALL)
    
    parsed: dict[str, Any] = {}
    for key, val in matches:
        val = val.strip()
        if val.startswith("<|\"|>") and val.endswith("<|\"|>"):
            parsed[key] = val[5:-5]
        elif val.startswith("[") and val.endswith("]"):
            elements_str = val[1:-1]
            elem_pattern = r"<\|\"\|>(.*?)<\|\"\|>"
            elems = re.findall(elem_pattern, elements_str)
            if not elems:
                elems = [e.strip() for e in elements_str.split(",") if e.strip()]
            parsed[key] = elems
        else:
            val_lower = val.lower()
            if val_lower == "true":
                parsed[key] = True
            elif val_lower == "false":
                parsed[key] = False
            elif val_lower == "null":
                parsed[key] = None
            else:
                try:
                    if "." in val:
                        parsed[key] = float(val)
                    else:
                        parsed[key] = int(val)
                except ValueError:
                    parsed[key] = val
    return parsed


class MLXAgentEngine:
    def __init__(self, config: dict[str, Any]) -> None:
        model_id: str = config.get("model_id", DEFAULT_CONFIG["model_id"])
        self.kv_bits: float | None = config.get("kv_bits", DEFAULT_CONFIG["kv_bits"])
        self.kv_quant_scheme: str | None = config.get("kv_quant_scheme", None)
        if self.kv_quant_scheme == "normal":
            self.kv_quant_scheme = None

        logger.info(f"Loading MLX model from {model_id}...")
        self.model, self.processor = load(model_id)
        self.config_data: dict[str, Any] = load_config(model_id)
        logger.info("Model loaded successfully.")

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 10000,
        temperature: float = 0.2,
    ) -> str:
        # Normalize role names
        normalized: list[dict[str, str]] = []
        for m in messages:
            role = m["role"]
            if role == "model":
                role = "assistant"
            normalized.append({"role": role, "content": m["content"]})

        # Apply chat template to the full conversation with tools
        prompt: str = apply_chat_template(
            self.processor,
            self.config_data,
            normalized,
            num_images=0,
            tools=tools,
        )

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
        text = text.strip()

        # Check for native tool calls
        parsed_calls: list[dict[str, Any]] = []

        # 1. Parse Qwen-style XML tool calls: <tool_call> <function=...> <parameter=...> ... </tool_call>
        if "<tool_call>" in text:
            qwen_matches = re.finditer(r'<tool_call>\s*<function=(\w+)>\s*(.*?)\s*</tool_call>', text, re.DOTALL)
            for qwen_match in qwen_matches:
                name = qwen_match.group(1)
                param_str = qwen_match.group(2)
                args = {}
                param_matches = re.finditer(r'<parameter=(\w+)>\s*(.*?)(?=\s*<parameter=|\s*$)', param_str, re.DOTALL)
                for pm in param_matches:
                    p_name = pm.group(1)
                    p_val = pm.group(2).strip()
                    
                    while True:
                        stripped = re.sub(r'</\w+>\s*$', '', p_val).strip()
                        if stripped == p_val:
                            break
                        p_val = stripped
                        
                    try:
                        args[p_name] = json.loads(p_val)
                    except Exception:
                        try:
                            if "." in p_val:
                                args[p_name] = float(p_val)
                            else:
                                args[p_name] = int(p_val)
                        except Exception:
                            args[p_name] = p_val
                parsed_calls.append({
                    "name": name,
                    "arguments": args
                })

        # 2. Parse Gemma-style tool calls: <|tool_call|>call:(\w+)({...})<tool_call|>
        if "<|tool_call|>" in text:
            gemma_matches = re.finditer(r'<\|tool_call\|\>call:(\w+)(\{.*?\})(?:<tool_call\|>|<\|tool_call\|>)', text, re.DOTALL)
            for g_match in gemma_matches:
                name = g_match.group(1)
                args_str = g_match.group(2)
                try:
                    args = json.loads(args_str)
                except Exception:
                    try:
                        args = parse_gemma_args(args_str)
                        if not isinstance(args, dict):
                            raise ValueError("Not a dictionary")
                    except Exception:
                        try:
                            import yaml
                            args = yaml.safe_load(args_str)
                        except Exception:
                            args = args_str
                
                if isinstance(args, dict):
                    parsed_calls.append({
                        "name": name,
                        "arguments": args
                    })

        # 3. Fallback to tokenizer's tool parser if available and we haven't parsed anything yet
        if not parsed_calls:
            tokenizer = getattr(self.processor, "tokenizer", self.processor)
            if hasattr(tokenizer, "tool_call_start") and hasattr(tokenizer, "tool_parser"):
                tool_start_tok = tokenizer.tool_call_start
                tool_end_tok = getattr(tokenizer, "tool_call_end", None)

                if tool_start_tok in text:
                    try:
                        start_idx = text.find(tool_start_tok) + len(tool_start_tok)
                        if tool_end_tok and tool_end_tok in text:
                            end_idx = text.find(tool_end_tok)
                            raw_tool = text[start_idx:end_idx].strip()
                        else:
                            raw_tool = text[start_idx:].strip()

                        parsed = tokenizer.tool_parser(raw_tool)
                        if parsed:
                            if isinstance(parsed, dict):
                                parsed = [parsed]
                            for item in parsed:
                                name = item.get("name")
                                arguments = item.get("arguments") or {}
                                if name:
                                    parsed_calls.append({
                                        "name": name,
                                        "arguments": arguments
                                    })
                    except Exception as e:
                        logger.warning(f"Tokenizer tool parser failed: {e}")

        # Format parsed calls into standard JSON blocks and append to text
        if parsed_calls:
            formatted_blocks = []
            for call in parsed_calls:
                name = call.get("name")
                arguments = call.get("arguments") or {}
                if name:
                    coerced_args = coerce_tool_args(name, arguments)
                    tool_json = {
                        "tool": name,
                        "args": coerced_args
                    }
                    formatted_blocks.append(
                        f"\n```json\n{json.dumps(tool_json, indent=2)}\n```"
                    )
            if formatted_blocks:
                text += "\n" + "\n".join(formatted_blocks)

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

        messages: list[dict[str, str]] = req.get("messages", [])
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
        cap: dict[str, Any] = {
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
