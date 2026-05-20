# =========================================================================
# DEVELOPER WARNING: SINGLE SOURCE OF TRUTH (SST) FOR AGENT TOOLS
#
# If you add, modify, or remove any agent tools, you MUST update:
# 1. agent/tools.py (The dynamic dispatch & implementation)
# 2. agent/prompts.py (The text-based instructions for raw LLMs)
# 3. subprocesses/agent_api/agent_api.py (The JSON schema array)
# 4. subprocesses/agent_macos/agent_macos.py (The JSON schema array)
# 5. subprocesses/agent_hf/agent_hf.py (The JSON schema array)
# =========================================================================

from __future__ import annotations

import json
import signal
import sys
import logging
import threading
import os
import re
from typing import Any, TypedDict

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


tools: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "vector_search",
            "description": "Search across all indexed files in the local database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search phrase."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of research texts to return (default is 5).",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "file_read",
            "description": "Read lines from a specific file. Useful to read more context around a search result snippet or a wikilink target.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to read. Supports prefixed paths, bare stems, or bracketed links (e.g. 'books:History Of Rome.md', 'History Of Rome', '[[History Of Rome]]')."
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "The starting line number (1-indexed)."
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "The ending line number (use -1 to read to the end)."
                    }
                },
                "required": ["path", "start_line", "end_line"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "file_list",
            "description": "List files in a given directory prefix.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "The directory prefix (e.g. 'wiki:Notes')."
                    }
                },
                "required": ["directory"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "timeline_query",
            "description": "Query the parsed timeline events database. All arguments are optional. Returns a formatted markdown table of events.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_year": {
                        "type": "integer",
                        "description": "Start year filter (negative for BC)."
                    },
                    "end_year": {
                        "type": "integer",
                        "description": "End year filter (negative for BC)."
                    },
                    "context_filters": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of context strings to filter by."
                    },
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of keywords to filter by."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "file_write",
            "description": "Write (overwrite) content to a file in AINotes/Research/.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The relative destination path within AINotes/Research/ (e.g. 'Roman Law/General Principles.md')."
                    },
                    "content": {
                        "type": "string",
                        "description": "The complete markdown file content."
                    }
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "file_append",
            "description": "Append content to an existing file in AINotes/Research/. Use this for large documents to build them section by section.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The relative destination path within AINotes/Research/."
                    },
                    "content": {
                        "type": "string",
                        "description": "The markdown content to append."
                    }
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_directory",
            "description": "Create a subdirectory in AINotes/Research/.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The relative directory path to create within AINotes/Research/ (e.g. 'Roman Law')."
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "file_move",
            "description": "Move or rename a file or directory strictly within AINotes/Research/.",
            "parameters": {
                "type": "object",
                "properties": {
                    "src": {
                        "type": "string",
                        "description": "Source path relative to AINotes/Research/."
                    },
                    "dst": {
                        "type": "string",
                        "description": "Destination path relative to AINotes/Research/."
                    }
                },
                "required": ["src", "dst"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": "Search Wikipedia for an article title. Returns the top 10 matching article titles and a brief snippet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search term query."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia_fetch",
            "description": "Download a full Wikipedia article in plain text and automatically save it to AINotes/Research/Wikipedia/. Returns a preview and the local file path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The exact Wikipedia article title."
                    }
                },
                "required": ["title"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wikilink_search",
            "description": "Search for wiki documents whose title matches a query. Returns stems usable as [[wikilink]] targets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_wiki",
            "description": "Ensure all files in AINotes/Research/ have proper metadata (uuid, context, creation_date). Call this after writing research files.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_index",
            "description": "Run the full vector index update so newly written research becomes searchable. Call this after writing research files and running check_wiki.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_timeline",
            "description": "Extract timeline tables from a research file and merge into the timeline database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The relative path within AINotes/Research/."
                    }
                },
                "required": ["path"]
            }
        }
    }
]

def coerce_tool_args(name: str, args: dict[str, Any]) -> dict[str, Any]:
    # Find the tool in the tools list
    tool_schema = None
    for t in tools:
        if t.get("function", {}).get("name") == name:
            tool_schema = t.get("function", {})
            break
            
    if not tool_schema:
        return args
        
    properties = tool_schema.get("parameters", {}).get("properties", {})
    coerced = {}
    for k, v in args.items():
        prop_type = properties.get(k, {}).get("type")
        if prop_type == "integer":
            try:
                coerced[k] = int(v)
            except Exception:
                coerced[k] = v
        elif prop_type == "number":
            try:
                coerced[k] = float(v)
            except Exception:
                coerced[k] = v
        elif prop_type == "boolean":
            if isinstance(v, str):
                if v.lower() in ("true", "1", "yes"):
                    coerced[k] = True
                elif v.lower() in ("false", "0", "no"):
                    coerced[k] = False
                else:
                    coerced[k] = v
            else:
                coerced[k] = bool(v)
        else:
            coerced[k] = v
    return coerced

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

    def __init__(self, config: dict[str, Any]) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id: str = config.get("model_id", DEFAULT_CONFIG["model_id"])
        device = self._get_device(config.get("device", DEFAULT_CONFIG["device"]))
        dtype_str: str = config.get("torch_dtype", DEFAULT_CONFIG["torch_dtype"])

        # Resolve torch dtype
        torch_dtype: str | torch.dtype
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
        messages: list[dict[str, str]],
        max_tokens: int = 10000,
        temperature: float = 0.2,
    ) -> str:
        import torch

        # Normalize role names
        normalized: list[dict[str, str]] = []
        for m in messages:
            role = m["role"]
            if role == "model":
                role = "assistant"
            normalized.append({"role": role, "content": m["content"]})

        # Apply chat template with tools fallback
        try:
            input_text: str = self.tokenizer.apply_chat_template(
                normalized,
                tokenize=False,
                add_generation_prompt=True,
                tools=tools,
            )
        except Exception as e:
            logger.warning(f"Could not apply chat template with tools: {e}. Falling back to standard template.")
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
        response: str = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        text = response.strip()
        
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
            tokenizer = self.tokenizer
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
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
            
        return text.strip()


def run_persistent(engine: HFAgentEngine) -> None:
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

    parser = argparse.ArgumentParser(description="Agent backend using HuggingFace Transformers")
    parser.add_argument("--capabilities", action="store_true", help="Print backend capabilities as JSON")
    args = parser.parse_args()

    if args.capabilities:
        config = get_platform_config(CONFIG_FILE, DEFAULT_CONFIG)
        cap: dict[str, Any] = {
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
