# =========================================================================
# DEVELOPER WARNING: SINGLE SOURCE OF TRUTH (SST) FOR AGENT TOOLS
#
# If you add, modify, or remove any agent tools, you MUST update:
# 1. agent/tools.py (The dynamic dispatch & implementation)
# 2. agent/prompts.py (The text-based instructions for raw LLMs)
# 3. subprocesses/agent_api/agent_api.py (The JSON schema array)
# 4. subprocesses/agent_macos/agent_macos.py (The JSON schema array)
# =========================================================================

from __future__ import annotations


import json
import re
import signal
import sys
import logging
import threading
from typing import Any, Dict, List, TypedDict

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

CONFIG_SCHEMA: Dict[str, str] = {
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

tools: List[Dict[str, Any]] = [
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

class MLXAgentEngine:
    def __init__(self, config: Dict[str, Any]) -> None:
        model_id: str = config.get("model_id", DEFAULT_CONFIG["model_id"])
        self.kv_bits: float | None = config.get("kv_bits", DEFAULT_CONFIG["kv_bits"])
        self.kv_quant_scheme: str | None = config.get("kv_quant_scheme", None)
        if self.kv_quant_scheme == "normal":
            self.kv_quant_scheme = None

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
        parsed_calls: List[Dict[str, Any]] = []

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
                    
                    if p_val.endswith("</parameter>"):
                        p_val = p_val[:-12].strip()
                        
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
                    tool_json = {
                        "tool": name,
                        "args": arguments
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
