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
from typing import Any, Dict, List, Optional, TypedDict

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
from summarizer_core.utils import get_platform_config, get_llm_server_config, watchdog, validate_config

# ── Config schema ──────────────────────────────────────────────────────────────

CONFIG_FILE = "agent_config_api.json"

class AgentApiConfig(TypedDict):
    model_name: str
    default_max_tokens: int
    default_temperature: float

CONFIG_SCHEMA: Dict[str, str] = {
    "model_name": "str",
    "default_max_tokens": "int",
    "default_temperature": "float",
}

DEFAULT_LOCAL_CONFIG: AgentApiConfig = {
    "model_name": "Qwen3.6-35B-A3B-UD-Q4_K_XL",
    "default_max_tokens": 10000,
    "default_temperature": 0.2,
}

tools = [
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

class ApiAgentEngine:
    def __init__(self, server_cfg: Dict[str, Any], local_cfg: Dict[str, Any]) -> None:
        self.url: str = server_cfg["url"].rstrip("/")
        self.api_key: Optional[str] = server_cfg.get("api_key")
        self.model: str = local_cfg["model_name"]
        logger.info(f"Agent API backend: {self.url} (Model: {self.model})")

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 10000,
        temperature: float = 0.2,
    ) -> str:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Normalize role names for OpenAI API
        formatted: List[Dict[str, str]] = []
        for m in messages:
            role = m["role"]
            if role == "model":
                role = "assistant"
            formatted.append({"role": role, "content": m["content"]})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": formatted,
            "tools": tools,
            "tool_choice": "auto",
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
            
            message = data["choices"][0]["message"]
            content = message.get("content") or ""
            
            tool_calls = message.get("tool_calls")
            if tool_calls:
                # Format native tool calls into our existing JSON format!
                tool_lines = []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name")
                    args_str = fn.get("arguments") or "{}"
                    try:
                        args = json.loads(args_str)
                    except Exception:
                        args = args_str
                    
                    tool_json = {
                        "tool": name,
                        "args": args
                    }
                    tool_lines.append(json.dumps(tool_json, indent=2))
                
                # Combine content and tool calls
                if content:
                    content += "\n\n" + "\n\n".join(tool_lines)
                else:
                    content = "\n\n".join(tool_lines)
                    
            return content.strip()
        except requests.HTTPError as e:
            logger.error(f"API request failed: {e}")
            try:
                if e.response is not None:
                    logger.error(f"Response: {e.response.text}")
            except Exception:
                pass
            raise
        except Exception as e:
            logger.error(f"API request failed: {e}")
            raise


def run_persistent(engine: ApiAgentEngine) -> None:
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

        messages: List[Dict[str, str]] = req.get("messages", [])
        max_tokens: int = req.get("max_tokens", 10000)
        temperature: float = req.get("temperature", 0.2)

        try:
            text = engine.chat(messages, max_tokens, temperature)
            write_response(text)
        except Exception as e:
            write_error(str(e))


def main() -> None:
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
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        available_models: List[str] = []
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
        cap: Dict[str, Any] = {
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
