from __future__ import annotations

import os

# Silence verbose progress bar output from huggingface and tqdm
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYSPROGRESS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

# Also programmatically disable them to be extra safe
try:
    import tqdm
    from functools import partialmethod
    tqdm.tqdm.__init__ = partialmethod(tqdm.tqdm.__init__, disable=True)
except Exception:
    pass

try:
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()
except Exception:
    pass

import sys
import argparse
import signal
import logging
import threading
import hashlib
import json
from typing import Any

# Setup logging (to stderr so stdout stays clean for capabilities)
from rich.logging import RichHandler
from rich.console import Console

logging.basicConfig(
    level=logging.INFO if os.environ.get("KNRS_VERBOSE") == "1" else logging.WARNING,
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
logger = logging.getLogger("summarizer_macos")

# Noise filter for external libraries
class NoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "AFC is enabled" in msg: return False
        if "HTTP Request" in msg and "200 OK" in msg: return False
        return True

for handler in logging.root.handlers:
    handler.addFilter(NoiseFilter())

# Suppress KeyboardInterrupt globally
signal.signal(signal.SIGINT, signal.SIG_DFL)

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from mlx_vlm.speculative.drafters import load_drafter

from summarizer_core.engine import BaseEngine
from summarizer_core.cache import WorkCache
from summarizer_core.markdown import parse_markdown, assemble_markdown
from summarizer_core.summarizer import chunked_summarize
from summarizer_core.utils import get_platform_config, watchdog

# new: https://huggingface.co/mlx-community/gemma-4-31B-it-assistant-bf16

# Constants
VERSION = "0.1.0"
DEFAULT_CONFIG: dict[str, Any] = {
    "chunk_size": 200000,
    "model_id": "mlx-community/gemma-4-31b-it-4bit",
    "model_name": "gemma-4-31b-it-mlx",
    "assistant_id": "mlx-community/gemma-4-31B-it-assistant-bf16",
    "summary_max_tokens": 2500
}

class MLXEngine(BaseEngine):
    def __init__(self, config: dict[str, Any]) -> None:
        model_id: str = config.get("model_id", DEFAULT_CONFIG["model_id"])
        assistant_id: str | None = config.get("assistant_id", DEFAULT_CONFIG.get("assistant_id"))
        logger.info(f"Loading MLX model from {model_id}...")
        self.model, self.processor = load(model_id)
        self.config_data: dict[str, Any] = load_config(model_id)
        self.drafter: Any | None = None
        self.draft_kind: str | None = None
        if assistant_id:
            logger.info(f"Loading MTP assistant from {assistant_id}...")
            self.drafter, self.draft_kind = load_drafter(assistant_id, kind="mtp")

    def format_prompt(self, messages: list[dict[str, str]]) -> str:
        return apply_chat_template(
            self.processor,
            self.config_data,
            messages,
            num_images=0,
            chat_template_kwargs={"enable_thinking": False}
        )

    def generate(self, prompt: str, max_tokens: int = 2500, temp: float = 0.2, repetition_penalty: float = 1.1) -> str:
        gen_kwargs: dict[str, Any] = {
            "max_tokens": max_tokens,
            "temp": temp,
            "repetition_penalty": repetition_penalty,
            "kv_bits": 3.5,
            "kv_quant_scheme": "turboquant",
            "verbose": False
        }
        if self.drafter is not None:
            gen_kwargs["draft_model"] = self.drafter
            gen_kwargs["draft_kind"] = self.draft_kind
            gen_kwargs["draft_block_size"] = 3

        output: Any = generate(
            self.model, self.processor, prompt, [],
            **gen_kwargs
        )
        if hasattr(output, "text"):
            text = str(getattr(output, "text"))
        else:
            text = str(output)
        return text

def summarize_file(source_file: str, destination_file: str, config: dict[str, Any], summary_max_tokens: int) -> None:
    if not os.path.exists(source_file):
        logger.error(f"Error: Source file does not exist: {source_file}")
        sys.exit(1)
        
    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        chunk_size: int = config.get("chunk_size", DEFAULT_CONFIG["chunk_size"])
        model_name: str = config.get("model_name", DEFAULT_CONFIG["model_name"])
        
        metadata, md_text = parse_markdown(content)
        
        source_md_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        
        cache = WorkCache()
        cache.cleanup_old_entries()
        
        logger.info(f"Initializing MLX Engine...")
        engine = MLXEngine(config)
        
        summary_text = chunked_summarize(engine, md_text, source_file, chunk_size, source_md_hash, final_sum_tokens=summary_max_tokens)
        
        sum_metadata: dict[str, Any] = {}
        if metadata:
            for key in ['title', 'authors', 'tags', 'uuid']:
                if key in metadata:
                    sum_metadata[key] = metadata[key]
        
        sum_metadata['summary_version'] = f"{model_name} {VERSION}"
        sum_metadata['source_md_hash'] = source_md_hash
                    
        full_summary = assemble_markdown(sum_metadata, summary_text)
        
        target_dir = os.path.dirname(destination_file)
        if target_dir and not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
            
        temp_file = destination_file + ".tmp"
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(full_summary)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_file, destination_file)
            
        logger.info(f"Successfully wrote summary to: {destination_file}")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Error during summarization: {e}")
        sys.exit(1)

def answer_query(query: str, source_file: str, destination_file: str, config: dict[str, Any], summary_max_tokens: int) -> None:
    if not os.path.exists(source_file):
        logger.error(f"Error: Source file does not exist: {source_file}")
        sys.exit(1)
        
    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        logger.info(f"Initializing MLX Engine for Q&A...")
        engine = MLXEngine(config)
        
        prompt_text = f"Based on the following context, please answer the query: '{query}'.\n\nContext:\n{content}"
        
        prompt: str | list[dict[str, str]] = prompt_text
        if hasattr(engine, 'format_prompt'):
            formatted = engine.format_prompt([{"role": "user", "content": prompt_text}])
            if formatted:
                prompt = formatted
                
        output = engine.generate(prompt, max_tokens=summary_max_tokens)
        
        target_dir = os.path.dirname(destination_file)
        if target_dir and not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
            
        temp_file = destination_file + ".tmp"
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(output)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_file, destination_file)
            
        logger.info(f"Successfully wrote answer to: {destination_file}")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Error during Q&A: {e}")
        sys.exit(1)

def main() -> None:
    w = threading.Thread(target=watchdog, daemon=True)
    w.start()

    parser = argparse.ArgumentParser(description="Summarize Markdown using MLX (macOS)")
    parser.add_argument("source", nargs="?", help="Path to the source markdown file")
    parser.add_argument("destination", nargs="?", help="Path to the destination summary markdown file")
    parser.add_argument("--query", type=str, help="If provided, answer this query based on the source file instead of summarizing it.", default=None)
    parser.add_argument("--summary_max_tokens", type=int, help="Max tokens for the final summary.", default=None)
    parser.add_argument("--capabilities", action="store_true", help="Print backend capabilities as JSON")
    parser.add_argument("--unload", action="store_true", help="Unload active model (no-op for this backend)")
    
    args = parser.parse_args()

    if args.unload:
        sys.exit(0)

    if args.capabilities:
        cap: dict[str, Any] = {
            "name": "summarizer_macos",
            "type": "summarizer",
            "config_file": "summarizer_config_macos.json",
            "platform": "macos",
            "validated_models": [DEFAULT_CONFIG["model_id"]],
            "available_models": [DEFAULT_CONFIG["model_id"]],
            "parameters": {
                "chunk_size":   {"type": "int", "min": 1000, "max": 500000},
                "model_id":     {"type": "str"},
                "model_name":   {"type": "str"},
                "assistant_id": {"type": "str"},
                "summary_max_tokens": {"type": "int", "min": 100, "max": 100000},
            },
        }
        print(json.dumps(cap))
        sys.exit(0)
        
    if not args.source or not args.destination:
        parser.error("source and destination are required unless --capabilities or --unload is passed")
    config = get_platform_config("summarizer_config_macos.json", DEFAULT_CONFIG)
    
    summary_tokens = args.summary_max_tokens if args.summary_max_tokens is not None else config.get("summary_max_tokens", DEFAULT_CONFIG["summary_max_tokens"])
    
    if args.query:
        answer_query(args.query, args.source, args.destination, config, summary_tokens)
    else:
        summarize_file(args.source, args.destination, config, summary_tokens)

if __name__ == "__main__":
    main()
