from __future__ import annotations

import os
import sys
import argparse
import signal
import logging
import threading
import hashlib
from typing import Any, Dict, List, Optional, Union

# Setup logging
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            show_path=False,
            markup=False,
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

from summarizer_core.engine import BaseEngine
from summarizer_core.cache import WorkCache
from summarizer_core.markdown import parse_markdown, assemble_markdown
from summarizer_core.summarizer import chunked_summarize
from summarizer_core.utils import get_platform_config, watchdog

# Constants
VERSION = "0.1.0"
DEFAULT_CONFIG: Dict[str, Any] = {
    "chunk_size": 200000,
    "model_id": "mlx-community/gemma-4-26b-a4b-it-4bit",
    "model_name": "gemma-4-26b-it-mlx"
}

class MLXEngine(BaseEngine):
    def __init__(self, config: Dict[str, Any]) -> None:
        model_id: str = config.get("model_id", DEFAULT_CONFIG["model_id"])
        logger.info(f"Loading MLX model from {model_id}...")
        self.model, self.processor = load(model_id)
        self.config_data: Dict[str, Any] = load_config(model_id)

    def format_prompt(self, messages: List[Dict[str, str]]) -> str:
        return apply_chat_template(self.processor, self.config_data, messages, num_images=0)

    def generate(self, prompt: str, max_tokens: int = 1500, temp: float = 0.2, repetition_penalty: float = 1.1) -> str:
        output: Any = generate(
            self.model, self.processor, prompt, [],
            max_tokens=max_tokens,
            temp=temp,
            repetition_penalty=repetition_penalty,
            kv_bits=3.5,
            kv_quant_scheme="turboquant",
            verbose=False
        )
        if hasattr(output, "text"):
            text = str(getattr(output, "text"))
        else:
            text = str(output)
        return text

def summarize_file(source_file: str, destination_file: str, config: Dict[str, Any], summary_max_tokens: int) -> None:
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
        
        sum_metadata: Dict[str, Any] = {}
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

def answer_query(query: str, source_file: str, destination_file: str, config: Dict[str, Any], summary_max_tokens: int) -> None:
    if not os.path.exists(source_file):
        logger.error(f"Error: Source file does not exist: {source_file}")
        sys.exit(1)
        
    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        logger.info(f"Initializing MLX Engine for Q&A...")
        engine = MLXEngine(config)
        
        prompt_text = f"Based on the following context, please answer the query: '{query}'.\n\nContext:\n{content}"
        
        prompt: Union[str, List[Dict[str, str]]] = prompt_text
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
    parser.add_argument("--summary_max_tokens", type=int, help="Max tokens for the final summary.", default=1500)
    parser.add_argument("--capabilities", action="store_true", help="Print backend capabilities as JSON")
    
    args = parser.parse_args()

    if args.capabilities:
        cap: Dict[str, Any] = {
            "name": "summarizer_macos",
            "type": "summarizer",
            "config_file": "summarizer_config_macos.json",
            "platform": "macos",
            "validated_models": [DEFAULT_CONFIG["model_id"]],
            "available_models": [DEFAULT_CONFIG["model_id"]],
            "parameters": {
                "chunk_size":  {"type": "int", "min": 1000, "max": 500000},
                "model_id":    {"type": "str"},
                "model_name":  {"type": "str"},
            },
        }
        print(json.dumps(cap))
        sys.exit(0)
        
    if not args.source or not args.destination:
        parser.error("source and destination are required unless --capabilities is passed")
    config = get_platform_config("summarizer_config_macos.json", DEFAULT_CONFIG)
    
    if args.query:
        answer_query(args.query, args.source, args.destination, config, args.summary_max_tokens)
    else:
        summarize_file(args.source, args.destination, config, args.summary_max_tokens)

if __name__ == "__main__":
    main()
