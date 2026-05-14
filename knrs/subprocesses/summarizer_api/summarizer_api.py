import os
import sys
import argparse
import signal
import logging
import threading
import hashlib
import requests

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
logger = logging.getLogger("summarizer_api")

# Suppress KeyboardInterrupt globally
signal.signal(signal.SIGINT, signal.SIG_DFL)

from summarizer_core.engine import BaseEngine
from summarizer_core.cache import WorkCache
from summarizer_core.markdown import parse_markdown, assemble_markdown
from summarizer_core.summarizer import chunked_summarize
from summarizer_core.utils import get_platform_config, get_llm_server_config, watchdog

# Constants
VERSION = "0.1.0"
DEFAULT_LOCAL_CONFIG = {
    "chunk_size": 200000,
    "model_name": "gemma-4-26B-A4B-it-UD-Q4_K_XL"
}

class ApiEngine(BaseEngine):
    def __init__(self, server_cfg: dict, local_cfg: dict):
        self.url = server_cfg["url"].rstrip("/")
        self.api_key = server_cfg.get("api_key")
        self.model = local_cfg["model_name"]
        logger.info(f"Using LLM Server at {self.url} (Model: {self.model})")

    def format_prompt(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        # Return messages as-is for the chat API
        return messages

    def generate(self, prompt: str | list, max_tokens: int = 1500, temp: float = 0.2, repetition_penalty: float = 1.1) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # If prompt is still a string (not formatted by format_prompt), wrap it
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temp,
            "repeat_penalty": repetition_penalty
        }
        
        try:
            response = requests.post(f"{self.url}/v1/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"API request failed: {e}")
            raise

def summarize_file(source_file: str, destination_file: str, config: dict, server_config: dict, summary_max_tokens: int):
    if not os.path.exists(source_file):
        logger.error(f"Source file does not exist: {source_file}")
        sys.exit(1)
        
    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            content = f.read()

        chunk_size = config.get("chunk_size", DEFAULT_LOCAL_CONFIG["chunk_size"])
        model_name = config.get("model_name", DEFAULT_LOCAL_CONFIG["model_name"])
        
        metadata, md_text = parse_markdown(content)
        source_md_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        
        cache = WorkCache()
        cache.cleanup_old_entries()
        
        engine = ApiEngine(server_config, config)
        summary_text = chunked_summarize(engine, md_text, source_file, chunk_size, source_md_hash, final_sum_tokens=summary_max_tokens)
        
        sum_metadata = {}
        if metadata:
            for key in ['title', 'authors', 'tags', 'uuid']:
                if key in metadata:
                    sum_metadata[key] = metadata[key]
        
        sum_metadata['summary_version'] = f"{model_name} (API) {VERSION}"
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

def answer_query(query: str, source_file: str, destination_file: str, config: dict, server_config: dict, summary_max_tokens: int):
    if not os.path.exists(source_file):
        logger.error(f"Source file does not exist: {source_file}")
        sys.exit(1)
        
    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        engine = ApiEngine(server_config, config)
        
        prompt_text = f"Based on the following context, please answer the query: '{query}'.\n\nContext:\n{content}"
        output = engine.generate(prompt_text, max_tokens=summary_max_tokens)
        
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

def main():
    w = threading.Thread(target=watchdog, daemon=True)
    w.start()

    parser = argparse.ArgumentParser(description="Summarize Markdown using REST API")
    parser.add_argument("source", nargs="?", help="Path to the source markdown file")
    parser.add_argument("destination", nargs="?", help="Path to the destination summary markdown file")
    parser.add_argument("--query", type=str, help="If provided, answer this query based on the source file instead of summarizing it.", default=None)
    parser.add_argument("--summary_max_tokens", type=int, help="Max tokens for the final summary.", default=1500)
    parser.add_argument("--capabilities", action="store_true", help="Print backend capabilities as JSON")
    args = parser.parse_args()
    
    server_config = get_llm_server_config()
    
    if args.capabilities:
        import json
        url = server_config.get("url", "http://localhost:8180").rstrip("/")
        api_key = server_config.get("api_key")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        try:
            response = requests.get(f"{url}/v1/models", headers=headers, timeout=2)
            response.raise_for_status()
            data = response.json()
            available_models = [m["id"] for m in data.get("data", [])]
        except Exception as e:
            logger.error(f"Failed to query {url}/v1/models: {e}")
            available_models = []

        cap = {
            "name": "summarizer_api",
            "type": "summarizer",
            "config_file": "summarizer_config_api.json",
            "platform": "any",
            "validated_models": ["gemma-4-26B-A4B-it-UD-Q4_K_XL", "gemma-4-31B-it-UD-Q4_K_XL"],
            "available_models": available_models,
            "parameters": {
                "chunk_size":  {"type": "int", "min": 1000, "max": 500000},
                "model_name":  {"type": "str"},
            },
        }
        print(json.dumps(cap))
        sys.exit(0)
        
    if not args.source or not args.destination:
        parser.error("source and destination are required unless --capabilities is passed")

    config = get_platform_config("summarizer_config_api.json", DEFAULT_LOCAL_CONFIG)
    
    if args.query:
        answer_query(args.query, args.source, args.destination, config, server_config, args.summary_max_tokens)
    else:
        summarize_file(args.source, args.destination, config, server_config, args.summary_max_tokens)

if __name__ == "__main__":
    main()
