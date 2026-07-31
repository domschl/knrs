from __future__ import annotations

import os
import sys
import json
import signal
import argparse
import time
import re
import logging
import threading
import hashlib
from datetime import datetime, timedelta
from typing import Any

# Suppress KeyboardInterrupt globally
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Silence transformers advisory warnings (e.g. missing PyTorch)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from google import genai
from google.genai import types

from summarizer_core.engine import BaseEngine
from summarizer_core.cache import WorkCache
from summarizer_core.markdown import parse_markdown, assemble_markdown
from summarizer_core.summarizer import chunked_summarize
from summarizer_core.utils import watchdog

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
logger = logging.getLogger("gc_gemma4_31b")

# Noise filter for external libraries
class NoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "AFC is enabled" in msg: return False
        if "HTTP Request" in msg and "200 OK" in msg: return False
        return True

for handler in logging.root.handlers:
    handler.addFilter(NoiseFilter())

# Constants
VERSION = "0.1.0"
DEFAULT_CONFIG: dict[str, Any] = {
    "chunk_size": 45000,
    "model_name": "gemma-4-31b-it",
    "api_key": "",
    "rate_blocked_until": "",
    "summary_max_tokens": 2500
}

def get_platform_config() -> dict[str, Any]:
    # Local specialized config loader since it modifies the config too
    config_file = os.path.expanduser("~/.config/knrs/summarizer_config_gc_gemma4_31b.json")
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading platform config: {e}")

    # If not found or error, create default
    try:
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        logger.warning(f"Default config created at {config_file}")
    except Exception as e:
        logger.error(f"Failed to create default config at {config_file}: {e}")

    return DEFAULT_CONFIG.copy()

def update_block_until(timestamp_str: str) -> None:
    config_path = os.path.expanduser("~/.config/knrs/summarizer_config_gc_gemma4_31b.json")
    try:
        config = get_platform_config()
        current_blocked = config.get("rate_blocked_until", "")
        if timestamp_str > current_blocked:
            config["rate_blocked_until"] = timestamp_str
            temp_path = config_path + ".tmp"
            with open(temp_path, 'w') as f:
                json.dump(config, f, indent=4)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, config_path)
            logger.info(f"Updated rate limit block until: {timestamp_str}")
    except Exception as e:
        logger.error(f"Failed to update config file: {e}")

def check_rate_limit() -> None:
    while True:
        config = get_platform_config()
        blocked_until_str = config.get("rate_blocked_until", "")
        if not blocked_until_str: break
        try:
            blocked_until = datetime.fromisoformat(blocked_until_str)
            now = datetime.now()
            if now < blocked_until:
                wait_seconds = (blocked_until - now).total_seconds()
                if wait_seconds > 0:
                    logger.info(f"Rate limited. Waiting {wait_seconds:.1f}s until {blocked_until_str}...")
                    time.sleep(min(wait_seconds, 10))
                    continue
        except Exception:
            pass
        break

def parse_retry_delay(exception: Exception) -> float | None:
    try:
        if hasattr(exception, "details") and exception.details:
            for detail in exception.details:
                if isinstance(detail, dict) and "retry_delay" in detail:
                    delay_str = str(detail["retry_delay"])
                    match = re.search(r"([\d\.]+)", delay_str)
                    if match: return float(match.group(1))
        match = re.search(r"'retryDelay':\s*'([\d\.]+)s'", str(exception))
        if match: return float(match.group(1))
    except Exception: pass
    return None

_tokenizer: Any = None

def get_gemma_tokenizer() -> Any:
    global _tokenizer
    if _tokenizer is None:
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from transformers import AutoTokenizer, logging as tf_logging
                tf_logging.set_verbosity_error()
                _tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
        except Exception as e:
            logger.warning(f"Local Gemma tokenizer unavailable ({e}). Falling back to character estimation.")
            _tokenizer = False
    return _tokenizer if _tokenizer is not False else None

def count_tokens_local(text: str) -> int:
    tok = get_gemma_tokenizer()
    if tok:
        try:
            return len(tok.encode(text))
        except Exception:
            pass
    return int(len(text) / 3.8)

class GemmaEngine(BaseEngine):
    def __init__(self, api_key: str, model_name: str) -> None:
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.last_request_time: float = 0
        self.min_delay: float = 4.1
        self.backoff: float = 60.0

    def _estimate_tokens(self, prompt: str | list[dict[str, str]], max_tokens: int) -> int:
        if isinstance(prompt, str):
            prompt_tokens = count_tokens_local(prompt)
        else:
            full_text = "\n".join(m.get("content", "") for m in prompt)
            prompt_tokens = count_tokens_local(full_text)
        return prompt_tokens + max_tokens

    def generate(self, prompt: str | list[dict[str, str]], max_tokens: int = 2500, temp: float = 0.2, repetition_penalty: float = 1.1) -> str:
        attempts = 0
        max_attempts = 10
        est_tokens = self._estimate_tokens(prompt, max_tokens)
        required_delay = max(self.min_delay, (est_tokens / 15000.0) * 60.0)

        while attempts < max_attempts:
            check_rate_limit()
            
            elapsed = time.time() - self.last_request_time
            if elapsed < required_delay:
                wait_time = required_delay - elapsed
                logger.info(f"Rate pacing for 16k TPM limit: waiting {wait_time:.1f}s (est {est_tokens} tokens)...")
                time.sleep(wait_time)

            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=temp,
                        thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                    )
                )
                self.last_request_time = time.time()
                self.backoff = 60.0 # Reset backoff on success
                if not response.text: return "[Summary blocked or empty response]"
                return response.text.strip()
            except Exception as e:
                attempts += 1
                msg = str(e).lower()
                if ("quota" in msg and "daily" in msg) or "per day" in msg:  
                    logger.error("Daily API Quota reached.")
                    tomorrow = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0) + timedelta(days=1)
                    update_block_until(tomorrow.isoformat())
                    sys.exit(10)
                
                status_code = getattr(e, 'code', None) or getattr(e, 'status_code', None)
                if "429" in msg or "resource_exhausted" in msg or "rate limit" in msg or status_code == 429:
                    suggested = parse_retry_delay(e) or 0.0
                    delay = max(60.0, suggested, self.backoff)
                    logger.warning(f"Rate limit hit. Suggest retry in {delay:.1f}s. Attempt {attempts}/{max_attempts}")
                    time.sleep(delay)
                    self.backoff = max(self.backoff * 2, delay * 2)
                    continue
                
                is_transient = (
                    "503" in msg or "500" in msg or 
                    "unavailable" in msg or "internal_error" in msg or 
                    "deadline_exceeded" in msg or
                    status_code in (500, 503, 504)
                )
                if is_transient:
                    delay = max(60.0, self.backoff)
                    logger.warning(f"Transient error (code={status_code}): {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    self.backoff = max(self.backoff * 2, delay * 2)
                    continue
                raise e
        raise Exception("Max retry attempts reached.")

def summarize_file(source_file: str, destination_file: str, config: dict[str, Any], summary_max_tokens: int) -> None:
    api_key: str = config.get("api_key", "")
    chunk_size: int = config.get("chunk_size", DEFAULT_CONFIG["chunk_size"])
    model_name: str = config.get("model_name", DEFAULT_CONFIG["model_name"])

    if chunk_size > 45000:
        chunk_size = 45000

    if not api_key:
        logger.error("No api_key found in platform config.")
        sys.exit(1)

    with open(source_file, 'r', encoding='utf-8') as f:
        content = f.read()
    metadata, md_text = parse_markdown(content)
    
    # Dynamic token-density chunk size adjustment
    MAX_TARGET_INPUT_TOKENS = 11500
    total_tokens = count_tokens_local(md_text)
    if total_tokens > 0:
        tokens_per_char = total_tokens / max(1, len(md_text))
        token_chunk_size = int(MAX_TARGET_INPUT_TOKENS / tokens_per_char)
        if token_chunk_size < chunk_size:
            logger.info(
                f"Document token density adjustment: setting chunk_size to {token_chunk_size} chars "
                f"({total_tokens} tokens / {len(md_text)} chars) to enforce max {MAX_TARGET_INPUT_TOKENS} tokens per chunk."
            )
            chunk_size = max(4000, token_chunk_size)
    
    doc_hash = hashlib.sha256(md_text.encode('utf-8')).hexdigest()
    
    cache = WorkCache()
    cache.cleanup_old_entries()

    engine = GemmaEngine(api_key, model_name)
    summary_text = chunked_summarize(engine, md_text, source_file, chunk_size, doc_hash, final_sum_tokens=summary_max_tokens)
    
    sum_metadata: dict[str, Any] = {}
    if metadata:
        for key in ['title', 'authors', 'tags', 'uuid']:
            if key in metadata: sum_metadata[key] = metadata[key]
    sum_metadata['summary_version'] = f"{model_name} {VERSION}"
    sum_metadata['source_md_hash'] = doc_hash
    
    full_summary = assemble_markdown(sum_metadata, summary_text)
    
    os.makedirs(os.path.dirname(destination_file), exist_ok=True)
    temp_file = destination_file + ".tmp"
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(full_summary)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_file, destination_file)
    logger.info(f"Successfully wrote summary: {destination_file}")

def answer_query(query: str, source_file: str, destination_file: str, config: dict[str, Any], summary_max_tokens: int) -> None:
    api_key: str = config.get("api_key", "")
    model_name: str = config.get("model_name", DEFAULT_CONFIG["model_name"])

    if not api_key:
        logger.error("No api_key found in platform config.")
        sys.exit(1)

    if not os.path.exists(source_file):
        logger.error(f"Source file does not exist: {source_file}")
        sys.exit(1)

    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            content = f.read()

        engine = GemmaEngine(api_key, model_name)
        
        prompt_text = f"Based on the following context, please answer the query: '{query}'.\n\nContext:\n{content}"
        
        prompt: str | list[dict[str, str]] = prompt_text
        if hasattr(engine, 'format_prompt'):
            formatted = engine.format_prompt([{"role": "user", "content": prompt_text}])
            if formatted:
                prompt = formatted
                
        output = engine.generate(prompt, max_tokens=summary_max_tokens)
        
        os.makedirs(os.path.dirname(destination_file), exist_ok=True)
        temp_file = destination_file + ".tmp"
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(output)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_file, destination_file)
        
        logger.info(f"Successfully wrote answer: {destination_file}")
    except Exception as e:
        logger.exception(f"Error during Q&A: {e}")
        sys.exit(1)

def main() -> None:
    w = threading.Thread(target=watchdog, daemon=True)
    w.start()

    parser = argparse.ArgumentParser(description="Summarize using Gemma 4 31B")
    parser.add_argument("source", nargs="?", help="Source markdown file")
    parser.add_argument("destination", nargs="?", help="Destination summary file")
    parser.add_argument("--query", type=str, help="If provided, answer this query based on the source file instead of summarizing it.", default=None)
    parser.add_argument("--summary_max_tokens", type=int, help="Max tokens for the final summary.", default=None)
    parser.add_argument("--capabilities", action="store_true", help="Print backend capabilities as JSON")
    parser.add_argument("--unload", action="store_true", help="Unload active model (no-op for this backend)")
    args = parser.parse_args()
    
    if args.unload:
        sys.exit(0)
        
    if args.capabilities:
        cap: dict[str, Any] = {
            "name": "summarizer_gc_gemma4_31b",
            "type": "summarizer",
            "config_file": "summarizer_config_gc_gemma4_31b.json",
            "platform": "any",
            "validated_models": [DEFAULT_CONFIG["model_name"]],
            "available_models": [DEFAULT_CONFIG["model_name"]],
            "parameters": {
                "chunk_size":         {"type": "int", "min": 1000, "max": 500000},
                "model_name":         {"type": "str"},
                "api_key":            {"type": "str"},
                "rate_blocked_until": {"type": "str", "read_only": True},
                "summary_max_tokens": {"type": "int", "min": 100, "max": 100000},
            },
        }
        print(json.dumps(cap))
        sys.exit(0)

    if not args.source or not args.destination:
        parser.error("source and destination are required unless --capabilities or --unload is passed")

    config = get_platform_config()
    
    summary_tokens = args.summary_max_tokens if args.summary_max_tokens is not None else config.get("summary_max_tokens", DEFAULT_CONFIG["summary_max_tokens"])
    
    if args.query:
        answer_query(args.query, args.source, args.destination, config, summary_tokens)
    else:
        summarize_file(args.source, args.destination, config, summary_tokens)

if __name__ == "__main__":
    main()
