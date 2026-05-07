import json
import logging
import os
import urllib.request
import urllib.error
import time
import sys

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, model_name: str, config_dir: str = "~/.config/knrs/"):
        self.model_name = model_name
        self.config_dir = config_dir
        self._init_local()
        
    def _init_local(self):
        config_path = os.path.expanduser(f"{self.config_dir}/llm_server.json")
        try:
            with open(config_path, "r") as f:
                self.server_cfg = json.load(f)
        except Exception:
            self.server_cfg = {"url": "http://localhost:8180", "api_key": None}
            
        self.url = self.server_cfg.get("url", "http://localhost:8180").rstrip("/")
        self.api_key = self.server_cfg.get("api_key")
        
    def generate(self, messages: list[dict], max_tokens: int = 2000, temperature: float = 0.2) -> str:
        """Generate a response given a list of messages [{'role': 'user|model|system', 'content': '...'}]."""
        return self._generate_local(messages, max_tokens, temperature)
            
    def _generate_local(self, messages: list[dict], max_tokens: int, temperature: float) -> str:
        formatted_messages = []
        for m in messages:
            role = m["role"]
            if role == "model": role = "assistant"
            formatted_messages.append({"role": role, "content": m["content"]})
            
        payload = {
            "model": self.model_name,
            "messages": formatted_messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        req = urllib.request.Request(
            f"{self.url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        
        try:
            with urllib.request.urlopen(req, timeout=1800) as response:
                data = json.loads(response.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            logger.error(f"Local LLM request failed: {e}")
            try:
                err_text = e.read().decode("utf-8")
                logger.error(f"Response: {err_text}")
            except Exception:
                pass
            raise
        except Exception as e:
            logger.error(f"Local LLM request failed: {e}")
            raise
            
