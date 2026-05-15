from __future__ import annotations

from typing import Dict, List, Protocol, Union


class BaseEngine(Protocol):
    def format_prompt(self, messages: List[Dict[str, str]]) -> Union[str, List[Dict[str, str]]]:
        """Formats the chat messages into a model-compatible prompt."""
        ...

    def generate(self, prompt: Union[str, List[Dict[str, str]]], max_tokens: int = 1500, temp: float = 0.2, repetition_penalty: float = 1.1) -> str:
        """Generates a response from the model."""
        ...
