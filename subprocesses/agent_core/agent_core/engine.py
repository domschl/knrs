from __future__ import annotations

from typing import List, Dict, Protocol


class BaseAgentEngine(Protocol):
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 10000,
        temperature: float = 0.2,
    ) -> str:
        """Generate a response given the full conversation history.

        Args:
            messages:    List of {"role": "system|user|assistant", "content": "..."}.
            max_tokens:  Maximum tokens for the response.
            temperature: Sampling temperature.

        Returns:
            The model's response text.
        """
        ...
