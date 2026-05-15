"""
agent_core.engine — Base protocol for agent backends.

Unlike the summarizer's BaseEngine (single prompt → response), the agent
engine operates on full multi-turn conversation histories, which is
essential for:
  - OpenAI API backends that need the message array directly
  - HF/MLX backends that need to apply their own chat template to the
    full conversation history
"""

from typing import Protocol


class BaseAgentEngine(Protocol):
    def chat(
        self,
        messages: list[dict[str, str]],
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
