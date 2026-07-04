"""Anthropic provider integration."""

from __future__ import annotations

import os
from typing import Any, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BaseProvider


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(self) -> None:
        self._client = None

    def _get_client(self):
        if self._client is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set in the environment.")
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise RuntimeError(
                    "The `anthropic` package is required for AnthropicProvider. Install with `pip install anthropic`."
                ) from exc
            self._client = Anthropic(api_key=api_key)
        return self._client

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def _call(
        self, prompt: str, model: str, temperature: float, max_tokens: int, **kwargs: Any
    ) -> tuple[str, Optional[int], Optional[int]]:
        client = self._get_client()
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        usage = getattr(response, "usage", None)
        in_tok = getattr(usage, "input_tokens", None) if usage else None
        out_tok = getattr(usage, "output_tokens", None) if usage else None
        return text, in_tok, out_tok
