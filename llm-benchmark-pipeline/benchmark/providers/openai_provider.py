"""OpenAI provider integration."""

from __future__ import annotations

import os
from typing import Any, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BaseProvider


class OpenAIProvider(BaseProvider):
    name = "openai"

    def __init__(self) -> None:
        self._client = None

    def _get_client(self):
        if self._client is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "The `openai` package is required for OpenAIProvider. Install with `pip install openai`."
                ) from exc
            self._client = OpenAI(api_key=api_key)
        return self._client

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def _call(
        self, prompt: str, model: str, temperature: float, max_tokens: int, **kwargs: Any
    ) -> tuple[str, Optional[int], Optional[int]]:
        client = self._get_client()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        in_tok = getattr(usage, "prompt_tokens", None) if usage else None
        out_tok = getattr(usage, "completion_tokens", None) if usage else None
        return text, in_tok, out_tok
