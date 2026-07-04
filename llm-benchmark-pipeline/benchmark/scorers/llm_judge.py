"""LLM-as-judge scoring for open-ended tasks (reasoning, analytical, summarization)."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from benchmark.providers import get_provider

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_FALLBACK_SCORE_RE = re.compile(r"\b(10|[1-9])\b")

JUDGE_PROMPT_TEMPLATE = """You are an impartial grader evaluating an AI model's response to a task.

## Task category
{category}

## Original prompt given to the model
{prompt}
{context_block}
## Grading rubric
{rubric}

## Model's response to grade
{response}

Score the response from 1 (completely wrong / unusable) to 10 (fully correct and meets the rubric).
Be strict: a response should only score 8-10 if it fully satisfies the rubric, 4-7 for partially
correct responses with notable gaps, and 1-3 for responses that are largely wrong or off-task.

Respond with ONLY a JSON object in this exact format, no other text:
{{"score": <integer 1-10>, "rationale": "<one sentence explanation>"}}
"""


class LLMJudge:
    def __init__(self, provider: str, model: str, temperature: float = 0.0, max_tokens: int = 1024):
        self.provider_name = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _build_prompt(self, task: Any, response_text: str) -> str:
        context_parts = []
        if getattr(task, "source_text", None):
            context_parts.append(f"\n## Source text\n{task.source_text}\n")
        if getattr(task, "reference_answer", None):
            context_parts.append(f"\n## Reference answer (for grading only, not shown to the model)\n{task.reference_answer}\n")
        context_block = "".join(context_parts)

        rubric = getattr(task, "rubric", None) or "Judge general correctness, clarity, and completeness relative to the prompt."

        return JUDGE_PROMPT_TEMPLATE.format(
            category=task.category,
            prompt=task.prompt,
            context_block=context_block,
            rubric=rubric,
            response=response_text or "(empty response)",
        )

    def judge(self, task: Any, response_text: str) -> dict:
        if not response_text.strip():
            return {"score": 0.0, "raw_score": 1, "rationale": "Empty response.", "error": None}

        provider = get_provider(self.provider_name)
        prompt = self._build_prompt(task, response_text)
        result = provider.generate(prompt, model=self.model, temperature=self.temperature, max_tokens=self.max_tokens)

        if not result.ok:
            return {"score": None, "raw_score": None, "rationale": None, "error": result.raw_error}

        raw_score, rationale = self._parse_judgment(result.text)
        if raw_score is None:
            return {"score": None, "raw_score": None, "rationale": None, "error": f"Could not parse judge output: {result.text[:200]}"}

        normalized = (raw_score - 1) / 9.0  # map 1-10 -> 0-1
        return {"score": normalized, "raw_score": raw_score, "rationale": rationale, "error": None}

    @staticmethod
    def _parse_judgment(text: str) -> tuple[Optional[int], Optional[str]]:
        match = _JSON_BLOCK_RE.search(text)
        if match:
            try:
                data = json.loads(match.group(0))
                score = int(data.get("score"))
                if 1 <= score <= 10:
                    return score, data.get("rationale")
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        # Fallback: grab the first standalone 1-10 number in the text.
        fallback = _FALLBACK_SCORE_RE.search(text)
        if fallback:
            return int(fallback.group(1)), text.strip()[:200]
        return None, None
