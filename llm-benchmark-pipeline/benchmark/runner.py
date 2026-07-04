"""Orchestrates running every configured model against every loaded task and scoring results."""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from benchmark.providers import get_provider
from benchmark.scorers import LLMJudge, score_coding_task, score_reference_overlap
from benchmark.tasks import Task, load_tasks


@dataclass
class ModelConfig:
    provider: str
    model: str
    display_name: str
    temperature: float = 0.2
    max_tokens: int = 2048


@dataclass
class BenchmarkConfig:
    models: list[ModelConfig]
    judge: Optional[dict]
    category_weights: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def from_yaml(path: str | Path) -> "BenchmarkConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        models = [ModelConfig(**m) for m in data["models"]]
        return BenchmarkConfig(
            models=models,
            judge=data.get("judge"),
            category_weights=data.get("category_weights", {}),
        )


def _score_task_result(task: Task, response_text: str, judge: Optional[LLMJudge]) -> dict:
    """Category-aware scoring. Returns a dict with a normalized `final_score` (0-1, or None)."""
    scoring: dict[str, Any] = {"rule_based": None, "judge": None, "final_score": None}

    if task.category == "coding":
        rule_result = score_coding_task(task, response_text)
        scoring["rule_based"] = rule_result
        scoring["final_score"] = rule_result.get("score")
        return scoring

    if task.category == "summarization":
        if judge is not None:
            judge_result = judge.judge(task, response_text)
            scoring["judge"] = judge_result
            scoring["final_score"] = judge_result.get("score")
        return scoring

    # reasoning / analytical: reference overlap (supplementary) + judge (primary)
    if task.reference_answer:
        scoring["rule_based"] = score_reference_overlap(task.reference_answer, response_text)
    if judge is not None:
        judge_result = judge.judge(task, response_text)
        scoring["judge"] = judge_result
        scoring["final_score"] = judge_result.get("score")
    elif scoring["rule_based"] is not None:
        scoring["final_score"] = scoring["rule_based"].get("score")

    return scoring


def _run_one(model_cfg: ModelConfig, task: Task, judge: Optional[LLMJudge]) -> dict:
    provider = get_provider(model_cfg.provider)
    gen = provider.generate(
        prompt=task.prompt,
        model=model_cfg.model,
        temperature=model_cfg.temperature,
        max_tokens=model_cfg.max_tokens,
    )

    record = {
        "model_display_name": model_cfg.display_name,
        "provider": model_cfg.provider,
        "model": model_cfg.model,
        "task_id": task.id,
        "category": task.category,
        "prompt": task.prompt,
        "response": gen.text,
        "latency_seconds": round(gen.latency_seconds, 3),
        "input_tokens": gen.input_tokens,
        "output_tokens": gen.output_tokens,
        "generation_error": gen.raw_error,
        "scoring": None,
    }

    if not gen.ok:
        record["scoring"] = {"rule_based": None, "judge": None, "final_score": None}
        return record

    record["scoring"] = _score_task_result(task, gen.text, judge)
    return record


def run_benchmark(
    config: BenchmarkConfig,
    categories: Optional[list[str]] = None,
    model_names: Optional[list[str]] = None,
    use_judge: bool = True,
    workers: int = 4,
    tasks_dir: Optional[Path] = None,
    progress_callback: Optional[Any] = None,
) -> list[dict]:
    """Run the full benchmark matrix (models x tasks) and return a flat list of result records."""

    tasks = load_tasks(categories=categories, tasks_dir=tasks_dir)

    models = config.models
    if model_names:
        wanted = set(model_names)
        models = [m for m in models if m.model in wanted or m.display_name in wanted]
        if not models:
            raise ValueError(f"No models in config matched --models filter {model_names!r}")

    judge = None
    if use_judge and config.judge:
        judge = LLMJudge(
            provider=config.judge["provider"],
            model=config.judge["model"],
            temperature=config.judge.get("temperature", 0.0),
            max_tokens=config.judge.get("max_tokens", 1024),
        )

    jobs = [(m, t) for m in models for t in tasks]
    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_one, m, t, judge): (m, t) for m, t in jobs}
        completed = 0
        for future in as_completed(futures):
            m, t = futures[future]
            try:
                record = future.result()
            except Exception as exc:  # noqa: BLE001
                record = {
                    "model_display_name": m.display_name,
                    "provider": m.provider,
                    "model": m.model,
                    "task_id": t.id,
                    "category": t.category,
                    "prompt": t.prompt,
                    "response": "",
                    "latency_seconds": None,
                    "input_tokens": None,
                    "output_tokens": None,
                    "generation_error": f"{type(exc).__name__}: {exc}",
                    "scoring": {"rule_based": None, "judge": None, "final_score": None},
                }
            results.append(record)
            completed += 1
            if progress_callback:
                progress_callback(completed, len(jobs), record)
            else:
                print(f"[{completed}/{len(jobs)}] {record['model_display_name']} - {record['task_id']}", file=sys.stderr)

    return results
