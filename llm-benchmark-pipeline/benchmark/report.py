"""Aggregates raw benchmark results into CSV, JSON, and a Markdown leaderboard report."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any, Optional

from tabulate import tabulate


def _mean(values: list[float]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    return round(statistics.mean(clean), 4) if clean else None


def aggregate_by_model(results: list[dict], category_weights: dict[str, float]) -> list[dict]:
    """Compute per-model, per-category average scores plus a weighted overall score."""
    models = sorted({r["model_display_name"] for r in results})
    categories = sorted({r["category"] for r in results})

    summary = []
    for model in models:
        model_results = [r for r in results if r["model_display_name"] == model]
        row: dict[str, Any] = {"model": model}

        weighted_sum = 0.0
        weight_total = 0.0
        for category in categories:
            cat_results = [r for r in model_results if r["category"] == category]
            scores = [
                r["scoring"]["final_score"]
                for r in cat_results
                if r.get("scoring") and r["scoring"].get("final_score") is not None
            ]
            avg = _mean(scores)
            row[f"{category}_score"] = avg
            row[f"{category}_n"] = len(cat_results)
            if avg is not None:
                weight = category_weights.get(category, 1.0 / len(categories))
                weighted_sum += avg * weight
                weight_total += weight

        row["overall_score"] = round(weighted_sum / weight_total, 4) if weight_total else None
        row["avg_latency_seconds"] = _mean([r.get("latency_seconds") for r in model_results])
        row["errors"] = sum(1 for r in model_results if r.get("generation_error"))
        row["total_tasks"] = len(model_results)
        summary.append(row)

    summary.sort(key=lambda r: (r["overall_score"] is None, -(r["overall_score"] or 0)))
    return summary


def write_json(results: list[dict], summary: list[dict], output_dir: Path) -> Path:
    path = output_dir / "results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    return path


def write_csv(results: list[dict], output_dir: Path) -> Path:
    path = output_dir / "results.csv"
    fieldnames = [
        "model_display_name", "provider", "model", "task_id", "category",
        "final_score", "latency_seconds", "input_tokens", "output_tokens",
        "generation_error", "judge_rationale",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            scoring = r.get("scoring") or {}
            judge = scoring.get("judge") or {}
            writer.writerow({
                "model_display_name": r["model_display_name"],
                "provider": r["provider"],
                "model": r["model"],
                "task_id": r["task_id"],
                "category": r["category"],
                "final_score": scoring.get("final_score"),
                "latency_seconds": r.get("latency_seconds"),
                "input_tokens": r.get("input_tokens"),
                "output_tokens": r.get("output_tokens"),
                "generation_error": r.get("generation_error"),
                "judge_rationale": judge.get("rationale"),
            })
    return path


def write_markdown(summary: list[dict], output_dir: Path, categories: list[str]) -> Path:
    path = output_dir / "report.md"
    headers = ["Model", "Overall"] + [c.capitalize() for c in categories] + ["Avg Latency (s)", "Errors"]
    rows = []
    for row in summary:
        rows.append([
            row["model"],
            row["overall_score"] if row["overall_score"] is not None else "-",
            *[row.get(f"{c}_score") if row.get(f"{c}_score") is not None else "-" for c in categories],
            row["avg_latency_seconds"] if row["avg_latency_seconds"] is not None else "-",
            row["errors"],
        ])

    table = tabulate(rows, headers=headers, tablefmt="github", floatfmt=".3f")

    with open(path, "w", encoding="utf-8") as f:
        f.write("# LLM Benchmark Report\n\n")
        f.write("Scores are normalized to 0-1 (1 = perfect). Coding scores are the fraction of "
                "test cases passed; reasoning/analytical/summarization scores come from LLM-as-judge "
                "grading (1-10 rescaled to 0-1) unless run with `--no-judge`.\n\n")
        f.write(table)
        f.write("\n")
    return path


def generate_reports(results: list[dict], category_weights: dict[str, float], output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = sorted({r["category"] for r in results})
    summary = aggregate_by_model(results, category_weights)

    json_path = write_json(results, summary, output_dir)
    csv_path = write_csv(results, output_dir)
    md_path = write_markdown(summary, output_dir, categories)

    return {"json": json_path, "csv": csv_path, "markdown": md_path, "summary": summary}
