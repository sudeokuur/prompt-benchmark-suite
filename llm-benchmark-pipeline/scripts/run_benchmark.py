#!/usr/bin/env python
"""CLI entrypoint: run the benchmark pipeline and write reports.

Usage:
    python scripts/run_benchmark.py --config config/models.yaml --tasks all --output results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/run_benchmark.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from benchmark.report import generate_reports
from benchmark.runner import BenchmarkConfig, run_benchmark
from benchmark.tasks import CATEGORIES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LLM benchmark pipeline.")
    parser.add_argument("--config", default="config/models.yaml", help="Path to models config YAML.")
    parser.add_argument(
        "--tasks",
        default="all",
        help=f"Comma-separated categories to run ({', '.join(CATEGORIES)}) or 'all'.",
    )
    parser.add_argument("--models", default=None, help="Comma-separated model names/display names to restrict to.")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM-as-judge scoring.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker threads.")
    parser.add_argument("--output", default="results/", help="Output directory for reports.")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    categories = None if args.tasks == "all" else [c.strip() for c in args.tasks.split(",")]
    model_names = [m.strip() for m in args.models.split(",")] if args.models else None

    config = BenchmarkConfig.from_yaml(args.config)

    print(f"Running benchmark: {len(config.models)} model(s) x tasks in {categories or 'all categories'}...", file=sys.stderr)

    results = run_benchmark(
        config=config,
        categories=categories,
        model_names=model_names,
        use_judge=not args.no_judge,
        workers=args.workers,
    )

    paths = generate_reports(results, config.category_weights, args.output)

    print("\nDone. Reports written to:", file=sys.stderr)
    print(f"  {paths['json']}", file=sys.stderr)
    print(f"  {paths['csv']}", file=sys.stderr)
    print(f"  {paths['markdown']}", file=sys.stderr)
    print(file=sys.stderr)

    for row in paths["summary"]:
        print(f"  {row['model']:<20} overall={row['overall_score']}", file=sys.stderr)


if __name__ == "__main__":
    main()
