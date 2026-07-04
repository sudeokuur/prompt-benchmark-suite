"""Task schema and loading utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

CATEGORIES = ("coding", "reasoning", "summarization", "analytical")

TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"


@dataclass
class TestCase:
    call: str
    expected: str


@dataclass
class Task:
    id: str
    category: str
    prompt: str
    test_cases: list[TestCase] = field(default_factory=list)
    reference_answer: Optional[str] = None
    rubric: Optional[str] = None
    source_text: Optional[str] = None

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Task":
        test_cases = [TestCase(**tc) for tc in data.get("test_cases", [])]
        return Task(
            id=data["id"],
            category=data["category"],
            prompt=data["prompt"],
            test_cases=test_cases,
            reference_answer=data.get("reference_answer"),
            rubric=data.get("rubric"),
            source_text=data.get("source_text"),
        )


def load_tasks(categories: Optional[list[str]] = None, tasks_dir: Optional[Path] = None) -> list[Task]:
    """Load all task JSON files under `tasks/<category>/*.json`.

    Args:
        categories: restrict to these categories (defaults to all of CATEGORIES).
        tasks_dir: override the base tasks directory (defaults to the repo's `tasks/`).
    """
    base = tasks_dir or TASKS_DIR
    cats = categories or list(CATEGORIES)
    tasks: list[Task] = []
    for category in cats:
        cat_dir = base / category
        if not cat_dir.exists():
            continue
        for path in sorted(cat_dir.glob("*.json")):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            tasks.append(Task.from_dict(data))
    if not tasks:
        raise ValueError(
            f"No tasks found under {base} for categories {cats}. "
            "Add JSON task files or check --tasks filter."
        )
    return tasks
