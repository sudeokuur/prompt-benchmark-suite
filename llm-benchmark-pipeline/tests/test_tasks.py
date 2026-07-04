"""Unit tests for task loading."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.tasks import CATEGORIES, load_tasks


def test_load_all_sample_tasks():
    tasks = load_tasks()
    assert len(tasks) >= 8
    categories_found = {t.category for t in tasks}
    assert categories_found <= set(CATEGORIES)


def test_load_single_category():
    tasks = load_tasks(categories=["coding"])
    assert all(t.category == "coding" for t in tasks)
    assert len(tasks) >= 3


def test_coding_tasks_have_test_cases():
    tasks = load_tasks(categories=["coding"])
    for t in tasks:
        assert len(t.test_cases) > 0


def test_reasoning_tasks_have_reference_or_rubric():
    tasks = load_tasks(categories=["reasoning"])
    for t in tasks:
        assert t.reference_answer or t.rubric
