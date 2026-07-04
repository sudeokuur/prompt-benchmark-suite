"""Unit tests for rule-based scoring (no API calls required)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.scorers.rule_based import (
    extract_code_block,
    score_coding_task,
    score_reference_overlap,
)
from benchmark.tasks import Task, TestCase


def test_extract_code_block_with_fencing():
    text = "Here's the code:\n```python\ndef f():\n    return 1\n```\nDone."
    assert extract_code_block(text).strip() == "def f():\n    return 1"


def test_extract_code_block_without_fencing():
    text = "def f():\n    return 1"
    assert extract_code_block(text) == text


def test_score_coding_task_all_pass():
    task = Task(
        id="t1",
        category="coding",
        prompt="write is_even",
        test_cases=[
            TestCase(call="is_even(2)", expected="True"),
            TestCase(call="is_even(3)", expected="False"),
        ],
    )
    response = "```python\ndef is_even(n):\n    return n % 2 == 0\n```"
    result = score_coding_task(task, response)
    assert result["score"] == 1.0
    assert result["passed"] == 2
    assert result["total"] == 2


def test_score_coding_task_partial_pass():
    task = Task(
        id="t2",
        category="coding",
        prompt="write is_even (buggy)",
        test_cases=[
            TestCase(call="is_even(2)", expected="True"),
            TestCase(call="is_even(3)", expected="False"),
        ],
    )
    response = "```python\ndef is_even(n):\n    return True\n```"
    result = score_coding_task(task, response)
    assert result["score"] == 0.5
    assert result["passed"] == 1


def test_score_coding_task_syntax_error():
    task = Task(
        id="t3",
        category="coding",
        prompt="broken",
        test_cases=[TestCase(call="f(1)", expected="1")],
    )
    response = "```python\ndef f(n)\n    return n\n```"  # missing colon
    result = score_coding_task(task, response)
    assert result["score"] == 0.0
    assert result["error"] is not None


def test_score_coding_task_timeout_protection():
    task = Task(
        id="t4",
        category="coding",
        prompt="infinite loop",
        test_cases=[TestCase(call="f(1)", expected="1")],
    )
    response = "```python\ndef f(n):\n    while True:\n        pass\n```"
    result = score_coding_task(task, response)
    assert result["score"] == 0.0
    assert "timed out" in (result["error"] or "").lower()


def test_score_reference_overlap():
    result = score_reference_overlap("Amara is 14, Ben is 15, Cleo is 16.", "The ages are 14, 15, and 16.")
    assert result["score"] > 0
