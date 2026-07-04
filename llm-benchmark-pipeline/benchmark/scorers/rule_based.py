"""Rule-based scoring: code execution against test cases, and simple
reference-overlap checks for reasoning/analytical tasks.

Code execution runs in a subprocess with a timeout so a single hung or
malicious generation can't block the whole benchmark run. This is a basic
isolation measure, not a hardened sandbox -- do not run untrusted task sets
against this without a container or stronger sandboxing.
"""

from __future__ import annotations

import json
import re
import string
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

EXEC_TIMEOUT_SECONDS = 5
_RUNNER_TEMPLATE = '''
import json, sys

# --- candidate code ---
{code}
# --- end candidate code ---

_test_calls = {calls!r}
_results = []
for _call in _test_calls:
    try:
        _value = eval(_call)
        _results.append({{"ok": True, "value": str(_value)}})
    except Exception as exc:  # noqa: BLE001
        _results.append({{"ok": False, "error": f"{{type(exc).__name__}}: {{exc}}"}})

print(json.dumps(_results))
'''


def extract_code_block(text: str) -> str:
    """Pull the first fenced code block out of a model response.

    Falls back to the raw text if no fenced block is found (some models
    reply with bare code and no markdown fencing).
    """
    match = CODE_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def score_coding_task(task: Any, response_text: str) -> dict:
    """Execute the candidate code against the task's test cases.

    Returns a dict with `score` (fraction of test cases passed, 0-1),
    `passed`, `total`, and per-case `details`.
    """
    code = extract_code_block(response_text)
    calls = [tc.call for tc in task.test_cases]
    expected = [tc.expected for tc in task.test_cases]

    if not calls:
        return {"score": None, "passed": 0, "total": 0, "details": [], "error": "No test cases defined."}

    script = _RUNNER_TEMPLATE.format(code=code, calls=calls)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name

    details = []
    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=EXEC_TIMEOUT_SECONDS,
        )
        if proc.returncode != 0:
            return {
                "score": 0.0,
                "passed": 0,
                "total": len(calls),
                "details": [{"call": c, "ok": False, "error": proc.stderr[-500:]} for c in calls],
                "error": f"Execution failed: {proc.stderr[-500:]}",
            }
        try:
            results = json.loads(proc.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as exc:
            return {
                "score": 0.0,
                "passed": 0,
                "total": len(calls),
                "details": [],
                "error": f"Could not parse execution output: {exc}",
            }
    except subprocess.TimeoutExpired:
        return {
            "score": 0.0,
            "passed": 0,
            "total": len(calls),
            "details": [{"call": c, "ok": False, "error": "timeout"} for c in calls],
            "error": f"Execution timed out after {EXEC_TIMEOUT_SECONDS}s.",
        }
    finally:
        Path(script_path).unlink(missing_ok=True)

    passed = 0
    for call, exp, result in zip(calls, expected, results):
        actual = result.get("value") if result.get("ok") else None
        is_pass = result.get("ok", False) and actual == exp
        passed += int(is_pass)
        details.append({
            "call": call,
            "expected": exp,
            "actual": actual,
            "ok": is_pass,
            "error": result.get("error"),
        })

    return {
        "score": passed / len(calls),
        "passed": passed,
        "total": len(calls),
        "details": details,
        "error": None,
    }


def _normalize(text: str) -> set[str]:
    stripped = text.lower().translate(str.maketrans("", "", string.punctuation))
    # Keep tokens longer than 2 chars, but always keep purely numeric tokens
    # (e.g. "14", "16") since short numbers are often the key fact to match.
    return {tok for tok in stripped.split() if len(tok) > 2 or tok.isdigit()}


def score_reference_overlap(reference_answer: str, response_text: str) -> dict:
    """Cheap heuristic: token overlap between reference answer and response.

    This is a *supplementary* signal only (not a substitute for the LLM
    judge) -- it flags cases where the response contains basically none of
    the reference's key terms/numbers, which usually means the answer is
    off track.
    """
    ref_tokens = _normalize(reference_answer)
    resp_tokens = _normalize(response_text)
    if not ref_tokens:
        return {"score": None, "overlap": 0, "reference_terms": 0}
    overlap = ref_tokens & resp_tokens
    return {
        "score": len(overlap) / len(ref_tokens),
        "overlap": len(overlap),
        "reference_terms": len(ref_tokens),
    }
