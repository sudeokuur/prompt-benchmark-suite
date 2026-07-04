# prompt-benchmark-suite
LLM Benchmark Pipeline
An automated pipeline for benchmarking multiple LLMs (OpenAI, Anthropic, Google) side by side across four task categories: coding, reasoning, summarization, and analytical tasks.

Each model is run against the same task set, scored automatically (rule-based checks where possible, LLM-as-judge for open-ended output), and the results are aggregated into CSV and Markdown reports.

Features
Pluggable provider layer (benchmark/providers/) — add a new model provider by subclassing BaseProvider.
Category-specific scoring: exact/unit-test checks for code, keyword/structure checks for analytical tasks, and an LLM judge rubric for reasoning and summarization.
Parallel execution across models and tasks with retry/backoff.
Markdown + CSV reports, plus a per-model leaderboard summary.
Sample task sets included so the pipeline runs out of the box; bring your own tasks by dropping JSON files into tasks/<category>/.


