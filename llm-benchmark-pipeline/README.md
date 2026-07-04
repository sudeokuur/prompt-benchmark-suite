# LLM Benchmark Pipeline

An automated pipeline for benchmarking multiple LLMs (OpenAI, Anthropic, Google) side by side across four task categories: **coding**, **reasoning**, **summarization**, and **analytical** tasks.

Each model is run against the same task set, scored automatically (rule-based checks where possible, LLM-as-judge for open-ended output), and the results are aggregated into CSV and Markdown reports.

## Features

- Pluggable provider layer (`benchmark/providers/`) — add a new model provider by subclassing `BaseProvider`.
- Category-specific scoring: exact/unit-test checks for code, keyword/structure checks for analytical tasks, and an LLM judge rubric for reasoning and summarization.
- Parallel execution across models and tasks with retry/backoff.
- Markdown + CSV reports, plus a per-model leaderboard summary.
- Sample task sets included so the pipeline runs out of the box; bring your own tasks by dropping JSON files into `tasks/<category>/`.

## Project layout

```
llm-benchmark-pipeline/
├── benchmark/
│   ├── providers/          # provider integrations (OpenAI, Anthropic, Google)
│   ├── scorers/            # rule-based + LLM-judge scoring
│   ├── tasks.py            # task loading/schema
│   ├── runner.py           # orchestrates model x task execution
│   └── report.py           # aggregates results into CSV/Markdown
├── config/
│   └── models.yaml         # which models to benchmark
├── tasks/
│   ├── coding/
│   ├── reasoning/
│   ├── summarization/
│   └── analytical/
├── scripts/
│   └── run_benchmark.py    # CLI entrypoint
├── results/                # output reports (gitignored except .gitkeep)
├── tests/
└── requirements.txt
```

## Setup

```bash
git clone <this-repo>
cd llm-benchmark-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your API keys
```

Required environment variables (in `.env`):

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
```

You only need keys for the providers you actually enable in `config/models.yaml`.

## Configuring models

Edit `config/models.yaml`:

```yaml
judge:
  provider: anthropic
  model: claude-opus-4-6

models:
  - provider: openai
    model: gpt-5.1
    display_name: "GPT-5.1"
  - provider: anthropic
    model: claude-sonnet-5
    display_name: "Claude Sonnet 5"
  - provider: google
    model: gemini-2.5-pro
    display_name: "Gemini 2.5 Pro"
```

The `judge` entry is the model used for LLM-as-judge scoring on reasoning and summarization tasks. Model name strings should be updated to whatever is current for each provider at the time you run this.

## Running the benchmark

```bash
python scripts/run_benchmark.py --config config/models.yaml --tasks all --output results/
```

Options:

- `--tasks all|coding|reasoning|summarization|analytical` — restrict to one category (default: `all`)
- `--models gpt-5.1,claude-sonnet-5` — restrict to specific model names from the config
- `--no-judge` — skip LLM-as-judge scoring and rely on rule-based checks only (cheaper, faster)
- `--workers 4` — number of parallel worker threads (default: 4)
- `--output results/` — directory to write `results.json`, `results.csv`, and `report.md`

## Adding tasks

Each task is a JSON file in `tasks/<category>/`. Schema:

```json
{
  "id": "coding_001",
  "category": "coding",
  "prompt": "Write a Python function `is_palindrome(s)` that returns True if s reads the same forwards and backwards, ignoring case and spaces.",
  "test_cases": [
    {"call": "is_palindrome('Race car')", "expected": "True"},
    {"call": "is_palindrome('hello')", "expected": "False"}
  ]
}
```

For `reasoning` and `analytical` tasks, include a `reference_answer` and/or `rubric` field instead of `test_cases`. For `summarization`, include `source_text` and a `rubric` describing what a good summary should cover. See existing files in `tasks/` for full examples of each category.

## Adding a new provider

Subclass `BaseProvider` in `benchmark/providers/base.py` and implement `generate(prompt, model, **kwargs)`. Register it in `benchmark/providers/__init__.py`'s `PROVIDER_REGISTRY`.

## Scoring methodology

- **Coding**: generated code is extracted and executed in a subprocess sandbox against the task's test cases (pass/fail + partial credit).
- **Reasoning / Analytical**: rule-based check against `reference_answer` (if present) combined with LLM-judge scoring (1-10) against a rubric for correctness and reasoning quality.
- **Summarization**: LLM-judge scoring (1-10) against a rubric (coverage, faithfulness, concision) since there's no single correct answer.

Final per-model score is a weighted average across categories, plus latency and (estimated) cost per task where token usage is available.

## Cost and safety notes

This pipeline makes real API calls to paid LLM providers. Running the full sample suite across 3 models costs a small amount in API credits (typically well under $1 with default sample tasks, but scales with task count, model choice, and judge usage). Code generated by models is executed locally in a subprocess with a timeout — review `benchmark/scorers/rule_based.py` before running untrusted/expanded task sets, and consider running in a container if you add tasks from untrusted sources.

## License

MIT — see `LICENSE`.
