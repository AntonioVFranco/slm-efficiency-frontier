# Tests

All tests are designed to run on Kaggle only. Do not execute them locally.

Run on Kaggle:
    pip install -e .
    pytest tests/

## Test files
- `test_metrics.py` — metrics, per-model metrics, task-family breakdown.
- `test_budget_guard.py` — budget guard eligibility and stop conditions.
- `test_validators.py` — per-task validators on golden examples.
- `test_schema.py` — schema validation and v0.1 English-only policy.
- `test_leaderboard.py` — per-model leaderboard ranking and tie-breakers.
- `test_json_utils.py` — strict JSON serialization (no Infinity/NaN).
- `test_english_only.py` — English-only scanner, false-positive avoidance.

## Kaggle-only rule
No test in this directory is intended to run on a local developer machine.
