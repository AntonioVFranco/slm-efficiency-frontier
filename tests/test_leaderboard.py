"""Tests for leaderboard generation. Run on Kaggle only."""

import importlib.util
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / "scripts" / "build_leaderboard.py"


def _load_build_module():
    spec = importlib.util.spec_from_file_location("build_leaderboard", BUILD_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_leaderboard"] = module
    spec.loader.exec_module(module)
    return module


def _report(per_model):
    return {"per_model_metrics": per_model}


def test_one_row_per_model():
    module = _load_build_module()
    per_model = {
        "m1": {"accuracy": 0.5, "cost_per_correct_answer": 0.04},
        "m2": {"accuracy": 1.0, "cost_per_correct_answer": 0.02},
    }
    rows = module.build_leaderboard(_report(per_model))
    assert len(rows) == 2
    ids = {r["model_id"] for r in rows}
    assert ids == {"m1", "m2"}


def test_ranks_by_cost_per_correct_answer_ascending():
    module = _load_build_module()
    per_model = {
        "m1": {"accuracy": 0.9, "cost_per_correct_answer": 0.10},
        "m2": {"accuracy": 0.5, "cost_per_correct_answer": 0.02},
    }
    rows = module.build_leaderboard(_report(per_model))
    assert rows[0]["model_id"] == "m2"
    assert rows[1]["model_id"] == "m1"


def test_tiebreaker_higher_accuracy_first():
    module = _load_build_module()
    per_model = {
        "m1": {"accuracy": 0.9, "cost_per_correct_answer": 0.05,
               "median_latency_ms": 10.0, "tokens_per_correct_answer": 100.0},
        "m2": {"accuracy": 0.5, "cost_per_correct_answer": 0.05,
               "median_latency_ms": 10.0, "tokens_per_correct_answer": 100.0},
    }
    rows = module.build_leaderboard(_report(per_model))
    assert rows[0]["model_id"] == "m1"


def test_tiebreaker_lower_latency_then_tokens_then_alpha():
    """Rule: lower cost, higher accuracy, lower latency, lower tokens, alpha.

    All three tie on cost and accuracy. Latency decides first:
      a_model latency 10, tokens 200
      b_model latency 10, tokens 50
      z_model latency 20, tokens 50

    a_model and b_model tie on latency (10 < 20 for z_model), so among them
    lower tokens decides: b_model (50) before a_model (200). z_model has the
    highest latency and comes last. Correct order: b, a, z.
    """
    module = _load_build_module()
    per_model = {
        "z_model": {"accuracy": 0.5, "cost_per_correct_answer": 0.05,
                    "median_latency_ms": 20.0, "tokens_per_correct_answer": 50.0},
        "a_model": {"accuracy": 0.5, "cost_per_correct_answer": 0.05,
                    "median_latency_ms": 10.0, "tokens_per_correct_answer": 200.0},
        "b_model": {"accuracy": 0.5, "cost_per_correct_answer": 0.05,
                    "median_latency_ms": 10.0, "tokens_per_correct_answer": 50.0},
    }
    rows = module.build_leaderboard(_report(per_model))
    order = [r["model_id"] for r in rows]
    assert order == ["b_model", "a_model", "z_model"], f"got {order}"


def test_infinite_cost_ranks_below_finite():
    module = _load_build_module()
    per_model = {
        "good": {"accuracy": 0.5, "cost_per_correct_answer": 0.05},
        "bad": {"accuracy": 1.0, "cost_per_correct_answer": math.inf},
    }
    rows = module.build_leaderboard(_report(per_model))
    assert rows[0]["model_id"] == "good"
    assert rows[1]["model_id"] == "bad"


def test_empty_when_no_per_model_metrics():
    module = _load_build_module()
    rows = module.build_leaderboard({})
    assert rows == []


def test_leaderboard_serializes_strict_json():
    """Leaderboard with non-finite cost must serialize with allow_nan=False."""
    module = _load_build_module()
    per_model = {
        "good": {"accuracy": 0.5, "cost_per_correct_answer": 0.05},
        "bad": {"accuracy": 1.0, "cost_per_correct_answer": math.inf},
    }
    rows = module.build_leaderboard(_report(per_model))
    safe = module.sanitize_for_json(rows)
    # Must not raise under strict JSON.
    dumped = json.dumps(safe, allow_nan=False)
    # Non-finite cost becomes None.
    bad_row = next(r for r in json.loads(dumped) if r["model_id"] == "bad")
    assert bad_row["cost_per_correct_answer"] is None
