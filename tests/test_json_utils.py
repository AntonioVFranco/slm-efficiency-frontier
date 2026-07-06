"""Tests for strict JSON serialization. Run on Kaggle only."""

import json
import math

from slm_efficiency_frontier.json_utils import sanitize_for_json
from slm_efficiency_frontier.metrics import compute_per_model_metrics
from slm_efficiency_frontier.schemas import RunResult


def _make(model_id="m", correct=False, cost=0.02):
    return RunResult(
        model_id=model_id,
        example_id="e",
        response="resp",
        task="json_validity",
        execution_model_id="local/dry-run-model",
        input_tokens=10,
        output_tokens=5,
        latency_ms=12.0,
        cost_usd=cost,
        correct=correct,
        valid=True,
    )


def test_inf_becomes_none():
    assert sanitize_for_json(math.inf) is None
    assert sanitize_for_json(-math.inf) is None


def test_nan_becomes_none():
    assert sanitize_for_json(math.nan) is None


def test_finite_floats_preserved():
    assert sanitize_for_json(0.05) == 0.05
    assert sanitize_for_json(3) == 3
    assert sanitize_for_json("text") == "text"
    assert sanitize_for_json(True) is True
    assert sanitize_for_json(None) is None


def test_recursive_dict_and_list():
    data = {"a": math.inf, "b": [math.nan, 1, "x"], "c": {"d": -math.inf}}
    safe = sanitize_for_json(data)
    assert safe == {"a": None, "b": [None, 1, "x"], "c": {"d": None}}


def test_report_with_zero_correct_serializes_strict():
    """A model with zero correct answers yields inf cost; serialization must
    produce strict JSON with None instead of Infinity."""
    results = [_make(model_id="m1", correct=False, cost=0.02)]
    per_model = compute_per_model_metrics(results)
    assert per_model["m1"]["cost_per_correct_answer"] == math.inf
    safe = sanitize_for_json(per_model)
    dumped = json.dumps(safe, allow_nan=False)
    parsed = json.loads(dumped)
    assert parsed["m1"]["cost_per_correct_answer"] is None


def test_dataclass_serialization():
    result = _make()
    safe = sanitize_for_json(result)
    # Dataclass flattened to dict; inf-free fields preserved.
    assert safe["model_id"] == "m"
    dumped = json.dumps(safe, allow_nan=False)
    assert json.loads(dumped)["model_id"] == "m"
