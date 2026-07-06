"""Tests for metrics. Run on Kaggle only."""

from slm_efficiency_frontier.metrics import (
    accuracy,
    cost_per_correct_answer,
    refusal_rate,
    overgeneration_rate,
    task_family_breakdown,
    compute_per_model_metrics,
    compute_per_model_task_breakdown,
)
from slm_efficiency_frontier.schemas import RunResult


def make(c=True, cost=0.01, refused=False, over=False, task="json_validity", model_id="m"):
    return RunResult(
        model_id=model_id,
        example_id="e",
        response="resp",
        task=task,
        input_tokens=10,
        output_tokens=5,
        latency_ms=12.0,
        cost_usd=cost,
        correct=c,
        valid=True,
        refused=refused,
        overgenerated=over,
    )


def test_accuracy():
    results = [make(c=True), make(c=False)]
    assert accuracy(results) == 0.5


def test_cost_per_correct_answer():
    results = [make(c=True, cost=0.02), make(c=True, cost=0.02)]
    assert cost_per_correct_answer(results) == 0.02


def test_refusal_and_overgeneration():
    results = [make(refused=True), make(over=True), make()]
    assert refusal_rate(results) > 0
    assert overgeneration_rate(results) > 0


def test_task_family_breakdown_groups_by_task():
    results = [
        make(task="json_validity", c=True),
        make(task="json_validity", c=False),
        make(task="tool_calling", c=True),
    ]
    breakdown = task_family_breakdown(results)
    assert "json_validity" in breakdown
    assert "tool_calling" in breakdown
    assert breakdown["json_validity"]["accuracy"] == 0.5
    assert breakdown["tool_calling"]["accuracy"] == 1.0
    assert breakdown["json_validity"]["count"] == 2.0


def test_per_model_metrics_keyed_by_model():
    results = [
        make(model_id="m1", c=True, cost=0.02),
        make(model_id="m1", c=False, cost=0.02),
        make(model_id="m2", c=True, cost=0.04),
    ]
    per_model = compute_per_model_metrics(results)
    assert set(per_model.keys()) == {"m1", "m2"}
    assert per_model["m1"]["accuracy"] == 0.5
    assert per_model["m2"]["accuracy"] == 1.0
    assert per_model["m1"]["cost_per_correct_answer"] == 0.04
    assert per_model["m2"]["cost_per_correct_answer"] == 0.04
    for metrics in per_model.values():
        for field in (
            "accuracy",
            "cost_per_correct_answer",
            "cost_per_1k_examples",
            "median_latency_ms",
            "tokens_per_correct_answer",
            "json_validity_rate",
            "refusal_rate",
            "overgeneration_rate",
            "num_examples",
        ):
            assert field in metrics


def test_per_model_task_breakdown():
    results = [
        make(model_id="m1", task="json_validity", c=True),
        make(model_id="m1", task="tool_calling", c=False),
        make(model_id="m2", task="json_validity", c=True),
    ]
    breakdown = compute_per_model_task_breakdown(results)
    assert set(breakdown.keys()) == {"m1", "m2"}
    assert "json_validity" in breakdown["m1"]
    assert "tool_calling" in breakdown["m1"]
    assert breakdown["m1"]["json_validity"]["accuracy"] == 1.0
    assert breakdown["m2"]["json_validity"]["accuracy"] == 1.0


def test_per_model_metrics_infinite_cost_when_no_correct():
    results = [make(model_id="m1", c=False, cost=0.02)]
    per_model = compute_per_model_metrics(results)
    assert per_model["m1"]["cost_per_correct_answer"] == float("inf")


def test_per_model_groups_by_candidate_model_id_not_backend():
    """Per-model metrics must group by the candidate model_id, not by the
    synthetic execution backend, so dry-run preserves candidate identity."""
    results = [
        make(model_id="candidate-a", c=True, cost=0.0),
        make(model_id="candidate-b", c=False, cost=0.0),
    ]
    # Simulate dry-run backend identity on execution_model_id.
    results[0].execution_model_id = "local/dry-run-model"
    results[1].execution_model_id = "local/dry-run-model"
    per_model = compute_per_model_metrics(results)
    assert set(per_model.keys()) == {"candidate-a", "candidate-b"}
    assert per_model["candidate-a"]["accuracy"] == 1.0
    assert per_model["candidate-b"]["accuracy"] == 0.0
