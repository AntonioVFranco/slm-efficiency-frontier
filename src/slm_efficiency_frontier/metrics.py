"""Metric registry with PyTorch tensor aggregation."""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any, Callable

import torch

from .schemas import RunResult

# Metrics computed for every model in per-model reports.
PER_MODEL_METRIC_NAMES = (
    "accuracy",
    "cost_per_correct_answer",
    "cost_per_1k_examples",
    "median_latency_ms",
    "tokens_per_correct_answer",
    "json_validity_rate",
    "refusal_rate",
    "overgeneration_rate",
)


def accuracy(results: list[RunResult]) -> float:
    if not results:
        return 0.0
    correct = torch.tensor([int(r.correct) for r in results], dtype=torch.float32)
    return float(correct.mean().item())


def exact_match(results: list[RunResult]) -> float:
    return accuracy(results)


def json_validity_rate(results: list[RunResult]) -> float:
    if not results:
        return 0.0
    valid = torch.tensor([int(r.valid) for r in results], dtype=torch.float32)
    return float(valid.mean().item())


def tool_selection_accuracy(results: list[RunResult]) -> float:
    return accuracy(results)


def argument_exact_match(results: list[RunResult]) -> float:
    return accuracy(results)


def cost_per_correct_answer(results: list[RunResult]) -> float:
    correct = sum(1 for r in results if r.correct)
    total = sum(r.cost_usd for r in results)
    if correct == 0:
        return float("inf")
    return total / correct


def cost_per_1k_examples(results: list[RunResult]) -> float:
    if not results:
        return 0.0
    total = sum(r.cost_usd for r in results)
    return total / (len(results) / 1000.0)


def median_latency_ms(results: list[RunResult]) -> float:
    if not results:
        return 0.0
    return float(statistics.median(r.latency_ms for r in results))


def tokens_per_correct_answer(results: list[RunResult]) -> float:
    correct = sum(1 for r in results if r.correct)
    total_tokens = sum(r.input_tokens + r.output_tokens for r in results)
    if correct == 0:
        return float("inf")
    return total_tokens / correct


def overgeneration_rate(results: list[RunResult]) -> float:
    if not results:
        return 0.0
    over = torch.tensor([int(r.overgenerated) for r in results], dtype=torch.float32)
    return float(over.mean().item())


def refusal_rate(results: list[RunResult]) -> float:
    if not results:
        return 0.0
    ref = torch.tensor([int(r.refused) for r in results], dtype=torch.float32)
    return float(ref.mean().item())


def task_family_breakdown(results: list[RunResult]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[RunResult]] = defaultdict(list)
    for r in results:
        groups[r.task].append(r)
    breakdown: dict[str, dict[str, float]] = {}
    for task, group in groups.items():
        breakdown[task] = {
            "accuracy": accuracy(group),
            "cost_per_correct_answer": cost_per_correct_answer(group),
            "count": float(len(group)),
        }
    return breakdown


METRIC_REGISTRY: dict[str, Callable[[list[RunResult]], Any]] = {
    "accuracy": accuracy,
    "exact_match": exact_match,
    "json_validity_rate": json_validity_rate,
    "tool_selection_accuracy": tool_selection_accuracy,
    "argument_exact_match": argument_exact_match,
    "cost_per_correct_answer": cost_per_correct_answer,
    "cost_per_1k_examples": cost_per_1k_examples,
    "median_latency_ms": median_latency_ms,
    "tokens_per_correct_answer": tokens_per_correct_answer,
    "overgeneration_rate": overgeneration_rate,
    "refusal_rate": refusal_rate,
    "task_family_breakdown": task_family_breakdown,
}


def compute_metrics(results: list[RunResult]) -> dict[str, Any]:
    return {name: fn(results) for name, fn in METRIC_REGISTRY.items()}


def compute_per_model_metrics(results: list[RunResult]) -> dict[str, dict[str, float]]:
    """Compute the per-model metric set, keyed by model_id.

    Each model gets accuracy, cost_per_correct_answer, cost_per_1k_examples,
    median_latency_ms, tokens_per_correct_answer, json_validity_rate,
    refusal_rate, and overgeneration_rate.
    """
    by_model: dict[str, list[RunResult]] = defaultdict(list)
    for r in results:
        by_model[r.model_id].append(r)
    per_model: dict[str, dict[str, float]] = {}
    for model_id, group in by_model.items():
        per_model[model_id] = {
            "accuracy": accuracy(group),
            "cost_per_correct_answer": cost_per_correct_answer(group),
            "cost_per_1k_examples": cost_per_1k_examples(group),
            "median_latency_ms": median_latency_ms(group),
            "tokens_per_correct_answer": tokens_per_correct_answer(group),
            "json_validity_rate": json_validity_rate(group),
            "refusal_rate": refusal_rate(group),
            "overgeneration_rate": overgeneration_rate(group),
            "num_examples": float(len(group)),
        }
    return per_model


def compute_per_model_task_breakdown(
    results: list[RunResult],
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute per-model, per-task-family breakdown.

    Outer key: model_id. Inner key: task family. Values: accuracy,
    cost_per_correct_answer, and count.
    """
    by_model: dict[str, list[RunResult]] = defaultdict(list)
    for r in results:
        by_model[r.model_id].append(r)
    breakdown: dict[str, dict[str, dict[str, float]]] = {}
    for model_id, group in by_model.items():
        breakdown[model_id] = task_family_breakdown(group)
    return breakdown
