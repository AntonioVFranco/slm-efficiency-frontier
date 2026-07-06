"""Build a leaderboard from an evaluation report. Run on Kaggle only.

Produces one row per model using per_model_metrics. Ranks models by
cost_per_correct_answer (ascending). Tie-breakers, in order:
    1. higher accuracy
    2. lower median latency
    3. lower tokens per correct answer
    4. alphabetical model_id

Models with infinite/NaN or unavailable cost_per_correct_answer are placed
below all models with a finite cost, and their cost is serialized as None for
strict JSON safety.
"""

from __future__ import annotations

import argparse
import math
from typing import Any

from slm_efficiency_frontier.json_utils import dump_json, sanitize_for_json


def _safe_float(value: Any, default: float = math.inf) -> float:
    if value is None:
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(f):
        return math.inf
    return f


def _sort_key(model_id: str, metrics: dict[str, Any]) -> tuple:
    cost = _safe_float(metrics.get("cost_per_correct_answer"))
    accuracy = _safe_float(metrics.get("accuracy"), default=0.0)
    latency = _safe_float(metrics.get("median_latency_ms"), default=math.inf)
    tokens = _safe_float(metrics.get("tokens_per_correct_answer"), default=math.inf)
    # Finite costs rank above infinite costs (False < True, so is_inf sorts last).
    is_inf = math.isinf(cost)
    # Lower cost first; for tie-breakers: higher accuracy (negate),
    # lower latency, lower tokens, alphabetical model_id.
    return (is_inf, cost, -accuracy, latency, tokens, model_id)


def build_leaderboard(report: dict) -> list[dict]:
    per_model: dict[str, dict[str, Any]] = report.get("per_model_metrics", {})
    if not per_model:
        return []
    rows: list[dict] = []
    for model_id, metrics in per_model.items():
        rows.append({
            "model_id": model_id,
            "accuracy": metrics.get("accuracy"),
            "cost_per_correct_answer": metrics.get("cost_per_correct_answer"),
            "cost_per_1k_examples": metrics.get("cost_per_1k_examples"),
            "median_latency_ms": metrics.get("median_latency_ms"),
            "tokens_per_correct_answer": metrics.get("tokens_per_correct_answer"),
            "json_validity_rate": metrics.get("json_validity_rate"),
            "refusal_rate": metrics.get("refusal_rate"),
            "overgeneration_rate": metrics.get("overgeneration_rate"),
            "num_examples": metrics.get("num_examples"),
        })
    rows.sort(key=lambda row: _sort_key(row["model_id"], row))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", default="/kaggle/working/leaderboard.json")
    args = parser.parse_args()
    import json

    with open(args.report, "r", encoding="utf-8") as fh:
        report = json.load(fh)
    leaderboard = build_leaderboard(report)
    dump_json(leaderboard, args.output, indent=2)
    print(f"Leaderboard written to {args.output} ({len(leaderboard)} rows)")


if __name__ == "__main__":
    main()
