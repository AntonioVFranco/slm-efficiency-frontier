"""PyTorch-first evaluation engine."""

from __future__ import annotations

import random
import dataclasses
from datetime import datetime, timezone
from typing import Any

import torch

from .metrics import (
    compute_metrics,
    compute_per_model_metrics,
    compute_per_model_task_breakdown,
    task_family_breakdown,
)
from .openrouter import OpenRouterRunner
from .schemas import EvaluationReport, Example, RunResult


class Evaluator:
    def __init__(
        self,
        runner: OpenRouterRunner,
        validators: dict[str, Any],
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        self.runner = runner
        self.validators = validators
        self.device = device
        self.seed = seed

    def evaluate(self, model_ids: list[str], examples: list[Example]) -> EvaluationReport:
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        results: list[RunResult] = []
        for mid in model_ids:
            for ex in examples:
                try:
                    result = self.runner.run_example(mid, ex)
                except Exception as exc:
                    result = RunResult(
                        model_id=mid,
                        example_id=ex.id,
                        response="",
                        task=ex.task,
                        execution_model_id=mid,
                        error=str(exc)[:300],
                    )
                validator = self.validators.get(ex.validator)
                if validator is not None:
                    result.correct, result.valid = validator(ex, result.response)
                    result.validator_name = ex.validator
                result.raw_output = result.response
                result.normalized_prediction = result.response.strip()
                results.append(result)

        global_metrics = compute_metrics(results)
        per_model = compute_per_model_metrics(results)
        per_model_task = compute_per_model_task_breakdown(results)
        breakdown = task_family_breakdown(results)
        total_cost = float(sum(r.cost_usd for r in results))

        # The list of models that actually produced results, preserving order.
        seen: list[str] = []
        for r in results:
            if r.model_id not in seen:
                seen.append(r.model_id)

        ledger_entries = [
            dataclasses.asdict(e) for e in self.runner.ledger.entries
        ]

        report = EvaluationReport(
            run_id=str(random.randint(100000, 999999)),
            created_at=datetime.now(timezone.utc).isoformat(),
            dry_run=self.runner.dry_run,
            models=seen,
            metrics=global_metrics,
            per_model_metrics=per_model,
            task_family_breakdown=breakdown,
            per_model_task_breakdown=per_model_task,
            total_cost_usd=total_cost,
            results=results,
            cost_ledger=ledger_entries,
        )
        report.budget = {
            "max_total_usd": self.runner.budget_guard.config.max_total_usd,
            "target_total_usd": self.runner.budget_guard.config.target_total_usd,
            "dry_run_default": self.runner.budget_guard.config.dry_run_default,
            "remaining_usd": self.runner.budget_guard.remaining_usd(),
            "ledger_total": self.runner.ledger.total,
        }
        return report
