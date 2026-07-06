"""Budget guard and cost ledger for OpenRouter spend control."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import BudgetConfig, ModelEntry


@dataclass
class LedgerEntry:
    model_id: str
    example_id: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    meta: dict[str, Any] = field(default_factory=dict)


class CostLedger:
    def __init__(self) -> None:
        self.entries: list[LedgerEntry] = []

    def record(self, entry: LedgerEntry) -> None:
        self.entries.append(entry)

    @property
    def total(self) -> float:
        return sum(e.cost_usd for e in self.entries)


class BudgetGuard:
    def __init__(self, config: BudgetConfig, ledger: CostLedger, models: list[ModelEntry]) -> None:
        self.config = config
        self.ledger = ledger
        self._models = {m.model_id: m for m in models}

    def is_eligible(self, model_id: str) -> bool:
        model = self._models.get(model_id)
        if model is None:
            return False
        if not model.eligible:
            return False
        if model.input_price_per_million is None or model.output_price_per_million is None:
            return False
        if model.input_price_per_million >= self.config.max_input_price_per_million_usd:
            return False
        if model.output_price_per_million >= self.config.max_output_price_per_million_usd:
            return False
        return True

    def estimate_call(
        self, model_id: str, input_tokens: int, max_output_tokens: int
    ) -> float:
        model = self._models.get(model_id)
        if model is None or model.input_price_per_million is None:
            if self.config.stop_on_unknown_price:
                raise ValueError(f"Unknown price for model {model_id}")
            return 0.0
        out_price = model.output_price_per_million or 0.0
        cost = (
            input_tokens * model.input_price_per_million
            + max_output_tokens * out_price
        ) / 1_000_000.0
        return cost

    def estimate_battery(
        self, model_ids: list[str], input_tokens_per_example: int, max_output_tokens: int, n_examples: int
    ) -> float:
        total = 0.0
        for mid in model_ids:
            total += self.estimate_call(mid, input_tokens_per_example, max_output_tokens) * n_examples
        return total

    def can_spend(self, amount_usd: float) -> bool:
        if self.config.stop_on_budget_exceeded:
            return (self.ledger.total + amount_usd) <= self.config.max_total_usd
        return True

    def remaining_usd(self) -> float:
        return max(0.0, self.config.max_total_usd - self.ledger.total)
