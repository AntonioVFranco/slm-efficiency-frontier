"""Configuration loading for the benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BudgetConfig:
    max_total_usd: float = 4.0
    target_total_usd: float = 2.0
    max_input_price_per_million_usd: float = 0.12
    max_output_price_per_million_usd: float = 0.30
    dry_run_default: bool = True
    stop_on_unknown_price: bool = True
    stop_on_budget_exceeded: bool = True
    require_price_verification: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BudgetConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class ModelEntry:
    model_id: str
    provider: str
    context_window: int
    input_price_per_million: float | None
    output_price_per_million: float | None
    price_source_url: str | None
    price_checked_at: str | None
    eligible: bool = False
    eligibility_reason: str = "price not verified"
    notes: str = ""


@dataclass
class Config:
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    models: list[ModelEntry] = field(default_factory=list)
    tasks: dict[str, Any] = field(default_factory=dict)
    kaggle: dict[str, Any] = field(default_factory=dict)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_config(config_dir: str | Path) -> Config:
    config_dir = Path(config_dir)
    budget = BudgetConfig.from_dict(load_yaml(config_dir / "budget.yaml"))
    model_pool = load_yaml(config_dir / "model_pool.yaml")
    models = [ModelEntry(**m) for m in model_pool.get("models", [])]
    tasks = load_yaml(config_dir / "tasks.yaml")
    kaggle = load_yaml(config_dir / "kaggle.yaml")
    return Config(budget=budget, models=models, tasks=tasks, kaggle=kaggle)
