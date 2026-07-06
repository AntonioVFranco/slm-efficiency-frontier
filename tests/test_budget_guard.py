"""Tests for budget guard. Run on Kaggle only."""

import pytest

from slm_efficiency_frontier.config import BudgetConfig, ModelEntry
from slm_efficiency_frontier.budget import BudgetGuard, CostLedger, LedgerEntry


def make_guard(max_total=0.05):
    cfg = BudgetConfig(max_total_usd=max_total)
    ledger = CostLedger()
    models = [ModelEntry("m1", "p", 8000, 0.01, 0.05, "url", "2026-01-01", True, "ok")]
    return BudgetGuard(cfg, ledger, models), ledger


def test_eligible():
    guard, _ = make_guard()
    assert guard.is_eligible("m1")


def test_ineligible_unknown_price():
    cfg = BudgetConfig()
    ledger = CostLedger()
    models = [ModelEntry("m2", "p", 8000, None, None, None, None, False, "price not verified")]
    guard = BudgetGuard(cfg, ledger, models)
    assert not guard.is_eligible("m2")


def test_budget_exceeded_stops():
    guard, ledger = make_guard(max_total=0.001)
    ledger.record(LedgerEntry("m1", "e", 10, 5, 0.001))
    assert not guard.can_spend(0.001)
