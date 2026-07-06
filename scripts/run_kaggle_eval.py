"""Run a Kaggle evaluation battery. Run on Kaggle only.

Usage (Kaggle):
    python scripts/run_kaggle_eval.py --config configs --dry-run

Dry-run mode (default):
- Uses all configured model ids from model_pool.yaml as candidates to exercise
  the per-model pipeline, even when no model is eligible for a real run.
- Uses the synthetic local backend when a candidate is not eligible, so no API
  call is made and no money is spent.
- If no models are configured at all, falls back to the synthetic backend as
  the single candidate.
- Output is strict JSON (no Infinity/NaN tokens).

Real mode (dry_run=False):
- Requires at least one OpenRouter model with a verified eligible price.
- Requires the API key from Kaggle Secrets.
- Fails before any call if no eligible verified model exists.
- Uses _real_call() to make actual OpenRouter Chat Completions requests.

All output files use strict JSON (allow_nan=False). Raw results are persisted
as JSONL (one line per call) and the cost ledger as a separate JSON file.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from slm_efficiency_frontier.config import load_config
from slm_efficiency_frontier.budget import BudgetGuard, CostLedger
from slm_efficiency_frontier.datasets import BenchmarkDataset
from slm_efficiency_frontier.evaluator import Evaluator
from slm_efficiency_frontier.json_utils import dump_json, sanitize_for_json
from slm_efficiency_frontier.openrouter import OpenRouterRunner, DRY_RUN_BACKEND
from slm_efficiency_frontier.validators import VALIDATORS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs")
    parser.add_argument("--data", default="data/sample/examples.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="/kaggle/working/evaluation_report.json")
    parser.add_argument("--leaderboard", default="/kaggle/working/leaderboard.json")
    parser.add_argument("--raw-results", default="/kaggle/working/raw_results.jsonl")
    parser.add_argument("--cost-ledger", default="/kaggle/working/cost_ledger.json")
    args = parser.parse_args()

    config = load_config(args.config)
    dry_run = args.dry_run or config.budget.dry_run_default
    api_key = None
    if not dry_run:
        import os

        api_key = os.environ.get("OPENROUTER_API_KEY")

    ledger = CostLedger()
    guard = BudgetGuard(config.budget, ledger, config.models)
    configured_ids = [m.model_id for m in config.models]
    eligible_ids = [mid for mid in configured_ids if guard.is_eligible(mid)]

    if dry_run:
        candidates = configured_ids if configured_ids else [DRY_RUN_BACKEND]
        print(f"Dry run with candidate models: {candidates}")
    else:
        if not eligible_ids:
            print(
                "ERROR: no OpenRouter models with a verified eligible price. "
                "Refusing to run without dry_run. No remote call will be made.",
                file=sys.stderr,
            )
            sys.exit(2)
        if not api_key:
            print("ERROR: OPENROUTER_API_KEY is not set for a real run.", file=sys.stderr)
            sys.exit(2)
        candidates = eligible_ids
        print(f"Real run with eligible models: {candidates}")

    dataset = BenchmarkDataset(args.data)
    runner = OpenRouterRunner(api_key, guard, ledger, dry_run=dry_run)
    evaluator = Evaluator(runner, VALIDATORS)
    report = evaluator.evaluate(candidates, dataset.examples)

    # Persist the full report (including raw results and cost ledger).
    report_dict = dataclasses.asdict(report)
    dump_json(report_dict, args.output, indent=2)
    print(f"Report written to {args.output}")
    print(f"Dry run: {report.dry_run}")
    print(f"Models: {report.models}")
    print(f"Total cost (USD): {report.total_cost_usd}")
    print(f"Results count: {len(report.results)}")
    print(f"Cost ledger entries: {len(report.cost_ledger)}")

    # Persist raw results as JSONL (one line per call).
    raw_path = Path(args.raw_results)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "w", encoding="utf-8") as fh:
        for result in report.results:
            line = json.dumps(sanitize_for_json(result.to_raw_result_dict()), allow_nan=False)
            fh.write(line + "\n")
    print(f"Raw results written to {raw_path} ({len(report.results)} lines)")

    # Persist cost ledger as JSON.
    dump_json(report.cost_ledger, args.cost_ledger, indent=2)
    print(f"Cost ledger written to {args.cost_ledger} ({len(report.cost_ledger)} entries)")


if __name__ == "__main__":
    main()
