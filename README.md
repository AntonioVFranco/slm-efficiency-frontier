# SLM Efficiency Frontier

A PyTorch-first benchmark for measuring when small language models beat larger
cheap models in **cost-per-correct-answer** on verifiable agentic tasks.

## Why this benchmark

Model selection today is dominated by quality leaderboards that ignore cost.
For verifiable agentic tasks—tool calling, JSON validity, structured
extraction, short reasoning, clause conflict detection, table reasoning, and
safe action selection—the question that matters is: *which model gives the
most correct answers per dollar?*

This benchmark answers that question with a PyTorch-first evaluation engine,
local PyTorch baselines for calibration, and OpenRouter-hosted models gated by
strict price eligibility.

## Primary metric

`cost_per_correct_answer = total_cost_usd / num_correct`

## Key features

- PyTorch evaluator, dataset loaders, metrics, and baselines.
- Per-model metrics and per-model leaderboard ranking.
- OpenRouter runner with `dry_run` default and pre-flight cost estimation.
- Budget guard with hard cap US$4 (target US$2) and price eligibility gates.
- Per-task validators for automatic scoring.
- Reproducible leaderboard generation with deterministic tie-breakers.
- Strict JSON output (no Infinity/NaN tokens).
- Kaggle-first execution; no local runtime required.
- Hugging Face dataset, Space, and model repos ready for publication.

## v0.1 task families

json_validity, tool_calling, short_reasoning, structured_extraction,
clause_conflict, table_reasoning, semantic_classification, judge_reranker,
safe_action_selection.

Future multilingual extensions are out of scope for v0.1.

## Execution status

- Notebook 01 completed on Kaggle: Hugging Face duplication sweep (13 adjacent
  projects, 0 exact/near duplicates, max risk score 45, decision PROCEED).
- Notebook 02 completed on Kaggle: OpenRouter price verification (65 model
  entries, 21 eligible models, DeepSeek V4 Flash eligible at 0.09/0.18 USD/M).
- Notebook 03 completed on Kaggle: dry-run pipeline validation (8 candidate
  models, 10 examples, 80 synthetic results, strict JSON validated, per-model
  leaderboard generated).
- Notebook 04 completed on Kaggle: real OpenRouter smoke evaluation (20 calls
  attempted, 15 succeeded, 5 failed due to rate limit on a free model, total
  token-based cost US$0.000088, budget stayed below US$0.50 cap).
- Notebook 05 completed on Kaggle: final QA (44 tests passed, 121 QA checks
  passed, 0 BLOCKER, 0 MAJOR, 0 MINOR, 0 NIT).
- Final release gate: READY_FOR_FINAL_PACKAGING.
- Publication status: not published; awaiting explicit user confirmation.

The smoke-run results from notebook 04 are tiny validation results from a
5-example, 4-model evaluation. They confirm that the real call path works
end-to-end. They are not a definitive model leaderboard.

## Execution

All code is designed to run on Kaggle. Do not run it locally.

1. Attach this repository to a Kaggle notebook.
2. Run `notebooks/00_kaggle_environment_check.ipynb` to verify the environment.
3. Run `notebooks/01_kaggle_research_sweep.ipynb` for the HF duplication sweep.
4. Run `notebooks/02_kaggle_openrouter_price_check.ipynb` to verify prices.
5. Run `notebooks/03_kaggle_dry_run_evaluation.ipynb` for a dry run.
6. Run `notebooks/04_kaggle_public_eval_run.ipynb` for a real smoke evaluation.
7. Run `notebooks/05_kaggle_final_qa.ipynb` for final QA.

Kaggle-first runtime policy remains. Dry-run is still the default for safety.
Real OpenRouter evaluation requires Kaggle Secrets for the API key and budget
guard approval before each call.

See `docs/kaggle_execution.md` for details.

## Dry-run behavior

The dry-run mode (default) does not call any remote API and does not require an
eligible OpenRouter model. It uses all configured candidate model ids to
exercise the per-model pipeline, substituting a synthetic in-memory backend
with zero price when a candidate is not eligible. Candidate model identity is
preserved for ranking. Synthetic responses are generated from
`example.expected` to exercise validators. This is pipeline testing behavior,
not model performance.

## Budget policy

See `docs/budget_policy.md` and `configs/budget.yaml`. No model is used unless
its price is verified and below the thresholds (input < US$0.12/M, output <
US$0.30/M).

## License

MIT. See `LICENSE`.
