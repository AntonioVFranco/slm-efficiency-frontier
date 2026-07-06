# Benchmark Design

## Theme
SLM Efficiency Frontier: A PyTorch-first benchmark for measuring when small
language models beat larger cheap models in cost-per-correct-answer on
verifiable agentic tasks.

## Primary metric
cost_per_correct_answer = total_cost_usd / num_correct

## Task families (v0.1)
json_validity, tool_calling, short_reasoning, structured_extraction,
clause_conflict, table_reasoning, semantic_classification, judge_reranker,
safe_action_selection.

Future multilingual extensions are out of scope for v0.1.

## Evaluator
PyTorch-first engine with deterministic seeding, tensor-based metric
aggregation, per-model metrics, and per-task validators.

## Baselines
PyTorch tiny classifier, cross-encoder, and reranker provide local calibration
and auxiliary reference points.
