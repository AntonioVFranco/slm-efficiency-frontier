# Sample Dataset

This is a tiny synthetic sample dataset demonstrating the schema and validators.
It is not the full benchmark. All content is synthetic and free of copyright
restrictions.

## Format
JSONL, one Example per line. See `src/slm_efficiency_frontier/schemas.py` for
the schema and `data_schema.md` for per-task expected shapes.

## Task families present (v0.1)
json_validity, tool_calling, short_reasoning, structured_extraction,
clause_conflict, table_reasoning, semantic_classification,
safe_action_selection, judge_reranker.

## Language
All v0.1 examples use `language: "en"`. The v0.1 release is strictly
English-only. Future multilingual extensions are out of scope for v0.1.

## Loading
Use `BenchmarkDataset` from `datasets.py`. Run loading on Kaggle only.
