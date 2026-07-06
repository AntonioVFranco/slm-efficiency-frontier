"""Data schemas for dataset examples and run results.

The v0.1 release is strictly English-only. Example.validate() accepts only
language == "en". Future multilingual extensions are out of scope for v0.1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Example:
    id: str
    task: str
    language: str
    prompt: str
    expected: dict[str, Any]
    validator: str
    max_output_tokens: int = 256

    def validate(self) -> bool:
        if not self.id or not self.task or not self.prompt:
            return False
        # v0.1 is strictly English-only.
        if self.language != "en":
            return False
        if not isinstance(self.expected, dict):
            return False
        if not self.validator:
            return False
        return True


@dataclass
class RunResult:
    # The candidate model being evaluated/ranked. In dry-run this is the
    # requested model id even when a synthetic backend is used.
    model_id: str
    example_id: str
    response: str
    task: str = ""
    # The backend that actually produced the response. In dry-run with no
    # eligible remote model this is "local/dry-run-model". In real mode this
    # is normally equal to model_id.
    execution_model_id: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    correct: bool = False
    valid: bool = False
    refused: bool = False
    overgenerated: bool = False
    # Release-schema fields for raw results JSONL persistence.
    # raw_output is the raw model response (same content as response, kept
    # for the release schema name). normalized_prediction is a stripped or
    # normalized version of the response used by validators. validator_name
    # records which validator scored this result. error is non-empty when
    # the call failed.
    raw_output: str = ""
    normalized_prediction: str = ""
    validator_name: str = ""
    error: str = ""

    def to_raw_result_dict(self) -> dict[str, Any]:
        """Return a dict matching the release raw-results JSONL schema."""
        return {
            "example_id": self.example_id,
            "task": self.task,
            "model_id": self.model_id,
            "execution_model_id": self.execution_model_id,
            "prompt_token_count": self.input_tokens,
            "completion_token_count": self.output_tokens,
            "estimated_cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "raw_output": self.raw_output or self.response,
            "normalized_prediction": self.normalized_prediction or self.response.strip(),
            "correct": self.correct,
            "validator_name": self.validator_name,
            "error": self.error,
        }


@dataclass
class EvaluationReport:
    run_id: str
    created_at: str
    dry_run: bool
    models: list[str]
    metrics: dict[str, Any] = field(default_factory=dict)
    per_model_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    task_family_breakdown: dict[str, Any] = field(default_factory=dict)
    per_model_task_breakdown: dict[str, dict[str, Any]] = field(default_factory=dict)
    total_cost_usd: float = 0.0
    budget: dict[str, Any] = field(default_factory=dict)
    # Raw run results (one entry per model x example call).
    results: list[RunResult] = field(default_factory=list)
    # Serializable cost ledger entries (as dicts, not LedgerEntry objects).
    cost_ledger: list[dict[str, Any]] = field(default_factory=list)
