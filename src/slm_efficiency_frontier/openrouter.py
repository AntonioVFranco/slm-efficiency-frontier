"""OpenRouter runner with dry_run support and budget integration.

The runner supports a synthetic dry-run mode that does not call any remote API
and does not require an eligible OpenRouter model. In dry-run mode a temporary
in-memory local backend with zero price is used so the pipeline can be
exercised end-to-end on Kaggle without spending money.

Model identity is preserved: RunResult.model_id is always the requested
candidate model id used for ranking. RunResult.execution_model_id records the
backend that produced the response ("local/dry-run-model" in synthetic mode,
or the real model id in real mode).

Real OpenRouter execution is implemented via _real_call() using the official
OpenRouter Chat Completions endpoint with temperature 0 and non-streaming
requests. The API key is loaded from Kaggle Secrets and is never written to
disk or logged. Any real call path requires dry_run=False, a verified eligible
model, the API key from Kaggle Secrets, and budget guard approval before the
call."""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
import time
from typing import Any

from .budget import BudgetGuard, CostLedger, LedgerEntry
from .schemas import Example, RunResult

# A synthetic in-memory backend used only in dry-run mode. It is never treated
# as a real eligible OpenRouter model.
DRY_RUN_BACKEND = "local/dry-run-model"


class OpenRouterRunner:
    def __init__(
        self,
        api_key: str | None,
        budget_guard: BudgetGuard,
        ledger: CostLedger,
        dry_run: bool = True,
        base_url: str = "https://openrouter.ai/api/v1",
    ) -> None:
        self.api_key = api_key
        self.budget_guard = budget_guard
        self.ledger = ledger
        self.dry_run = dry_run
        self.base_url = base_url

    def resolve_backend(self, model_id: str) -> str:
        """Return the backend id that will produce the response.

        In dry-run mode, if the requested model is not eligible (or no model is
        eligible), substitute the synthetic local dry-run backend. In real
        mode, fail loudly if the model is not eligible with a verified price.
        The candidate model_id is always preserved on the RunResult regardless
        of the backend chosen here.
        """
        if self.dry_run:
            if not self.budget_guard.is_eligible(model_id):
                return DRY_RUN_BACKEND
            return model_id
        if not self.budget_guard.is_eligible(model_id):
            raise RuntimeError(
                f"Model {model_id} is not eligible for a real run: "
                "price is unverified or above thresholds. "
                "No remote call will be made."
            )
        return model_id

    def run_example(self, model_id: str, example: Example) -> RunResult:
        backend = self.resolve_backend(model_id)
        input_tokens = len(example.prompt.split())
        max_out = example.max_output_tokens

        if backend == DRY_RUN_BACKEND:
            estimated = 0.0
        else:
            estimated = self.budget_guard.estimate_call(backend, input_tokens, max_out)
            if not self.budget_guard.can_spend(estimated):
                raise RuntimeError("Budget exceeded: stopping before call.")

        start = time.monotonic()
        if self.dry_run:
            response = self._dry_run_response(example)
            output_tokens = max(1, max_out // 4)
            cost = 0.0 if backend == DRY_RUN_BACKEND else estimated
            latency = 10.0
        else:
            response, api_input_tokens, output_tokens, latency = self._real_call(backend, example)
            input_tokens = api_input_tokens if api_input_tokens > 0 else input_tokens
            cost = self.budget_guard.estimate_call(backend, input_tokens, output_tokens)

        self.ledger.record(
            LedgerEntry(
                backend,
                example.id,
                input_tokens,
                output_tokens,
                cost,
                meta={"requested_model_id": model_id, "dry_run": self.dry_run},
            )
        )
        return RunResult(
            model_id=model_id,
            example_id=example.id,
            response=response,
            task=example.task,
            execution_model_id=backend,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency,
            cost_usd=cost,
        )

    def _dry_run_response(self, example: Example) -> str:
        """Generate a synthetic valid response from example.expected.

        This is pipeline testing behavior only. It is NOT model performance.
        The synthetic response is constructed so validators can exercise the
        scoring path during a dry run without any API call.
        """
        expected = example.expected
        task = example.task
        if task == "json_validity":
            schema = expected.get("schema", {})
            obj: dict[str, Any] = {}
            for key, value in schema.items():
                obj[key] = "" if isinstance(value, str) and value == "string" else 0
            if not obj:
                obj = {"key": "value"}
            return json.dumps(obj)
        if task == "tool_calling":
            return json.dumps({
                "tool": expected.get("tool", ""),
                "arguments": expected.get("arguments", {}),
            })
        if task == "clause_conflict":
            return "true" if expected.get("conflict") else "false"
        if "answer" in expected:
            return str(expected["answer"])
        if "label" in expected:
            return str(expected["label"])
        if "action" in expected:
            return str(expected["action"])
        return ""

    def _real_call(self, model_id: str, example: Example) -> tuple[str, int, int, float]:
        """Make a real OpenRouter Chat Completions call.

        Requirements enforced by the caller (run_example):
        - dry_run must be False.
        - model_id must be eligible with a verified price.
        - BudgetGuard must approve the estimated cost before this call.

        This method:
        - Uses temperature 0 and non-streaming requests.
        - Uses max_tokens from the example (capped at 128).
        - Does not use provider fallback.
        - Retries at most once for transient network/429 errors.
        - Never logs or prints the API key.
        """
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is missing. Cannot make a real call. "
                "Load it from Kaggle Secrets before running."
            )
        if self.dry_run:
            raise RuntimeError("_real_call must not be called in dry-run mode.")

        max_tokens = min(example.max_output_tokens, 128)
        payload = {
            "model": model_id,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a precise assistant. Answer with only the "
                        "requested output. Do not add extra prose or "
                        "explanations."
                    ),
                },
                {"role": "user", "content": example.prompt},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
        }
        body = json.dumps(payload).encode("utf-8")
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://kaggle.com",
            "X-Title": "slm-efficiency-frontier-benchmark",
        }

        max_attempts = 2
        last_error = None
        for attempt in range(1, max_attempts + 1):
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            start = time.monotonic()
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    raw = resp.read().decode("utf-8")
                    latency = (time.monotonic() - start) * 1000.0
                    data = json.loads(raw)
                    break
            except urllib.error.HTTPError as exc:
                latency = (time.monotonic() - start) * 1000.0
                code = exc.code
                err_body = ""
                try:
                    err_body = exc.read().decode("utf-8", errors="replace")[:500]
                except Exception:
                    pass
                # Retry only on 429 or 5xx transient errors.
                if code == 429 or 500 <= code < 600:
                    last_error = f"HTTP {code}: {err_body}"
                    if attempt < max_attempts:
                        time.sleep(2.0 * attempt)
                        continue
                # Non-retryable error: surface immediately.
                raise RuntimeError(
                    f"OpenRouter call failed (HTTP {code}) for {model_id}: {err_body}"
                ) from exc
            except urllib.error.URLError as exc:
                latency = (time.monotonic() - start) * 1000.0
                last_error = str(exc)
                if attempt < max_attempts:
                    time.sleep(2.0 * attempt)
                    continue
                raise RuntimeError(
                    f"OpenRouter network error for {model_id}: {exc}"
                ) from exc
        else:
            raise RuntimeError(
                f"OpenRouter call failed after {max_attempts} attempts for "
                f"{model_id}: {last_error}"
            )

        choices = data.get("choices", [])
        text = ""
        if choices:
            msg = choices[0].get("message", {})
            text = msg.get("content", "") or ""
        usage = data.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        output_tokens = completion_tokens if completion_tokens > 0 else max(1, max_tokens // 4)
        return text, prompt_tokens, output_tokens, latency

    def run_battery(self, model_ids: list[str], examples: list[Example]) -> list[RunResult]:
        results: list[RunResult] = []
        for mid in model_ids:
            for ex in examples:
                results.append(self.run_example(mid, ex))
        return results
