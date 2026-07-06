"""Exact match validator for short reasoning, extraction, classification, safe action."""

from __future__ import annotations

from typing import Any


def validate_exact_match(example: Any, response: str) -> tuple[bool, bool]:
    expected = example.expected
    target = None
    if "answer" in expected:
        target = expected["answer"]
    elif "label" in expected:
        target = expected["label"]
    elif "action" in expected:
        target = expected["action"]
    if target is None:
        return False, True
    return response.strip() == str(target).strip(), True
