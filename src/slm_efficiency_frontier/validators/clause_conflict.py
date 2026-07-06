"""Clause conflict validator."""

from __future__ import annotations

from typing import Any


def validate_clause_conflict(example: Any, response: str) -> tuple[bool, bool]:
    expected = example.expected.get("conflict")
    if expected is None:
        return False, True
    text = response.strip().lower()
    predicted = None
    if "true" in text or "yes" in text:
        predicted = True
    elif "false" in text or "no" in text:
        predicted = False
    if predicted is None:
        return False, False
    return predicted == expected, True
