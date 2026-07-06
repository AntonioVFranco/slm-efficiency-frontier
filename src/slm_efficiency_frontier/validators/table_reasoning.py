"""Table reasoning validator (exact answer match)."""

from __future__ import annotations

from typing import Any


def validate_table_reasoning(example: Any, response: str) -> tuple[bool, bool]:
    target = example.expected.get("answer")
    if target is None:
        return False, True
    return response.strip() == str(target).strip(), True
