"""Tool calling validator."""

from __future__ import annotations

import json
from typing import Any


def validate_tool_calling(example: Any, response: str) -> tuple[bool, bool]:
    try:
        parsed = json.loads(response)
    except (json.JSONDecodeError, TypeError):
        return False, False
    expected = example.expected
    tool_ok = parsed.get("tool") == expected.get("tool")
    args_ok = parsed.get("arguments") == expected.get("arguments")
    return bool(tool_ok and args_ok), True
