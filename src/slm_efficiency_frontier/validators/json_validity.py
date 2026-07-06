"""JSON validity validator."""

from __future__ import annotations

import json
from typing import Any


def validate_json_validity(example: Any, response: str) -> tuple[bool, bool]:
    try:
        parsed = json.loads(response)
    except (json.JSONDecodeError, TypeError):
        return False, False
    schema = example.expected.get("schema")
    if schema and isinstance(schema, dict):
        for key in schema:
            if key not in parsed:
                return False, True
    return True, True
