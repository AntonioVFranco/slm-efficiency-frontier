"""Strict, JSON-safe serialization helpers.

Metrics such as cost_per_correct_answer and tokens_per_correct_answer can
return float("inf") when a model has zero correct answers. Standard
json.dump emits Infinity/-Infinity/NaN tokens, which are not strict JSON and
break Hugging Face Spaces, dataset parsers, and JSON validators.

sanitize_for_json() recursively converts non-finite floats to None and
flattens dataclasses to dicts. Use it before every json.dump and always pass
allow_nan=False.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any


def sanitize_for_json(value: Any) -> Any:
    """Recursively convert a value to a strict-JSON-safe structure.

    - math.inf, -math.inf, and NaN become None.
    - dicts, lists, and tuples are processed recursively (tuples become lists).
    - dataclasses are converted to dicts via asdict then sanitized.
    - strings, ints, finite floats, bools, and None are preserved.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, int):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return sanitize_for_json(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(v) for v in value]
    # Fallback: stringify unknown types to avoid non-serializable objects.
    return str(value)


def dump_json(obj: Any, path: str, *, indent: int = 2) -> None:
    """Sanitize obj and write strict JSON to path with allow_nan=False."""
    import json

    safe = sanitize_for_json(obj)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(safe, fh, indent=indent, allow_nan=False)
