"""Per-task validators for automatic scoring."""

from .json_validity import validate_json_validity
from .tool_calling import validate_tool_calling
from .exact_match import validate_exact_match
from .clause_conflict import validate_clause_conflict
from .table_reasoning import validate_table_reasoning

VALIDATORS = {
    "json_validity": validate_json_validity,
    "tool_calling": validate_tool_calling,
    "exact_match": validate_exact_match,
    "clause_conflict": validate_clause_conflict,
    "table_reasoning": validate_table_reasoning,
}

__all__ = ["VALIDATORS"]
