"""Tests for validators. Run on Kaggle only."""

from slm_efficiency_frontier.validators import (
    validate_json_validity, validate_tool_calling, validate_exact_match,
    validate_clause_conflict, validate_table_reasoning,
)


class Ex:
    def __init__(self, expected):
        self.expected = expected


def test_json_validity():
    ex = Ex({"schema": {"name": "string"}})
    assert validate_json_validity(ex, '{"name":"Ana"}')[0] is True
    assert validate_json_validity(ex, 'not json')[0] is False


def test_tool_calling():
    ex = Ex({"tool": "get_weather", "arguments": {"city": "Lisbon"}})
    assert validate_tool_calling(ex, '{"tool":"get_weather","arguments":{"city":"Lisbon"}}')[0] is True
    assert validate_tool_calling(ex, '{"tool":"wrong","arguments":{}}')[0] is False


def test_exact_match():
    ex = Ex({"answer": "A"})
    assert validate_exact_match(ex, "A")[0] is True
    assert validate_exact_match(ex, "B")[0] is False


def test_clause_conflict():
    ex = Ex({"conflict": True})
    assert validate_clause_conflict(ex, "true")[0] is True
    assert validate_clause_conflict(ex, "false")[0] is False


def test_table_reasoning():
    ex = Ex({"answer": "B"})
    assert validate_table_reasoning(ex, "B")[0] is True
