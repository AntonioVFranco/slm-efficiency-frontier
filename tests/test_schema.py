"""Tests for schemas. Run on Kaggle only.

The v0.1 release is strictly English-only. Example.validate() accepts only
language == "en". Future multilingual extensions are out of scope for v0.1.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from slm_efficiency_frontier.schemas import Example


def test_valid_english_example():
    ex = Example("e1", "json_validity", "en", "prompt", {"schema": {}}, "json_validity")
    assert ex.validate()


def test_non_english_language_rejected():
    ex = Example("e1", "json_validity", "fr", "prompt", {"schema": {}}, "json_validity")
    assert not ex.validate()


def test_empty_prompt_rejected():
    ex = Example("e1", "json_validity", "en", "", {"schema": {}}, "json_validity")
    assert not ex.validate()


def test_missing_validator_rejected():
    ex = Example("e1", "json_validity", "en", "prompt", {"schema": {}}, "")
    assert not ex.validate()


def test_v0_1_sample_dataset_is_english_only():
    """All sample dataset examples must use language 'en' in v0.1."""
    data_path = Path(__file__).resolve().parents[1] / "data" / "sample" / "examples.jsonl"
    languages = set()
    with open(data_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            languages.add(record["language"])
    assert languages == {"en"}, f"v0.1 sample must be English-only, got {languages}"


def test_v0_1_task_configs_have_no_multilingual_families():
    """v0.1 active task families must not include any multilingual family.

    A multilingual family id is detected by substrings commonly used for
    language-specific tasks, without naming any particular language.
    """
    config_path = Path(__file__).resolve().parents[1] / "configs" / "tasks.yaml"
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    active_ids = {t["id"] for t in data.get("task_families", [])}
    forbidden_substrings = ("multilingual", "ml_", "i18n", "cross_lingual", "non_english")
    offenders = [i for i in active_ids if any(s in i.lower() for s in forbidden_substrings)]
    assert not offenders, f"v0.1 must not include multilingual families: {offenders}"
