"""English-only artifact test. Run on Kaggle only.

Wraps scripts/check_english_only.py to scan the GitHub-ready and Hugging
Face-ready artifacts for non-English text leakage, and to verify the scanner
avoids false positives while still detecting real non-English content.

Test fixtures use Unicode escape sequences and generated character sets so
that no readable non-English natural language appears in this source file.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_SCRIPT = ROOT / "scripts" / "check_english_only.py"


def _load_scan_module():
    spec = importlib.util.spec_from_file_location("check_english_only", SCAN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_english_only"] = module
    spec.loader.exec_module(module)
    return module


def test_no_non_english_in_github_artifacts():
    module = _load_scan_module()
    findings = module.scan_tree(ROOT)
    assert not findings, f"Non-English leakage detected: {findings}"


def test_no_non_english_in_hf_artifacts():
    hf_root = ROOT.parent / "hugging_face"
    if not hf_root.exists():
        return
    module = _load_scan_module()
    findings = module.scan_tree(hf_root)
    assert not findings, f"Non-English leakage detected in HF artifacts: {findings}"


def test_scanner_excludes_itself():
    """The scanner must not flag its own detection data as violations."""
    module = _load_scan_module()
    self_findings = module.scan_file(SCAN_SCRIPT)
    assert self_findings == [], f"Scanner flags itself: {self_findings}"


def test_scanner_excludes_test_wrapper(tmp_path):
    """The test wrapper path must also be excluded."""
    module = _load_scan_module()
    wrapper = tmp_path / "tests" / "test_english_only.py"
    wrapper.parent.mkdir(parents=True)
    # Use Unicode escapes to avoid embedding readable non-English text.
    wrapper.write_text("x = \"\\u00e0\\u00e1\\u00e2\"\n", encoding="utf-8")
    assert module.scan_file(wrapper) == []


def test_scanner_does_not_flag_example_dot_com(tmp_path):
    module = _load_scan_module()
    f = tmp_path / "ok.py"
    f.write_text('url = "https://example.com/path"\n', encoding="utf-8")
    assert module.scan_file(f) == []


def test_scanner_does_not_flag_english_capital_phrase(tmp_path):
    module = _load_scan_module()
    f = tmp_path / "ok.md"
    f.write_text(
        "The capital budget and capital allocation are English finance terms.\n",
        encoding="utf-8",
    )
    assert module.scan_file(f) == []


def test_scanner_does_not_flag_country_city_names(tmp_path):
    module = _load_scan_module()
    f = tmp_path / "ok.jsonl"
    f.write_text(
        '{"prompt":"What is the capital of Portugal? Answer Lisbon.","language":"en"}\n',
        encoding="utf-8",
    )
    assert module.scan_file(f) == []


def test_scanner_flags_non_english_accented_chars(tmp_path):
    """Scanner must detect accented characters typical of non-English scripts."""
    module = _load_scan_module()
    f = tmp_path / "bad.md"
    # Generate accented chars from Unicode codepoints to avoid readable text.
    chars = "".join(chr(c) for c in range(0xE0, 0xF0))
    f.write_text(f"content = {chars}\n", encoding="utf-8")
    findings = module.scan_file(f)
    assert findings, "Scanner failed to detect accented characters"


def test_scanner_flags_non_english_language_tag(tmp_path):
    """Scanner must detect non-English language tags in JSON files."""
    module = _load_scan_module()
    f = tmp_path / "bad.jsonl"
    f.write_text(
        '{"id":"e1","language":"xx","prompt":"test"}\n',
        encoding="utf-8",
    )
    findings = module.scan_file(f)
    assert findings, "Scanner failed to detect non-English language tag"
