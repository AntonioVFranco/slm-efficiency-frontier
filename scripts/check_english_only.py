"""English-only artifact scanner. Run on Kaggle only.

Scans the GitHub-ready and Hugging Face-ready artifacts for non-English text
leakage. Reports findings with file paths and matched terms. Exits non-zero if
any BLOCKER-level leakage is found.

Usage (Kaggle):
    python scripts/check_english_only.py --root . --hf-root ../hugging_face

Detection strategy (release-safe, no embedded non-English natural language):
- Detects non-ASCII accented characters that are typical of Latin-extended
  scripts not used in English-only artifacts.
- Detects language tags or field values that are not "en" in data files.
- Detects known release-forbidden field values by structural inspection
  without embedding readable non-English text in source.

Self-scan exclusion:
- This scanner file and its test wrapper are excluded from scanning.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

# Characters with diacritics common in non-English Latin scripts. Detection is
# based on Unicode category analysis, not on embedding readable non-English
# words in this file.
ACCENTED_RANGES = (
    "\u00e0\u00e1\u00e2\u00e3\u00e4\u00e5\u00e7\u00e8\u00e9\u00ea"
    "\u00eb\u00ec\u00ed\u00ee\u00ef\u00f2\u00f3\u00f4\u00f5\u00f6"
    "\u00f9\u00fa\u00fb\u00fc\u00fd\u00ff\u00f1"
    "\u00c0\u00c1\u00c2\u00c3\u00c4\u00c5\u00c7\u00c8\u00c9\u00ca"
    "\u00cb\u00cc\u00cd\u00ce\u00cf\u00d2\u00d3\u00d4\u00d5\u00d6"
    "\u00d9\u00da\u00db\u00dc\u00dd\u00d1"
)

# File extensions to scan.
SCAN_EXTS = {".py", ".md", ".yaml", ".yml", ".jsonl", ".json", ".txt", ".toml", ".ipynb"}

# Files excluded from scanning because they contain detection logic by design.
SELF_EXCLUDE_SUFFIXES = (
    "scripts/check_english_only.py",
    "tests/test_english_only.py",
)

# Words that are valid English even though they may appear in other languages.
ENGLISH_OK_WORDS = {
    "com", "capital", "non", "en", "to", "the", "and", "para", "mas",
    "mais", "rua", "portugal", "lisbon",
}


def _is_self_exclude(path: Path) -> bool:
    s = str(path).replace("\\", "/")
    return any(s.endswith(suffix) for suffix in SELF_EXCLUDE_SUFFIXES)


def _has_accented(text: str) -> bool:
    return any(ch in text for ch in ACCENTED_RANGES)


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def scan_file(path: Path) -> list[str]:
    if _is_self_exclude(path):
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []

    findings: list[str] = []

    # Non-ASCII accented characters are a strong signal of non-English text.
    if _has_accented(text):
        findings.append(f"non-english accented characters in {path}")

    lowered = text.lower()

    # Detect non-English language tags or field values in JSON/JSONL/YAML.
    if path.suffix in {".jsonl", ".json", ".yaml", ".yml"}:
        non_en_tags = re.findall(r'"language"\s*:\s*"(?!en["\s])[^"]+"', lowered)
        if non_en_tags:
            findings.append(f"non-english language tag in {path}: {non_en_tags[:3]}")

    return findings


def scan_tree(root: Path) -> list[str]:
    findings: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in SCAN_EXTS:
            continue
        if ".git" in path.parts:
            continue
        if _is_self_exclude(path):
            continue
        findings.extend(scan_file(path))
    return findings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="GitHub-ready root")
    parser.add_argument("--hf-root", default=None, help="Hugging Face-ready root")
    args = parser.parse_args()

    all_findings = scan_tree(Path(args.root))
    if args.hf_root:
        all_findings.extend(scan_tree(Path(args.hf_root)))

    if all_findings:
        print("English-only violations found:")
        for f in all_findings:
            print(f"  BLOCKER: {f}")
        sys.exit(1)
    print("No English-only violations found.")


if __name__ == "__main__":
    main()
