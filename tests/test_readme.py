"""The README's headline numbers are quoted from results/SUMMARY.md, never typed afresh."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_every_headline_number_is_in_the_summary() -> None:
    readme = (ROOT / "README.md").read_text()
    section = readme.split("## Headline results", 1)[1].split("\n## ", 1)[0]
    summary = (ROOT / "results" / "SUMMARY.md").read_text()
    numbers = re.findall(r"-?\d+(?:\.\d+)?", section)
    assert numbers
    missing = [n for n in numbers if n not in summary]
    assert not missing, f"headline numbers not in SUMMARY.md: {missing}"
