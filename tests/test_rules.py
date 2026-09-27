"""Every dataset rule id the repo cites ("spec S2", "rule Q0b") is defined in docs/dataset.md."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULES = ROOT / "docs" / "dataset.md"
ID = r"[A-Z][0-9]*[a-z]?"
CITATION = re.compile(rf"\b(?:spec|dataset|rules?)(?: rules?)? ({ID}(?:(?:, | and | to ){ID})*)\b")


def cited_ids() -> dict[str, set[str]]:
    """Rule ids cited per file, over src, tests, docs, results, notebooks and the README."""
    files = [
        *ROOT.glob("src/**/*.py"),
        *ROOT.glob("tests/*.py"),
        *ROOT.glob("docs/*.md"),
        *ROOT.glob("results/*.md"),
        *ROOT.glob("notebooks/*.py"),
        ROOT / "README.md",
    ]
    found: dict[str, set[str]] = {}
    for path in files:
        for match in CITATION.finditer(path.read_text()):
            for rule_id in re.split(r", | and | to ", match.group(1)):
                found.setdefault(rule_id, set()).add(str(path.relative_to(ROOT)))
    return found


def defined_ids() -> set[str]:
    """Rule ids that head a section of docs/dataset.md ("## T: ...", "### T2 ...")."""
    return set(re.findall(rf"^#{{2,4}} ({ID})[: ]", RULES.read_text(), flags=re.MULTILINE))


def test_the_citation_scan_finds_the_known_ids() -> None:
    cited = cited_ids()
    for rule_id in ("S2", "Q0b", "N7", "M2b", "T", "Q"):
        assert rule_id in cited


def test_every_cited_rule_id_is_defined() -> None:
    defined = defined_ids()
    missing = {k: sorted(v) for k, v in cited_ids().items() if k not in defined}
    assert not missing, f"rule ids cited but not defined in docs/dataset.md: {missing}"
