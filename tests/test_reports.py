"""Every committed report's .md is its .json rendered: the two cannot drift."""

from pathlib import Path

import pytest

from sionna_twin_ops.reports import RENDERERS, read_report, render

RESULTS = Path(__file__).resolve().parents[1] / "results"
REPORTS = sorted(RESULTS.glob("*.json"))


def test_there_are_reports_to_check() -> None:
    assert REPORTS, f"no report data in {RESULTS}"


@pytest.mark.parametrize("path", REPORTS, ids=[p.stem for p in REPORTS])
def test_the_markdown_is_the_rendered_data(path: Path) -> None:
    data = read_report(path)
    assert data["kind"] in RENDERERS
    assert render(data) == path.with_suffix(".md").read_text()


def test_an_unknown_kind_is_refused() -> None:
    with pytest.raises(ValueError, match="no renderer"):
        render({"kind": "nothing"})
