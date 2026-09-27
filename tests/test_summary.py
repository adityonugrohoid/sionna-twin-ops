"""results/SUMMARY.md is exactly what `twin summary` writes from the committed data."""

from pathlib import Path

from sionna_twin_ops.summary import summary_markdown

RESULTS = Path(__file__).resolve().parents[1] / "results"


def test_the_committed_summary_is_up_to_date() -> None:
    assert summary_markdown(RESULTS) == (RESULTS / "SUMMARY.md").read_text()
