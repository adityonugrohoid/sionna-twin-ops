"""Evaluation arithmetic (spec rule E)."""

from pathlib import Path

import numpy as np
import pytest

from sionna_twin_ops.evaluate import (
    COVERED_GAIN_DB,
    POWER_PER_RE_DBM,
    Tally,
    error_stats,
    quoted_uncertainty,
    strata_masks,
    tally_map,
)

REPO = Path(__file__).resolve().parents[1]


def test_power_per_resource_element_and_coverage_threshold() -> None:
    # 43 dBm over 1,200 REs is 43 - 30.79 = 12.21 dBm per RE; -110 dBm needs -122.21 dB.
    assert pytest.approx(12.208, abs=1e-3) == POWER_PER_RE_DBM
    assert pytest.approx(-122.208, abs=1e-3) == COVERED_GAIN_DB


def test_strata_split_lit_cells_by_path_and_los() -> None:
    power = np.array([True, True, True, True, False])
    los = np.array([True, True, False, False, True])
    residual = np.array([1.0, 5.0, 1.0, 5.0, 0.0], dtype=np.float32)
    masks = strata_masks(power, los, residual)
    assert masks["LOS direct"].tolist() == [True, False, False, False, False]
    assert masks["LOS reflection"].tolist() == [False, True, False, False, False]
    assert masks["NLOS"].tolist() == [False, False, True, True, False]
    assert masks["all"].tolist() == power.tolist()


def test_error_stats_are_absolute_except_the_bias() -> None:
    mean, median, p95, bias, cells = error_stats([np.array([-2.0, 2.0, 4.0], dtype=np.float32)])
    assert mean == pytest.approx(8 / 3)
    assert median == 2.0
    assert bias == pytest.approx(4 / 3)
    assert cells == 3
    assert p95 == pytest.approx(np.percentile([2.0, 2.0, 4.0], 95))
    assert error_stats([])[4] == 0


def test_tally_counts_power_classes_and_coverage() -> None:
    tally = Tally()
    power = np.array([True, True, False, False])
    traced = np.array([-100.0, -130.0, 0.0, 0.0], dtype=np.float32)
    predicted = np.array([-101.0, -100.0, -100.0, -140.0], dtype=np.float32)
    says_power = np.array([True, True, True, False])
    masks = strata_masks(power, np.ones(4, dtype=bool), np.zeros(4, dtype=np.float32))
    tally_map(tally, ["all"], predicted, says_power, traced, masks)
    # TP 2, FP 1 (cell 2), TN 1 (cell 3), FN 0.
    assert tally.power["all"] == [2, 1, 1, 0]
    # Covered: traced cell 0 only; predicted cells 0, 1, 2 (all above -122.2 dB with power).
    assert tally.coverage == [3, 1, 1, 3]
    assert tally.iou == [pytest.approx(1 / 3)]
    assert np.concatenate(tally.errors[("all", "all")]).tolist() == [-1.0, 30.0]


def test_uncertainty_is_quoted_from_the_committed_fold_check() -> None:
    quoted = quoted_uncertainty(REPO / "results" / "fold_check.md")
    assert len(quoted.commit) == 7
    for group in ("all", "hilltop", "slope", "valley"):
        for stratum in ("LOS direct", "LOS reflection", "NLOS"):
            fold, sampling = quoted.terms[(group, stratum)].values()
            assert fold[0] <= fold[1] and sampling[0] <= sampling[1]


def test_a_report_without_the_rows_is_refused(tmp_path: Path) -> None:
    report = tmp_path / "fold_check.md"
    report.write_text("| commit | abcdef1234 |\n")
    with pytest.raises(ValueError, match="lacks the rows"):
        quoted_uncertainty(report)
