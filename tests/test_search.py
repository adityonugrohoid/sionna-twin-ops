"""Tilt and power search arithmetic (spec rule Q)."""

from pathlib import Path

import numpy as np
import pytest

from sionna_twin_ops.dataset import AZIMUTHS_DEG
from sionna_twin_ops.evaluate import COVERED_GAIN_DB
from sionna_twin_ops.search import (
    POWERS_DBM,
    SURROGATE_TILTS_DEG,
    TRACE_TILTS_DEG,
    covered_gain_db,
    first_best,
    objectives,
    rule_of_thumb_tilt_deg,
    surrogate_choices,
    tilts_to_trace,
    vertical_hpbw_deg,
)


def test_grids_match_the_spec() -> None:
    assert len(TRACE_TILTS_DEG) == 13 and TRACE_TILTS_DEG[-1] == 12.0
    assert len(SURROGATE_TILTS_DEG) == 25 and SURROGATE_TILTS_DEG[1] == 0.5
    assert tuple(float(p) for p in range(37, 47)) == POWERS_DBM


def test_threshold_agrees_with_the_evaluation_at_43_dbm() -> None:
    assert covered_gain_db(43.0) == pytest.approx(COVERED_GAIN_DB)
    assert covered_gain_db(46.0) == pytest.approx(COVERED_GAIN_DB - 3.0)


def test_hpbw_is_near_the_uniform_array_estimate_and_sets_the_rule() -> None:
    # Uniform 8-element column at 0.8 wavelength: about 0.886 / (8 * 0.8) rad = 7.9 deg.
    hpbw = vertical_hpbw_deg()
    assert 7.0 < hpbw < 9.0
    assert rule_of_thumb_tilt_deg() == pytest.approx(np.degrees(np.arctan(0.01)) + hpbw / 2)


def test_objective_counts_near_cover_minus_spill() -> None:
    gain = np.array([-100.0, -124.0, -100.0, -200.0])
    has_power = np.array([True, True, True, False])
    near = np.array([True, True, False, False])
    values = objectives(gain, has_power, near)
    at_43 = POWERS_DBM.index(43.0)
    at_46 = POWERS_DBM.index(46.0)
    assert values[at_43] == 1 - 1  # -124 dB is not covered at 43 dBm (needs -122.2)
    assert values[at_46] == 2 - 1  # it is at 46 dBm (needs -125.2)


def test_ties_go_to_the_first_in_c_order() -> None:
    values = np.zeros((2, 3))
    values[0, 2] = values[1, 0] = 5.0
    assert first_best(values) == (0, 2)


def test_choices_and_the_trace_plan() -> None:
    grid = np.zeros((len(AZIMUTHS_DEG), len(SURROGATE_TILTS_DEG), len(POWERS_DBM)))
    grid[:, 3, 9] = 1.0  # every azimuth: tilt 1.5, 46 dBm
    grid[2, 4, 9] = 2.0  # azimuth 90: tilt 2.0 is the whole-grid best
    record = {"objectives": {"0": {"52": grid.tolist()}}}
    choices = surrogate_choices(record)
    assert choices["cases"]["0"]["52/0"] == [1.5, 46.0]
    assert choices["cases"]["0"]["52/90"] == [2.0, 46.0]
    assert choices["whole_grid"]["0"]["52"] == [90.0, 2.0, 46.0]
    plan = tilts_to_trace([choices], 4.5346)
    assert plan["52/0"] == [1.5, 4.5346]
    assert plan["52/90"] == [4.5346]  # 2.0 is on the 1 deg grid already
    assert len(plan) == len(AZIMUTHS_DEG)


def test_search_map_names_keep_the_rule_tilt_apart_from_a_half_degree() -> None:
    from sionna_twin_ops.search import search_map_name

    assert search_map_name(52, 90.0, 4.5) == "t052_a090_t04.5000.npy"
    assert search_map_name(52, 90.0, 4.5346) == "t052_a090_t04.5346.npy"
    with pytest.raises(ValueError, match="1e-4"):
        search_map_name(52, 90.0, 4.53461)


def test_cpu_raytracer_medians_are_quoted_from_the_committed_evaluation() -> None:
    from sionna_twin_ops.search import cpu_raytracer_seconds

    report = Path(__file__).resolve().parents[1] / "results" / "evaluation_test.md"
    first, further = cpu_raytracer_seconds(report)
    assert first > 1.0 and further > 1.0


def test_share_row_counts_cases_within_one_percent() -> None:
    from sionna_twin_ops.search import _share_row

    row = _share_row("x", [1.0, 0.995, 0.98, 1.02])
    assert row.startswith("| x | 0.9975 |")
    assert row.endswith("| 0.9800 | 3 of 4 |")


def test_terrain_objectives_read_every_search_map(tmp_path: Path) -> None:
    from sionna_twin_ops.search import search_map_name, terrain_objectives

    (tmp_path / "maps").mkdir()
    gain = np.full((2, 2), 1e-9)
    for tilt in (4.5, 4.5346):
        np.save(tmp_path / "maps" / search_map_name(52, 90.0, tilt), gain.astype(np.float32))
    values = terrain_objectives(tmp_path, 52, np.array([[True, True], [False, False]]))
    assert set(values) == {(90.0, 4.5), (90.0, 4.5346)}
    assert values[(90.0, 4.5)][-1] == 2 - 2  # -90 dB is covered everywhere
