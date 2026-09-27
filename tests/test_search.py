"""Tilt and power search arithmetic (spec rule Q)."""

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
