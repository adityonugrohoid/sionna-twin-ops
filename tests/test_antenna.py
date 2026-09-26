"""Antenna math (spec rule A), independent of Sionna."""

import numpy as np
import pytest

from sionna_twin_ops.antenna import (
    G_E_MAX_DBI,
    NUM_ROWS,
    VERTICAL_SPACING_WL,
    array_gain_db,
    element_gain_db,
    element_heights_wl,
    tilt_weights,
    yaw_for_azimuth,
)


def test_element_gain_follows_tr38901_table_7_3_1() -> None:
    zenith = np.array([90.0, 90.0, 90.0 + 65.0 / 2, 90.0, 0.0, 0.0])
    off = np.array([0.0, 65.0 / 2, 0.0, 180.0, 0.0, 90.0])
    gain = element_gain_db(zenith, off)
    assert gain[0] == pytest.approx(G_E_MAX_DBI)  # boresight
    assert gain[1] == pytest.approx(G_E_MAX_DBI - 3.0)  # half the 3 dB beamwidth
    assert gain[2] == pytest.approx(G_E_MAX_DBI - 3.0)
    assert gain[3] == pytest.approx(G_E_MAX_DBI - 30.0)  # A_max floor at the back
    # Zenith alone loses 12 (90/65)**2 = 23 dB, short of SLA_V; adding the horizontal cut
    # reaches the 30 dB cap on the sum.
    assert gain[4] == pytest.approx(G_E_MAX_DBI - 12.0 * (90.0 / 65.0) ** 2)
    assert gain[5] == pytest.approx(G_E_MAX_DBI - 30.0)


def test_element_heights_are_centred_rows_top_first() -> None:
    z = element_heights_wl()
    assert z.shape == (NUM_ROWS,)
    assert z.sum() == pytest.approx(0.0)
    assert np.allclose(np.diff(z), -VERTICAL_SPACING_WL)


@pytest.mark.parametrize("tilt", [0.0, 3.0, 12.0])
def test_tilt_weights_have_unit_power_and_a_linear_phase_taper(tilt: float) -> None:
    w = tilt_weights(tilt)
    assert np.sum(np.abs(w) ** 2) == pytest.approx(1.0)
    step = np.angle(w[1:] * np.conj(w[:-1]))
    expected = -2.0 * np.pi * VERTICAL_SPACING_WL * np.sin(np.radians(tilt))
    assert np.allclose(step, expected)


@pytest.mark.parametrize("tilt", [0.0, 6.0, 12.0])
def test_array_gain_peaks_at_the_commanded_tilt(tilt: float) -> None:
    elevation = np.linspace(-30.0, 10.0, 4001)
    gain = array_gain_db(90.0 - elevation, np.zeros_like(elevation), tilt_weights(tilt))
    peak = elevation[np.argmax(gain)]
    # The element pattern pulls the product's peak slightly toward the horizon.
    assert -tilt <= peak <= -tilt + 0.5
    # On boresight at zero tilt: element 8 dBi plus array gain 10 log10(8).
    if tilt == 0.0:
        assert gain.max() == pytest.approx(G_E_MAX_DBI + 10.0 * np.log10(NUM_ROWS), abs=0.01)


@pytest.mark.parametrize(
    ("azimuth", "yaw"), [(0.0, 90.0), (90.0, 0.0), (180.0, -90.0), (270.0, -180.0)]
)
def test_azimuth_is_clockwise_from_north(azimuth: float, yaw: float) -> None:
    assert np.degrees(yaw_for_azimuth(azimuth)) == pytest.approx(yaw)
