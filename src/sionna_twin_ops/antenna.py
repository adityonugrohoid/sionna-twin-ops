"""Sector antenna (spec rule A): an 8 x 1 vertical column of TR 38.901 elements.

Pure numpy, so baselines and model features do not need Sionna. The Sionna array objects
are built from these constants in `scene.py`, and a test holds the two geometries equal.

Element pattern: 3GPP TR 38.901 version 19.5.0 Release 19 (ETSI TR 138 901 V19.5.0,
2026-09), clause 7.3.0, Table 7.3-1: theta_3dB = phi_3dB = 65 deg, SLA_V = A_max = 30 dB,
G_E,max = 8 dBi, as implemented by Sionna RT's "tr38901" pattern. Read from
https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/19.05.00_60/tr_138901v190500p.pdf
The array size and spacing are ASSUMPTION: TR 38.901 does not fix a macro panel.
"""

import numpy as np
from numpy.typing import NDArray

CARRIER_HZ = 1.8e9  # ASSUMPTION (spec A1): a common rural band
NUM_ROWS = 8  # ASSUMPTION (spec A2)
VERTICAL_SPACING_WL = 0.8  # ASSUMPTION (spec A2): row spacing in wavelengths

# TR 38.901 Table 7.3-1.
THETA_3DB_DEG = 65.0
PHI_3DB_DEG = 65.0
SLA_V_DB = 30.0
A_MAX_DB = 30.0
G_E_MAX_DBI = 8.0


def element_heights_wl() -> NDArray[np.float64]:
    """Element heights above the array centre, in wavelengths, row 0 at the top.

    Returns:
        Heights, shape (NUM_ROWS,), matching Sionna's PlanarArray layout.
    """
    rows = np.arange(NUM_ROWS, dtype=np.float64)
    heights: NDArray[np.float64] = VERTICAL_SPACING_WL * ((NUM_ROWS - 1) / 2.0 - rows)
    return heights


def tilt_weights(tilt_deg: float) -> NDArray[np.complex128]:
    """Per-element weights that steer the main lobe `tilt_deg` below the horizon (spec A3).

    Sionna forms the equivalent channel as sum_k a exp(j 2 pi r.d_k / lambda) p_k with
    no conjugation, so a lobe at elevation -tilt needs p_k = exp(+j 2 pi z_k sin(tilt)),
    with z_k in wavelengths, scaled to unit total power.

    Args:
        tilt_deg: Electrical downtilt, positive below the horizon.

    Returns:
        Complex weights, one per element, with unit total power.
    """
    z_wl = element_heights_wl()
    phase = 2.0 * np.pi * z_wl * np.sin(np.radians(tilt_deg))
    weights: NDArray[np.complex128] = np.exp(1j * phase) / np.sqrt(NUM_ROWS)
    return weights


def yaw_for_azimuth(azimuth_deg: float) -> float:
    """Sionna yaw angle for a boresight azimuth measured clockwise from true north.

    The scene has x east and y north, and yaw turns the boresight counter-clockwise from
    +x, so yaw = 90 degrees - azimuth.

    Args:
        azimuth_deg: Boresight azimuth, clockwise from north.

    Returns:
        Yaw in radians.
    """
    return float(np.radians(90.0 - azimuth_deg))


def element_gain_db(
    zenith_deg: NDArray[np.float64], off_boresight_deg: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Element gain from TR 38.901 Table 7.3-1, in dBi.

    Args:
        zenith_deg: Zenith angle in the antenna frame (90 = horizon).
        off_boresight_deg: Azimuth from boresight in the antenna frame, in [-180, 180].

    Returns:
        A''_dB(theta'', phi'') + G_E,max.
    """
    vertical = -np.minimum(12.0 * ((zenith_deg - 90.0) / THETA_3DB_DEG) ** 2, SLA_V_DB)
    horizontal = -np.minimum(12.0 * (off_boresight_deg / PHI_3DB_DEG) ** 2, A_MAX_DB)
    gain: NDArray[np.float64] = -np.minimum(-(vertical + horizontal), A_MAX_DB) + G_E_MAX_DBI
    return gain


def array_gain_db(
    zenith_deg: NDArray[np.float64],
    off_boresight_deg: NDArray[np.float64],
    weights: NDArray[np.complex128],
) -> NDArray[np.float64]:
    """Gain of the weighted column toward each direction, in dBi.

    The element gain times |sum_k p_k exp(j 2 pi z_k cos(theta))|**2, the same product
    Sionna forms for a synthetic array with a precoding vector.

    Args:
        zenith_deg: Zenith angle in the antenna frame.
        off_boresight_deg: Azimuth from boresight in the antenna frame.
        weights: Per-element weights with unit total power.

    Returns:
        Gain in dBi, shaped like the angles.
    """
    z_wl = element_heights_wl()
    cos_theta = np.cos(np.radians(zenith_deg))[..., None]
    factor = np.abs((weights * np.exp(1j * 2.0 * np.pi * z_wl * cos_theta)).sum(axis=-1)) ** 2
    with np.errstate(divide="ignore"):
        factor_db = 10.0 * np.log10(factor)
    gain: NDArray[np.float64] = element_gain_db(zenith_deg, off_boresight_deg) + factor_db
    return gain
