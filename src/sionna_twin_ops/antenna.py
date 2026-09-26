"""Sector antenna (spec rule A): an 8 x 1 vertical column of TR 38.901 elements.

Element pattern: 3GPP TR 38.901 V19.5.0 (2026-09), Release 19, clause 7.3, Table 7.3-1
(unchanged since V14.0.0; 65 degree 3 dB beamwidth in both cuts,
30 dB side-lobe and front-to-back limits, 8 dBi maximum element gain), as implemented by
Sionna RT's "tr38901" pattern. The array size and spacing are ASSUMPTION: TR 38.901 does
not fix a macro panel.
"""

import numpy as np

from sionna_twin_ops.sionna_rt import mi, rt

CARRIER_HZ = 1.8e9  # ASSUMPTION (spec A1): a common rural band
NUM_ROWS = 8  # ASSUMPTION (spec A2)
VERTICAL_SPACING_WL = 0.8  # ASSUMPTION (spec A2): row spacing in wavelengths


def sector_array() -> rt.PlanarArray:
    """The sector antenna: 8 rows x 1 column, 0.8 wavelength spacing, vertical polarization.

    Returns:
        The array, boresight along its local +x axis.
    """
    return rt.PlanarArray(
        num_rows=NUM_ROWS,
        num_cols=1,
        vertical_spacing=VERTICAL_SPACING_WL,
        pattern="tr38901",
        polarization="V",
    )


def receiver_array() -> rt.PlanarArray:
    """Receive side of the map: one isotropic, vertically polarized element.

    Returns:
        A single-element array.
    """
    return rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")


def tilt_weights(array: rt.PlanarArray, tilt_deg: float) -> np.ndarray:
    """Per-element weights that steer the main lobe `tilt_deg` below the horizon (spec A3).

    Sionna forms the equivalent channel as sum_k a exp(j 2 pi r.d_k / lambda) p_k with
    no conjugation, so a lobe at elevation -tilt needs p_k = exp(+j 2 pi z_k sin(tilt)),
    with z_k in wavelengths, scaled to unit total power.

    Args:
        array: The transmit array; element heights are read from its geometry.
        tilt_deg: Electrical downtilt, positive below the horizon.

    Returns:
        Complex weights, one per element, with unit total power.
    """
    z_wl = np.asarray(array.normalized_positions.z.numpy(), dtype=np.float64)
    phase = 2.0 * np.pi * z_wl * np.sin(np.radians(tilt_deg))
    weights: np.ndarray = np.exp(1j * phase) / np.sqrt(len(z_wl))
    return weights


def precoding_vec(weights: np.ndarray) -> tuple[mi.Float, mi.Float]:
    """Weights in the (real, imaginary) form the radio map solver takes.

    Args:
        weights: Complex per-element weights.

    Returns:
        (real parts, imaginary parts).
    """
    return mi.Float(weights.real.astype(np.float32)), mi.Float(weights.imag.astype(np.float32))


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
