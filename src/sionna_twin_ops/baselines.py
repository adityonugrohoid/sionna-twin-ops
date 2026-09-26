"""Baselines and the LOS mask (spec rule B), from the heightmap alone; no ray tracing.

B0: free-space gain plus the array's gain at the commanded tilt toward each cell.
    Free-space basic transmission loss: Recommendation ITU-R P.525-5 (11/2024), Annex 1,
    section 2.3, equation (5), L_bf = 20 log10(4 pi d / lambda). Read from
    https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.525-5-202411-I!!PDF-E.pdf
B1: B0 minus the Bullington diffraction loss along the terrain profile from the antenna
    to the cell: Recommendation ITU-R P.526-16 (11/2025), section 4.5.1, equations (49) to (57),
    with the knife-edge loss J(nu) of section 4.1, equation (31), for nu > -0.78 and 0 dB
    otherwise. Flat Earth (spec T5): the effective Earth curvature C_e is 0, so the section 4.5.2
    spherical-Earth term does not arise. Read from
    https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.526-16-202511-I!!PDF-E.pdf
LOS: a cell is in line of sight when the straight line from the antenna to the receiver
    clears every profile sample, which is Bullington's own test S_tim < S_tr (section 4.5.1).

Receivers sit SURFACE_HEIGHT_M above the ground at each map cell centre. Synthetic terrain.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sionna_twin_ops.antenna import CARRIER_HZ, array_gain_db, tilt_weights
from sionna_twin_ops.site import MAP_CELL_M, MAP_CELLS, MAP_SIZE_M, SURFACE_HEIGHT_M, Site
from sionna_twin_ops.terrain import Terrain

SPEED_OF_LIGHT_M_S = 299_792_458.0  # exact: SI defining constant (SI Brochure, 9th ed., 2019)
WAVELENGTH_M = SPEED_OF_LIGHT_M_S / CARRIER_HZ
PROFILE_SAMPLES = 256  # intermediate profile points per path; spacing under 15 m at 3.6 km
NU_MIN = -0.78  # P.526-16 section 4.1: equation (31) holds above this; J = 0 at or below


@dataclass(frozen=True)
class MapGeometry:
    """Receiver positions of one map and the profiles from the antenna to each.

    Attributes:
        site: The map centre and antenna.
        x_m: Cell centre x, shape (128, 128), rows from the south edge.
        y_m: Cell centre y, same shape.
        rx_m: Receiver height (ground plus SURFACE_HEIGHT_M), same shape.
        distance_km: Horizontal distance from the antenna, same shape.
        profile_km: Distance of each intermediate profile point, shape (128, 128, n).
        profile_m: Ground height at each profile point, same shape.
    """

    site: Site
    x_m: NDArray[np.float64]
    y_m: NDArray[np.float64]
    rx_m: NDArray[np.float64]
    distance_km: NDArray[np.float64]
    profile_km: NDArray[np.float64]
    profile_m: NDArray[np.float64]


def ground_at(
    terrain: Terrain, x: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Ground height by bilinear interpolation of the heightmap.

    Args:
        terrain: The terrain.
        x: East coordinates.
        y: North coordinates, same shape as x.

    Returns:
        Heights, same shape as x.

    Raises:
        ValueError: If a point lies outside the tile.
    """
    c = terrain.coords_m
    if (x < c[0]).any() or (x > c[-1]).any() or (y < c[0]).any() or (y > c[-1]).any():
        raise ValueError("point outside the terrain tile")
    fx = (x - c[0]) / terrain.spacing_m
    fy = (y - c[0]) / terrain.spacing_m
    j = np.minimum(np.floor(fx).astype(int), len(c) - 2)
    i = np.minimum(np.floor(fy).astype(int), len(c) - 2)
    tx, ty = fx - j, fy - i
    h = terrain.heights_m
    heights: NDArray[np.float64] = (
        h[i, j] * (1 - tx) * (1 - ty)
        + h[i, j + 1] * tx * (1 - ty)
        + h[i + 1, j] * (1 - tx) * ty
        + h[i + 1, j + 1] * tx * ty
    )
    return heights


def map_geometry(terrain: Terrain, site: Site) -> MapGeometry:
    """Receivers at the cell centres of the site's map and the profiles leading to them.

    Args:
        terrain: The terrain.
        site: The map centre and antenna.

    Returns:
        The geometry.
    """
    offsets = (np.arange(MAP_CELLS) + 0.5) * MAP_CELL_M - MAP_SIZE_M / 2.0
    x, y = np.meshgrid(site.x_m + offsets, site.y_m + offsets)
    rx = ground_at(terrain, x, y) + SURFACE_HEIGHT_M
    distance_km = np.hypot(x - site.x_m, y - site.y_m) / 1000.0
    u = np.arange(1, PROFILE_SAMPLES + 1) / (PROFILE_SAMPLES + 1)  # excludes both ends
    px = site.x_m + u * (x - site.x_m)[..., None]
    py = site.y_m + u * (y - site.y_m)[..., None]
    return MapGeometry(
        site=site,
        x_m=x,
        y_m=y,
        rx_m=rx,
        distance_km=distance_km,
        profile_km=u * distance_km[..., None],
        profile_m=ground_at(terrain, px, py),
    )


def knife_edge_loss_db(nu: NDArray[np.float64]) -> NDArray[np.float64]:
    """J(nu), P.526-16 section 4.1 equation (31); 0 dB at or below nu = -0.78.

    Args:
        nu: Diffraction parameter.

    Returns:
        Loss in dB.
    """
    above = nu > NU_MIN
    safe = np.where(above, nu, 0.0)
    j = 6.9 + 20.0 * np.log10(np.sqrt((safe - 0.1) ** 2 + 1.0) + safe - 0.1)
    loss: NDArray[np.float64] = np.where(above, j, 0.0)
    return loss


def slopes(geometry: MapGeometry) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """S_tim and S_tr of P.526-16 equations (49) and (50) with C_e = 0, in m/km.

    Args:
        geometry: Map geometry.

    Returns:
        (S_tim, S_tr), each shaped like the map.
    """
    hts = geometry.site.antenna_m
    s_tim = ((geometry.profile_m - hts) / geometry.profile_km).max(axis=-1)
    s_tr = (geometry.rx_m - hts) / geometry.distance_km
    return s_tim, s_tr


def los_mask(geometry: MapGeometry) -> NDArray[np.bool_]:
    """Cells whose straight line from the antenna clears every profile sample.

    Args:
        geometry: Map geometry.

    Returns:
        Boolean mask shaped like the map.
    """
    s_tim, s_tr = slopes(geometry)
    los: NDArray[np.bool_] = s_tim < s_tr
    return los


def bullington_nu(geometry: MapGeometry) -> NDArray[np.float64]:
    """Diffraction parameter of the Bullington point, P.526-16 section 4.5.1, C_e = 0.

    Equation (51) on line-of-sight paths (the profile point with the highest nu) and
    equations (53) to (55) on trans-horizon paths (the Bullington point).

    Args:
        geometry: Map geometry.

    Returns:
        nu, shaped like the map.

    Raises:
        ValueError: If the construction produces a non-finite nu.
    """
    hts = geometry.site.antenna_m
    hrs = geometry.rx_m
    d = geometry.distance_km
    di = geometry.profile_km
    hi = geometry.profile_m
    s_tim, s_tr = slopes(geometry)
    los = s_tim < s_tr

    # Case 1, LoS: equation (51), the profile point with the highest nu.
    chord = (hts * (d[..., None] - di) + hrs[..., None] * di) / d[..., None]
    nu_points = (hi - chord) * np.sqrt(
        0.002 * d[..., None] / (WAVELENGTH_M * di * (d[..., None] - di))
    )
    nu_los = nu_points.max(axis=-1)

    # Case 2, trans-horizon: equations (53) to (55), the Bullington point.
    s_rim = ((hi - hrs[..., None]) / (d[..., None] - di)).max(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        db = (hrs - hts + s_rim * d) / (s_tim + s_rim)
        nu_b = (hts + s_tim * db - (hts * (d - db) + hrs * db) / d) * np.sqrt(
            0.002 * d / (WAVELENGTH_M * db * (d - db))
        )

    nu: NDArray[np.float64] = np.where(los, nu_los, nu_b)
    if not np.isfinite(nu).all():
        raise ValueError("non-finite diffraction parameter in the Bullington construction")
    return nu


def bullington_loss_db(geometry: MapGeometry) -> NDArray[np.float64]:
    """Bullington diffraction loss L_b, P.526-16 section 4.5.1 equations (52), (56) and (57).

    Args:
        geometry: Map geometry.

    Returns:
        L_b in dB, shaped like the map.
    """
    d = geometry.distance_km
    l_uc = knife_edge_loss_db(bullington_nu(geometry))  # equations (52) and (56)
    loss: NDArray[np.float64] = l_uc + (1.0 - np.exp(-l_uc / 6.0)) * (10.0 + 0.02 * d)  # (57)
    return loss


def free_space_gain_db(distance_m: NDArray[np.float64]) -> NDArray[np.float64]:
    """Minus the free-space basic transmission loss, P.525-5 equation (5).

    Args:
        distance_m: Path length in metres.

    Returns:
        -20 log10(4 pi d / lambda), in dB.
    """
    gain: NDArray[np.float64] = -20.0 * np.log10(4.0 * np.pi * distance_m / WAVELENGTH_M)
    return gain


def b0_gain_db(geometry: MapGeometry, azimuth_deg: float, tilt_deg: float) -> NDArray[np.float64]:
    """B0: free-space gain plus the array gain toward each receiver; no terrain.

    Args:
        geometry: Map geometry.
        azimuth_deg: Boresight azimuth, clockwise from north.
        tilt_deg: Electrical downtilt.

    Returns:
        Path gain in dB, shaped like the map.
    """
    site = geometry.site
    dx = geometry.x_m - site.x_m
    dy = geometry.y_m - site.y_m
    dz = geometry.rx_m - site.antenna_m
    distance_m = np.sqrt(dx**2 + dy**2 + dz**2)
    bearing = np.degrees(np.arctan2(dx, dy))  # clockwise from north
    off_boresight = (bearing - azimuth_deg + 180.0) % 360.0 - 180.0
    zenith = np.degrees(np.arccos(dz / distance_m))
    gain: NDArray[np.float64] = free_space_gain_db(distance_m) + array_gain_db(
        zenith, off_boresight, tilt_weights(tilt_deg)
    )
    return gain


def b1_gain_db(geometry: MapGeometry, azimuth_deg: float, tilt_deg: float) -> NDArray[np.float64]:
    """B1: B0 minus the Bullington diffraction loss along the terrain profile.

    Args:
        geometry: Map geometry.
        azimuth_deg: Boresight azimuth, clockwise from north.
        tilt_deg: Electrical downtilt.

    Returns:
        Path gain in dB, shaped like the map.
    """
    gain: NDArray[np.float64] = b0_gain_db(geometry, azimuth_deg, tilt_deg) - bullington_loss_db(
        geometry
    )
    return gain
