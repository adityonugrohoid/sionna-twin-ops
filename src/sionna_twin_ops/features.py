"""Model inputs and targets per map (spec M1, M1b, M2), from the heightmap and B0.

Inputs, one 128 x 128 channel each, with fixed physical scalings so no statistics are
fitted on any split:
    0  receiver height relative to the antenna, per 100 m
    1  log10 of horizontal distance from the antenna in km
    2  sin of the horizontal angle off boresight
    3  cos of the horizontal angle off boresight
    4  elevation angle from the antenna, per 10 degrees
    5  LOS mask (1 in line of sight)
    6  B0 path gain in dB, shifted by +100 and scaled per 20 dB
Targets: the ray-traced path gain in dB minus B0 in dB (the residual the model predicts,
spec M1), and the power mask (1 where the ray tracer has power, spec M1b). Cells without
power have no residual; the path-gain loss skips them (spec M2).
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sionna_twin_ops.baselines import MapGeometry, b0_gain_db, los_mask, map_geometry
from sionna_twin_ops.site import Site
from sionna_twin_ops.terrain import GRID_SPACING_M, generate_terrain

INPUT_CHANNELS = 7
B0_OFFSET_DB = 100.0
B0_SCALE_DB = 20.0


@dataclass(frozen=True)
class TerrainFeatures:
    """The parts of the inputs that depend only on the terrain and site.

    Attributes:
        geometry: Receivers and profiles of the map.
        height: Channel 0.
        log_distance: Channel 1.
        elevation: Channel 4.
        los: Channel 5.
    """

    geometry: MapGeometry
    height: NDArray[np.float32]
    log_distance: NDArray[np.float32]
    elevation: NDArray[np.float32]
    los: NDArray[np.float32]


def terrain_features(terrain_id: int, site: Site) -> TerrainFeatures:
    """Terrain-only channels of one terrain's map.

    Args:
        terrain_id: Terrain id.
        site: Its site.

    Returns:
        The channels.
    """
    geometry = map_geometry(generate_terrain(terrain_id, GRID_SPACING_M), site)
    dz = geometry.rx_m - site.antenna_m
    horizontal_m = geometry.distance_km * 1000.0
    return TerrainFeatures(
        geometry=geometry,
        height=(dz / 100.0).astype(np.float32),
        log_distance=np.log10(geometry.distance_km).astype(np.float32),
        elevation=(np.degrees(np.arctan2(dz, horizontal_m)) / 10.0).astype(np.float32),
        los=los_mask(geometry).astype(np.float32),
    )


def map_inputs(
    terrain: TerrainFeatures, azimuth_deg: float, tilt_deg: float
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """All input channels of one map, and B0 in dB.

    Args:
        terrain: Terrain-only channels.
        azimuth_deg: Boresight azimuth, clockwise from north.
        tilt_deg: Electrical downtilt.

    Returns:
        (inputs of shape (INPUT_CHANNELS, 128, 128), B0 in dB of shape (128, 128)).
    """
    g = terrain.geometry
    bearing = np.degrees(np.arctan2(g.x_m - g.site.x_m, g.y_m - g.site.y_m))
    off = np.radians(bearing - azimuth_deg)
    b0 = b0_gain_db(g, azimuth_deg, tilt_deg).astype(np.float32)
    inputs = np.stack(
        [
            terrain.height,
            terrain.log_distance,
            np.sin(off).astype(np.float32),
            np.cos(off).astype(np.float32),
            terrain.elevation,
            terrain.los,
            (b0 + B0_OFFSET_DB) / B0_SCALE_DB,
        ]
    ).astype(np.float32)
    return inputs, b0


def targets(
    path_gain: NDArray[np.float32], b0_db: NDArray[np.float32]
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Residual target and power mask of one ray-traced map.

    Args:
        path_gain: Linear path gain from the ray tracer.
        b0_db: B0 in dB for the same map.

    Returns:
        (residual in dB, 0 where there is no power; power mask, 1 where there is power).
    """
    power = path_gain > 0
    with np.errstate(divide="ignore"):
        traced_db = np.where(power, 10.0 * np.log10(np.where(power, path_gain, 1.0)), 0.0)
    residual = np.where(power, traced_db - b0_db, 0.0).astype(np.float32)
    return residual, power.astype(np.float32)
