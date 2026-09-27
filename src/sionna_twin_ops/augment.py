"""Symmetries of a map: 4 rotations by 90 degrees times 2 mirrors (8 variants), of which
the 4 that keep the mesh fold are exact for the ray tracer (spec M2b).

The map is a 128 x 128 grid centred on the site, so its centre is a cell corner and every
variant maps cells onto cells. Rasters are indexed [row, column] with row 0 at the south
edge and column 0 at the west edge. A variant is (k, mirror): first mirror east-west when
`mirror` is set, then rotate k quarter turns clockwise (seen from above). The boresight
azimuth moves with the map: a mirror sends azimuth a to 360 - a, a clockwise quarter turn
adds 90. The model inputs are rasters too, except channel 2 (sin of the angle off
boresight), which also changes sign under a mirror; the tests prove this against features
recomputed from the transformed terrain and azimuth.
"""

import numpy as np
from numpy.typing import NDArray

from sionna_twin_ops.site import MAP_SIZE_M, Site
from sionna_twin_ops.terrain import Terrain, TerrainParams

ALL_VARIANTS = tuple((k, mirror) for mirror in (False, True) for k in range(4))
# The mesh splits every cell along its SW-NE diagonal (scene.grid_faces). Only the variants
# that keep that diagonal move the traced surface exactly: the identity, the half turn and
# the two diagonal mirrors (mirror then 1 or 3 quarter turns). Quarter turns and axis
# mirrors flip every cell's fold, which changes the traced map (spec M2b); training uses
# only these four.
VARIANTS = ((0, False), (2, False), (1, True), (3, True))


def keeps_fold(k: int, mirror: bool) -> bool:
    """Whether a variant maps each cell's SW-NE fold diagonal onto itself.

    Args:
        k: Clockwise quarter turns.
        mirror: Mirror east-west first.

    Returns:
        True for the four exact variants.
    """
    return (k, mirror) in VARIANTS


SIN_CHANNEL = 2  # features.py: sin of the horizontal angle off boresight
CROP_HALF_M = 3480.0  # largest half-width on the 40 m grid inside the tile for any site


def transform_raster[Scalar: np.generic](
    raster: NDArray[Scalar], k: int, mirror: bool
) -> NDArray[Scalar]:
    """Apply a variant to a raster (or a stack whose last two axes are the map).

    Args:
        raster: Array (..., rows, columns), row 0 south, column 0 west.
        k: Clockwise quarter turns.
        mirror: Mirror east-west first.

    Returns:
        The transformed array.
    """
    out = raster[..., ::-1] if mirror else raster
    # Clockwise seen from above is anticlockwise in [row, column] with row 0 at the south.
    turned: NDArray[Scalar] = np.rot90(out, k=k, axes=(-2, -1)).copy()
    return turned


def transform_azimuth(azimuth_deg: float, k: int, mirror: bool) -> float:
    """Boresight azimuth after a variant.

    Args:
        azimuth_deg: Azimuth, clockwise from north.
        k: Clockwise quarter turns.
        mirror: Mirror east-west first.

    Returns:
        The azimuth in [0, 360).
    """
    a = (360.0 - azimuth_deg) % 360.0 if mirror else azimuth_deg
    return (a + 90.0 * k) % 360.0


def transform_inputs(inputs: NDArray[np.float32], k: int, mirror: bool) -> NDArray[np.float32]:
    """Apply a variant to a stack of model inputs (channels, rows, columns).

    Args:
        inputs: Model inputs of one map.
        k: Clockwise quarter turns.
        mirror: Mirror east-west first.

    Returns:
        The transformed inputs.
    """
    out = transform_raster(inputs, k, mirror)
    if mirror:
        out[SIN_CHANNEL] = -out[SIN_CHANNEL]
    return out


def centred_crop(terrain: Terrain, site: Site) -> tuple[Terrain, Site]:
    """The square of terrain CROP_HALF_M around the site, re-centred so the site is at 0, 0.

    Args:
        terrain: A generated terrain.
        site: A site on one of its grid vertices.

    Returns:
        (cropped terrain, the same site at the origin).

    Raises:
        ValueError: If the crop does not fit in the tile or does not cover the map.
    """
    if MAP_SIZE_M / 2.0 + terrain.spacing_m > CROP_HALF_M:
        raise ValueError("crop does not cover the map")
    steps = round(CROP_HALF_M / terrain.spacing_m)
    i = int(np.searchsorted(terrain.coords_m, site.y_m))
    j = int(np.searchsorted(terrain.coords_m, site.x_m))
    n = len(terrain.coords_m)
    if min(i, j) - steps < 0 or max(i, j) + steps >= n:
        raise ValueError(f"a {CROP_HALF_M} m crop around the site leaves the tile")
    heights = terrain.heights_m[i - steps : i + steps + 1, j - steps : j + steps + 1]
    coords: NDArray[np.float64] = np.arange(-steps, steps + 1, dtype=np.float64) * terrain.spacing_m
    params = TerrainParams(
        terrain_id=terrain.params.terrain_id,
        relief_m=terrain.params.relief_m,
        beta=terrain.params.beta,
    )
    return (
        Terrain(params=params, spacing_m=terrain.spacing_m, coords_m=coords, heights_m=heights),
        Site(site.site_class, 0.0, 0.0, site.ground_m, site.antenna_m, site.percentile),
    )


def transform_terrain(crop: Terrain, k: int, mirror: bool) -> Terrain:
    """Apply a variant to a site-centred crop (the site stays at the origin).

    Args:
        crop: Output of `centred_crop`.
        k: Clockwise quarter turns.
        mirror: Mirror east-west first.

    Returns:
        The transformed terrain.
    """
    return Terrain(
        params=crop.params,
        spacing_m=crop.spacing_m,
        coords_m=crop.coords_m,
        heights_m=transform_raster(crop.heights_m, k, mirror),
    )
