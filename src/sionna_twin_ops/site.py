"""Site placement and map geometry (spec rule S).

The site sits on a terrain grid vertex inside the central 3 x 3 km of the tile, and the
128 x 128 map of 40 m cells is centred on it, so the site is always the map centre.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

from sionna_twin_ops.terrain import TILE_SIZE_M, Terrain, generate_terrain

SiteClass = Literal["hilltop", "slope", "valley"]
SITE_CLASSES: tuple[SiteClass, ...] = ("hilltop", "slope", "valley")

MAST_HEIGHT_M = 30.0  # ASSUMPTION (spec S1): antenna 30 m above local ground
SEARCH_HALF_WIDTH_M = 1500.0  # spec S2: central 3 x 3 km of the tile
DISC_RADIUS_M = 200.0  # spec S2: hilltop and valley are extremes of this disc
GRADIENT_BASELINE_M = 200.0  # spec S2: slope gradient by central difference over this
HILLTOP_MIN_PERCENTILE = 85.0  # spec S2
VALLEY_MAX_PERCENTILE = 15.0  # spec S2
SLOPE_PERCENTILE_RANGE = (35.0, 65.0)  # spec S2
MAP_CELLS = 128  # spec S3
MAP_CELL_M = 40.0  # spec S3
MAP_SIZE_M = MAP_CELLS * MAP_CELL_M


class NoSiteError(ValueError):
    """The terrain has no candidate vertex for the requested site class."""


@dataclass(frozen=True)
class Site:
    """One sector site.

    Attributes:
        site_class: How the site was chosen.
        x_m: East coordinate of the site.
        y_m: North coordinate of the site.
        ground_m: Ground height at the site.
        antenna_m: Antenna height, MAST_HEIGHT_M above the ground.
        percentile: Midrank percentile of the site's height among its map's vertices.
    """

    site_class: SiteClass
    x_m: float
    y_m: float
    ground_m: float
    antenna_m: float
    percentile: float


def site_class_for(terrain_id: int) -> SiteClass:
    """Site class of a terrain: the classes cycle with the id, so any 3 consecutive ids
    hold one of each and the dataset stays balanced (spec S2).

    Args:
        terrain_id: Terrain index.

    Returns:
        The site class for that terrain.
    """
    return SITE_CLASSES[terrain_id % len(SITE_CLASSES)]


def map_percentile(terrain: Terrain, i: int, j: int) -> float:
    """Midrank percentile of vertex (i, j) among the vertices of a map centred on it.

    Args:
        terrain: The terrain.
        i: Row (y) index of the centre vertex.
        j: Column (x) index of the centre vertex.

    Returns:
        The percentile, from 0 to 100.
    """
    half = round(MAP_SIZE_M / 2.0 / terrain.spacing_m)
    heights = terrain.heights_m
    block = heights[i - half : i + half + 1, j - half : j + half + 1]
    below = np.count_nonzero(block < heights[i, j])
    ties = np.count_nonzero(block == heights[i, j])
    return float((below + 0.5 * ties) / block.size * 100.0)


def disc_extremes(
    heights: NDArray[np.float64], rows: slice, cols: slice, radius_cells: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Highest and lowest height within a disc around every vertex of a block.

    Args:
        heights: Full heightmap.
        rows: Rows of the block; the disc must fit inside the heightmap.
        cols: Columns of the block; the disc must fit inside the heightmap.
        radius_cells: Disc radius in grid steps.

    Returns:
        (disc maximum, disc minimum), each shaped like the block.
    """
    r = radius_cells
    padded = heights[rows.start - r : rows.stop + r, cols.start - r : cols.stop + r]
    windows = sliding_window_view(padded, (2 * r + 1, 2 * r + 1))
    offsets = np.arange(-r, r + 1)
    disc = np.hypot(offsets[None, :], offsets[:, None]) <= r
    return (
        np.where(disc, windows, -np.inf).max(axis=(-2, -1)),
        np.where(disc, windows, np.inf).min(axis=(-2, -1)),
    )


def baseline_gradient(
    heights: NDArray[np.float64], rows: slice, cols: slice, spacing_m: float
) -> NDArray[np.float64]:
    """Gradient magnitude by central differences over GRADIENT_BASELINE_M.

    Heights at +-half the baseline are linearly interpolated between grid vertices
    when half the baseline is not a whole number of grid steps.

    Args:
        heights: Full heightmap.
        rows: Rows of the block; the baseline must fit inside the heightmap.
        cols: Columns of the block; the baseline must fit inside the heightmap.
        spacing_m: Grid spacing.

    Returns:
        Gradient magnitude (m per m), shaped like the block.
    """
    steps = GRADIENT_BASELINE_M / 2.0 / spacing_m
    whole = int(np.floor(steps))
    frac = steps - whole
    i = np.arange(rows.start, rows.stop)[:, None]
    j = np.arange(cols.start, cols.stop)[None, :]

    def at(di: int, dj: int) -> NDArray[np.float64]:
        value: NDArray[np.float64] = heights[i + di, j + dj]
        return value

    def shifted(sign: int, axis: int) -> NDArray[np.float64]:
        near = (sign * whole, 0) if axis == 0 else (0, sign * whole)
        far = (sign * (whole + 1), 0) if axis == 0 else (0, sign * (whole + 1))
        if frac == 0.0:
            return at(*near)
        return (1.0 - frac) * at(*near) + frac * at(*far)

    grad_y = (shifted(1, 0) - shifted(-1, 0)) / GRADIENT_BASELINE_M
    grad_x = (shifted(1, 1) - shifted(-1, 1)) / GRADIENT_BASELINE_M
    return np.hypot(grad_x, grad_y)


def _ordered_candidates(terrain: Terrain, site_class: SiteClass) -> Iterator[tuple[int, int]]:
    """Candidate vertices for a class, most preferred first, before the percentile test."""
    half = int(SEARCH_HALF_WIDTH_M // terrain.spacing_m)
    centre = len(terrain.coords_m) // 2
    rows = cols = slice(centre - half, centre + half + 1)
    block = terrain.heights_m[rows, cols]

    if site_class == "slope":
        key = -baseline_gradient(terrain.heights_m, rows, cols, terrain.spacing_m)
        eligible = np.ones_like(block, dtype=bool)
    else:
        radius = round(DISC_RADIUS_M / terrain.spacing_m)
        disc_max, disc_min = disc_extremes(terrain.heights_m, rows, cols, radius)
        if site_class == "hilltop":
            eligible, key = block == disc_max, -block
        else:
            eligible, key = block == disc_min, block

    flat = np.flatnonzero(eligible.ravel())
    for index in flat[np.argsort(key.ravel()[flat], kind="stable")]:
        r, c = divmod(int(index), block.shape[1])
        yield rows.start + r, cols.start + c


def _passes(site_class: SiteClass, percentile: float) -> bool:
    """Whether a candidate's map percentile qualifies it for the class."""
    if site_class == "hilltop":
        return percentile >= HILLTOP_MIN_PERCENTILE
    if site_class == "valley":
        return percentile <= VALLEY_MAX_PERCENTILE
    low, high = SLOPE_PERCENTILE_RANGE
    return low <= percentile <= high


def place_site(terrain: Terrain, site_class: SiteClass) -> Site:
    """Choose the site vertex by the rule for its class (spec S2).

    Candidates lie inside the central 3 x 3 km; percentiles are taken over the vertices
    of the site's own map. Hilltop: the highest vertex that is the highest point of the
    200 m disc around it, at percentile >= 85. Valley: the lowest vertex that is the
    lowest point of its 200 m disc, at percentile <= 15. Slope: the vertex with the
    largest gradient over a 200 m baseline among those at percentile 35 to 65.

    Args:
        terrain: Terrain to place the site on.
        site_class: Which rule to apply.

    Returns:
        The chosen site.

    Raises:
        NoSiteError: If no vertex qualifies.
    """
    for i, j in _ordered_candidates(terrain, site_class):
        percentile = map_percentile(terrain, i, j)
        if _passes(site_class, percentile):
            ground = float(terrain.heights_m[i, j])
            return Site(
                site_class=site_class,
                x_m=float(terrain.coords_m[j]),
                y_m=float(terrain.coords_m[i]),
                ground_m=ground,
                antenna_m=ground + MAST_HEIGHT_M,
                percentile=percentile,
            )
    raise NoSiteError(f"terrain {terrain.params.terrain_id}: no {site_class} candidate")


def map_bounds(site: Site) -> tuple[float, float, float, float]:
    """Edges of the map centred on the site (spec S3).

    Args:
        site: The map centre.

    Returns:
        (x_min, x_max, y_min, y_max) in metres.
    """
    half = MAP_SIZE_M / 2.0
    return (site.x_m - half, site.x_m + half, site.y_m - half, site.y_m + half)


def guard_band_m(site: Site) -> float:
    """Smallest distance from a map edge to the tile edge.

    Args:
        site: The map centre.

    Returns:
        The guard band in metres; at least 940 m for any site in the search window.
    """
    x_min, x_max, y_min, y_max = map_bounds(site)
    tile_half = TILE_SIZE_M / 2.0
    return min(x_min + tile_half, tile_half - x_max, y_min + tile_half, tile_half - y_max)


@dataclass(frozen=True)
class ClassCheck:
    """Outcome of placing one site class over a set of terrain ids.

    Attributes:
        site_class: The class checked.
        ids: Terrain ids assigned this class.
        skipped: Ids with no qualifying vertex.
        percentiles: Map percentile of each placed site, in id order.
    """

    site_class: SiteClass
    ids: tuple[int, ...]
    skipped: tuple[int, ...]
    percentiles: tuple[float, ...]


def site_check(ids: range, spacing_m: float) -> list[ClassCheck]:
    """Place each id's own class (spec S2) and collect percentiles and skips per class.

    Args:
        ids: Terrain ids to check.
        spacing_m: Grid spacing of the terrains.

    Returns:
        One entry per site class, in SITE_CLASSES order.
    """
    checks = []
    for site_class in SITE_CLASSES:
        own = tuple(i for i in ids if site_class_for(i) == site_class)
        skipped, percentiles = [], []
        for terrain_id in own:
            try:
                site = place_site(generate_terrain(terrain_id, spacing_m), site_class)
            except NoSiteError:
                skipped.append(terrain_id)
                continue
            percentiles.append(site.percentile)
        checks.append(ClassCheck(site_class, own, tuple(skipped), tuple(percentiles)))
    return checks
