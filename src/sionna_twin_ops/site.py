"""Site placement and map geometry (spec rule S).

The site sits on a terrain grid vertex inside the central 1 x 1 km of the tile, and the
128 x 128 map of 40 m cells is centred on it, so the site is always the map centre.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from sionna_twin_ops.terrain import TILE_SIZE_M, Terrain

SiteClass = Literal["hilltop", "slope", "valley"]
SITE_CLASSES: tuple[SiteClass, ...] = ("hilltop", "slope", "valley")

MAST_HEIGHT_M = 30.0  # ASSUMPTION (spec S1): antenna 30 m above local ground
SEARCH_HALF_WIDTH_M = 500.0  # spec S2: central 1 x 1 km of the tile
MAP_CELLS = 128  # spec S3
MAP_CELL_M = 40.0  # spec S3
MAP_SIZE_M = MAP_CELLS * MAP_CELL_M


@dataclass(frozen=True)
class Site:
    """One sector site.

    Attributes:
        site_class: How the site was chosen.
        x_m: East coordinate of the site.
        y_m: North coordinate of the site.
        ground_m: Ground height at the site.
        antenna_m: Antenna height, MAST_HEIGHT_M above the ground.
    """

    site_class: SiteClass
    x_m: float
    y_m: float
    ground_m: float
    antenna_m: float


def site_class_for(terrain_id: int) -> SiteClass:
    """Site class of a terrain: the classes cycle with the id, so any 3 consecutive ids
    hold one of each and the dataset stays balanced (spec S2).

    Args:
        terrain_id: Terrain index.

    Returns:
        The site class for that terrain.
    """
    return SITE_CLASSES[terrain_id % len(SITE_CLASSES)]


def place_site(terrain: Terrain, site_class: SiteClass) -> Site:
    """Choose the site vertex by the rule for its class (spec S2).

    Inside the central 1 x 1 km: hilltop is the highest vertex, valley the lowest, and
    slope the vertex with the largest gradient among those whose height lies in the
    middle third of the height range inside that same window.

    Args:
        terrain: Terrain to place the site on.
        site_class: Which rule to apply.

    Returns:
        The chosen site.

    Raises:
        ValueError: For a slope site on a flat window, or if no vertex lies in the middle
            third.
    """
    coords = terrain.coords_m
    inside = np.flatnonzero(np.abs(coords) <= SEARCH_HALF_WIDTH_M)
    window = terrain.heights_m[np.ix_(inside, inside)]

    if site_class == "hilltop":
        flat_index = int(np.argmax(window))
    elif site_class == "valley":
        flat_index = int(np.argmin(window))
    else:
        grad_y, grad_x = np.gradient(terrain.heights_m, terrain.spacing_m)
        gradient = np.hypot(grad_x, grad_y)[np.ix_(inside, inside)]
        low, high = float(window.min()), float(window.max())
        third = (high - low) / 3.0
        middle = (window >= low + third) & (window <= high - third)
        if third <= 0.0 or not middle.any():
            raise ValueError(
                f"terrain {terrain.params.terrain_id}: no vertex in the middle third of "
                f"the window height range [{low:.1f}, {high:.1f}] m"
            )
        flat_index = int(np.argmax(np.where(middle, gradient, -np.inf)))

    row, col = np.unravel_index(flat_index, window.shape)
    i, j = int(inside[row]), int(inside[col])
    ground = float(terrain.heights_m[i, j])
    return Site(
        site_class=site_class,
        x_m=float(coords[j]),
        y_m=float(coords[i]),
        ground_m=ground,
        antenna_m=ground + MAST_HEIGHT_M,
    )


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
