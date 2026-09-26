"""Site placement and map geometry (spec rule S)."""

from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from sionna_twin_ops.site import (
    MAP_SIZE_M,
    MAST_HEIGHT_M,
    SEARCH_HALF_WIDTH_M,
    SITE_CLASSES,
    guard_band_m,
    map_bounds,
    place_site,
    site_class_for,
)
from sionna_twin_ops.terrain import Terrain, generate_terrain


def window(terrain: Terrain) -> tuple[np.ndarray, np.ndarray]:
    inside = np.flatnonzero(np.abs(terrain.coords_m) <= SEARCH_HALF_WIDTH_M)
    return inside, terrain.heights_m[np.ix_(inside, inside)]


@pytest.mark.parametrize("terrain_id", range(12))
def test_each_class_follows_its_rule(terrain_id: int) -> None:
    terrain = generate_terrain(terrain_id, 40.0)
    inside, heights = window(terrain)
    sites = {c: place_site(terrain, c) for c in SITE_CLASSES}

    for site in sites.values():
        assert abs(site.x_m) <= SEARCH_HALF_WIDTH_M
        assert abs(site.y_m) <= SEARCH_HALF_WIDTH_M
        assert site.antenna_m == site.ground_m + MAST_HEIGHT_M

    assert sites["hilltop"].ground_m == heights.max()
    assert sites["valley"].ground_m == heights.min()

    low, high = heights.min(), heights.max()
    third = (high - low) / 3.0
    slope = sites["slope"]
    assert low + third <= slope.ground_m <= high - third
    grad_y, grad_x = np.gradient(terrain.heights_m, terrain.spacing_m)
    gradient = np.hypot(grad_x, grad_y)[np.ix_(inside, inside)]
    middle = (heights >= low + third) & (heights <= high - third)
    i = int(np.searchsorted(terrain.coords_m, slope.y_m))
    j = int(np.searchsorted(terrain.coords_m, slope.x_m))
    assert np.hypot(grad_x, grad_y)[i, j] == gradient[middle].max()


def test_map_is_centred_on_the_site_with_a_guard_band() -> None:
    for terrain_id in range(30):
        terrain = generate_terrain(terrain_id, 40.0)
        site = place_site(terrain, site_class_for(terrain_id))
        x_min, x_max, y_min, y_max = map_bounds(site)
        assert x_max - x_min == MAP_SIZE_M == 5120.0
        assert (x_min + x_max) / 2 == site.x_m
        assert (y_min + y_max) / 2 == site.y_m
        assert guard_band_m(site) >= 940.0


def test_classes_are_balanced_over_consecutive_ids() -> None:
    counts = Counter(site_class_for(i) for i in range(60))
    assert counts == {"hilltop": 20, "slope": 20, "valley": 20}
    for start in range(0, 60, 3):
        assert {site_class_for(i) for i in range(start, start + 3)} == set(SITE_CLASSES)


def test_flat_window_has_no_slope_site() -> None:
    terrain = generate_terrain(0, 40.0)
    flat = replace(terrain, heights_m=np.zeros_like(terrain.heights_m))
    with pytest.raises(ValueError, match="middle third"):
        place_site(flat, "slope")
