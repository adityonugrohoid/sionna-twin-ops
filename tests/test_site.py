"""Site placement and map geometry (spec rule S)."""

from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from sionna_twin_ops.site import (
    DISC_RADIUS_M,
    HILLTOP_MIN_PERCENTILE,
    MAP_SIZE_M,
    MAST_HEIGHT_M,
    SEARCH_HALF_WIDTH_M,
    SITE_CLASSES,
    SLOPE_PERCENTILE_RANGE,
    VALLEY_MAX_PERCENTILE,
    NoSiteError,
    Site,
    baseline_gradient,
    guard_band_m,
    map_bounds,
    map_percentile,
    place_site,
    site_check,
    site_class_for,
)
from sionna_twin_ops.terrain import Terrain, generate_terrain


def index_of(terrain: Terrain, site: Site) -> tuple[int, int]:
    """Grid (row, column) of the site vertex."""
    return (
        int(np.searchsorted(terrain.coords_m, site.y_m)),
        int(np.searchsorted(terrain.coords_m, site.x_m)),
    )


def disc(terrain: Terrain, i: int, j: int) -> np.ndarray:
    """Heights within DISC_RADIUS_M of vertex (i, j)."""
    x, y = np.meshgrid(terrain.coords_m, terrain.coords_m)
    inside = np.hypot(x - terrain.coords_m[j], y - terrain.coords_m[i]) <= DISC_RADIUS_M
    heights: np.ndarray = terrain.heights_m[inside]
    return heights


def placed(terrain_id: int) -> tuple[Terrain, Site]:
    terrain = generate_terrain(terrain_id, 40.0)
    return terrain, place_site(terrain, site_class_for(terrain_id))


@pytest.mark.parametrize("terrain_id", [0, 3, 6, 9])
def test_hilltop_rule(terrain_id: int) -> None:
    terrain, site = placed(terrain_id)
    i, j = index_of(terrain, site)
    assert site.ground_m == disc(terrain, i, j).max()
    assert site.percentile >= HILLTOP_MIN_PERCENTILE
    assert site.percentile == map_percentile(terrain, i, j)


@pytest.mark.parametrize("terrain_id", [5, 8, 11, 14])
def test_valley_rule(terrain_id: int) -> None:
    terrain, site = placed(terrain_id)
    i, j = index_of(terrain, site)
    assert site.ground_m == disc(terrain, i, j).min()
    assert site.percentile <= VALLEY_MAX_PERCENTILE


@pytest.mark.parametrize("terrain_id", [1, 4, 7])
def test_slope_rule_picks_the_steepest_qualifying_vertex(terrain_id: int) -> None:
    terrain, site = placed(terrain_id)
    low, high = SLOPE_PERCENTILE_RANGE
    assert low <= site.percentile <= high
    half = int(SEARCH_HALF_WIDTH_M // terrain.spacing_m)
    centre = len(terrain.coords_m) // 2
    rows = cols = slice(centre - half, centre + half + 1)
    gradient = baseline_gradient(terrain.heights_m, rows, cols, terrain.spacing_m)
    i, j = index_of(terrain, site)
    site_gradient = gradient[i - rows.start, j - cols.start]
    # Every steeper vertex in the window fails the percentile band.
    for r, c in zip(*np.nonzero(gradient > site_gradient), strict=True):
        p = map_percentile(terrain, rows.start + int(r), cols.start + int(c))
        assert not low <= p <= high


def test_baseline_gradient_is_exact_on_a_plane() -> None:
    terrain = generate_terrain(0, 40.0)
    x, y = np.meshgrid(terrain.coords_m, terrain.coords_m)
    plane = replace(terrain, heights_m=0.03 * x - 0.04 * y)
    gradient = baseline_gradient(plane.heights_m, slice(100, 150), slice(100, 150), 40.0)
    assert np.allclose(gradient, 0.05)


@pytest.mark.parametrize("terrain_id", range(12))
def test_site_lies_in_window_with_guard_band(terrain_id: int) -> None:
    terrain = generate_terrain(terrain_id, 40.0)
    try:
        site = place_site(terrain, site_class_for(terrain_id))
    except NoSiteError:
        pytest.skip("no candidate for this id; counted by the site check")
    assert abs(site.x_m) <= SEARCH_HALF_WIDTH_M
    assert abs(site.y_m) <= SEARCH_HALF_WIDTH_M
    assert site.antenna_m == site.ground_m + MAST_HEIGHT_M
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


def test_flat_terrain_has_no_hilltop_or_valley() -> None:
    # Every vertex of a flat map sits at midrank percentile 50. S2 sets no minimum
    # gradient, so the slope rule is not tested here.
    terrain = generate_terrain(0, 40.0)
    flat = replace(terrain, heights_m=np.zeros_like(terrain.heights_m))
    for site_class in ("hilltop", "valley"):
        with pytest.raises(NoSiteError, match=site_class):
            place_site(flat, site_class)


def test_site_check_accounts_for_every_id() -> None:
    checks = site_check(range(12), 40.0)
    assert [c.site_class for c in checks] == list(SITE_CLASSES)
    for c in checks:
        assert len(c.ids) == 4
        assert len(c.percentiles) + len(c.skipped) == len(c.ids)
        assert set(c.skipped) <= set(c.ids)
