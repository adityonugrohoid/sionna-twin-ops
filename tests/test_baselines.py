"""Baselines B0, B1 and the LOS mask (spec rule B)."""

from dataclasses import replace

import numpy as np
import pytest

from sionna_twin_ops.baselines import (
    WAVELENGTH_M,
    MapGeometry,
    b0_gain_db,
    b1_gain_db,
    bullington_loss_db,
    bullington_nu,
    free_space_gain_db,
    ground_at,
    knife_edge_loss_db,
    los_mask,
    map_geometry,
)
from sionna_twin_ops.site import MAST_HEIGHT_M, Site, place_site, site_class_for
from sionna_twin_ops.terrain import generate_terrain


def fresnel_j(nu: float) -> float:
    """J(nu) from P.526-16 equation (30), Fresnel integrals by numerical quadrature."""
    t = np.linspace(0.0, nu, 200_001)
    c = np.trapezoid(np.cos(np.pi * t**2 / 2), t)
    s = np.trapezoid(np.sin(np.pi * t**2 / 2), t)
    return float(-20.0 * np.log10(np.sqrt((1 - c - s) ** 2 + (c - s) ** 2) / 2))


def flat_geometry() -> tuple[MapGeometry, Site]:
    terrain = generate_terrain(0, 40.0)
    flat = replace(terrain, heights_m=np.zeros_like(terrain.heights_m))
    site = Site("slope", 0.0, 0.0, 0.0, MAST_HEIGHT_M, 50.0)
    return map_geometry(flat, site), site


@pytest.mark.parametrize("nu", [-0.7, -0.3, 0.0, 0.5, 1.0, 2.0, 3.5, 5.0])
def test_equation_31_tracks_equation_30(nu: float) -> None:
    approx = float(knife_edge_loss_db(np.array([nu]))[0])
    assert approx == pytest.approx(fresnel_j(nu), abs=0.15)


def test_knife_edge_loss_is_zero_at_and_below_the_limit() -> None:
    assert np.all(knife_edge_loss_db(np.array([-0.78, -1.0, -5.0])) == 0.0)
    assert knife_edge_loss_db(np.array([0.0]))[0] == pytest.approx(6.03, abs=0.01)


def test_single_knife_edge_gives_the_equation_26_nu() -> None:
    # One 60 m edge 800 m along a 2 km flat path; antenna 30 m, receiver 1.5 m.
    n = 99
    d_km = 2.0
    di = np.arange(1, n + 1) / (n + 1) * d_km
    hi = np.zeros(n)
    k = int(np.argmin(np.abs(di - 0.8)))
    hi[k] = 60.0
    site = Site("slope", 0.0, 0.0, 0.0, 30.0, 50.0)
    geometry = MapGeometry(
        site=site,
        x_m=np.array([[2000.0]]),
        y_m=np.array([[0.0]]),
        rx_m=np.array([[1.5]]),
        distance_km=np.array([[d_km]]),
        profile_km=di[None, None, :],
        profile_m=hi[None, None, :],
    )
    assert not los_mask(geometry)[0, 0]
    d1, d2 = di[k] * 1000.0, (d_km - di[k]) * 1000.0
    h = 60.0 - (30.0 + (1.5 - 30.0) * d1 / (d1 + d2))  # edge height above the direct line
    nu_26 = h * np.sqrt(2.0 / WAVELENGTH_M * (1.0 / d1 + 1.0 / d2))
    assert bullington_nu(geometry)[0, 0] == pytest.approx(nu_26, rel=1e-9)
    j = float(knife_edge_loss_db(np.array([nu_26]))[0])
    expected = j + (1.0 - np.exp(-j / 6.0)) * (10.0 + 0.02 * d_km)  # equation (57)
    assert bullington_loss_db(geometry)[0, 0] == pytest.approx(expected, rel=1e-9)


def test_flat_tile_is_all_line_of_sight_with_little_diffraction_loss() -> None:
    geometry, _ = flat_geometry()
    assert los_mask(geometry).all()
    loss = bullington_loss_db(geometry)
    # A 1.5 m receiver over flat ground does not clear its first Fresnel zone at the map
    # corners, so nu rises just above -0.78 there and J(nu) is a few hundredths of a dB.
    assert np.all(loss[geometry.distance_km < 3.0] == 0.0)
    assert loss.max() < 0.1
    b0, b1 = b0_gain_db(geometry, 90.0, 6.0), b1_gain_db(geometry, 90.0, 6.0)
    assert np.allclose(b1, b0 - loss)


def test_cells_behind_a_ridge_are_not_line_of_sight() -> None:
    terrain = generate_terrain(0, 40.0)
    x, _ = np.meshgrid(terrain.coords_m, terrain.coords_m)
    ridge = replace(terrain, heights_m=np.where(np.abs(x - 1000.0) <= 40.0, 60.0, 0.0))
    site = Site("slope", 0.0, 0.0, 0.0, MAST_HEIGHT_M, 50.0)
    geometry = map_geometry(ridge, site)
    los = los_mask(geometry)
    assert los[:, geometry.x_m[0] < 900.0].all()
    assert not los[:, geometry.x_m[0] > 1300.0].any()
    assert (bullington_loss_db(geometry)[:, geometry.x_m[0] > 1300.0] > 10.0).all()


def test_los_mask_matches_an_independent_ray_march() -> None:
    terrain = generate_terrain(3, 40.0)
    site = place_site(terrain, site_class_for(3))
    geometry = map_geometry(terrain, site)
    rng = np.random.default_rng(0)
    rows, cols = rng.integers(0, 128, 200), rng.integers(0, 128, 200)
    u = np.linspace(0.0, 1.0, 2001)[1:-1]
    for r, c in zip(rows, cols, strict=True):
        x = site.x_m + u * (geometry.x_m[r, c] - site.x_m)
        y = site.y_m + u * (geometry.y_m[r, c] - site.y_m)
        line = site.antenna_m + u * (geometry.rx_m[r, c] - site.antenna_m)
        clear = bool((ground_at(terrain, x, y) < line).all())
        # The mask samples 256 points per path; disagreement is possible only where the
        # line grazes the ground between samples, so allow none at a 1 m margin.
        margin = (line - ground_at(terrain, x, y)).min()
        if abs(margin) > 1.0:
            assert los_mask(geometry)[r, c] == clear


def test_free_space_gain_follows_p525_equation_5() -> None:
    # 1 km at 1.8 GHz: 20 log10(4 pi 1000 / lambda) = 97.55 dB.
    assert free_space_gain_db(np.array([1000.0]))[0] == pytest.approx(-97.55, abs=0.01)


@pytest.mark.sionna
def test_b0_matches_sionna_in_free_space() -> None:
    from sionna_twin_ops.antenna import CARRIER_HZ, tilt_weights, yaw_for_azimuth
    from sionna_twin_ops.checks import planar_map
    from sionna_twin_ops.scene import receiver_array, sector_array
    from sionna_twin_ops.sionna_rt import mi, rt
    from sionna_twin_ops.solve import specular_settings

    geometry, site = flat_geometry()
    scene = rt.load_scene()  # empty: no ground, so free space
    scene.frequency = CARRIER_HZ
    scene.tx_array = sector_array()
    scene.rx_array = receiver_array()
    scene.add(
        rt.Transmitter(
            "sector",
            position=mi.Point3f(site.x_m, site.y_m, site.antenna_m),
            orientation=mi.Point3f(yaw_for_azimuth(90.0), 0.0, 0.0),
        )
    )
    # 1e8 rays: at 1e7 the far free-space cells scatter by about 2 dB from sampling alone.
    traced = planar_map(scene, tilt_weights(6.0), specular_settings(10**8, 1))
    b0 = b0_gain_db(geometry, 90.0, 6.0)
    far = geometry.distance_km > 0.5
    assert (traced[far] > 0).all()
    diff = 10.0 * np.log10(traced[far]) - b0[far]
    assert abs(np.median(diff)) < 0.05
    assert np.percentile(np.abs(diff), 95) < 0.5
