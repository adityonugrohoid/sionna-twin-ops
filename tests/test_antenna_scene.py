"""Antenna weights and scene geometry (spec rules A and S4). No ray tracing."""

import numpy as np
import pytest

pytestmark = pytest.mark.sionna

from sionna_twin_ops.antenna import (  # noqa: E402
    NUM_ROWS,
    VERTICAL_SPACING_WL,
    sector_array,
    tilt_weights,
    yaw_for_azimuth,
)
from sionna_twin_ops.scene import SURFACE_HEIGHT_M, grid_faces, measurement_surface  # noqa: E402
from sionna_twin_ops.site import MAP_CELL_M, MAP_CELLS, place_site, site_class_for  # noqa: E402
from sionna_twin_ops.terrain import generate_terrain  # noqa: E402


@pytest.mark.parametrize("tilt", [0.0, 3.0, 12.0])
def test_tilt_weights_have_unit_power_and_a_linear_phase_taper(tilt: float) -> None:
    w = tilt_weights(sector_array(), tilt)
    assert w.shape == (NUM_ROWS,)
    assert np.sum(np.abs(w) ** 2) == pytest.approx(1.0)
    # Row 0 is the top element; the phase steps down by 2 pi d sin(tilt) per row.
    step = np.angle(w[1:] * np.conj(w[:-1]))
    expected = -2.0 * np.pi * VERTICAL_SPACING_WL * np.sin(np.radians(tilt))
    assert np.allclose(step, expected)


def test_zero_tilt_is_uniform() -> None:
    w = tilt_weights(sector_array(), 0.0)
    assert np.allclose(w, 1.0 / np.sqrt(NUM_ROWS))


@pytest.mark.parametrize(
    ("azimuth", "yaw"), [(0.0, 90.0), (90.0, 0.0), (180.0, -90.0), (270.0, -180.0)]
)
def test_azimuth_is_clockwise_from_north(azimuth: float, yaw: float) -> None:
    assert np.degrees(yaw_for_azimuth(azimuth)) == pytest.approx(yaw)


def test_grid_faces_cover_each_cell_with_two_triangles() -> None:
    faces = grid_faces(3, 4)  # 2 x 3 cells
    assert faces.shape == (12, 3)
    # Cell (1, 2): vertices 6, 7, 10, 11; its triangles are faces 10 and 11.
    assert set(faces[10]) | set(faces[11]) == {6, 7, 10, 11}
    assert set(faces[10]) & set(faces[11]) == {6, 11}  # the shared diagonal


def test_measurement_surface_follows_the_terrain_over_the_map() -> None:
    terrain = generate_terrain(3, 40.0)
    site = place_site(terrain, site_class_for(3))
    mesh = measurement_surface(terrain, site)
    assert mesh.face_count() == 2 * MAP_CELLS * MAP_CELLS
    v = np.asarray(mesh.vertex_positions_buffer().numpy()).reshape(-1, 3)
    n = MAP_CELLS + 1
    x, y, z = (v[:, k].reshape(n, n) for k in range(3))
    half = MAP_CELLS * MAP_CELL_M / 2
    assert x[0, 0] == pytest.approx(site.x_m - half)
    assert y[-1, -1] == pytest.approx(site.y_m + half)
    # The centre vertex sits 1.5 m above the site's ground.
    assert z[n // 2, n // 2] == pytest.approx(site.ground_m + SURFACE_HEIGHT_M, abs=1e-3)
    i = np.searchsorted(terrain.coords_m, y[:, 0])
    j = np.searchsorted(terrain.coords_m, x[0, :])
    assert np.allclose(z, terrain.heights_m[np.ix_(i, j)] + SURFACE_HEIGHT_M, atol=1e-3)
