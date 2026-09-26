"""Sionna array and scene geometry (spec rules A and S4). No ray tracing."""

import numpy as np
import pytest

pytestmark = pytest.mark.sionna

from sionna_twin_ops.antenna import element_heights_wl  # noqa: E402
from sionna_twin_ops.scene import grid_faces, measurement_surface, sector_array  # noqa: E402
from sionna_twin_ops.site import (  # noqa: E402
    MAP_CELL_M,
    MAP_CELLS,
    SURFACE_HEIGHT_M,
    place_site,
    site_class_for,
)
from sionna_twin_ops.terrain import generate_terrain  # noqa: E402


def test_numpy_element_heights_match_sionna_array_geometry() -> None:
    sionna_z = np.asarray(sector_array().normalized_positions.z.numpy(), dtype=np.float64)
    assert np.allclose(element_heights_wl(), sionna_z)


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
