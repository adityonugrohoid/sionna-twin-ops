"""Sionna RT scene: terrain mesh, measurement surface and the sector transmitter.

Both meshes are built in memory from the heightmap; no scene file is read or written.
"""

import numpy as np
from numpy.typing import NDArray

from sionna_twin_ops.antenna import (
    CARRIER_HZ,
    receiver_array,
    sector_array,
    yaw_for_azimuth,
)
from sionna_twin_ops.sionna_rt import mi, rt
from sionna_twin_ops.site import MAP_CELL_M, MAP_CELLS, Site
from sionna_twin_ops.terrain import Terrain

GROUND_MATERIAL = "medium_dry_ground"  # ASSUMPTION (spec T4): one ITU material everywhere
# ASSUMPTION: Sionna models a material as a slab; 10 m of medium dry ground at 1.8 GHz is
# opaque, so the slab stands in for a half-space.
GROUND_THICKNESS_M = 10.0
SURFACE_HEIGHT_M = 1.5  # spec S4: measurement surface above ground


def grid_faces(rows: int, cols: int) -> NDArray[np.uint32]:
    """Two triangles per cell of a vertex grid, row-major by cell.

    Cell (r, c) has vertices (r, c), (r, c+1), (r+1, c), (r+1, c+1) and is split along the
    diagonal from (r, c) to (r+1, c+1). Its triangles are faces 2 * (r * (cols - 1) + c)
    and the one after, both counter-clockwise seen from above.

    Args:
        rows: Vertex rows.
        cols: Vertex columns.

    Returns:
        Face indices, shape (2 * (rows - 1) * (cols - 1), 3).
    """
    r, c = np.meshgrid(np.arange(rows - 1), np.arange(cols - 1), indexing="ij")
    v00 = (r * cols + c).ravel()
    v01 = v00 + 1
    v10 = v00 + cols
    v11 = v10 + 1
    first = np.stack([v00, v01, v11], axis=1)
    second = np.stack([v00, v11, v10], axis=1)
    return np.stack([first, second], axis=1).reshape(-1, 3).astype(np.uint32)


def grid_mesh(
    name: str, x: NDArray[np.float64], y: NDArray[np.float64], z: NDArray[np.float64]
) -> mi.Mesh:
    """A triangle mesh over a regular vertex grid.

    Args:
        name: Mesh id.
        x: Vertex x coordinates along columns.
        y: Vertex y coordinates along rows.
        z: Vertex heights, shape (len(y), len(x)).

    Returns:
        The Mitsuba mesh.
    """
    xx, yy = np.meshgrid(x, y)
    vertices = np.stack([xx.ravel(), yy.ravel(), z.ravel()], axis=1).astype(np.float32)
    faces = grid_faces(len(y), len(x))
    mesh = mi.Mesh(name, len(vertices), len(faces))
    params = mi.traverse(mesh)
    params["vertex_positions"] = mi.Float(vertices.ravel())
    params["faces"] = mi.UInt32(faces.ravel())
    params.update()
    return mesh


def terrain_mesh(terrain: Terrain) -> mi.Mesh:
    """The whole terrain tile as one mesh, no walls or skirts (spec T6).

    Args:
        terrain: The terrain.

    Returns:
        The mesh.
    """
    return grid_mesh("terrain", terrain.coords_m, terrain.coords_m, terrain.heights_m)


def measurement_surface(terrain: Terrain, site: Site) -> mi.Mesh:
    """The map's measurement surface: 1.5 m above ground, two triangles per cell (spec S4).

    Cell corners sit on terrain vertices, so the map grid spacing must be a whole multiple
    of the terrain grid spacing. Cell (r, c) is row r from the south edge, column c from
    the west edge; its two triangles are faces 2 * (r * 128 + c) and the one after.

    Args:
        terrain: The terrain.
        site: The map centre.

    Returns:
        The mesh.

    Raises:
        ValueError: If the map grid does not land on terrain vertices.
    """
    step = MAP_CELL_M / terrain.spacing_m
    if step != round(step):
        raise ValueError(f"{MAP_CELL_M} m cells need a terrain spacing that divides them")
    step_i = round(step)
    i0 = int(np.searchsorted(terrain.coords_m, site.y_m)) - MAP_CELLS // 2 * step_i
    j0 = int(np.searchsorted(terrain.coords_m, site.x_m)) - MAP_CELLS // 2 * step_i
    rows = slice(i0, i0 + MAP_CELLS * step_i + 1, step_i)
    cols = slice(j0, j0 + MAP_CELLS * step_i + 1, step_i)
    x = terrain.coords_m[cols]
    y = terrain.coords_m[rows]
    if (
        x[0] != site.x_m - MAP_CELLS * MAP_CELL_M / 2
        or y[0] != site.y_m - MAP_CELLS * MAP_CELL_M / 2
    ):
        raise ValueError("map grid is not centred on the site")
    z = terrain.heights_m[rows, cols] + SURFACE_HEIGHT_M
    return grid_mesh("measurement-surface", x, y, z)


def build_scene(terrain: Terrain, site: Site, azimuth_deg: float) -> rt.Scene:
    """Terrain plus one sector transmitter at the site, boresight at `azimuth_deg`.

    Args:
        terrain: The terrain.
        site: Where the mast stands; the antenna is at `site.antenna_m`.
        azimuth_deg: Boresight azimuth, clockwise from true north.

    Returns:
        The scene, ready for the radio map solver.
    """
    scene = rt.load_scene()
    scene.frequency = CARRIER_HZ
    material = rt.ITURadioMaterial("ground", GROUND_MATERIAL, thickness=GROUND_THICKNESS_M)
    scene.edit(
        add=rt.SceneObject(mi_mesh=terrain_mesh(terrain), name="terrain", radio_material=material)
    )
    scene.tx_array = sector_array()
    scene.rx_array = receiver_array()
    scene.add(
        rt.Transmitter(
            "sector",
            position=mi.Point3f(site.x_m, site.y_m, site.antenna_m),
            orientation=mi.Point3f(yaw_for_azimuth(azimuth_deg), 0.0, 0.0),
        )
    )
    return scene
