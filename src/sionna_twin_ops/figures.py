"""Figures for review: sample terrain hillshades and site placement."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LightSource, Normalize
from matplotlib.patches import Rectangle

from sionna_twin_ops.site import SEARCH_HALF_WIDTH_M, SITE_CLASSES, map_bounds, place_site
from sionna_twin_ops.terrain import BASE_SEED, RELIEF_RANGE_M, Terrain, generate_terrain

HEIGHT_NORM = Normalize(vmin=0.0, vmax=RELIEF_RANGE_M[1])
CMAP = "gist_earth"
SITE_MARKERS = {"hilltop": "^", "slope": "D", "valley": "v"}


def hillshade(terrain: Terrain) -> np.ndarray:
    """RGB hillshade coloured by height on one scale shared by every terrain.

    Args:
        terrain: Terrain to shade.

    Returns:
        RGB array of shape (n, n, 3), row 0 at the south edge.
    """
    light = LightSource(azdeg=315, altdeg=45)
    rgb: np.ndarray = light.shade(
        terrain.heights_m,
        cmap=plt.get_cmap(CMAP),
        norm=HEIGHT_NORM,
        blend_mode="overlay",
        vert_exag=1.0,
        dx=terrain.spacing_m,
        dy=terrain.spacing_m,
    )
    return rgb[..., :3]


def _show(ax: Axes, terrain: Terrain) -> None:
    """Draw the terrain hillshade on `ax` with axes in km."""
    half_km = terrain.coords_m[-1] / 1000.0
    ax.imshow(hillshade(terrain), origin="lower", extent=(-half_km, half_km, -half_km, half_km))


def terrain_grid(ids: list[int], spacing_m: float, path: Path) -> None:
    """Hillshades of several terrains, each titled with its drawn parameters.

    Args:
        ids: Terrain ids to draw; laid out 4 per row.
        spacing_m: Grid spacing of the terrains.
        path: Output image file (JPEG).
    """
    cols = 4
    rows = -(-len(ids) // cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(2.6 * cols, 3.0 * rows), squeeze=False, layout="constrained"
    )
    for ax, terrain_id in zip(axes.flat, ids, strict=False):
        terrain = generate_terrain(terrain_id, spacing_m)
        p = terrain.params
        _show(ax, terrain)
        ridges = "".join(
            f"\nridge {np.degrees(r.angle_rad):.0f} deg, amp {r.amplitude:.2f}, "
            f"sigma {r.width_m:.0f} m"
            for r in p.ridges
        )
        ax.set_title(
            f"id {terrain_id}: relief {p.relief_m:.0f} m, beta {p.beta:.2f}" + ridges,
            fontsize=6,
            loc="left",
        )
        ax.tick_params(labelsize=6)
    for ax in axes.flat[len(ids) :]:
        ax.axis("off")
    fig.colorbar(
        plt.cm.ScalarMappable(norm=HEIGHT_NORM, cmap=CMAP),
        ax=axes,
        shrink=0.5,
        label="height above lowest vertex (m)",
    )
    fig.suptitle(
        f"Synthetic terrain, 8 x 8 km tiles, {spacing_m:.0f} m grid, base seed {BASE_SEED}. "
        "Axes in km. Shared colour scale, no vertical exaggeration.",
        fontsize=8,
    )
    fig.savefig(path, dpi=90, pil_kwargs={"quality": 85})
    plt.close(fig)


def site_placement(terrain_id: int, spacing_m: float, path: Path) -> None:
    """All three site classes placed on one terrain, with their map extents.

    Args:
        terrain_id: Terrain to place the sites on.
        spacing_m: Grid spacing of the terrain.
        path: Output image file (JPEG).
    """
    terrain = generate_terrain(terrain_id, spacing_m)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    _show(ax, terrain)
    w = SEARCH_HALF_WIDTH_M / 1000.0
    ax.add_patch(Rectangle((-w, -w), 2 * w, 2 * w, fill=False, ls="--", ec="white", lw=1))
    colours = {"hilltop": "tab:red", "slope": "tab:orange", "valley": "tab:blue"}
    for site_class in SITE_CLASSES:
        site = place_site(terrain, site_class)
        x_min, x_max, y_min, y_max = (v / 1000.0 for v in map_bounds(site))
        ax.add_patch(
            Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min, fill=False, ec=colours[site_class]
            )
        )
        ax.plot(
            site.x_m / 1000.0,
            site.y_m / 1000.0,
            SITE_MARKERS[site_class],
            color=colours[site_class],
            mec="black",
            ms=9,
            label=f"{site_class}: ({site.x_m:.0f}, {site.y_m:.0f}) m, ground {site.ground_m:.0f} m",
        )
    ax.legend(loc="lower left", fontsize=8)
    p = terrain.params
    ax.set_title(
        f"Synthetic terrain id {terrain_id} (relief {p.relief_m:.0f} m, beta {p.beta:.2f}, "
        f"ridges {len(p.ridges)}).\nDashed: 1 x 1 km search window. "
        "Boxes: 5.12 km map of each site.",
        fontsize=9,
    )
    ax.set_xlabel("x east (km)")
    ax.set_ylabel("y north (km)")
    fig.savefig(path, dpi=100, bbox_inches="tight", pil_kwargs={"quality": 85})
    plt.close(fig)
