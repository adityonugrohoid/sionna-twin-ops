"""Figures for review: sample terrain hillshades and site placement."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LightSource, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from sionna_twin_ops.site import SEARCH_HALF_WIDTH_M, map_bounds, place_site, site_class_for
from sionna_twin_ops.terrain import BASE_SEED, RELIEF_RANGE_M, Terrain, generate_terrain

HEIGHT_NORM = Normalize(vmin=0.0, vmax=RELIEF_RANGE_M[1])
CMAP = "gist_earth"
SITE_STYLE = {
    "hilltop": ("^", "tab:red"),
    "slope": ("D", "tab:orange"),
    "valley": ("v", "tab:blue"),
}


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


def _colourbar(fig: Figure, axes: np.ndarray) -> None:
    """Add the shared height colour bar."""
    fig.colorbar(
        plt.cm.ScalarMappable(norm=HEIGHT_NORM, cmap=CMAP),
        ax=axes,
        shrink=0.5,
        label="height above lowest vertex (m)",
    )


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
        rows, cols, figsize=(2.6 * cols, 2.8 * rows), squeeze=False, layout="constrained"
    )
    for ax, terrain_id in zip(axes.flat, ids, strict=False):
        terrain = generate_terrain(terrain_id, spacing_m)
        p = terrain.params
        _show(ax, terrain)
        ax.set_title(
            f"id {terrain_id}: relief {p.relief_m:.0f} m\n"
            f"beta {p.beta:.2f}, ridge weight {p.ridge_weight:.2f}",
            fontsize=6,
            loc="left",
        )
        ax.tick_params(labelsize=6)
    for ax in axes.flat[len(ids) :]:
        ax.axis("off")
    _colourbar(fig, axes)
    fig.suptitle(
        f"Synthetic terrain, 10 x 10 km tiles, {spacing_m:.0f} m grid, base seed {BASE_SEED}. "
        "Axes in km. Shared colour scale, no vertical exaggeration.",
        fontsize=8,
    )
    fig.savefig(path, dpi=90, pil_kwargs={"quality": 85})
    plt.close(fig)


def site_placement(ids: list[int], spacing_m: float, path: Path) -> None:
    """One panel per terrain with the site of its own class and the site's map extent.

    Args:
        ids: Terrain ids, one panel each.
        spacing_m: Grid spacing of the terrains.
        path: Output image file (JPEG).
    """
    fig, axes = plt.subplots(
        1, len(ids), figsize=(4.6 * len(ids), 5.0), squeeze=False, layout="constrained"
    )
    w = SEARCH_HALF_WIDTH_M / 1000.0
    for ax, terrain_id in zip(axes.flat, ids, strict=True):
        terrain = generate_terrain(terrain_id, spacing_m)
        site = place_site(terrain, site_class_for(terrain_id))
        marker, colour = SITE_STYLE[site.site_class]
        _show(ax, terrain)
        ax.add_patch(Rectangle((-w, -w), 2 * w, 2 * w, fill=False, ls="--", ec="white", lw=1))
        x_min, x_max, y_min, y_max = (v / 1000.0 for v in map_bounds(site))
        ax.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, ec=colour))
        ax.plot(site.x_m / 1000.0, site.y_m / 1000.0, marker, color=colour, mec="black", ms=10)
        p = terrain.params
        ax.set_title(
            f"id {terrain_id} {site.site_class}: ({site.x_m:.0f}, {site.y_m:.0f}) m, "
            f"ground {site.ground_m:.0f} m, map percentile {site.percentile:.0f}\n"
            f"relief {p.relief_m:.0f} m, beta {p.beta:.2f}, ridge weight {p.ridge_weight:.2f}",
            fontsize=7,
            loc="left",
        )
        ax.set_xlabel("x east (km)", fontsize=7)
        ax.set_ylabel("y north (km)", fontsize=7)
        ax.tick_params(labelsize=6)
    _colourbar(fig, axes)
    fig.suptitle(
        "Synthetic terrain. Dashed: 3 x 3 km search window. Box: the site's 5.12 km map.",
        fontsize=8,
    )
    fig.savefig(path, dpi=100, pil_kwargs={"quality": 85})
    plt.close(fig)
