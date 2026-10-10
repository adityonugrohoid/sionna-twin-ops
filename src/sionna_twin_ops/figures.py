"""Figures for review: sample terrain hillshades, site placement, and a still of the 3D viewer."""

import textwrap
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LightSource, Normalize, to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D

from sionna_twin_ops.site import (
    MAP_CELL_M,
    SEARCH_HALF_WIDTH_M,
    map_bounds,
    place_site,
    site_class_for,
)
from sionna_twin_ops.terrain import BASE_SEED, RELIEF_RANGE_M, Terrain, generate_terrain
from sionna_twin_ops.viewer import BEAM_LENGTH_M, unpack_int16

matplotlib.rcParams["axes.unicode_minus"] = False  # ASCII hyphen-minus in tick labels

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
            f"id {terrain_id}: relief {p.relief_m:.0f} m, beta {p.beta:.2f}",
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
            f"relief {p.relief_m:.0f} m, beta {p.beta:.2f}",
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


GAIN_RANGE_DB = (-150.0, -60.0)  # the viewer page's shared path-gain scale
NO_SIGNAL_GREY = "#8a8a8a"  # the viewer page's ground colour where the ray tracer has no power
VIEWER_BG = "#111111"
VIEWER_TEXT = "#dddddd"
VIEWER_AXIS = "#888888"
VIEWER_EYE = (-0.95, -1.1, 0.65)  # the page's camera, as Plotly's eye vector


def viewer_still(
    data: dict[str, Any], case_index: int, exaggeration: float, caption: str, path: Path
) -> None:
    """One case of the 3D viewer as a still: the ray-traced path gain draped over its terrain.

    Draws from the same packed data the viewer page receives, so the still and the page agree.
    Cells without traced power are the page's grey; the mast and the boresight are drawn as
    on the page; the camera matches the page's starting view.

    Args:
        data: Output of `viewer.viewer_data`.
        case_index: Index into `data["cases"]`.
        exaggeration: Vertical exaggeration of the heights, as the page's buttons offer.
        caption: Text under the figure stating what is shown and its settings.
        path: Output image file (JPEG).

    Raises:
        ValueError: If a case's maps are not square or disagree with the terrain's size.
    """
    case = data["cases"][case_index]
    terrain = data["terrains"][str(case["terrain"])]
    ground = unpack_int16(terrain["ground"])
    traced = unpack_int16(case["traced"])
    n = round(float(np.sqrt(ground.size)))
    if n * n != ground.size or traced.size != ground.size:
        raise ValueError(f"maps of {ground.size} and {traced.size} cells are not one square grid")
    axis_km = ((np.arange(n) + 0.5) * MAP_CELL_M - n * MAP_CELL_M / 2) / 1000.0
    x, y = np.meshgrid(axis_km, axis_km)
    z = ground.reshape(n, n)
    gain = traced.reshape(n, n)

    norm = Normalize(vmin=GAIN_RANGE_DB[0], vmax=GAIN_RANGE_DB[1])
    colours = plt.get_cmap("viridis")(norm(np.nan_to_num(gain, nan=GAIN_RANGE_DB[0])))
    colours[np.isnan(gain)] = to_rgba(NO_SIGNAL_GREY)
    light = LightSource(azdeg=315, altdeg=45).hillshade(z, vert_exag=exaggeration, dx=1, dy=1)
    colours[..., :3] *= (0.55 + 0.45 * light)[..., None]

    fig = plt.figure(figsize=(10.5, 6.4), facecolor=VIEWER_BG)
    ax = cast(Axes3D, fig.add_subplot(projection="3d"))
    ax.set_facecolor(VIEWER_BG)
    ax.computed_zorder = False  # the mast and boresight are drawn after, and over, the ground
    ax.plot_surface(
        x,
        y,
        z,
        facecolors=colours,
        rstride=1,
        cstride=1,
        shade=False,
        linewidth=0,
        antialiased=False,
    )
    ground_m, antenna_m = terrain["site_ground_m"], terrain["antenna_m"]
    ax.plot([0, 0], [0, 0], [ground_m, antenna_m], color="#ff3030", lw=3)
    az, tl = np.radians(case["azimuth"]), np.radians(case["tilt"])
    beam_km = BEAM_LENGTH_M / 1000.0
    ax.plot(
        [0, beam_km * np.sin(az) * np.cos(tl)],
        [0, beam_km * np.cos(az) * np.cos(tl)],
        [antenna_m, antenna_m - BEAM_LENGTH_M * np.sin(tl)],
        color="#ffd000",
        lw=2,
    )
    lo, hi = float(z.min()), max(float(z.max()), float(antenna_m))
    ax.set_zlim(lo - 10, hi + 40)
    ax.set_box_aspect((1.0, 1.0, exaggeration * (hi - lo + 50) / (n * MAP_CELL_M)), zoom=1.6)
    ex, ey, ez = VIEWER_EYE
    ax.view_init(
        elev=float(np.degrees(np.arctan2(ez, np.hypot(ex, ey)))),
        azim=float(np.degrees(np.arctan2(ey, ex))),
    )
    ax.set_xlabel("east (km)")
    ax.set_ylabel("north (km)")
    ax.set_zlabel("height (m)")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color(to_rgba(VIEWER_BG))
        axis.label.set_color(VIEWER_AXIS)
        axis.line.set_color(VIEWER_AXIS)
    ax.tick_params(colors=VIEWER_AXIS, labelsize=7)
    ax.grid(False)
    bar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, shrink=0.5, pad=0.02
    )
    bar.set_label("path gain (dB)", color=VIEWER_TEXT)
    bar.ax.yaxis.set_tick_params(color=VIEWER_AXIS, labelcolor=VIEWER_TEXT)
    fig.text(
        0.01,
        0.01,
        "\n".join(textwrap.wrap(caption, 150)),
        fontsize=7,
        color=VIEWER_TEXT,
        va="bottom",
    )
    fig.subplots_adjust(left=0.0, right=0.98, bottom=0.12, top=1.0)
    fig.savefig(path, dpi=110, pil_kwargs={"quality": 85}, facecolor=VIEWER_BG)
    plt.close(fig)


SERIES = "#2a78d6"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
SURFACE = "#fcfcfb"


def tilt_check_figure(
    tilts_deg: list[float],
    lobe_elevations_deg: list[float],
    far_field_medians_db: list[float],
    caption: str,
    path: Path,
) -> None:
    """The flat-plain tilt check (spec A4) as two panels.

    Args:
        tilts_deg: Commanded electrical tilts.
        lobe_elevations_deg: Measured main-lobe elevations (negative is down).
        far_field_medians_db: Far-field median path gain at each tilt.
        caption: Provenance and reading notes, printed under the panels.
        path: Output image file.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.0, 3.9), layout="constrained")
    fig.patch.set_facecolor(SURFACE)
    downtilt = [-e for e in lobe_elevations_deg]
    left.plot(tilts_deg, tilts_deg, ls="--", lw=1, color=INK_MUTED, label="commanded")
    left.plot(tilts_deg, downtilt, "o-", lw=2, ms=8, color=SERIES, mec=SURFACE, mew=2)
    for t, d in zip(tilts_deg, downtilt, strict=True):
        left.annotate(
            f"{d - t:+.2f}",
            (t, d),
            textcoords="offset points",
            xytext=(6, -12),
            fontsize=7,
            color=INK_MUTED,
        )
    left.set_xlabel("commanded tilt (deg)", color=INK)
    left.set_ylabel("measured main-lobe downtilt (deg)", color=INK)
    left.set_title("Main lobe follows the commanded tilt", fontsize=9, color=INK, loc="left")
    right.plot(tilts_deg, far_field_medians_db, "o-", lw=2, ms=8, color=SERIES, mec=SURFACE, mew=2)
    right.set_xlabel("commanded tilt (deg)", color=INK)
    right.set_ylabel("far-field median path gain (dB)", color=INK)
    right.set_title("Far-field gain across tilt", fontsize=9, color=INK, loc="left")
    for ax in (left, right):
        ax.set_facecolor(SURFACE)
        ax.grid(color="#e4e3df", lw=0.8)
        ax.tick_params(colors=INK_MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK_MUTED)
    fig.text(0.01, -0.02, caption, fontsize=7, color=INK_MUTED, ha="left", va="top", wrap=True)
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
