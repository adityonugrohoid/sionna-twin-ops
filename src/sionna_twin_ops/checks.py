"""Ray-tracer checks: the flat-plain tilt check (spec A4), the measurement-surface check
(spec S5), and time per map plus the sampling floor (spec N3). All on synthetic geometry."""

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sionna_twin_ops.antenna import CARRIER_HZ, tilt_weights, yaw_for_azimuth
from sionna_twin_ops.baselines import los_mask, map_geometry
from sionna_twin_ops.scene import (
    DATASET_FOLD,
    build_scene,
    measurement_surface,
    receiver_array,
    sector_array,
)
from sionna_twin_ops.sionna_rt import mi, rt
from sionna_twin_ops.site import (
    MAP_CELL_M,
    MAP_CELLS,
    MAP_SIZE_M,
    MAST_HEIGHT_M,
    SURFACE_HEIGHT_M,
    Site,
    SiteClass,
    place_site,
    site_class_for,
)
from sionna_twin_ops.solve import (
    DATASET_SAMPLES,
    DEV_SAMPLES,
    MAX_SAMPLES_PER_TX,
    REFERENCE_SAMPLES,
    SolverSettings,
    precoding_vec,
    solve_map,
    specular_settings,
)
from sionna_twin_ops.terrain import (
    GRID_SPACING_M,
    TILE_SIZE_M,
    Terrain,
    TerrainParams,
    generate_terrain,
)

TILTS_DEG = (0.0, 3.0, 6.0, 9.0, 12.0)  # spec C2
CHECK_AZIMUTH_DEG = 90.0  # boresight east, along +x

# Main-lobe measurement: a vertical planar map on boresight, 1 m tall cells.
LOBE_DISTANCE_M = 200.0  # far field: the 1.1 m array's 2 D**2 / lambda is about 14 m
LOBE_ELEVATION_SPAN_DEG = (-20.0, 5.0)

# ASSUMPTION: "far field" for the A4 gain criterion is the cells 1.5 to 2.5 km from the
# site within 60 degrees of boresight.
FAR_FIELD_RANGE_M = (1500.0, 2500.0)
FAR_FIELD_HALF_ANGLE_DEG = 60.0


def flat_terrain() -> Terrain:
    """A flat tile at height 0 on the standard grid; not one of the generated terrains."""
    n = round(TILE_SIZE_M / GRID_SPACING_M) + 1
    coords = np.linspace(-TILE_SIZE_M / 2.0, TILE_SIZE_M / 2.0, n)
    return Terrain(
        params=TerrainParams(terrain_id=-1, relief_m=0.0, beta=0.0),
        spacing_m=GRID_SPACING_M,
        coords_m=coords,
        heights_m=np.zeros((n, n)),
    )


def centre_site() -> Site:
    """A mast at the centre of the flat tile (the class label is unused on flat ground)."""
    return Site("slope", 0.0, 0.0, 0.0, MAST_HEIGHT_M, 50.0)


def to_db(gain: NDArray[np.float64]) -> NDArray[np.float64]:
    """Linear path gain to dB; cells with no hit become NaN.

    Args:
        gain: Linear path gain.

    Returns:
        Gain in dB, NaN where the gain is 0.
    """
    with np.errstate(divide="ignore"):
        db: NDArray[np.float64] = np.where(gain > 0, 10.0 * np.log10(gain), np.nan)
    return db


def main_lobe_elevation(tilt_deg: float, samples_per_tx: int, seed: int) -> float:
    """Elevation of the main lobe, measured by the ray tracer in free space.

    The sector antenna radiates into an empty scene; a vertical planar radio map on
    boresight at LOBE_DISTANCE_M, with 1 m tall cells, samples the elevation cut. The
    peak is refined by a parabola through the highest cell and its neighbours, in dB.

    Args:
        tilt_deg: Commanded electrical downtilt.
        samples_per_tx: Rays launched.
        seed: Solver seed.

    Returns:
        Main-lobe elevation in degrees, negative below the horizon.

    Raises:
        ValueError: If the peak lies on the edge of the measured span.
    """
    scene = rt.load_scene()
    scene.frequency = CARRIER_HZ
    array = sector_array()
    scene.tx_array = array
    scene.rx_array = receiver_array()
    scene.add(
        rt.Transmitter(
            "sector",
            position=mi.Point3f(0.0, 0.0, MAST_HEIGHT_M),
            orientation=mi.Point3f(yaw_for_azimuth(CHECK_AZIMUTH_DEG), 0.0, 0.0),
        )
    )
    low, high = (float(LOBE_DISTANCE_M * np.tan(np.radians(e))) for e in LOBE_ELEVATION_SPAN_DEG)
    height = float(np.ceil(high - low))
    # Rotating the plane -90 degrees about y turns its normal to -x (towards the antenna)
    # and its local x axis to global +z, so cells run up the plane.
    radio_map = rt.RadioMapSolver()(
        scene,
        center=mi.Point3f(LOBE_DISTANCE_M, 0.0, MAST_HEIGHT_M + (low + high) / 2.0),
        orientation=mi.Point3f(0.0, -np.pi / 2.0, 0.0),
        size=mi.Point2f(height, 20.0),
        cell_size=mi.Point2f(1.0, 20.0),
        precoding_vec=precoding_vec(tilt_weights(tilt_deg)),
        samples_per_tx=samples_per_tx,
        max_depth=0,
        seed=seed,
    )
    gain = np.asarray(radio_map.path_gain.numpy(), dtype=np.float64)[0].ravel()
    centres = np.asarray(radio_map.cell_centers.numpy(), dtype=np.float64).reshape(-1, 3)
    elevation = np.degrees(np.arctan2(centres[:, 2] - MAST_HEIGHT_M, centres[:, 0]))
    order = np.argsort(elevation)
    elevation, db = elevation[order], to_db(gain[order])
    k = int(np.nanargmax(db))
    if k == 0 or k == len(db) - 1:
        raise ValueError(f"tilt {tilt_deg}: main lobe at the edge of the measured span")
    y0, y1, y2 = db[k - 1], db[k], db[k + 1]
    shift = 0.5 * (y0 - y2) / (y0 - 2.0 * y1 + y2)
    step = elevation[k + 1] - elevation[k]
    return float(elevation[k] + shift * step)


def far_field_mask() -> NDArray[np.bool_]:
    """Map cells in the A4 far-field sector, for a site at the map centre.

    Returns:
        Boolean mask of shape (128, 128), rows from the south edge.
    """
    offsets = (np.arange(MAP_CELLS) + 0.5) * MAP_CELL_M - MAP_SIZE_M / 2.0
    x, y = np.meshgrid(offsets, offsets)
    distance = np.hypot(x, y)
    bearing = np.degrees(np.arctan2(x, y)) % 360.0  # clockwise from north
    off_axis = np.abs((bearing - CHECK_AZIMUTH_DEG + 180.0) % 360.0 - 180.0)
    low, high = FAR_FIELD_RANGE_M
    mask: NDArray[np.bool_] = (
        (distance >= low) & (distance <= high) & (off_axis <= FAR_FIELD_HALF_ANGLE_DEG)
    )
    return mask


@dataclass(frozen=True)
class TiltRow:
    """One tilt of the A4 check.

    Attributes:
        tilt_deg: Commanded tilt.
        lobe_elevation_deg: Measured main-lobe elevation (negative is down).
        far_field_median_db: Median path gain over the far-field cells on the flat tile.
        far_field_valid: Share of far-field cells with a hit.
    """

    tilt_deg: float
    lobe_elevation_deg: float
    far_field_median_db: float
    far_field_valid: float


def tilt_check(settings: SolverSettings, lobe_samples: int) -> list[TiltRow]:
    """The flat-plain tilt check (spec A4) at every tilt of the C2 grid.

    Args:
        settings: Solver settings for the flat-tile maps.
        lobe_samples: Rays for each free-space main-lobe measurement.

    Returns:
        One row per tilt.
    """
    terrain, site = flat_terrain(), centre_site()
    scene = build_scene(terrain, site, CHECK_AZIMUTH_DEG, DATASET_FOLD)
    surface = measurement_surface(terrain, site, DATASET_FOLD)
    mask = far_field_mask()
    rows = []
    for tilt in TILTS_DEG:
        gain = solve_map(scene, surface, tilt_weights(tilt), settings).path_gain
        cells = to_db(gain)[mask]
        rows.append(
            TiltRow(
                tilt_deg=tilt,
                lobe_elevation_deg=main_lobe_elevation(tilt, lobe_samples, settings.seed),
                far_field_median_db=float(np.nanmedian(cells)),
                far_field_valid=float(np.isfinite(cells).mean()),
            )
        )
    return rows


@dataclass(frozen=True)
class DiffStats:
    """Per-cell absolute difference in dB between two maps, over cells valid in both.

    Attributes:
        median_db: Median absolute difference.
        p95_db: 95th percentile absolute difference.
        bias_db: Mean signed difference (first minus second).
        valid: Share of the compared region valid in both maps.
        cells: Number of cells valid in both maps.
    """

    median_db: float
    p95_db: float
    bias_db: float
    valid: float
    cells: int


def diff_stats_in(
    first: NDArray[np.float64], second: NDArray[np.float64], region: NDArray[np.bool_]
) -> DiffStats:
    """Compare two linear path-gain maps cell by cell, in dB, inside a region.

    Args:
        first: Linear path gain map.
        second: Linear path gain map of the same shape.
        region: Cells to compare.

    Returns:
        The statistics over region cells with a hit in both maps.

    Raises:
        ValueError: If no region cell has a hit in both maps.
    """
    a, b = to_db(first), to_db(second)
    both = np.isfinite(a) & np.isfinite(b) & region
    if not both.any():
        raise ValueError("no cell in the region has power in both maps")
    d = a[both] - b[both]
    return DiffStats(
        median_db=float(np.median(np.abs(d))),
        p95_db=float(np.percentile(np.abs(d), 95)),
        bias_db=float(d.mean()),
        valid=float(both.sum() / region.sum()),
        cells=int(both.sum()),
    )


def diff_stats(first: NDArray[np.float64], second: NDArray[np.float64]) -> DiffStats:
    """Compare two linear path-gain maps cell by cell, in dB, over the whole map.

    Args:
        first: Linear path gain map.
        second: Linear path gain map of the same shape.

    Returns:
        The statistics over cells with a hit in both maps.
    """
    return diff_stats_in(first, second, np.ones(first.shape, dtype=bool))


def planar_map(
    scene: rt.Scene, weights: np.ndarray, settings: SolverSettings
) -> NDArray[np.float64]:
    """Planar radio map at SURFACE_HEIGHT_M over the map area of a site at the origin.

    Args:
        scene: Scene with the transmitter.
        weights: Transmit weights.
        settings: Solver settings.

    Returns:
        Linear path gain, shape (128, 128), rows from the south edge.

    Raises:
        ValueError: If the planar cells do not line up with the mesh map cells.
    """
    radio_map = rt.RadioMapSolver()(
        scene,
        center=mi.Point3f(0.0, 0.0, SURFACE_HEIGHT_M),
        orientation=mi.Point3f(0.0, 0.0, 0.0),
        size=mi.Point2f(MAP_SIZE_M, MAP_SIZE_M),
        cell_size=mi.Point2f(MAP_CELL_M, MAP_CELL_M),
        precoding_vec=precoding_vec(weights),
        samples_per_tx=settings.samples_per_tx,
        max_depth=settings.max_depth,
        los=settings.los,
        specular_reflection=settings.specular_reflection,
        diffuse_reflection=settings.diffuse_reflection,
        refraction=settings.refraction,
        diffraction=settings.diffraction,
        edge_diffraction=settings.edge_diffraction,
        seed=settings.seed,
    )
    centres = np.asarray(radio_map.cell_centers.numpy(), dtype=np.float64)
    expected = (np.arange(MAP_CELLS) + 0.5) * MAP_CELL_M - MAP_SIZE_M / 2.0
    if not (np.allclose(centres[0, :, 0], expected) and np.allclose(centres[:, 0, 1], expected)):
        raise ValueError("planar map cells do not match the mesh map cells")
    gain: NDArray[np.float64] = np.asarray(radio_map.path_gain.numpy(), dtype=np.float64)[0]
    return gain


@dataclass(frozen=True)
class SurfaceCheck:
    """The measurement-surface check (spec S5) on the flat tile.

    Attributes:
        surface_vs_planar: Mesh surface map against the planar map, same seed.
        planar_noise: Planar map, seed against seed + 1 (the noise floor).
        surface_noise: Mesh surface map, seed against seed + 1.
    """

    surface_vs_planar: DiffStats
    planar_noise: DiffStats
    surface_noise: DiffStats


def surface_check(settings: SolverSettings, tilt_deg: float) -> SurfaceCheck:
    """Mesh measurement surface against a planar map at the same height (spec S5).

    Args:
        settings: Solver settings; `seed` and `seed + 1` are used.
        tilt_deg: Electrical tilt of the sector.

    Returns:
        The three comparisons.
    """
    terrain, site = flat_terrain(), centre_site()
    scene = build_scene(terrain, site, CHECK_AZIMUTH_DEG, DATASET_FOLD)
    surface = measurement_surface(terrain, site, DATASET_FOLD)
    weights = tilt_weights(tilt_deg)
    reseeded = replace(settings, seed=settings.seed + 1)
    mesh_a = solve_map(scene, surface, weights, settings).path_gain
    mesh_b = solve_map(scene, surface, weights, reseeded).path_gain
    planar_a = planar_map(scene, weights, settings)
    planar_b = planar_map(scene, weights, reseeded)
    return SurfaceCheck(
        surface_vs_planar=diff_stats(mesh_a, planar_a),
        planar_noise=diff_stats(planar_a, planar_b),
        surface_noise=diff_stats(mesh_a, mesh_b),
    )


FLOOR_TERRAIN_IDS = (3, 1, 5)  # one hilltop, one slope, one valley
FLOOR_TILT_DEG = 6.0
CONTEXT_SAMPLES = 10**8  # the earlier 1e8 vs 1e9 floor, kept beside the dataset floor
FLOOR_SAMPLES = (DEV_SAMPLES, CONTEXT_SAMPLES, DATASET_SAMPLES, REFERENCE_SAMPLES)


@dataclass(frozen=True)
class FloorRow:
    """Time per map and the sampling floor on one terrain (spec N3).

    Attributes:
        terrain_id: Terrain id.
        site_class: Its site class.
        seconds: Wall time per map at each of FLOOR_SAMPLES.
        floor: DATASET_SAMPLES against REFERENCE_SAMPLES, over cells with power in both.
        floor_los: The same over line-of-sight cells.
        floor_nlos: The same over non-line-of-sight cells (reached by reflection).
        context_los: CONTEXT_SAMPLES against DATASET_SAMPLES over line-of-sight cells.
        context_nlos: The same over non-line-of-sight cells.
        no_hit: Share of cells with no ray at the dataset and the reference sample count.
        hit_only_at_reference: Cells with power at the reference count but none at the
            dataset count.
        hit_only_at_dataset: Cells with power at the dataset count but none at the reference.
    """

    terrain_id: int
    site_class: SiteClass
    seconds: tuple[float, ...]
    floor: DiffStats
    floor_los: DiffStats
    floor_nlos: DiffStats
    context_los: DiffStats
    context_nlos: DiffStats
    no_hit: tuple[float, float]
    hit_only_at_reference: int
    hit_only_at_dataset: int


def sampling_floor(seed: int) -> list[FloorRow]:
    """Solve each floor terrain at each of FLOOR_SAMPLES with the ruled settings.

    Args:
        seed: Solver seed (the ray lattice does not depend on it).

    Returns:
        One row per terrain in FLOOR_TERRAIN_IDS.
    """
    weights = tilt_weights(FLOOR_TILT_DEG)
    rows = []
    for terrain_id in FLOOR_TERRAIN_IDS:
        terrain = generate_terrain(terrain_id, GRID_SPACING_M)
        site = place_site(terrain, site_class_for(terrain_id))
        scene = build_scene(terrain, site, CHECK_AZIMUTH_DEG, DATASET_FOLD)
        surface = measurement_surface(terrain, site, DATASET_FOLD)
        results = [
            solve_map(scene, surface, weights, specular_settings(samples, seed))
            for samples in FLOOR_SAMPLES
        ]
        by_count = {n: r.path_gain for n, r in zip(FLOOR_SAMPLES, results, strict=True)}
        g8, g9 = by_count[DATASET_SAMPLES], by_count[REFERENCE_SAMPLES]
        context = by_count[CONTEXT_SAMPLES]
        los = los_mask(map_geometry(terrain, site))
        rows.append(
            FloorRow(
                terrain_id=terrain_id,
                site_class=site.site_class,
                seconds=tuple(r.seconds for r in results),
                floor=diff_stats(g8, g9),
                floor_los=diff_stats_in(g8, g9, los),
                floor_nlos=diff_stats_in(g8, g9, ~los),
                context_los=diff_stats_in(context, g8, los),
                context_nlos=diff_stats_in(context, g8, ~los),
                no_hit=(float((g8 == 0).mean()), float((g9 == 0).mean())),
                hit_only_at_reference=int(((g8 == 0) & (g9 > 0)).sum()),
                hit_only_at_dataset=int(((g8 > 0) & (g9 == 0)).sum()),
            )
        )
    return rows


LOBE_TOLERANCE_DEG = 1.0  # spec A4
MIN_FAR_FIELD_SPAN_DB = 6.0  # spec A4


def solver_check_data(
    tilt_rows: list[TiltRow],
    surface: SurfaceCheck,
    floor_rows: list[FloorRow],
    settings: SolverSettings,
    lobe_samples: int,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """The solver checks' report data; pass or fail is computed from the numbers.

    Args:
        tilt_rows: Output of `tilt_check`.
        surface: Output of `surface_check`.
        floor_rows: Output of `sampling_floor`.
        settings: Settings of the A4 and S5 maps.
        lobe_samples: Rays per free-space main-lobe measurement.
        provenance: Where the checks ran (`provenance.provenance()`).

    Returns:
        JSON-safe data for `reports.solver_check_markdown`.
    """
    lobe_error = max(abs(r.lobe_elevation_deg + r.tilt_deg) for r in tilt_rows)
    medians = [r.far_field_median_db for r in tilt_rows]
    span = max(medians) - min(medians)
    return {
        "kind": "solver_check",
        "provenance": provenance,
        "settings": settings.record(),
        "lobe_samples": lobe_samples,
        "lobe_distance_m": LOBE_DISTANCE_M,
        "lobe_tolerance_deg": LOBE_TOLERANCE_DEG,
        "min_far_field_span_db": MIN_FAR_FIELD_SPAN_DB,
        "far_field_range_m": list(FAR_FIELD_RANGE_M),
        "far_field_half_angle_deg": FAR_FIELD_HALF_ANGLE_DEG,
        "check_azimuth_deg": CHECK_AZIMUTH_DEG,
        "floor_tilt_deg": FLOOR_TILT_DEG,
        "dataset_samples": DATASET_SAMPLES,
        "reference_samples": REFERENCE_SAMPLES,
        "max_samples_per_tx": MAX_SAMPLES_PER_TX,
        "context_samples": CONTEXT_SAMPLES,
        "floor_samples": list(FLOOR_SAMPLES),
        "lobe_error_deg": lobe_error,
        "far_field_span_db": span,
        "a4_pass": lobe_error <= LOBE_TOLERANCE_DEG and span >= MIN_FAR_FIELD_SPAN_DB,
        "tilt_rows": [asdict(r) for r in tilt_rows],
        "surface": asdict(surface),
        "floor_rows": [asdict(f) for f in floor_rows],
    }
