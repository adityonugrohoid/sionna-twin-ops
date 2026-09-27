"""Cross-backend agreement: the same maps solved on two Mitsuba variants, compared.

`floor_maps` runs on one backend (after `backend.select_variant`) and saves the maps of the
solver-check floor configuration with their timings and provenance. `compare_markdown`
needs no Sionna: it loads two such runs and reports per-cell agreement, split by the LOS
mask. Synthetic terrain.
"""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sionna_twin_ops.baselines import b0_gain_db, los_mask, map_geometry
from sionna_twin_ops.site import place_site, site_class_for
from sionna_twin_ops.terrain import GRID_SPACING_M, generate_terrain

TERRAIN_IDS = (3, 1, 5)  # one hilltop, one slope, one valley (as the solver-check floor)
AZIMUTH_DEG = 90.0
TILT_DEG = 6.0
META = "meta.json"


def floor_maps(samples: list[int], seed: int, out: Path) -> dict[str, Any]:
    """Solve and save the floor maps on the selected backend.

    Each map is solved twice: the first time includes JIT compilation (cold), the second
    is the one saved and timed (warm). Whether the two are bit-identical is recorded.

    Args:
        samples: Rays per map, one run each.
        seed: Solver seed.
        out: Directory for the maps and META.

    Returns:
        The metadata written to META.
    """
    from sionna_twin_ops.antenna import tilt_weights
    from sionna_twin_ops.provenance import gpu_memory_used_mib, provenance
    from sionna_twin_ops.scene import DATASET_FOLD, build_scene, measurement_surface
    from sionna_twin_ops.solve import solve_map, specular_settings

    out.mkdir(parents=True, exist_ok=True)
    weights = tilt_weights(TILT_DEG)
    rows = []
    for terrain_id in TERRAIN_IDS:
        terrain = generate_terrain(terrain_id, GRID_SPACING_M)
        site = place_site(terrain, site_class_for(terrain_id))
        scene = build_scene(terrain, site, AZIMUTH_DEG, DATASET_FOLD)
        surface = measurement_surface(terrain, site, DATASET_FOLD)
        for n in samples:
            settings = specular_settings(n, seed)
            start = time.perf_counter()
            first = solve_map(scene, surface, weights, settings)
            cold = time.perf_counter() - start
            second = solve_map(scene, surface, weights, settings)
            np.save(out / f"t{terrain_id}_{n:.0e}.npy", second.path_gain)
            rows.append(
                {
                    "terrain_id": terrain_id,
                    "samples": n,
                    "cold_s": round(cold, 3),
                    "warm_s": round(second.seconds, 3),
                    "repeat_identical": bool(np.array_equal(first.path_gain, second.path_gain)),
                    "gpu_memory_mib": gpu_memory_used_mib(),
                }
            )
            print(json.dumps(rows[-1]), flush=True)
    meta = {
        "provenance": provenance(),
        "settings": specular_settings(samples[0], seed).record() | {"samples_per_tx": samples},
        "azimuth_deg": AZIMUTH_DEG,
        "tilt_deg": TILT_DEG,
        "rows": rows,
    }
    (out / META).write_text(json.dumps(meta, indent=1), newline="\n")
    return meta


def load(run: Path) -> tuple[dict[str, Any], dict[tuple[int, int], NDArray[np.float64]]]:
    """A saved run: its metadata and maps keyed by (terrain id, samples).

    Args:
        run: Directory written by `floor_maps`.

    Returns:
        (metadata, maps).
    """
    meta: dict[str, Any] = json.loads((run / META).read_text())
    maps = {
        (r["terrain_id"], r["samples"]): np.load(run / f"t{r['terrain_id']}_{r['samples']:.0e}.npy")
        for r in meta["rows"]
    }
    return meta, maps


def _stats(a: NDArray[np.float64], b: NDArray[np.float64], region: NDArray[np.bool_]) -> str:
    both = region & (a > 0) & (b > 0)
    if not both.any():
        return "- | - | - | 0"
    d = np.abs(10.0 * np.log10(a[both]) - 10.0 * np.log10(b[both]))
    return f"{np.median(d):.3f} | {np.percentile(d, 95):.3f} | {d.max():.3f} | {int(both.sum())}"


def compare_markdown(reference: Path, candidate: Path) -> str:
    """Per-cell agreement of two backends on the same maps, as markdown.

    Args:
        reference: Run directory of the reference backend.
        candidate: Run directory of the backend under test.

    Returns:
        Markdown text.

    Raises:
        ValueError: If the runs differ in anything but the backend.
    """
    ref_meta, ref = load(reference)
    cand_meta, cand = load(candidate)
    for key in ("azimuth_deg", "tilt_deg"):
        if ref_meta[key] != cand_meta[key]:
            raise ValueError(f"runs differ in {key}")
    ref_settings = dict(ref_meta["settings"])
    cand_settings = dict(cand_meta["settings"])
    ref_settings.pop("variant")
    cand_settings.pop("variant")
    if ref_settings != cand_settings or set(ref) != set(cand):
        raise ValueError("runs differ in settings or in the maps they hold")

    ref_prov, cand_prov = ref_meta["provenance"], cand_meta["provenance"]
    lines = [
        "# Backend check",
        "",
        "The solver-check floor maps (synthetic terrain ids "
        f"{', '.join(str(t) for t in TERRAIN_IDS)}, azimuth {AZIMUTH_DEG:.0f}, tilt "
        f"{TILT_DEG:.0f}, line of sight and specular reflection, max_depth 3) solved on two "
        "backends and compared cell by cell. Written by `twin backend-check`.",
        "",
        "| | reference | candidate |",
        "|---|---|---|",
        *(f"| {k} | {ref_prov[k]} | {cand_prov[k]} |" for k in ref_prov),
        "",
        "## Agreement",
        "",
        "Absolute dB difference over cells with power on both backends, split by the LOS "
        "mask. The last columns count cells with power on only one backend.",
        "",
        "| terrain | samples | LOS median | LOS p95 | LOS max | LOS cells | NLOS median | "
        "NLOS p95 | NLOS max | NLOS cells | power only on reference | power only on candidate |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    los_by_terrain = {}
    for terrain_id in TERRAIN_IDS:
        terrain = generate_terrain(terrain_id, GRID_SPACING_M)
        site = place_site(terrain, site_class_for(terrain_id))
        los_by_terrain[terrain_id] = los_mask(map_geometry(terrain, site))
    for terrain_id, n in sorted(ref, key=lambda k: (TERRAIN_IDS.index(k[0]), k[1])):
        a, b = ref[(terrain_id, n)], cand[(terrain_id, n)]
        los = los_by_terrain[terrain_id]
        lines.append(
            f"| {terrain_id} | {n:.0e} | {_stats(a, b, los)} | {_stats(a, b, ~los)} | "
            f"{int(((a > 0) & (b == 0)).sum())} | {int(((a == 0) & (b > 0)).sum())} |"
        )
    lines += [
        "",
        "## Time per map",
        "",
        "Cold includes JIT compilation; warm is a second solve of the same map. Repeat "
        "identical says whether the two solves gave bit-identical maps.",
        "",
        "| terrain | samples | reference warm s | reference cold s | reference repeat identical "
        "| candidate warm s | candidate cold s | candidate repeat identical | candidate GPU MiB |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    cand_rows = {(r["terrain_id"], r["samples"]): r for r in cand_meta["rows"]}
    for r in ref_meta["rows"]:
        c = cand_rows[(r["terrain_id"], r["samples"])]
        lines.append(
            f"| {r['terrain_id']} | {r['samples']:.0e} | {r['warm_s']:.2f} | {r['cold_s']:.2f} | "
            f"{r['repeat_identical']} | {c['warm_s']:.2f} | {c['cold_s']:.2f} | "
            f"{c['repeat_identical']} | {c['gpu_memory_mib']} |"
        )
    return "\n".join(lines) + "\n"


def symmetry_check_markdown(terrain_id: int, azimuth_deg: float, tilt_deg: float) -> str:
    """Trace a site-centred crop and its 8 symmetric variants; compare with the moved map.

    Must run after `backend.select_variant`. Each variant's traced map is compared with the
    original traced map moved by the same variant, cell by cell in dB over cells with power
    in both, split by the LOS mask of the variant. Ray lattices are not rotation symmetric,
    so agreement is expected within the sampling floor, not exactly.

    Args:
        terrain_id: Terrain id (its own site class).
        azimuth_deg: Boresight azimuth of the original map.
        tilt_deg: Electrical tilt.

    Returns:
        Markdown text.
    """
    from sionna_twin_ops.antenna import tilt_weights
    from sionna_twin_ops.augment import (
        ALL_VARIANTS,
        centred_crop,
        keeps_fold,
        transform_azimuth,
        transform_raster,
        transform_terrain,
    )
    from sionna_twin_ops.provenance import provenance
    from sionna_twin_ops.scene import DATASET_FOLD, build_scene, measurement_surface
    from sionna_twin_ops.solve import DATASET_SAMPLES, solve_map, specular_settings

    settings = specular_settings(DATASET_SAMPLES, 1)
    weights = tilt_weights(tilt_deg)
    terrain = generate_terrain(terrain_id, GRID_SPACING_M)
    crop, site = centred_crop(terrain, place_site(terrain, site_class_for(terrain_id)))

    def trace(k: int, mirror: bool) -> NDArray[np.float64]:
        moved = transform_terrain(crop, k, mirror)
        azimuth = transform_azimuth(azimuth_deg, k, mirror)
        scene = build_scene(moved, site, azimuth, DATASET_FOLD)
        surface = measurement_surface(moved, site, DATASET_FOLD)
        return solve_map(scene, surface, weights, settings).path_gain

    original = trace(0, False)
    lines = [
        "# Symmetry check",
        "",
        f"Synthetic terrain id {terrain_id}, a {2 * 3480:.0f} m square centred on its "
        f"{site.site_class} site; azimuth {azimuth_deg:.0f}, tilt {tilt_deg:.0f}; line of sight "
        f"and specular reflection, {DATASET_SAMPLES:.0e} rays. Each variant (k clockwise "
        "quarter turns after an optional east-west mirror) moves the terrain and the "
        "azimuth together and is traced afresh; its map is compared with the original map "
        "moved the same way. Written by `twin symmetry-check`.",
        "",
        "The terrain mesh and the measurement surface split every cell along its SW-NE "
        "diagonal. Variants that keep that fold (the identity, the half turn and the two "
        "diagonal mirrors) move the traced surface exactly and should agree within the "
        "sampling floor. Quarter turns and axis mirrors flip every cell's fold: the vertex "
        "heights are the same but the surface between them is not, so those maps differ by "
        "more than the floor. Training augments with the four fold-keeping variants only "
        "(spec M2b).",
        "",
        "| provenance | |",
        "|---|---|",
        *(f"| {k} | {v} |" for k, v in provenance().items()),
        "",
        "| k | mirror | keeps fold | azimuth | LOS median | LOS p95 | LOS max | LOS cells | "
        "NLOS median | NLOS p95 | NLOS max | NLOS cells | power only in moved original | "
        "power only in traced |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for k, mirror in ALL_VARIANTS:
        expected = transform_raster(original, k, mirror)
        traced = original if (k, mirror) == (0, False) else trace(k, mirror)
        los = los_mask(map_geometry(transform_terrain(crop, k, mirror), site))
        lines.append(
            f"| {k} | {mirror} | {keeps_fold(k, mirror)} | "
            f"{transform_azimuth(azimuth_deg, k, mirror):.0f} | "
            f"{_stats(expected, traced, los)} | {_stats(expected, traced, ~los)} | "
            f"{int(((expected > 0) & (traced == 0)).sum())} | "
            f"{int(((expected == 0) & (traced > 0)).sum())} |"
        )
    return "\n".join(lines) + "\n"


FOLD_TERRAIN_IDS = (3, 6, 1, 4, 5, 8)  # two training terrains per site class
# ASSUMPTION: a cell whose traced gain exceeds B0 (direct path and antenna pattern, no
# terrain) by this much is reflection-dominated; below it, direct-dominated.
REFLECTION_EXCESS_DB = 3.0
SHADOW_DISTANCE_BINS = ((1.0, 2.0), (2.0, 4.0), (4.0, float("inf")))  # in cells


def distance_to_shadow(los: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Euclidean distance, in cells, from each cell to the nearest non-LOS cell.

    Only non-LOS cells with a line-of-sight 4-neighbour can be nearest (a step towards a
    lit cell always gets closer), so the search runs over those boundary cells.

    Args:
        los: LOS mask of a map.

    Returns:
        Distance per cell; 0 on non-LOS cells, infinite if the map has none.
    """
    nlos = ~los
    padded = np.pad(los, 1, constant_values=False)
    lit_neighbour = padded[:-2, 1:-1] | padded[2:, 1:-1] | padded[1:-1, :-2] | padded[1:-1, 2:]
    boundary = np.argwhere(nlos & lit_neighbour).astype(np.float64)
    distance = np.zeros(los.shape)
    lit = np.argwhere(los)
    if len(boundary) == 0:
        distance[los] = np.inf
        return distance
    for start in range(0, len(lit), 2048):
        chunk = lit[start : start + 2048].astype(np.float64)
        d = np.sqrt(((chunk[:, None, :] - boundary[None, :, :]) ** 2).sum(-1)).min(axis=1)
        distance[lit[start : start + 2048, 0], lit[start : start + 2048, 1]] = d
    return distance


def _pooled(
    pairs: list[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]],
) -> str:
    """Median / p95 / max |dB| over cells with power in both maps, pooled over pairs."""
    diffs = []
    for a, b, m in pairs:
        both = m & (a > 0) & (b > 0)
        diffs.append(np.abs(10.0 * np.log10(a[both]) - 10.0 * np.log10(b[both])))
    d = np.concatenate(diffs)
    if d.size == 0:
        return "- | - | - | 0"
    return f"{np.median(d):.3f} | {np.percentile(d, 95):.3f} | {d.max():.3f} | {d.size}"


def fold_check_markdown(azimuth_deg: float, tilt_deg: float) -> str:
    """The fold term beside the sampling floor, on training terrains (spec N3b).

    Must run after `backend.select_variant`. Each terrain is traced three times with the
    dataset's settings: SW-NE fold at DATASET_SAMPLES (the dataset's own map), NW-SE fold
    at DATASET_SAMPLES (the fold term), and SW-NE at REFERENCE_SAMPLES (the sampling floor).

    Args:
        azimuth_deg: Boresight azimuth.
        tilt_deg: Electrical tilt.

    Returns:
        Markdown text.
    """
    from sionna_twin_ops.antenna import tilt_weights
    from sionna_twin_ops.provenance import provenance
    from sionna_twin_ops.scene import DATASET_FOLD, build_scene, measurement_surface
    from sionna_twin_ops.solve import (
        DATASET_SAMPLES,
        REFERENCE_SAMPLES,
        solve_map,
        specular_settings,
    )

    weights = tilt_weights(tilt_deg)

    def trace(terrain: Any, site: Any, fold: Any, samples: int) -> NDArray[np.float64]:
        scene = build_scene(terrain, site, azimuth_deg, fold)
        surface = measurement_surface(terrain, site, fold)
        return solve_map(scene, surface, weights, specular_settings(samples, 1)).path_gain

    rows: dict[str, list[tuple[Any, Any, Any, Any, Any, Any]]] = {}
    for terrain_id in FOLD_TERRAIN_IDS:
        terrain = generate_terrain(terrain_id, GRID_SPACING_M)
        site = place_site(terrain, site_class_for(terrain_id))
        dataset_map = trace(terrain, site, DATASET_FOLD, DATASET_SAMPLES)
        other_fold = trace(terrain, site, "nw-se", DATASET_SAMPLES)
        reference = trace(terrain, site, DATASET_FOLD, REFERENCE_SAMPLES)
        geometry = map_geometry(terrain, site)
        los = los_mask(geometry)
        with np.errstate(divide="ignore"):
            excess = 10.0 * np.log10(dataset_map) - b0_gain_db(geometry, azimuth_deg, tilt_deg)
        reflection = (dataset_map > 0) & (excess >= REFLECTION_EXCESS_DB)
        rows.setdefault(site.site_class, []).append(
            (dataset_map, other_fold, reference, los, distance_to_shadow(los), reflection)
        )
        print(f"terrain {terrain_id} ({site.site_class}) traced", flush=True)

    groups = [(name, maps) for name, maps in rows.items()]
    groups.append(("all", [m for maps in rows.values() for m in maps]))
    lines = [
        "# Fold check",
        "",
        "Synthetic terrain ids "
        f"{', '.join(str(i) for i in FOLD_TERRAIN_IDS)} (training terrains, two per site "
        f"class), azimuth {azimuth_deg:.0f}, tilt {tilt_deg:.0f}; line of sight and specular "
        f"reflection. The terrain mesh and the measurement surface split every cell into two "
        "triangles along one diagonal; the dataset uses the SW-NE one. The fold term compares "
        f"the dataset's SW-NE maps with NW-SE maps at {DATASET_SAMPLES:.0e} rays; the sampling "
        f"floor compares the same SW-NE maps with {REFERENCE_SAMPLES:.0e} rays. Both are "
        "absolute dB differences over cells with power in both maps, pooled over the "
        "terrains of each row. Written by `twin fold-check`.",
        "",
        "| provenance | |",
        "|---|---|",
        *(f"| {k} | {v} |" for k, v in provenance().items()),
        "",
        "## Truth uncertainty: fold term beside the sampling floor",
        "",
        "| site class | term | LOS median | LOS p95 | LOS max | LOS cells | NLOS median | "
        "NLOS p95 | NLOS max | NLOS cells | power only with SW-NE | power only in the other map |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, maps in groups:
        for term, index in (("fold", 1), ("sampling", 2)):
            one_sided_a = sum(int(((m[0] > 0) & (m[index] == 0)).sum()) for m in maps)
            one_sided_b = sum(int(((m[0] == 0) & (m[index] > 0)).sum()) for m in maps)
            lines.append(
                f"| {name} | {term} | "
                f"{_pooled([(m[0], m[index], m[3]) for m in maps])} | "
                f"{_pooled([(m[0], m[index], ~m[3]) for m in maps])} | "
                f"{one_sided_a} | {one_sided_b} |"
            )
    lines += [
        "",
        "## Fold term on LOS cells by distance to the nearest shadow cell",
        "",
        "Distance is Euclidean, in cells (40 m), from a LOS cell to the nearest non-LOS cell.",
        "",
        "| site class | distance (cells) | median | p95 | max | cells |",
        "|---|---|---|---|---|---|",
    ]
    for name, maps in groups:
        for low, high in SHADOW_DISTANCE_BINS:
            label = f"{low:.0f} to {high:.0f}" if np.isfinite(high) else f"{low:.0f} or more"
            lines.append(
                f"| {name} | {label} | "
                + _pooled([(m[0], m[1], m[3] & (m[4] >= low) & (m[4] < high)) for m in maps])
                + " |"
            )
    lines += [
        "",
        "## Fold term by dominant path",
        "",
        f"A cell with power in the SW-NE map is reflection-dominated when its traced gain "
        f"exceeds B0 (direct path and antenna pattern, no terrain) by {REFLECTION_EXCESS_DB:.0f} "
        "dB or more (ASSUMPTION), and direct-dominated otherwise. Median / p95 of the fold "
        "term, and the cells compared. B0 has no terrain, so shadowed cells rarely exceed "
        "it: the NLOS split says little, and nearly all NLOS cells fall under direct.",
        "",
        "| site class | LOS direct | LOS reflection | NLOS direct | NLOS reflection | "
        "LOS cells reflection-dominated |",
        "|---|---|---|---|---|---|",
    ]

    def cut(maps: list[Any], region: str, kind: str) -> str:
        pairs = []
        for m in maps:
            place = m[3] if region == "LOS" else ~m[3]
            path = m[5] if kind == "reflection" else ~m[5]
            pairs.append((m[0], m[1], place & path))
        median, p95, _, count = _pooled(pairs).split(" | ")
        return f"{median} / {p95} ({count})"

    for name, maps in groups:
        lit = sum(int((m[3] & (m[0] > 0)).sum()) for m in maps)
        reflected = sum(int((m[3] & m[5]).sum()) for m in maps)
        lines.append(
            f"| {name} | {cut(maps, 'LOS', 'direct')} | {cut(maps, 'LOS', 'reflection')} | "
            f"{cut(maps, 'NLOS', 'direct')} | {cut(maps, 'NLOS', 'reflection')} | "
            f"{reflected / lit * 100:.1f}% |"
        )
    return "\n".join(lines) + "\n"
