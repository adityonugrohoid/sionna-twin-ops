"""Cross-backend agreement: the same maps solved on two Mitsuba variants, compared.

`floor_maps` runs on one backend (after `backend.select_variant`) and saves the maps of the
solver-check floor configuration with their timings and provenance. `compare_data`
needs no Sionna: it loads two such runs and reports per-cell agreement, split by the LOS
mask. Synthetic terrain.
"""

import json
import statistics
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


def _stats(
    a: NDArray[np.float64], b: NDArray[np.float64], region: NDArray[np.bool_]
) -> dict[str, float | int | None]:
    both = region & (a > 0) & (b > 0)
    if not both.any():
        return {"median": None, "p95": None, "max": None, "cells": 0}
    d = np.abs(10.0 * np.log10(a[both]) - 10.0 * np.log10(b[both]))
    return {
        "median": float(np.median(d)),
        "p95": float(np.percentile(d, 95)),
        "max": float(d.max()),
        "cells": int(both.sum()),
    }


def compare_data(reference: Path, candidate: Path) -> dict[str, Any]:
    """Per-cell agreement of two backends on the same maps.

    Args:
        reference: Run directory of the reference backend.
        candidate: Run directory of the backend under test.

    Returns:
        JSON-safe data for `reports.backend_check_markdown`.

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

    los_by_terrain = {}
    for terrain_id in TERRAIN_IDS:
        terrain = generate_terrain(terrain_id, GRID_SPACING_M)
        site = place_site(terrain, site_class_for(terrain_id))
        los_by_terrain[terrain_id] = los_mask(map_geometry(terrain, site))
    agreement = []
    for terrain_id, n in sorted(ref, key=lambda k: (TERRAIN_IDS.index(k[0]), k[1])):
        a, b = ref[(terrain_id, n)], cand[(terrain_id, n)]
        los = los_by_terrain[terrain_id]
        agreement.append(
            {
                "terrain_id": terrain_id,
                "samples": n,
                "los": _stats(a, b, los),
                "nlos": _stats(a, b, ~los),
                "power_only_reference": int(((a > 0) & (b == 0)).sum()),
                "power_only_candidate": int(((a == 0) & (b > 0)).sum()),
            }
        )
    cand_rows = {(r["terrain_id"], r["samples"]): r for r in cand_meta["rows"]}
    return {
        "kind": "backend_check",
        "terrain_ids": list(TERRAIN_IDS),
        "azimuth_deg": AZIMUTH_DEG,
        "tilt_deg": TILT_DEG,
        "reference_provenance": ref_meta["provenance"],
        "candidate_provenance": cand_meta["provenance"],
        "agreement": agreement,
        "timing": [
            {"reference": r, "candidate": cand_rows[(r["terrain_id"], r["samples"])]}
            for r in ref_meta["rows"]
        ],
    }


def symmetry_check_data(terrain_id: int, azimuth_deg: float, tilt_deg: float) -> dict[str, Any]:
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
        JSON-safe data for `reports.symmetry_check_markdown`.
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
    rows = []
    for k, mirror in ALL_VARIANTS:
        expected = transform_raster(original, k, mirror)
        traced = original if (k, mirror) == (0, False) else trace(k, mirror)
        los = los_mask(map_geometry(transform_terrain(crop, k, mirror), site))
        rows.append(
            {
                "k": k,
                "mirror": mirror,
                "keeps_fold": keeps_fold(k, mirror),
                "azimuth_deg": transform_azimuth(azimuth_deg, k, mirror),
                "los": _stats(expected, traced, los),
                "nlos": _stats(expected, traced, ~los),
                "power_only_moved_original": int(((expected > 0) & (traced == 0)).sum()),
                "power_only_traced": int(((expected == 0) & (traced > 0)).sum()),
            }
        )
    return {
        "kind": "symmetry_check",
        "terrain_id": terrain_id,
        "crop_side_m": 2 * 3480,
        "site_class": site.site_class,
        "azimuth_deg": azimuth_deg,
        "tilt_deg": tilt_deg,
        "samples_per_tx": DATASET_SAMPLES,
        "provenance": provenance(),
        "rows": rows,
    }


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
) -> dict[str, float | int | None]:
    """Median / p95 / max |dB| over cells with power in both maps, pooled over pairs."""
    diffs = []
    for a, b, m in pairs:
        both = m & (a > 0) & (b > 0)
        diffs.append(np.abs(10.0 * np.log10(a[both]) - 10.0 * np.log10(b[both])))
    d = np.concatenate(diffs)
    if d.size == 0:
        return {"median": None, "p95": None, "max": None, "cells": 0}
    return {
        "median": float(np.median(d)),
        "p95": float(np.percentile(d, 95)),
        "max": float(d.max()),
        "cells": int(d.size),
    }


def fold_check_data(azimuth_deg: float, tilt_deg: float) -> dict[str, Any]:
    """The fold term beside the sampling floor, on training terrains (spec N3b).

    Must run after `backend.select_variant`. Each terrain is traced three times with the
    dataset's settings: SW-NE fold at DATASET_SAMPLES (the dataset's own map), NW-SE fold
    at DATASET_SAMPLES (the fold term), and SW-NE at REFERENCE_SAMPLES (the sampling floor).

    Args:
        azimuth_deg: Boresight azimuth.
        tilt_deg: Electrical tilt.

    Returns:
        JSON-safe data for `reports.fold_check_markdown`.
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
    truth = []
    for name, maps in groups:
        for term, index in (("fold", 1), ("sampling", 2)):
            truth.append(
                {
                    "group": name,
                    "term": term,
                    "los": _pooled([(m[0], m[index], m[3]) for m in maps]),
                    "nlos": _pooled([(m[0], m[index], ~m[3]) for m in maps]),
                    "power_only_sw_ne": sum(
                        int(((m[0] > 0) & (m[index] == 0)).sum()) for m in maps
                    ),
                    "power_only_other": sum(
                        int(((m[0] == 0) & (m[index] > 0)).sum()) for m in maps
                    ),
                }
            )
    shadow = []
    for name, maps in groups:
        for low, high in SHADOW_DISTANCE_BINS:
            shadow.append(
                {
                    "group": name,
                    "low": low,
                    "high": high if np.isfinite(high) else None,
                    "fold": _pooled(
                        [(m[0], m[1], m[3] & (m[4] >= low) & (m[4] < high)) for m in maps]
                    ),
                }
            )

    def cut(maps: list[Any], region: str, kind: str) -> dict[str, float | int | None]:
        pairs = []
        for m in maps:
            place = m[3] if region == "LOS" else ~m[3]
            path = m[5] if kind == "reflection" else ~m[5]
            pairs.append((m[0], m[1], place & path))
        return _pooled(pairs)

    paths = []
    for name, maps in groups:
        paths.append(
            {
                "group": name,
                "los_direct": cut(maps, "LOS", "direct"),
                "los_reflection": cut(maps, "LOS", "reflection"),
                "nlos_direct": cut(maps, "NLOS", "direct"),
                "nlos_reflection": cut(maps, "NLOS", "reflection"),
                "los_lit_cells": sum(int((m[3] & (m[0] > 0)).sum()) for m in maps),
                "los_reflection_cells": sum(int((m[3] & m[5]).sum()) for m in maps),
            }
        )
    return {
        "kind": "fold_check",
        "terrain_ids": list(FOLD_TERRAIN_IDS),
        "azimuth_deg": azimuth_deg,
        "tilt_deg": tilt_deg,
        "dataset_samples": DATASET_SAMPLES,
        "reference_samples": REFERENCE_SAMPLES,
        "reflection_excess_db": REFLECTION_EXCESS_DB,
        "provenance": provenance(),
        "truth": truth,
        "shadow": shadow,
        "paths": paths,
    }


def raytracer_timing(dataset: Path, split: str, terrains: int, further_maps: int) -> dict[str, Any]:
    """Ray tracer seconds per map, end to end on a new terrain and per further map (spec E5).

    Must run after `backend.select_variant`. One untimed solve on another terrain first
    compiles the kernels (the surrogate's timing skips its warm-up the same way). Then, for
    each of the first `terrains` terrains of the split: terrain generation, scene and mesh
    build, measurement surface and the first solve are timed together; `further_maps` more
    maps of that terrain are timed as solves only (the scene is kept, as a sweep would).

    Args:
        dataset: Dataset directory (its manifest names the maps and settings).
        split: Split whose terrains are used.
        terrains: New terrains timed end to end.
        further_maps: Further maps timed per terrain.

    Returns:
        Medians, the samples behind them, the variant and the GPU record.
    """
    from sionna_twin_ops.antenna import tilt_weights
    from sionna_twin_ops.dataset import read_manifest
    from sionna_twin_ops.provenance import provenance
    from sionna_twin_ops.scene import DATASET_FOLD, build_scene, measurement_surface
    from sionna_twin_ops.site import Site
    from sionna_twin_ops.solve import solve_map, specular_settings

    lines = sorted(
        (line for line in read_manifest(dataset) if line["split"] == split),
        key=lambda line: line["file"],
    )
    by_terrain: dict[int, list[dict[str, Any]]] = {}
    for line in lines:
        by_terrain.setdefault(line["terrain_id"], []).append(line)
    ids = sorted(by_terrain)
    settings = specular_settings(lines[0]["settings"]["samples_per_tx"], 1)

    def first_map(line: dict[str, Any]) -> tuple[Any, Any, Any, float]:
        start = time.perf_counter()
        terrain = generate_terrain(line["terrain_id"], GRID_SPACING_M)
        site = Site(**line["site"])
        scene = build_scene(terrain, site, line["azimuth_deg"], DATASET_FOLD)
        surface = measurement_surface(terrain, site, DATASET_FOLD)
        solve_map(scene, surface, tilt_weights(line["tilt_deg"]), settings)
        return scene, surface, site, time.perf_counter() - start

    first_map(by_terrain[ids[-1]][0])  # warm-up: kernel compilation, not timed
    new_terrain, further = [], []
    for terrain_id in ids[:terrains]:
        azimuth = by_terrain[terrain_id][0]["azimuth_deg"]
        maps = [m for m in by_terrain[terrain_id] if m["azimuth_deg"] == azimuth]
        scene, surface, _, seconds = first_map(maps[0])
        new_terrain.append(seconds)
        for line in maps[1 : 1 + further_maps]:
            result = solve_map(scene, surface, tilt_weights(line["tilt_deg"]), settings)
            further.append(result.seconds)
    return {
        "provenance": provenance(),
        "samples_per_tx": settings.samples_per_tx,
        "terrains": ids[:terrains],
        "new_terrain_first_map_s": new_terrain,
        "further_map_s": further,
        "new_terrain_first_map_median_s": statistics.median(new_terrain),
        "further_map_median_s": statistics.median(further),
    }
