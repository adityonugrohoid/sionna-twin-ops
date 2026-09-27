"""Tilt and power search (spec rule Q): the surrogate and a rule of thumb against the ray tracer.

Per (terrain, azimuth) case the variables are electrical tilt and sector power. The
objective (Q1) counts covered cells within RADIUS_M of the site minus SPILL_WEIGHT times
covered cells beyond it; a cell is covered where RSRP (path gain plus power per resource
element, as in spec E2) reaches the evaluation's RSRP threshold. The ray tracer searches tilt
on a 1 deg grid, the surrogate on a 0.5 deg grid; power is post-processing on both sides.
Every chosen setting is scored with the ray tracer. Synthetic terrain.
"""

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sionna_twin_ops.antenna import array_gain_db, tilt_weights
from sionna_twin_ops.dataset import AZIMUTHS_DEG, read_manifest
from sionna_twin_ops.site import MAST_HEIGHT_M, Site
from sionna_twin_ops.terrain import GRID_SPACING_M, generate_terrain

TRACE_TILTS_DEG = tuple(float(t) for t in range(0, 13))  # ray tracer: 0 to 12 deg in 1 deg
SURROGATE_TILTS_DEG = tuple(t / 2.0 for t in range(0, 25))  # surrogate: 0 to 12 deg in 0.5 deg
POWERS_DBM = tuple(float(p) for p in range(37, 47))  # 37 to 46 dBm in 1 dB
RULE_POWER_DBM = 46.0
RADIUS_M = 3000.0  # ASSUMPTION (spec Q1): the service radius
SPILL_WEIGHT = 1.0  # ASSUMPTION (spec Q1): one cell of spill beyond RADIUS_M costs one cell
RESOURCE_ELEMENTS = 1200  # ASSUMPTION (spec E2): 20 MHz carrier, as in evaluate.py
RSRP_THRESHOLD_DBM = -110.0  # ASSUMPTION (spec E2), as in evaluate.py
NEAR_OPTIMUM = 0.99  # "within 1% of the optimum"
HPBW_STEP_DEG = 1e-4
META = "meta.json"


def search_map_name(terrain_id: int, azimuth_deg: float, tilt_deg: float) -> str:
    """File name of one search map, tilt to 1e-4 deg (the dataset's names round to 0.1 deg,
    which would put the rule of thumb's 4.5346 deg and a chosen 4.5 deg in one file).

    Args:
        terrain_id: Terrain id.
        azimuth_deg: Azimuth.
        tilt_deg: Tilt.

    Returns:
        The file name, e.g. t052_a090_t04.5346.npy.

    Raises:
        ValueError: If the tilt is not a multiple of 1e-4 deg.
    """
    if abs(round(tilt_deg, 4) - tilt_deg) > 1e-9:
        raise ValueError(f"tilt {tilt_deg} is not a multiple of 1e-4 deg")
    return f"t{terrain_id:03d}_a{azimuth_deg:03.0f}_t{tilt_deg:07.4f}.npy"


def covered_gain_db(power_dbm: float) -> float:
    """Path gain at which a cell reaches the RSRP threshold for a sector power.

    Args:
        power_dbm: Sector power, spread evenly over RESOURCE_ELEMENTS.

    Returns:
        The path gain in dB.
    """
    return RSRP_THRESHOLD_DBM - (power_dbm - 10.0 * float(np.log10(RESOURCE_ELEMENTS)))


def vertical_hpbw_deg() -> float:
    """Half-power beamwidth in elevation of the untilted column at boresight.

    Computed from `array_gain_db` (the TR 38.901 element times the array factor) on a
    HPBW_STEP_DEG grid of zenith angles around the horizon.

    Returns:
        The beamwidth in degrees.

    Raises:
        ValueError: If the pattern does not peak at the horizon.
    """
    zenith = np.arange(60.0, 120.0 + HPBW_STEP_DEG / 2, HPBW_STEP_DEG)
    gain = array_gain_db(zenith, np.zeros_like(zenith), tilt_weights(0.0))
    peak = int(np.argmax(gain))
    if abs(zenith[peak] - 90.0) > HPBW_STEP_DEG:
        raise ValueError(f"untilted column peaks at zenith {zenith[peak]:.4f} deg, not 90")
    above = gain >= gain[peak] - 3.0
    low = peak
    while above[low - 1]:
        low -= 1
    high = peak
    while above[high + 1]:
        high += 1
    return float(zenith[high] - zenith[low])


def rule_of_thumb_tilt_deg() -> float:
    """Tilt that puts the upper half-power edge on the cell edge (spec Q, rule of thumb).

    Returns:
        arctan(MAST_HEIGHT_M / RADIUS_M) plus half the vertical HPBW, in degrees.
    """
    return float(np.degrees(np.arctan(MAST_HEIGHT_M / RADIUS_M)) + vertical_hpbw_deg() / 2.0)


def objectives(
    gain_db: NDArray[np.floating[Any]], has_power: NDArray[np.bool_], near: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Objective Q1 of one map at every power of POWERS_DBM.

    Args:
        gain_db: Path gain in dB (ignored where has_power is False).
        has_power: Cells with power (the ray tracer's, or the surrogate's power head).
        near: Cells within RADIUS_M.

    Returns:
        One objective per power.
    """
    out = np.empty(len(POWERS_DBM))
    for k, power in enumerate(POWERS_DBM):
        covered = has_power & (gain_db >= covered_gain_db(power))
        out[k] = (covered & near).sum() - SPILL_WEIGHT * (covered & ~near).sum()
    return out


def first_best(values: NDArray[np.float64]) -> tuple[int, ...]:
    """Index of the maximum; ties go to the first in C order (lower azimuth, tilt, power).

    Args:
        values: Objectives.

    Returns:
        The index tuple.
    """
    return tuple(int(i) for i in np.unravel_index(int(np.argmax(values)), values.shape))


def split_sites(dataset: Path, split: str) -> dict[int, Site]:
    """Terrains of a split and their sites, from the dataset manifest.

    Args:
        dataset: Dataset directory.
        split: Split.

    Returns:
        Site per terrain id, in id order.

    Raises:
        ValueError: If the split has no maps.
    """
    sites = {
        line["terrain_id"]: Site(**line["site"])
        for line in read_manifest(dataset)
        if line["split"] == split
    }
    if not sites:
        raise ValueError(f"{dataset} has no {split} maps")
    return dict(sorted(sites.items()))


def surrogate_search(dataset: Path, split: str, runs: list[Path], device: str) -> dict[str, Any]:
    """The surrogate's objectives on every terrain of a split, timed per terrain.

    Per terrain and model the timer covers terrain generation, the terrain channels and,
    per azimuth, the channels of the 25 tilts and one batched inference of them (200
    inferences per terrain), then the objectives at every power. One untimed pass over the
    first terrain comes first (warm-up).

    Args:
        dataset: Dataset directory (its manifest gives the sites).
        split: Split.
        runs: Training runs, one model each (best checkpoint).
        device: "cuda" or "cpu", explicit.

    Returns:
        {"commit", "torch", "platform", "gpu", "device", "seeds", "runs", "terrains",
        "objectives": {seed: {terrain: (8, 25, 10) list}}, "seconds": {seed: {terrain: s}}}.
    """
    import platform
    from importlib.metadata import version

    import torch

    from sionna_twin_ops.evaluate import load_models
    from sionna_twin_ops.features import map_inputs, terrain_features
    from sionna_twin_ops.model import RESIDUAL_SCALE_DB
    from sionna_twin_ops.provenance import commit, gpu

    torch_device = torch.device(device)
    sites = split_sites(dataset, split)
    models = load_models(runs, torch_device)

    def search(terrain_id: int, model: Any) -> NDArray[np.float64]:
        features = terrain_features(generate_terrain(terrain_id, GRID_SPACING_M), sites[terrain_id])
        near = features.geometry.distance_km * 1000.0 <= RADIUS_M
        result = np.empty((len(AZIMUTHS_DEG), len(SURROGATE_TILTS_DEG), len(POWERS_DBM)))
        for a, azimuth in enumerate(AZIMUTHS_DEG):
            pairs = [map_inputs(features, azimuth, tilt) for tilt in SURROGATE_TILTS_DEG]
            x = torch.from_numpy(np.stack([p[0] for p in pairs])).to(torch_device)
            with torch.no_grad():
                out = model(x).cpu().numpy()
            for t, (_, b0) in enumerate(pairs):
                predicted = b0 + out[t, 0] * RESIDUAL_SCALE_DB
                result[a, t] = objectives(predicted, out[t, 1] > 0, near)
        return result

    ids = list(sites)
    search(ids[0], models[0][1])  # warm-up, not timed
    found: dict[str, dict[str, Any]] = {}
    seconds: dict[str, dict[str, float]] = {}
    for seed, model in models:
        found[str(seed)], seconds[str(seed)] = {}, {}
        for terrain_id in ids:
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            values = search(terrain_id, model)
            seconds[str(seed)][str(terrain_id)] = time.perf_counter() - start
            found[str(seed)][str(terrain_id)] = values.tolist()
    return {
        "commit": commit(),
        "torch": version("torch"),
        "platform": platform.platform(),
        "gpu": gpu(),
        "device": device,
        "seeds": [seed for seed, _ in models],
        "runs": [str(run) for run in runs],
        "terrains": ids,
        "objectives": found,
        "seconds": seconds,
    }


def surrogate_choices(record: dict[str, Any]) -> dict[str, Any]:
    """The surrogate's choices from `surrogate_search` objectives.

    Args:
        record: Output of `surrogate_search`.

    Returns:
        {"cases": {seed: {"terrain/azimuth": [tilt, power]}},
        "whole_grid": {seed: {terrain: [azimuth, tilt, power]}}}.
    """
    cases: dict[str, dict[str, list[float]]] = {}
    whole: dict[str, dict[str, list[float]]] = {}
    for seed, per_terrain in record["objectives"].items():
        cases[seed], whole[seed] = {}, {}
        for terrain, values in per_terrain.items():
            grid = np.asarray(values)
            for a, azimuth in enumerate(AZIMUTHS_DEG):
                t, p = first_best(grid[a])
                cases[seed][f"{terrain}/{azimuth:.0f}"] = [
                    SURROGATE_TILTS_DEG[t],
                    POWERS_DBM[p],
                ]
            a, t, p = first_best(grid)
            whole[seed][terrain] = [AZIMUTHS_DEG[a], SURROGATE_TILTS_DEG[t], POWERS_DBM[p]]
    return {"cases": cases, "whole_grid": whole}


def tilts_to_trace(
    choice_sets: list[dict[str, Any]], rule_tilt_deg: float
) -> dict[str, list[float]]:
    """Tilts off the 1 deg grid that some chooser picked, per (terrain, azimuth).

    Args:
        choice_sets: Outputs of `surrogate_choices` (one per device).
        rule_tilt_deg: The rule of thumb's tilt (picked for every case).

    Returns:
        {"terrain/azimuth": sorted tilts}, every case of the choice sets present.
    """
    extra: dict[str, set[float]] = {}
    for choices in choice_sets:
        for per_case in choices["cases"].values():
            for key, (tilt, _) in per_case.items():
                extra.setdefault(key, set()).add(tilt)
        for per_terrain in choices["whole_grid"].values():
            for terrain, (azimuth, tilt, _) in per_terrain.items():
                extra.setdefault(f"{terrain}/{azimuth:.0f}", set()).add(tilt)
    for key in extra:
        extra[key].add(round(rule_tilt_deg, 4))
    return {k: sorted(t for t in v if t not in TRACE_TILTS_DEG) for k, v in sorted(extra.items())}


def trace_search(
    dataset: Path, split: str, extra: dict[str, list[float]], samples_per_tx: int, out: Path
) -> dict[str, Any]:
    """The ray tracer's search: every 1 deg tilt at every azimuth, timed per terrain, then
    the extra tilts other choosers picked (untimed).

    Must run after `backend.select_variant`. Per terrain the timer covers terrain
    generation, the measurement surface, one scene per azimuth and the 104 solves. One
    untimed solve on the last terrain comes first (kernel compilation). Maps are written
    as the dataset writes them, under `out/maps`, with the settings in `out/meta.json`.

    Args:
        dataset: Dataset directory (its manifest gives the sites).
        split: Split.
        extra: {"terrain/azimuth": tilts} to trace after the timed search.
        samples_per_tx: Rays per map.
        out: Output directory.

    Returns:
        The meta record written to `out/meta.json`.
    """
    from sionna_twin_ops.dataset import SOLVER_SEED
    from sionna_twin_ops.provenance import provenance
    from sionna_twin_ops.scene import DATASET_FOLD, build_scene, measurement_surface
    from sionna_twin_ops.solve import solve_map, specular_settings

    settings = specular_settings(samples_per_tx, SOLVER_SEED)
    sites = split_sites(dataset, split)
    maps_dir = out / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    def save(terrain_id: int, azimuth: float, tilt: float, gain: NDArray[np.float64]) -> None:
        name = search_map_name(terrain_id, azimuth, tilt)
        partial = maps_dir / f"{name}.partial"
        with partial.open("wb") as handle:
            np.save(handle, gain.astype(np.float32))
        os.replace(partial, maps_dir / name)

    ids = list(sites)
    warm = generate_terrain(ids[-1], GRID_SPACING_M)
    solve_map(
        build_scene(warm, sites[ids[-1]], AZIMUTHS_DEG[0], DATASET_FOLD),
        measurement_surface(warm, sites[ids[-1]], DATASET_FOLD),
        tilt_weights(0.0),
        settings,
    )
    seconds: dict[str, float] = {}
    solve_seconds: dict[str, list[float]] = {}
    for terrain_id in ids:
        site = sites[terrain_id]
        start = time.perf_counter()
        terrain = generate_terrain(terrain_id, GRID_SPACING_M)
        surface = measurement_surface(terrain, site, DATASET_FOLD)
        results = []
        for azimuth in AZIMUTHS_DEG:
            scene = build_scene(terrain, site, azimuth, DATASET_FOLD)
            for tilt in TRACE_TILTS_DEG:
                result = solve_map(scene, surface, tilt_weights(tilt), settings)
                results.append((azimuth, tilt, result))
        seconds[str(terrain_id)] = time.perf_counter() - start
        solve_seconds[str(terrain_id)] = [r.seconds for _, _, r in results]
        for azimuth, tilt, result in results:
            save(terrain_id, azimuth, tilt, result.path_gain)
        for azimuth in AZIMUTHS_DEG:
            tilts = extra.get(f"{terrain_id}/{azimuth:.0f}", [])
            if not tilts:
                continue
            scene = build_scene(terrain, site, azimuth, DATASET_FOLD)
            for tilt in tilts:
                save(
                    terrain_id,
                    azimuth,
                    tilt,
                    solve_map(scene, surface, tilt_weights(tilt), settings).path_gain,
                )
        print(f"terrain {terrain_id}: {seconds[str(terrain_id)]:.1f} s for 104 maps", flush=True)
    meta = {
        "split": split,
        "settings": settings.record(),
        "provenance": provenance(),
        "sites": {str(k): asdict(v) for k, v in sites.items()},
        "extra": extra,
        "terrain_seconds": seconds,
        "solve_seconds": solve_seconds,
    }
    (out / META).write_text(json.dumps(meta, indent=1), newline="\n")
    return meta
