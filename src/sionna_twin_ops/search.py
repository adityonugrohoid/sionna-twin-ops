"""Tilt and power search (spec rule Q): the surrogate and a rule of thumb against the ray tracer.

Per (terrain, azimuth) case the variables are electrical tilt and sector power. The
objective (Q0b) is the covered fraction of the cells within a service radius of the site
minus SPILL_WEIGHT times the covered fraction of the map's cells beyond it; a cell is covered
where RSRP (path gain plus power per resource element, as in spec E2) reaches the
evaluation's RSRP threshold. The ray tracer searches tilt on a 1 deg grid, the surrogate on
a 0.5 deg grid; power is post-processing on both sides. Every chosen setting is scored with
the ray tracer. Synthetic terrain.
"""

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sionna_twin_ops.antenna import array_gain_db, tilt_weights
from sionna_twin_ops.dataset import AZIMUTHS_DEG, read_manifest
from sionna_twin_ops.site import MAP_SIZE_M, MAST_HEIGHT_M, Site
from sionna_twin_ops.terrain import GRID_SPACING_M, generate_terrain

TRACE_TILTS_DEG = tuple(float(t) for t in range(0, 13))  # ray tracer: 0 to 12 deg in 1 deg
SURROGATE_TILTS_DEG = tuple(t / 2.0 for t in range(0, 25))  # surrogate: 0 to 12 deg in 0.5 deg
POWERS_DBM = tuple(float(p) for p in range(28, 47))  # 28 to 46 dBm in 1 dB (spec Q0b)
RULE_POWER_DBM = 46.0
RADIUS_M = 1500.0  # ASSUMPTION (spec Q0): the service radius, inside the map's half-width
FIRST_RADIUS_M = 3000.0  # the first Q1 definition, beyond the map's half-width (spec Q0)
SPILL_WEIGHT = 1.0  # ASSUMPTION (spec Q0b): the two area-normalised fractions weigh the same
FIXED_SETTING = (0.0, 46.0)  # no search: tilt in deg, power in dBm (spec Q0)
RESOURCE_ELEMENTS = 1200  # ASSUMPTION (spec E2): 20 MHz carrier, as in evaluate.py
RSRP_THRESHOLD_DBM = -110.0  # ASSUMPTION (spec E2), as in evaluate.py
WITHIN_POINTS = 0.01  # spec Q0b: "within 1 point" of the optimum on the -1..1 scale
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


def rule_of_thumb_tilt_deg(radius_m: float) -> float:
    """Tilt that puts the upper half-power edge on the cell edge (spec Q, rule of thumb).

    Args:
        radius_m: Service radius.

    Returns:
        arctan(MAST_HEIGHT_M / radius_m) plus half the vertical HPBW, in degrees.
    """
    return float(np.degrees(np.arctan(MAST_HEIGHT_M / radius_m)) + vertical_hpbw_deg() / 2.0)


def radius_beyond_map(radius_m: float) -> bool:
    """Whether a service radius reaches past the map's half-width (spec Q0): then spill exists
    only in the corners and the objective rewards covering the whole map.

    Args:
        radius_m: Service radius.

    Returns:
        True if the radius is at least half the map's side.
    """
    return radius_m >= MAP_SIZE_M / 2.0


def objectives(
    gain_db: NDArray[np.floating[Any]], has_power: NDArray[np.bool_], near: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Objective Q0b of one map at every power of POWERS_DBM, on a -1..1 scale.

    Args:
        gain_db: Path gain in dB (ignored where has_power is False).
        has_power: Cells with power (the ray tracer's, or the surrogate's power head).
        near: Cells within the service radius.

    Returns:
        One objective per power: covered fraction inside minus SPILL_WEIGHT times covered
        fraction outside.

    Raises:
        ValueError: If either region is empty.
    """
    inside, outside = int(near.sum()), int((~near).sum())
    if inside == 0 or outside == 0:
        raise ValueError(f"regions of {inside} and {outside} cells: both must be non-empty")
    out = np.empty(len(POWERS_DBM))
    for k, power in enumerate(POWERS_DBM):
        covered = has_power & (gain_db >= covered_gain_db(power))
        out[k] = (covered & near).sum() / inside - SPILL_WEIGHT * (covered & ~near).sum() / outside
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


def surrogate_search(
    dataset: Path, split: str, runs: list[Path], device: str, radius_m: float
) -> dict[str, Any]:
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
        radius_m: Service radius of the objective.

    Returns:
        {"commit", "torch", "platform", "gpu", "device", "radius_m", "seeds", "runs", "terrains",
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
        near = features.geometry.distance_km * 1000.0 <= radius_m
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
        "radius_m": radius_m,
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
    dataset: Path,
    split: str,
    extra: dict[str, list[float]],
    samples_per_tx: int,
    grid: bool,
    out: Path,
) -> dict[str, Any]:
    """The ray tracer's search: every 1 deg tilt at every azimuth, timed per terrain, then
    the extra tilts other choosers picked (untimed).

    Must run after `backend.select_variant`. Per terrain the timer covers terrain
    generation, the measurement surface, one scene per azimuth and the 104 solves. One
    untimed solve on the last terrain comes first (kernel compilation). Maps are written
    as the dataset writes them, under `out/maps`, with the settings in `out/meta.json`.
    With `grid` False only the extra tilts are traced (for choices under a new objective,
    whose grid maps an earlier run already holds), and nothing is timed.

    Args:
        dataset: Dataset directory (its manifest gives the sites).
        split: Split.
        extra: {"terrain/azimuth": tilts} to trace after the timed search.
        samples_per_tx: Rays per map.
        grid: Whether to trace (and time) the 1 deg grid.
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
    if grid:
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
        if grid:
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
            print(
                f"terrain {terrain_id}: {seconds[str(terrain_id)]:.1f} s for 104 maps", flush=True
            )
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
    meta = {
        "split": split,
        "grid": grid,
        "settings": settings.record(),
        "provenance": provenance(),
        "sites": {str(k): asdict(v) for k, v in sites.items()},
        "extra": extra,
        "terrain_seconds": seconds,
        "solve_seconds": solve_seconds,
    }
    (out / META).write_text(json.dumps(meta, indent=1), newline="\n")
    return meta


def cpu_raytracer_seconds(evaluation_report: Path) -> tuple[float, float]:
    """The CPU ray tracer's measured medians, quoted from the committed evaluation report.

    Args:
        evaluation_report: results/evaluation_test.md (its data, the .json, is read).

    Returns:
        (new terrain first map, each further map) in seconds.

    Raises:
        ValueError: If the report is not an evaluation.
    """
    from sionna_twin_ops.reports import read_report

    data = read_report(evaluation_report)
    if data["kind"] != "evaluation":
        raise ValueError(f"{evaluation_report} is a {data['kind']} report, not an evaluation")
    return float(data["timing"]["rt_cpu_new"]), float(data["timing"]["rt_cpu_further"])


def traced_objectives(
    trace_dir: Path, terrain_id: int, azimuth: float, tilt: float, near: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Objective Q1 of one ray-traced search map at every power.

    Args:
        trace_dir: Output of `trace_search`.
        terrain_id: Terrain id.
        azimuth: Azimuth.
        tilt: Tilt.
        near: Cells within the service radius.

    Returns:
        One objective per power of POWERS_DBM.
    """
    gain = np.load(trace_dir / "maps" / search_map_name(terrain_id, azimuth, tilt)).astype(
        np.float64
    )
    has_power = gain > 0
    with np.errstate(divide="ignore"):
        gain_db = np.where(has_power, 10.0 * np.log10(np.where(has_power, gain, 1.0)), -np.inf)
    return objectives(gain_db, has_power, near)


_SEARCH_MAP = re.compile(r"^t(\d{3})_a(\d{3})_t(\d{2}\.\d{4})\.npy$")


def terrain_objectives(
    trace_dirs: list[Path], terrain_id: int, near: NDArray[np.bool_]
) -> dict[tuple[float, float], NDArray[np.float64]]:
    """Objectives of every search map of one terrain.

    Args:
        trace_dirs: Outputs of `trace_search`; where two hold the same map, the first wins.
        terrain_id: Terrain id.
        near: Cells within the service radius.

    Returns:
        {(azimuth, tilt): one objective per power}.
    """
    out: dict[tuple[float, float], NDArray[np.float64]] = {}
    for trace_dir in trace_dirs:
        for path in sorted((trace_dir / "maps").glob(f"t{terrain_id:03d}_*.npy")):
            m = _SEARCH_MAP.match(path.name)
            if m is None:
                raise ValueError(f"{path.name} is not a search map name")
            key = (float(m.group(2)), float(m.group(3)))
            if key not in out:
                out[key] = traced_objectives(trace_dir, terrain_id, key[0], key[1], near)
    return out


@dataclass
class Scores:
    """Ray-traced objectives and the choosers' shortfalls from the optimum.

    Attributes:
        optimum: Per "terrain/azimuth": (tilt, power, objective) of the 1 deg grid optimum.
        whole_optimum: Per terrain: (azimuth, tilt, power, objective) over the whole grid.
        shortfalls: Per chooser: {"terrain/azimuth": optimum minus the chosen objective}.
        whole_shortfalls: Per chooser: {terrain: whole-grid optimum minus the chosen}.
        cells: Per terrain: (cells inside the radius, cells outside).
        chosen: Per chooser: {"terrain/azimuth": (tilt, power)}.
    """

    optimum: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    whole_optimum: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)
    shortfalls: dict[str, dict[str, float]] = field(default_factory=dict)
    whole_shortfalls: dict[str, dict[str, float]] = field(default_factory=dict)
    cells: dict[str, tuple[int, int]] = field(default_factory=dict)
    chosen: dict[str, dict[str, tuple[float, float]]] = field(default_factory=dict)


RULE = "rule of thumb"
FIXED = f"fixed {FIXED_SETTING[0]:.0f} deg, {FIXED_SETTING[1]:.0f} dBm (no search)"
TRACER = "ray tracer, 1 deg grid"


def score(trace_dirs: list[Path], plan: dict[str, Any]) -> Scores:
    """Score every chooser's settings with the ray tracer, under the plan's service radius.

    Choosers: the rule of thumb (its tilt, RULE_POWER_DBM), the fixed setting, the surrogate
    per seed with the CPU pass's choices and, separately, the GPU pass's, and the ray tracer
    itself (shortfall 0 by definition).

    Args:
        trace_dirs: Outputs of `trace_search`, the one with the timed grid first (its
            meta.json gives the sites).
        plan: plan.json of `twin search-surrogate`.

    Returns:
        The scores.
    """
    from sionna_twin_ops.baselines import map_geometry

    meta = json.loads((trace_dirs[0] / META).read_text())
    radius_m = plan["radius_m"]
    rule_tilt = round(plan["rule_tilt_deg"], 4)
    choosers: dict[str, dict[str, tuple[float, float]]] = {RULE: {}, FIXED: {}}
    for label, key in (("CPU", "choices"), ("GPU", "gpu_choices")):
        for seed, per_case in plan[key]["cases"].items():
            choosers[f"surrogate seed {seed} ({label})"] = {
                k: (v[0], v[1]) for k, v in per_case.items()
            }
    choosers[TRACER] = {}
    scores = Scores()
    for terrain, site_record in meta["sites"].items():
        terrain_id = int(terrain)
        site = Site(**site_record)
        geometry = map_geometry(generate_terrain(terrain_id, GRID_SPACING_M), site)
        near = geometry.distance_km * 1000.0 <= radius_m
        traced = terrain_objectives(trace_dirs, terrain_id, near)

        def value(azimuth: float, tilt: float, power: float, traced: Any = traced) -> float:
            return float(traced[(azimuth, tilt)][POWERS_DBM.index(power)])

        grid = np.array(
            [[[value(a, t, p) for p in POWERS_DBM] for t in TRACE_TILTS_DEG] for a in AZIMUTHS_DEG]
        )
        a, t, p = first_best(grid)
        whole_best = float(grid[a, t, p])
        scores.cells[terrain] = (int(near.sum()), int((~near).sum()))
        scores.whole_optimum[terrain] = (
            AZIMUTHS_DEG[a],
            TRACE_TILTS_DEG[t],
            POWERS_DBM[p],
            whole_best,
        )
        for a, azimuth in enumerate(AZIMUTHS_DEG):
            case = f"{terrain}/{azimuth:.0f}"
            t, p = first_best(grid[a])
            best = float(grid[a, t, p])
            scores.optimum[case] = (TRACE_TILTS_DEG[t], POWERS_DBM[p], best)
            choosers[RULE][case] = (rule_tilt, RULE_POWER_DBM)
            choosers[FIXED][case] = FIXED_SETTING
            choosers[TRACER][case] = (TRACE_TILTS_DEG[t], POWERS_DBM[p])
            for name, chosen in choosers.items():
                tilt, power = chosen[case]
                scores.shortfalls.setdefault(name, {})[case] = best - value(azimuth, tilt, power)
        for label, key in (("CPU", "choices"), ("GPU", "gpu_choices")):
            for seed, per_terrain in plan[key]["whole_grid"].items():
                azimuth, tilt, power = per_terrain[terrain]
                scores.whole_shortfalls.setdefault(f"surrogate seed {seed} ({label})", {})[
                    terrain
                ] = whole_best - value(azimuth, tilt, power)
    scores.chosen = choosers
    return scores


def dataset_agreement(trace_dir: Path, dataset: Path) -> tuple[int, int, float]:
    """Search maps against the dataset's maps at the shared grid tilts.

    Args:
        trace_dir: Output of `trace_search`.
        dataset: The dataset directory.

    Returns:
        (maps compared, maps bit-identical, largest absolute difference in dB where both
        have power).

    Raises:
        ValueError: If a pair disagrees on which cells have power.
    """
    from sionna_twin_ops.dataset import map_name

    meta = json.loads((trace_dir / META).read_text())
    compared = identical = 0
    worst = 0.0
    for line in read_manifest(dataset):
        if str(line["terrain_id"]) not in meta["sites"] or line["kind"] != "grid":
            continue
        ours = np.load(
            trace_dir
            / "maps"
            / search_map_name(line["terrain_id"], line["azimuth_deg"], line["tilt_deg"])
        )
        theirs = np.load(
            dataset / "maps" / map_name(line["terrain_id"], line["azimuth_deg"], line["tilt_deg"])
        )
        if not np.array_equal(ours > 0, theirs > 0):
            raise ValueError(f"{line['file']}: search and dataset maps differ in cells with power")
        compared += 1
        identical += int(np.array_equal(ours, theirs))
        power = ours > 0
        if power.any():
            worst = max(worst, float(np.abs(10.0 * np.log10(ours[power] / theirs[power])).max()))
    return compared, identical, worst


def _shortfall_row(name: str, shortfalls: list[float]) -> str:
    values = np.asarray(shortfalls)
    within = int((values <= WITHIN_POINTS).sum())
    return (
        f"| {name} | {np.median(values):.4f} | {np.percentile(values, 90):.4f} | "
        f"{values.max():.4f} | {within} of {len(values)} |"
    )


def _counts_table(label: str, values: list[float], unit: str) -> list[str]:
    counts = {v: values.count(v) for v in sorted(set(values))}
    return [
        f"| {label} | " + " | ".join(f"{v:g} {unit}" for v in counts) + " |",
        "|---|" + "---|" * len(counts),
        "| cases | " + " | ".join(str(c) for c in counts.values()) + " |",
    ]


def edge_counts(scores: Scores) -> dict[str, int]:
    """How many per-case optima sit on an edge of the ray tracer's grid.

    Args:
        scores: Output of `score`.

    Returns:
        Counts for the lowest and highest tilt and power, and for any edge.
    """
    optima = list(scores.optimum.values())
    edges = {
        f"tilt {TRACE_TILTS_DEG[0]:g} deg": [t == TRACE_TILTS_DEG[0] for t, _, _ in optima],
        f"tilt {TRACE_TILTS_DEG[-1]:g} deg": [t == TRACE_TILTS_DEG[-1] for t, _, _ in optima],
        f"power {POWERS_DBM[0]:g} dBm": [p == POWERS_DBM[0] for _, p, _ in optima],
        f"power {POWERS_DBM[-1]:g} dBm": [p == POWERS_DBM[-1] for _, p, _ in optima],
    }
    counts = {name: sum(flags) for name, flags in edges.items()}
    counts["any edge"] = sum(any(flags) for flags in zip(*edges.values(), strict=True))
    return counts


EDGE_READINGS = {
    "tilt low": "the optimum would lie at an uptilt, outside the box",
    "tilt high": "the optimum would lie at more downtilt than the box allows",
    "power high": "the optimum would use more than the sector maximum",
    "power low": "the optimum would use less than the box's power floor",
}


def edge_table(scores: Scores, site_classes: dict[str, str]) -> list[str]:
    """Optima on the feasible box's edges, per site class, with a reading of each edge.

    Args:
        scores: Output of `score`.
        site_classes: Site class per terrain.

    Returns:
        Markdown lines: a table, then one reading line per edge that binds.
    """
    edges = {
        "tilt low": f"tilt {TRACE_TILTS_DEG[0]:g} deg",
        "tilt high": f"tilt {TRACE_TILTS_DEG[-1]:g} deg",
        "power low": f"power {POWERS_DBM[0]:g} dBm",
        "power high": f"power {POWERS_DBM[-1]:g} dBm",
    }

    def on_edge(key: str, tilt: float, power: float) -> bool:
        return {
            "tilt low": tilt == TRACE_TILTS_DEG[0],
            "tilt high": tilt == TRACE_TILTS_DEG[-1],
            "power low": power == POWERS_DBM[0],
            "power high": power == POWERS_DBM[-1],
        }[key]

    classes = sorted(set(site_classes.values()))
    counts: dict[str, dict[str, int]] = {
        c: dict.fromkeys([*edges, "any", "cases"], 0) for c in classes
    }
    for case, (tilt, power, _) in scores.optimum.items():
        row = counts[site_classes[case.split("/")[0]]]
        row["cases"] += 1
        hits = [key for key in edges if on_edge(key, tilt, power)]
        for key in hits:
            row[key] += 1
        row["any"] += int(bool(hits))
    lines = [
        "| site class | cases | " + " | ".join(edges.values()) + " | any edge |",
        "|---|---|" + "---|" * (len(edges) + 1),
    ]
    for c in classes:
        row = counts[c]
        lines.append(
            f"| {c} | {row['cases']} | "
            + " | ".join(str(row[k]) for k in edges)
            + f" | {row['any']} |"
        )
    lines.append("")
    for c in classes:
        binding = [k for k in edges if counts[c][k] > 0]
        if binding:
            lines.append(
                f"- {c}: "
                + "; ".join(
                    f"{counts[c][k]} of {counts[c]['cases']} at {edges[k]} "
                    f"(reading: {EDGE_READINGS[k]})"
                    for k in sorted(binding, key=lambda k: -counts[c][k])
                )
                + "."
            )
    return lines


def objective_section(
    scores: Scores, plan: dict[str, Any], terrains: list[str], site_classes: dict[str, str]
) -> list[str]:
    """Report lines for one objective (one service radius).

    Args:
        scores: Output of `score` under the plan.
        plan: plan.json of `twin search-surrogate`.
        terrains: Terrain ids, in order.
        site_classes: Site class per terrain.

    Returns:
        Markdown lines.
    """
    radius_m = plan["radius_m"]
    cases = len(scores.optimum)
    all_names = list(scores.shortfalls)
    names = [n for n in all_names if not n.endswith("(GPU)")]
    gpu_names = [n for n in all_names if n.endswith("(GPU)")]
    inside = sorted({c[0] for c in scores.cells.values()})
    outside = sorted({c[1] for c in scores.cells.values()})
    lines = [f"## Objective with a {radius_m / 1000:g} km service radius", ""]
    if radius_beyond_map(radius_m):
        raise ValueError(f"radius {radius_m} m reaches beyond the map (spec Q0)")
    lines += [
        f"Objective Q0b: the covered fraction of the cells within {radius_m / 1000:g} km of the "
        f"site ({', '.join(str(n) for n in inside)} cells) minus {SPILL_WEIGHT:g} times the "
        f"covered fraction of the map's cells beyond it ({', '.join(str(n) for n in outside)} "
        "cells), so neither region wins by size; radius and equal weights are ASSUMPTION. It "
        "runs from -1 to 1. Rule of thumb: arctan(30 m / "
        f"{radius_m / 1000:g} km) + {plan['hpbw_deg']:.2f} / 2 = {plan['rule_tilt_deg']:.2f} "
        f"deg, at {RULE_POWER_DBM:.0f} dBm.",
        "",
        f"### Per (terrain, azimuth), {cases} cases",
        "",
        "Shortfall = the ray tracer's optimum on the 1 deg grid minus the chosen setting's "
        "ray-traced objective, in objective points; a half-degree choice can beat the grid and "
        f"go negative. Within 1 point: shortfall at most {WITHIN_POINTS:g}. The objective is "
        "not smooth in tilt (the column's nulls and sidelobes sweep over distant ground, so "
        "spill beyond the radius can rise and fall within 1 deg), so the 1 deg grid's optimum "
        "is a reference with that resolution, not the true optimum; the most negative "
        "shortfalls below measure how far a half-degree setting beat it.",
        "",
        "| chooser | median shortfall | p90 | max | within 1 point |",
        "|---|---|---|---|---|",
    ]
    lines += [_shortfall_row(name, list(scores.shortfalls[name].values())) for name in names]
    differ = sum(
        plan["choices"]["cases"][seed][case] != plan["gpu_choices"]["cases"][seed][case]
        for seed in plan["choices"]["cases"]
        for case in plan["choices"]["cases"][seed]
    )
    total = sum(len(v) for v in plan["choices"]["cases"].values())
    lines += [
        "",
        f"The GPU pass's choices differ from the CPU pass's in {differ} of {total} (seed, case) "
        "pairs; scored with the ray tracer they give:",
        "",
        "| chooser | median shortfall | p90 | max | within 1 point |",
        "|---|---|---|---|---|",
    ]
    lines += [_shortfall_row(name, list(scores.shortfalls[name].values())) for name in gpu_names]
    surrogate_names = [n for n in names if n.startswith("surrogate")]
    worst_name, worst_case = max(
        ((n, k) for n in surrogate_names for k in scores.shortfalls[n]),
        key=lambda nk: scores.shortfalls[nk[0]][nk[1]],
    )

    def beyond(name: str) -> str:
        found = sorted(
            {k.split("/")[0] for k, v in scores.shortfalls[name].items() if v > WITHIN_POINTS},
            key=int,
        )
        return ", ".join(found) if found else "none"

    short = {n: beyond(n) for n in surrogate_names}
    best_name, best_case = min(
        ((n, k) for n in surrogate_names for k in scores.shortfalls[n]),
        key=lambda nk: scores.shortfalls[nk[0]][nk[1]],
    )
    whole_best_name, whole_best_terrain = min(
        ((n, t) for n in scores.whole_shortfalls for t in scores.whole_shortfalls[n]),
        key=lambda nt: scores.whole_shortfalls[nt[0]][nt[1]],
    )
    lines += [
        "",
        f"Most negative surrogate shortfall (a half-degree setting beating the 1 deg grid): "
        f"{scores.shortfalls[best_name][best_case]:.4f} per case ({best_name}, terrain "
        f"{best_case.split('/')[0]}, azimuth {best_case.split('/')[1]} deg) and "
        f"{scores.whole_shortfalls[whole_best_name][whole_best_terrain]:.4f} over a whole "
        f"terrain ({whole_best_name}, terrain {whole_best_terrain}).",
        "",
        f"Largest surrogate shortfall: {scores.shortfalls[worst_name][worst_case]:.4f} "
        f"({worst_name}, terrain {worst_case.split('/')[0]}, azimuth "
        f"{worst_case.split('/')[1]} deg). Terrains with a case more than 1 point short: "
        + "; ".join(f"{n}: {t}" for n, t in short.items())
        + f"; rule of thumb: {beyond(RULE)}; fixed setting: {beyond(FIXED)}.",
        "",
        "Where the ray tracer's optimum lies:",
        "",
    ]
    lines += _counts_table("optimum tilt", [t for t, _, _ in scores.optimum.values()], "deg")
    lines += [""]
    lines += _counts_table("optimum power", [p for _, p, _ in scores.optimum.values()], "dBm")
    edges = edge_counts(scores)
    lines += [
        "",
        "#### Optima on the edge of the feasible box",
        "",
        "The box (tilt 0 to 12 deg, power 28 to 46 dBm, ASSUMPTION, spec Q0c) is the feasible "
        "set, so an optimum on its edge is a constrained optimum, not a search artefact. "
        f"On an edge: {edges['any edge']} of {cases}. Per site class:",
        "",
        *edge_table(scores, site_classes),
        "",
        "#### By terrain",
        "",
        "Largest shortfall over the 8 azimuths, and the number of azimuths more than 1 point "
        "short.",
        "",
        "| terrain | site class | optimum tilt, median (deg) | " + " | ".join(names) + " |",
        "|---|---|---|" + "---|" * len(names),
    ]
    for terrain in terrains:
        keys = [k for k in scores.optimum if k.split("/")[0] == terrain]
        tilts = [scores.optimum[k][0] for k in keys]
        cells = []
        for name in names:
            values = [scores.shortfalls[name][k] for k in keys]
            cells.append(f"{max(values):.4f} ({sum(v > WITHIN_POINTS for v in values)})")
        lines.append(
            f"| {terrain} | {site_classes[terrain]} | {np.median(tilts):.1f} | "
            + " | ".join(cells)
            + " |"
        )
    whole_names = [n for n in scores.whole_shortfalls if not n.endswith("(GPU)")]
    lines += [
        "",
        "### One choice per terrain over azimuth, tilt and power",
        "",
        "Shortfall from the ray tracer's whole-grid optimum.",
        "",
        "| terrain | ray tracer optimum (azimuth, tilt, power: objective) | "
        + " | ".join(whole_names)
        + " |",
        "|---|---|" + "---|" * len(whole_names),
    ]
    for terrain in terrains:
        a, t, p, best = scores.whole_optimum[terrain]
        lines.append(
            f"| {terrain} | {a:.0f} deg, {t:.0f} deg, {p:.0f} dBm: {best:.4f} | "
            + " | ".join(f"{scores.whole_shortfalls[n][terrain]:.4f}" for n in whole_names)
            + " |"
        )
    lines += [
        "",
        "| chooser | median shortfall | p90 | max | within 1 point |",
        "|---|---|---|---|---|",
    ]
    lines += [
        _shortfall_row(name, list(scores.whole_shortfalls[name].values())) for name in whole_names
    ]
    return [*lines, ""]


def first_objective_section(first_report: Path) -> list[str]:
    """The first objective's test results, quoted from its committed report and marked flawed.

    Args:
        first_report: The first run's report (results/search_test_first_objective.md).

    Returns:
        Markdown lines.

    Raises:
        ValueError: If the report lacks its primary table.
    """
    text = first_report.read_text().splitlines()
    start = text.index("## Primary: per (terrain, azimuth), 72 cases")
    table = []
    for line in text[start:]:
        if line.startswith("|"):
            table.append(line)
        elif table:
            break
    optimum = [line for line in text if line.startswith("The ray tracer's optimum uses")]
    if not table or len(optimum) != 1:
        raise ValueError(f"{first_report} lacks its primary table")
    return [
        f"## First objective, {FIRST_RADIUS_M / 1000:g} km radius: a flawed definition",
        "",
        f"The first definition (covered cells within {FIRST_RADIUS_M / 1000:g} km minus covered "
        f"cells beyond, raw counts, power 37 to 46 dBm, share of the optimum) is flawed: its "
        f"radius is beyond the map's half-width ({MAP_SIZE_M / 2000:g} km), so only the corners "
        "lie outside it, there is almost no spill to avoid, and the objective rewards covering "
        "the whole map. A fixed setting then nearly ties any search. It was run once on this "
        f"split; its report is kept verbatim in `{first_report.name}`. Its primary table, quoted:",
        "",
        *table,
        "",
        optimum[0],
        "",
    ]


def report_markdown(
    split: str,
    scores: Scores,
    plan: dict[str, Any],
    first_objective: list[str],
    surrogate: dict[str, dict[str, Any]],
    trace_meta: dict[str, Any],
    agreement: tuple[int, int, float],
    cpu_seconds: tuple[float, float],
    site_classes: dict[str, str],
) -> str:
    """The search report (spec rules Q, Q0b, Q0c): the objective, time and a check.

    Args:
        split: The split searched.
        scores: Output of `score` under the plan.
        plan: plan.json of `twin search-surrogate`.
        first_objective: Lines of `first_objective_section` (test), or none (validation).
        surrogate: {"cpu": record, "cuda": record} of `surrogate_search` (the timing).
        trace_meta: meta.json of the timed `trace_search`.
        agreement: Output of `dataset_agreement`.
        cpu_seconds: Output of `cpu_raytracer_seconds`.
        site_classes: Site class per terrain.

    Returns:
        Markdown.
    """
    settings = trace_meta["settings"]
    origin = trace_meta["provenance"]
    terrains = list(trace_meta["sites"])
    cases = len(AZIMUTHS_DEG) * len(terrains)
    design = (
        "The objective was designed on the validation split (`results/search_validation.md`, "
        "spec Q0, Q0b, Q0c) and this split was then searched once."
        if split == "test"
        else "This split was used to design the objective (spec Q0, Q0b, Q0c): the first "
        "definition's flaw, the area normalisation and the feasible box were settled on it "
        "before the test split was searched."
    )
    lines = [
        f"# Tilt and power search on the {split} split",
        "",
        "Synthetic terrain, not a real place; one sector, no vegetation, buildings or "
        "interference, flat Earth. Ray tracer: Sionna RT "
        f"{origin['sionna-rt']}, {settings['variant']}, {settings['samples_per_tx']:.0e} rays, "
        f"max depth {settings['max_depth']}, line of sight and specular reflection only, seed "
        f"{settings['seed']} (grid maps at commit {origin['commit'][:7]}, {origin['gpu']}); "
        f"pattern: 3GPP TR 38.901. Surrogate: {', '.join(surrogate['cpu']['runs'])} (best "
        f"epochs; torch {surrogate['cpu']['torch']}; search commit {plan['commit'][:7]}). "
        "Written by `twin search-report`.",
        "",
        design,
        "",
        f"Cases: {cases} (terrains {', '.join(terrains)}, {len(AZIMUTHS_DEG)} azimuths each). "
        "Variables: electrical tilt and sector power, within the feasible box: tilt "
        f"{TRACE_TILTS_DEG[0]:g} to {TRACE_TILTS_DEG[-1]:g} deg, power {POWERS_DBM[0]:g} to "
        f"{POWERS_DBM[-1]:g} dBm, {POWERS_DBM[-1]:g} dBm being the sector maximum (both "
        "ASSUMPTION, spec Q0c). A cell is covered where RSRP reaches "
        f"{RSRP_THRESHOLD_DBM:.0f} dBm, with the sector power spread over {RESOURCE_ELEMENTS} "
        "resource elements (both ASSUMPTION, as in the evaluation).",
        "",
        "Choosers:",
        f"- Ray tracer: tilt in 1 deg ({len(TRACE_TILTS_DEG)} traces per case), power in 1 dB "
        "(post-processing). Its best setting is the optimum every shortfall is taken from.",
        f"- Surrogate: tilt in 0.5 deg ({len(SURROGATE_TILTS_DEG)} inferences per case), the "
        "same powers; covered cells need the power head to say power. Choices come from the "
        "CPU pass; the GPU pass in float32 breaks some near-ties the other way, so its choices "
        "are scored too and reported separately.",
        "- Rule of thumb: tilt = arctan(30 m / radius) + half the column's vertical half-power "
        "beamwidth (from the TR 38.901 element times the 8 x 1 array factor), at "
        f"{RULE_POWER_DBM:.0f} dBm.",
        f"- Fixed setting, no search: {FIXED_SETTING[0]:.0f} deg at {FIXED_SETTING[1]:.0f} dBm.",
        "",
        "Every chosen setting is scored with the ray tracer: settings off the 1 deg grid "
        "(half-degree tilts, the rule's tilt) were traced for the purpose. Ties go to the lower "
        "azimuth, then the lower tilt, then the lower power.",
        "",
        *objective_section(scores, plan, terrains, site_classes),
        *first_objective,
    ]

    gpu_s = [s for per in surrogate["cuda"]["seconds"].values() for s in per.values()]
    cpu_s = [s for per in surrogate["cpu"]["seconds"].values() for s in per.values()]
    rt_s = list(trace_meta["terrain_seconds"].values())
    first, further = cpu_seconds
    maps = len(AZIMUTHS_DEG) * len(TRACE_TILTS_DEG)
    rt_cpu = first + (maps - 1) * further
    lines += [
        "## Wall time per terrain, same machine",
        "",
        f"Seconds per terrain for the whole search ({len(AZIMUTHS_DEG)} azimuths). Surrogate: "
        f"terrain generation, terrain channels, per-map channels and {len(AZIMUTHS_DEG)} "
        f"batched inferences of {len(SURROGATE_TILTS_DEG)} tilts "
        f"({len(AZIMUTHS_DEG) * len(SURROGATE_TILTS_DEG)} maps), and the objectives at every "
        f"power; {len(gpu_s)} timings each ({len(surrogate['cpu']['seeds'])} seeds x "
        f"{len(terrains)} terrains, {surrogate['cpu']['radius_m'] / 1000:g} km objective), after "
        "one untimed warm-up. Ray tracer: terrain generation, measurement surface, one scene "
        f"per azimuth and {maps} solves, through the Windows runner, after one untimed solve; "
        "the objectives take milliseconds and are not in it. The CPU ray tracer was not run for "
        "this: its figure is estimated as the evaluation's measured CPU medians "
        f"(`results/evaluation_test.md`), {first:.3f} s for a new terrain's first map plus "
        f"{maps - 1} x {further:.3f} s.",
        "",
        "| method | hardware | median | min | max |",
        "|---|---|---|---|---|",
        f"| surrogate | GPU ({surrogate['cuda']['gpu']}) | {np.median(gpu_s):.2f} | "
        f"{min(gpu_s):.2f} | {max(gpu_s):.2f} |",
        f"| surrogate | CPU | {np.median(cpu_s):.2f} | {min(cpu_s):.2f} | {max(cpu_s):.2f} |",
        f"| ray tracer | GPU ({origin['gpu']}) | {np.median(rt_s):.1f} | {min(rt_s):.1f} | "
        f"{max(rt_s):.1f} |",
        f"| ray tracer | CPU (llvm), estimated | {rt_cpu:.0f} | | |",
        "",
        f"Per terrain, the ray tracer's search takes {np.median(rt_s) / np.median(gpu_s):.0f} "
        f"times as long as the surrogate's on the GPU; without a GPU, an estimated "
        f"{rt_cpu / np.median(cpu_s):.0f} times.",
        "",
    ]
    compared, identical, worst = agreement
    lines += [
        "## Check: search maps against the dataset",
        "",
        f"The search re-traced the dataset's grid tilts (0, 3, 6, 9, 12 deg) with the same "
        f"settings: {identical} of {compared} maps are bit-identical; the largest difference "
        f"where both have power is {worst:.2e} dB.",
        "",
    ]
    return "\n".join(lines)


def search_figure(scores: Scores, caption: str, path: Path) -> None:
    """Shortfalls from the ray tracer's optimum: per case (left) and per terrain over the whole
    grid (right), for the rule of thumb, the fixed setting and the surrogate's CPU choices.

    Args:
        scores: Output of `score`.
        caption: Provenance caption printed under the panels.
        path: Output image.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["axes.unicode_minus"] = False
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.6), layout="constrained")
    styles = {
        RULE: ("#b8423a", "o"),
        FIXED: ("#c98a1b", "v"),
        "surrogate seed 0 (CPU)": ("#2a6fb0", "s"),
        "surrogate seed 1 (CPU)": ("#3f9a5a", "D"),
        "surrogate seed 2 (CPU)": ("#8a5bb5", "^"),
    }
    for name, (color, marker) in styles.items():
        values = np.sort(np.asarray(list(scores.shortfalls[name].values())))
        rank = np.arange(1, len(values) + 1)
        left.plot(values, rank, color=color, marker=marker, ms=3.5, lw=1.5, label=name)
    left.axvline(WITHIN_POINTS, color="#8c8a85", lw=1.0, ls="--")
    left.set_xlabel("shortfall from the ray tracer's optimum (objective points)", fontsize=9)
    left.set_ylabel(f"cases at or below the shortfall (of {len(scores.optimum)})", fontsize=9)
    left.set_title("per (terrain, azimuth): tilt and power", fontsize=10, loc="left")
    left.legend(fontsize=8, loc="lower right", frameon=False)
    terrains = list(scores.whole_optimum)
    x = np.arange(len(terrains))
    for k, name in enumerate(n for n in styles if n not in (RULE, FIXED)):
        color, marker = styles[name]
        right.plot(
            x + (k - 1) * 0.18,
            [scores.whole_shortfalls[name][t] for t in terrains],
            ls="none",
            marker=marker,
            color=color,
            ms=6,
            label=name,
        )
    right.axhline(0.0, color="#8c8a85", lw=1.0)
    right.axhline(WITHIN_POINTS, color="#8c8a85", lw=1.0, ls="--")
    right.set_xticks(x, terrains, fontsize=8)
    right.set_xlabel("terrain", fontsize=9)
    right.set_ylabel("shortfall from the ray tracer's optimum", fontsize=9)
    right.set_title("per terrain: azimuth, tilt and power", fontsize=10, loc="left")
    right.legend(fontsize=8, loc="upper left", frameon=False)
    for ax in (left, right):
        ax.tick_params(labelsize=8)
        ax.grid(color="#e4e2dc", lw=0.6)
        ax.set_axisbelow(True)
    fig.text(0.01, -0.02, caption, fontsize=7.5, color="#52514e", ha="left", va="top", wrap=True)
    fig.savefig(path, dpi=110, bbox_inches="tight", pil_kwargs={"quality": 88})
    plt.close(fig)


def report_data(
    split: str,
    scores: Scores,
    plan: dict[str, Any],
    first_objective: list[str],
    surrogate: dict[str, dict[str, Any]],
    trace_meta: dict[str, Any],
    agreement: tuple[int, int, float],
    cpu_seconds: tuple[float, float],
    site_classes: dict[str, str],
) -> dict[str, Any]:
    """The search report's data: everything `report_markdown` reads, JSON-safe.

    The surrogate records keep their timings and provenance, not their objective arrays;
    the trace record keeps its settings, provenance, sites and per-terrain seconds.

    Args:
        split: The split searched.
        scores: Output of `score` under the plan.
        plan: plan.json of `twin search-surrogate`.
        first_objective: Lines of `first_objective_section` (test), or none (validation).
        surrogate: {"cpu": record, "cuda": record} of `surrogate_search`.
        trace_meta: meta.json of the timed `trace_search`.
        agreement: Output of `dataset_agreement`.
        cpu_seconds: Output of `cpu_raytracer_seconds`.
        site_classes: Site class per terrain.

    Returns:
        Data for `report_markdown_from_data`.
    """
    return {
        "kind": "search",
        "split": split,
        "scores": asdict(scores),
        "plan": plan,
        "first_objective": first_objective,
        "surrogate": {
            device: {k: v for k, v in record.items() if k != "objectives"}
            for device, record in surrogate.items()
        },
        "trace_meta": {
            k: trace_meta[k] for k in ("settings", "provenance", "sites", "terrain_seconds")
        },
        "agreement": list(agreement),
        "cpu_seconds": list(cpu_seconds),
        "site_classes": site_classes,
    }


def report_markdown_from_data(data: dict[str, Any]) -> str:
    """The search report rendered from `report_data`.

    Args:
        data: Output of `report_data` (or its JSON).

    Returns:
        Markdown.
    """
    s = data["scores"]
    scores = Scores(
        optimum={k: tuple(v) for k, v in s["optimum"].items()},
        whole_optimum={k: tuple(v) for k, v in s["whole_optimum"].items()},
        shortfalls=s["shortfalls"],
        whole_shortfalls=s["whole_shortfalls"],
        cells={k: tuple(v) for k, v in s["cells"].items()},
        chosen={n: {k: tuple(v) for k, v in c.items()} for n, c in s["chosen"].items()},
    )
    first, further = data["cpu_seconds"]
    compared, identical, worst = data["agreement"]
    return report_markdown(
        data["split"],
        scores,
        data["plan"],
        data["first_objective"],
        data["surrogate"],
        data["trace_meta"],
        (compared, identical, worst),
        (first, further),
        data["site_classes"],
    )
