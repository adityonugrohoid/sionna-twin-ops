"""Tilt and power search (spec rule Q): the surrogate and a rule of thumb against the ray tracer.

Per (terrain, azimuth) case the variables are electrical tilt and sector power. The
objective (Q1) counts covered cells within a service radius of the site minus SPILL_WEIGHT
times covered cells beyond it, within the map; a cell is covered where RSRP (path gain plus
power per resource element, as in spec E2) reaches the evaluation's RSRP threshold. The ray
tracer searches tilt on a 1 deg grid, the surrogate on a 0.5 deg grid; power is
post-processing on both sides. Every chosen setting is scored with the ray tracer. Synthetic
terrain.
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
POWERS_DBM = tuple(float(p) for p in range(37, 47))  # 37 to 46 dBm in 1 dB
RULE_POWER_DBM = 46.0
RADIUS_M = 1500.0  # ASSUMPTION (spec Q0): the service radius, inside the map's half-width
FIRST_RADIUS_M = 3000.0  # the first Q1 definition, beyond the map's half-width (spec Q0)
SPILL_WEIGHT = 1.0  # ASSUMPTION (spec Q1): one cell of spill beyond the radius costs one cell
FIXED_SETTING = (0.0, 46.0)  # no search: tilt in deg, power in dBm (spec Q0)
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
    """Objective Q1 of one map at every power of POWERS_DBM.

    Args:
        gain_db: Path gain in dB (ignored where has_power is False).
        has_power: Cells with power (the ray tracer's, or the surrogate's power head).
        near: Cells within the service radius.

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


_CPU_ROW = re.compile(r"^\| CPU \| (new terrain, first map|each further map) \| ([\d.]+) \|")


def cpu_raytracer_seconds(evaluation_report: Path) -> tuple[float, float]:
    """The CPU ray tracer's measured medians, quoted from the committed evaluation report.

    Args:
        evaluation_report: results/evaluation_test.md.

    Returns:
        (new terrain first map, each further map) in seconds.

    Raises:
        ValueError: If the report lacks either row.
    """
    rows = {
        m.group(1): float(m.group(2))
        for line in evaluation_report.read_text().splitlines()
        if (m := _CPU_ROW.match(line))
    }
    if len(rows) != 2:
        raise ValueError(f"{evaluation_report} lacks the CPU ray tracer rows")
    return rows["new terrain, first map"], rows["each further map"]


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
    """Ray-traced objectives and the choosers' shares of the optimum.

    Attributes:
        optimum: Per "terrain/azimuth": (tilt, power, objective) of the 1 deg grid optimum.
        whole_optimum: Per terrain: (azimuth, tilt, power, objective) over the whole grid.
        shares: Per chooser: {"terrain/azimuth": share of the optimum}.
        whole_shares: Per chooser: {terrain: share of the whole-grid optimum}.
        chosen: Per chooser: {"terrain/azimuth": (tilt, power)}.
    """

    optimum: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    whole_optimum: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)
    shares: dict[str, dict[str, float]] = field(default_factory=dict)
    whole_shares: dict[str, dict[str, float]] = field(default_factory=dict)
    chosen: dict[str, dict[str, tuple[float, float]]] = field(default_factory=dict)


RULE = "rule of thumb"
FIXED = f"fixed {FIXED_SETTING[0]:.0f} deg, {FIXED_SETTING[1]:.0f} dBm (no search)"
TRACER = "ray tracer, 1 deg grid"


def score(trace_dirs: list[Path], plan: dict[str, Any]) -> Scores:
    """Score every chooser's settings with the ray tracer, under the plan's service radius.

    Choosers: the rule of thumb (its tilt, RULE_POWER_DBM), the fixed setting, the surrogate
    per seed with the CPU pass's choices and, separately, the GPU pass's, and the ray tracer
    itself (share 1 by definition).

    Args:
        trace_dirs: Outputs of `trace_search`, the one with the timed grid first (its
            meta.json gives the sites).
        plan: plan.json of `twin search-surrogate`.

    Returns:
        The scores.

    Raises:
        ValueError: If an optimum is not positive (a share would be meaningless).
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
        if whole_best <= 0:
            raise ValueError(f"terrain {terrain}: whole-grid optimum {whole_best} is not positive")
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
            if best <= 0:
                raise ValueError(f"case {case}: optimum {best} is not positive")
            scores.optimum[case] = (TRACE_TILTS_DEG[t], POWERS_DBM[p], best)
            choosers[RULE][case] = (rule_tilt, RULE_POWER_DBM)
            choosers[FIXED][case] = FIXED_SETTING
            choosers[TRACER][case] = (TRACE_TILTS_DEG[t], POWERS_DBM[p])
            for name, chosen in choosers.items():
                tilt, power = chosen[case]
                scores.shares.setdefault(name, {})[case] = value(azimuth, tilt, power) / best
        for label, key in (("CPU", "choices"), ("GPU", "gpu_choices")):
            for seed, per_terrain in plan[key]["whole_grid"].items():
                azimuth, tilt, power = per_terrain[terrain]
                scores.whole_shares.setdefault(f"surrogate seed {seed} ({label})", {})[terrain] = (
                    value(azimuth, tilt, power) / whole_best
                )
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


def _share_row(name: str, shares: list[float]) -> str:
    values = np.asarray(shares)
    near = int((values >= NEAR_OPTIMUM).sum())
    return (
        f"| {name} | {np.median(values):.4f} | {np.percentile(values, 10):.4f} | "
        f"{values.min():.4f} | {near} of {len(values)} |"
    )


def _counts_table(label: str, values: list[float], unit: str) -> list[str]:
    counts = {v: values.count(v) for v in sorted(set(values))}
    return [
        f"| {label} | " + " | ".join(f"{v:g} {unit}" for v in counts) + " |",
        "|---|" + "---|" * len(counts),
        "| cases | " + " | ".join(str(c) for c in counts.values()) + " |",
    ]


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
    names = list(scores.shares)
    lines = [f"## Objective with a {radius_m / 1000:g} km service radius", ""]
    if radius_beyond_map(radius_m):
        lines += [
            f"FLAWED DEFINITION. The radius ({radius_m / 1000:g} km) is beyond the map's "
            f"half-width ({MAP_SIZE_M / 2000:g} km): only the corners lie outside it, so there "
            "is almost no spill to avoid and the objective rewards covering as much of the map "
            "as possible. It cannot show what a search buys. Kept as run, for the record "
            "(spec Q0).",
            "",
        ]
    lines += [
        f"Objective: covered cells within {radius_m / 1000:g} km of the site (ASSUMPTION) minus "
        f"{SPILL_WEIGHT:g} (ASSUMPTION) times covered cells beyond it, within the map. Rule of "
        f"thumb: arctan(30 m / {radius_m / 1000:g} km) + {plan['hpbw_deg']:.2f} / 2 = "
        f"{plan['rule_tilt_deg']:.2f} deg, at {RULE_POWER_DBM:.0f} dBm.",
        "",
        f"### Per (terrain, azimuth), {cases} cases",
        "",
        "Share = the chosen setting's ray-traced objective over the ray tracer's optimum on the "
        "1 deg grid; a half-degree choice can exceed 1.",
        "",
        "| chooser | median share | p10 | min | within 1% of the optimum |",
        "|---|---|---|---|---|",
    ]
    lines += [_share_row(name, list(scores.shares[name].values())) for name in names]
    surrogate_names = [n for n in names if n.startswith("surrogate")]
    worst_name, worst_case = min(
        ((n, k) for n in surrogate_names for k in scores.shares[n]),
        key=lambda nk: scores.shares[nk[0]][nk[1]],
    )
    short = {
        k.split("/")[0]
        for n in surrogate_names
        for k, v in scores.shares[n].items()
        if v < NEAR_OPTIMUM
    }

    def below(name: str) -> str:
        found = sorted(
            {k.split("/")[0] for k, v in scores.shares[name].items() if v < NEAR_OPTIMUM}, key=int
        )
        return ", ".join(found) if found else "none"

    lines += [
        "",
        f"Lowest surrogate share: {scores.shares[worst_name][worst_case]:.4f} ({worst_name}, "
        f"terrain {worst_case.split('/')[0]}, azimuth {worst_case.split('/')[1]} deg). "
        + (
            "No surrogate choice falls below 99% of the optimum on any terrain."
            if not short
            else "Surrogate choices fall below 99% on terrains "
            f"{', '.join(sorted(short, key=int))}."
        )
        + f" Terrains where the rule of thumb falls below 99%: {below(RULE)} (lowest "
        f"{min(scores.shares[RULE].values()):.4f}); the fixed setting: {below(FIXED)} (lowest "
        f"{min(scores.shares[FIXED].values()):.4f}).",
        "",
        "Where the ray tracer's optimum lies:",
        "",
    ]
    lines += _counts_table("optimum tilt", [t for t, _, _ in scores.optimum.values()], "deg")
    lines += [""]
    lines += _counts_table("optimum power", [p for _, p, _ in scores.optimum.values()], "dBm")
    lines += [
        "",
        "#### By terrain",
        "",
        "Lowest share over the 8 azimuths, and the number of azimuths below 99% of the optimum.",
        "",
        "| terrain | site class | optimum tilt, median (deg) | " + " | ".join(names) + " |",
        "|---|---|---|" + "---|" * len(names),
    ]
    for terrain in terrains:
        keys = [k for k in scores.optimum if k.split("/")[0] == terrain]
        tilts = [scores.optimum[k][0] for k in keys]
        cells = []
        for name in names:
            values = [scores.shares[name][k] for k in keys]
            cells.append(f"{min(values):.4f} ({sum(v < NEAR_OPTIMUM for v in values)})")
        lines.append(
            f"| {terrain} | {site_classes[terrain]} | {np.median(tilts):.1f} | "
            + " | ".join(cells)
            + " |"
        )
    whole_names = list(scores.whole_shares)
    lines += [
        "",
        "### One choice per terrain over azimuth, tilt and power",
        "",
        "| terrain | ray tracer optimum (azimuth, tilt, power: objective) | "
        + " | ".join(whole_names)
        + " |",
        "|---|---|" + "---|" * len(whole_names),
    ]
    for terrain in terrains:
        a, t, p, best = scores.whole_optimum[terrain]
        lines.append(
            f"| {terrain} | {a:.0f} deg, {t:.0f} deg, {p:.0f} dBm: {best:.0f} | "
            + " | ".join(f"{scores.whole_shares[n][terrain]:.4f}" for n in whole_names)
            + " |"
        )
    lines += [
        "",
        "| chooser | median share | p10 | min | within 1% of the optimum |",
        "|---|---|---|---|---|",
    ]
    lines += [_share_row(name, list(scores.whole_shares[name].values())) for name in whole_names]
    return [*lines, ""]


def report_markdown(
    split: str,
    sections: list[tuple[Scores, dict[str, Any]]],
    surrogate: dict[str, dict[str, Any]],
    trace_meta: dict[str, Any],
    agreement: tuple[int, int, float],
    cpu_seconds: tuple[float, float],
    site_classes: dict[str, str],
) -> str:
    """The search report (spec rule Q): one section per objective, then time and a check.

    Args:
        split: The split searched.
        sections: (scores, plan) per objective, in report order.
        surrogate: {"cpu": record, "cuda": record} of `surrogate_search` for the timing (the
            last objective's).
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
    agree = [plan["gpu_choices_agree"] for _, plan in sections]
    lines = [
        f"# Tilt and power search on the {split} split",
        "",
        "Synthetic terrain, not a real place; one sector, no vegetation, buildings or "
        "interference, flat Earth. Ray tracer: Sionna RT "
        f"{origin['sionna-rt']}, {settings['variant']}, {settings['samples_per_tx']:.0e} rays, "
        f"max depth {settings['max_depth']}, line of sight and specular reflection only, seed "
        f"{settings['seed']} (grid maps at commit {origin['commit'][:7]}, {origin['gpu']}); "
        f"pattern: 3GPP TR 38.901. Surrogate: {', '.join(surrogate['cpu']['runs'])} (best "
        f"epochs; torch {surrogate['cpu']['torch']}; search commits "
        f"{', '.join(sorted({plan['commit'][:7] for _, plan in sections}))}). Written by "
        "`twin search-report`.",
        "",
        f"Cases: {cases} (terrains {', '.join(terrains)}, {len(AZIMUTHS_DEG)} azimuths each). "
        "Variables: tilt and sector power. Objective Q1: covered cells within a service radius "
        f"minus {SPILL_WEIGHT:g} (ASSUMPTION) times covered cells beyond it, within the map "
        f"({MAP_SIZE_M / 1000:.2f} km square, centred on the site); a cell is covered where RSRP "
        f"reaches {RSRP_THRESHOLD_DBM:.0f} dBm, with the sector power spread over "
        f"{RESOURCE_ELEMENTS} resource elements (both ASSUMPTION, as in the evaluation).",
        "",
        "Choosers:",
        f"- Ray tracer: tilt 0 to 12 deg in 1 deg ({len(TRACE_TILTS_DEG)} traces per case), power "
        f"{POWERS_DBM[0]:.0f} to {POWERS_DBM[-1]:.0f} dBm in 1 dB (post-processing). Its best "
        "setting is the optimum every share is taken of.",
        f"- Surrogate: tilt 0 to 12 deg in 0.5 deg ({len(SURROGATE_TILTS_DEG)} inferences per "
        "case), the same powers; covered cells need the power head to say power. Choices come "
        "from the CPU pass; the GPU pass in float32 breaks some near-ties the other way "
        f"(choices agree, per objective: {', '.join(str(a) for a in agree)}), so its choices "
        "are scored too.",
        "- Rule of thumb: tilt = arctan(30 m / radius) + half the column's vertical half-power "
        "beamwidth (from the TR 38.901 element times the 8 x 1 array factor), at "
        f"{RULE_POWER_DBM:.0f} dBm.",
        f"- Fixed setting, no search: {FIXED_SETTING[0]:.0f} deg at {FIXED_SETTING[1]:.0f} dBm.",
        "",
        "Every chosen setting is scored with the ray tracer: settings off the 1 deg grid "
        "(half-degree tilts, the rule's tilt) were traced for the purpose. Ties go to the lower "
        "azimuth, then the lower tilt, then the lower power.",
        "",
    ]
    for scores, plan in sections:
        lines += objective_section(scores, plan, terrains, site_classes)

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
    """Shares of the ray tracer's optimum: per case (left) and per terrain over the whole grid
    (right), for the rule of thumb, the fixed setting and the surrogate's CPU choices.

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
        values = np.sort(np.asarray(list(scores.shares[name].values())))
        rank = np.arange(1, len(values) + 1)
        left.plot(values, rank, color=color, marker=marker, ms=3.5, lw=1.5, label=name)
    left.axvline(NEAR_OPTIMUM, color="#8c8a85", lw=1.0, ls="--")
    left.set_xlabel("share of the ray tracer's optimum (ray-traced objective)", fontsize=9)
    left.set_ylabel(f"cases at or below the share (of {len(scores.optimum)})", fontsize=9)
    left.set_title("per (terrain, azimuth): tilt and power", fontsize=10, loc="left")
    left.legend(fontsize=8, loc="upper left", frameon=False)
    terrains = list(scores.whole_optimum)
    x = np.arange(len(terrains))
    for k, name in enumerate(n for n in styles if n not in (RULE, FIXED)):
        color, marker = styles[name]
        right.plot(
            x + (k - 1) * 0.18,
            [scores.whole_shares[name][t] for t in terrains],
            ls="none",
            marker=marker,
            color=color,
            ms=6,
            label=name,
        )
    right.axhline(1.0, color="#8c8a85", lw=1.0)
    right.axhline(NEAR_OPTIMUM, color="#8c8a85", lw=1.0, ls="--")
    right.set_xticks(x, terrains, fontsize=8)
    right.set_xlabel("terrain", fontsize=9)
    right.set_ylabel("share of the ray tracer's optimum", fontsize=9)
    right.set_title("per terrain: azimuth, tilt and power", fontsize=10, loc="left")
    right.legend(fontsize=8, loc="lower left", frameon=False)
    for ax in (left, right):
        ax.tick_params(labelsize=8)
        ax.grid(color="#e4e2dc", lw=0.6)
        ax.set_axisbelow(True)
    fig.text(0.01, -0.02, caption, fontsize=7.5, color="#52514e", ha="left", va="top", wrap=True)
    fig.savefig(path, dpi=110, bbox_inches="tight", pil_kwargs={"quality": 88})
    plt.close(fig)
