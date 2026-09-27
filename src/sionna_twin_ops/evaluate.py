"""Evaluation (spec rule E): the surrogate beside B0, B1 and the ray tracer's own uncertainty.

Scored per map of one split, per training seed, on cells where the ray tracer has power (the
dB errors) and on all cells (the no-signal class). Strata: LOS direct-dominated, LOS
reflection-dominated (traced gain at least REFLECTION_EXCESS_DB above B0) and NLOS, overall
and per site class; off-grid tilts separately. Synthetic terrain.
"""

import json
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from sionna_twin_ops.baselines import bullington_loss_db
from sionna_twin_ops.dataset import read_manifest
from sionna_twin_ops.features import map_inputs, targets, terrain_features
from sionna_twin_ops.model import RESIDUAL_SCALE_DB, UNet
from sionna_twin_ops.site import SITE_CLASSES, Site
from sionna_twin_ops.terrain import GRID_SPACING_M, generate_terrain
from sionna_twin_ops.train import REFLECTION_EXCESS_DB

SECTOR_POWER_DBM = 43.0  # ASSUMPTION (spec E2)
RESOURCE_ELEMENTS = 1200  # ASSUMPTION (spec E2): 20 MHz carrier
RSRP_THRESHOLD_DBM = -110.0  # ASSUMPTION (spec E2)
POWER_PER_RE_DBM = SECTOR_POWER_DBM - 10.0 * np.log10(RESOURCE_ELEMENTS)
COVERED_GAIN_DB = RSRP_THRESHOLD_DBM - POWER_PER_RE_DBM  # path gain that clears the threshold
INFERENCE_WARMUP = 20
INFERENCE_TIMED = 200
STRATA = ("all", "LOS direct", "LOS reflection", "NLOS")
BASELINES = ("B0", "B1")


@dataclass
class Tally:
    """Everything accumulated for one method over a split.

    Attributes:
        errors: (group, stratum) to per-cell signed errors in dB, predicted minus traced.
        power: group to [true power, false power, true no-signal, false no-signal] counts.
        coverage: [predicted covered, truly covered, both, either] cell counts.
        iou: Per-map IoU of the covered masks.
    """

    errors: dict[tuple[str, str], list[NDArray[np.float32]]] = field(default_factory=dict)
    power: dict[str, list[int]] = field(default_factory=dict)
    coverage: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    iou: list[float] = field(default_factory=list)


def load_models(runs: list[Path], device: torch.device) -> list[tuple[int, UNet]]:
    """Best-epoch models of the training runs.

    Args:
        runs: Run directories (model.pt and meta.json).
        device: Where to put them.

    Returns:
        (seed, model) per run, in eval mode.
    """
    models = []
    for run in runs:
        meta = json.loads((run / "meta.json").read_text())
        model = UNet(meta["hyperparameters"]["width"])
        model.load_state_dict(torch.load(run / "model.pt", map_location=device))
        models.append((meta["hyperparameters"]["seed"], model.to(device).eval()))
    return models


def strata_masks(
    power: NDArray[np.bool_], los: NDArray[np.bool_], residual: NDArray[np.float32]
) -> dict[str, NDArray[np.bool_]]:
    """Cells of each stratum, among cells where the ray tracer has power.

    Args:
        power: Power mask.
        los: LOS mask.
        residual: Traced minus B0, in dB.

    Returns:
        Stratum name to mask.
    """
    reflected = residual >= REFLECTION_EXCESS_DB
    return {
        "all": power,
        "LOS direct": power & los & ~reflected,
        "LOS reflection": power & los & reflected,
        "NLOS": power & ~los,
    }


def tally_map(
    tally: Tally,
    groups: list[str],
    predicted_db: NDArray[np.float32],
    predicted_power: NDArray[np.bool_],
    traced_db: NDArray[np.float32],
    masks: dict[str, NDArray[np.bool_]],
) -> None:
    """Add one map's errors, power-class counts and coverage to a method's tally.

    Args:
        tally: The method's tally.
        groups: Groups the map belongs to ("all" and its site class, or "off-grid").
        predicted_db: The method's path gain in dB.
        predicted_power: Where the method says the ray tracer has power.
        traced_db: The ray tracer's path gain in dB (any value where it has none).
        masks: Output of `strata_masks`.
    """
    power = masks["all"]
    for group in groups:
        for stratum, mask in masks.items():
            tally.errors.setdefault((group, stratum), []).append(
                (predicted_db - traced_db)[mask].astype(np.float32)
            )
        counts = tally.power.setdefault(group, [0, 0, 0, 0])
        counts[0] += int((predicted_power & power).sum())
        counts[1] += int((predicted_power & ~power).sum())
        counts[2] += int((~predicted_power & ~power).sum())
        counts[3] += int((~predicted_power & power).sum())
    if "off-grid" in groups:
        return
    covered_pred = predicted_power & (predicted_db >= COVERED_GAIN_DB)
    covered_true = power & (traced_db >= COVERED_GAIN_DB)
    both = int((covered_pred & covered_true).sum())
    either = int((covered_pred | covered_true).sum())
    tally.coverage[0] += int(covered_pred.sum())
    tally.coverage[1] += int(covered_true.sum())
    tally.coverage[2] += both
    tally.coverage[3] += either
    if either:
        tally.iou.append(both / either)


def evaluate_split(
    dataset: Path, split: str, runs: list[Path], device: torch.device
) -> tuple[dict[str, Tally], dict[str, Any]]:
    """Score every map of a split for each seed and for B0 and B1.

    Args:
        dataset: Dataset directory.
        split: "validation" or "test".
        runs: Training run directories, one per seed.
        device: Where the models run.

    Returns:
        (method name to tally, facts about the split: maps, terrains, provenance).

    Raises:
        ValueError: If the split has no maps.
    """
    lines = sorted(
        (line for line in read_manifest(dataset) if line["split"] == split),
        key=lambda line: line["file"],
    )
    if not lines:
        raise ValueError(f"no {split} maps in {dataset}")
    models = load_models(runs, device)
    tallies = {f"seed {seed}": Tally() for seed, _ in models}
    tallies.update({name: Tally() for name in BASELINES})
    by_terrain: dict[int, list[dict[str, Any]]] = {}
    for line in lines:
        by_terrain.setdefault(line["terrain_id"], []).append(line)
    for terrain_id, maps in by_terrain.items():
        site = Site(**maps[0]["site"])
        features = terrain_features(generate_terrain(terrain_id, GRID_SPACING_M), site)
        loss = bullington_loss_db(features.geometry).astype(np.float32)
        los = features.los > 0.5
        stack = [map_inputs(features, m["azimuth_deg"], m["tilt_deg"]) for m in maps]
        x = torch.from_numpy(np.stack([s[0] for s in stack])).to(device)
        with torch.no_grad():
            outputs = {seed: model(x).cpu().numpy() for seed, model in models}
        for index, line in enumerate(maps):
            b0 = stack[index][1]
            residual, power_f = targets(np.load(dataset / "maps" / line["file"]), b0)
            power = power_f > 0.5
            traced = (residual + b0).astype(np.float32)
            masks = strata_masks(power, los, residual)
            groups = ["off-grid"] if line["kind"] == "off-grid" else ["all", site.site_class]
            everywhere = np.ones_like(power)
            tally_map(tallies["B0"], groups, b0, everywhere, traced, masks)
            tally_map(tallies["B1"], groups, b0 - loss, everywhere, traced, masks)
            for seed, out in outputs.items():
                predicted = (b0 + out[index, 0] * RESIDUAL_SCALE_DB).astype(np.float32)
                says_power = out[index, 1] > 0
                tally_map(tallies[f"seed {seed}"], groups, predicted, says_power, traced, masks)
    facts = {
        "split": split,
        "maps": len(lines),
        "grid maps": sum(line["kind"] == "grid" for line in lines),
        "off-grid maps": sum(line["kind"] == "off-grid" for line in lines),
        "terrains": sorted(by_terrain),
        "ray tracing": sorted(
            {
                f"{line['provenance']['variant']}, {line['settings']['samples_per_tx']:.0e} rays, "
                f"Sionna RT {line['provenance']['sionna-rt']}"
                for line in lines
            }
        ),
        "commits": sorted({line["provenance"]["commit"][:7] for line in lines}),
        "runs": [str(run) for run in runs],
    }
    return tallies, facts


def error_stats(errors: list[NDArray[np.float32]]) -> tuple[float, float, float, float, int]:
    """Mean, median and p95 of |error|, the bias, and the cell count.

    Args:
        errors: Per-map signed errors.

    Returns:
        (mean, median, p95, bias, cells); NaNs when there are no cells.
    """
    e = np.concatenate(errors) if errors else np.zeros(0, dtype=np.float32)
    if e.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0
    a = np.abs(e)
    p95 = float(np.percentile(a, 95))
    return float(a.mean()), float(np.median(a)), p95, float(e.mean()), int(e.size)


GROUPS = ("all", *SITE_CLASSES)
_TRUTH_ROW = re.compile(
    r"^\| (all|hilltop|slope|valley) \| (fold|sampling) \| ([\d.]+) \| ([\d.]+) \| [\d.]+ \| "
    r"\d+ \| ([\d.]+) \| ([\d.]+) \|"
)
_PATH_ROW = re.compile(
    r"^\| (all|hilltop|slope|valley) \| ([\d.]+) / ([\d.]+) \(\d+\) \| ([\d.]+) / ([\d.]+) "
    r"\(\d+\) \|"
)


@dataclass(frozen=True)
class Uncertainty:
    """The ray tracer's own uncertainty per stratum, quoted from the fold check.

    Attributes:
        commit: Commit that wrote the fold check.
        terms: (group, stratum) to {"fold": (median, p95), "sampling": (median, p95)}.
    """

    commit: str
    terms: dict[tuple[str, str], dict[str, tuple[float, ...]]]


def quoted_uncertainty(fold_report: Path) -> Uncertainty:
    """The fold term and sampling floor per stratum, quoted from the committed fold check.

    Args:
        fold_report: results/fold_check.md.

    Returns:
        The fold term and sampling floor per (group, stratum).

    Raises:
        ValueError: If the report does not have the expected rows.
    """
    text = fold_report.read_text().splitlines()
    commits = [m.group(1) for line in text if (m := re.match(r"^\| commit \| (\w+) \|", line))]
    commit = commits[0] if commits else None
    if commit is None:
        raise ValueError(f"no commit in {fold_report}")
    truth: dict[tuple[str, str], tuple[float, ...]] = {}
    paths: dict[str, tuple[float, ...]] = {}
    for line in text:
        if m := _TRUTH_ROW.match(line):
            truth[(m.group(1), m.group(2))] = tuple(float(v) for v in m.groups()[2:])
        elif m := _PATH_ROW.match(line):
            paths[m.group(1)] = tuple(float(v) for v in m.groups()[1:])
    terms: dict[tuple[str, str], dict[str, tuple[float, ...]]] = {}
    for group in GROUPS:
        if (group, "fold") not in truth or (group, "sampling") not in truth or group not in paths:
            raise ValueError(f"{fold_report} lacks the rows for {group}")
        fold, sampling, path = truth[(group, "fold")], truth[(group, "sampling")], paths[group]
        terms[(group, "LOS direct")] = {"fold": path[0:2], "sampling": sampling[0:2]}
        terms[(group, "LOS reflection")] = {"fold": path[2:4], "sampling": sampling[0:2]}
        terms[(group, "NLOS")] = {"fold": fold[2:4], "sampling": sampling[2:4]}
    return Uncertainty(commit=commit[:7], terms=terms)


@dataclass(frozen=True)
class Timing:
    """Seconds per map (spec E5), medians.

    Attributes:
        rt_gpu_new: Ray tracer on the GPU, new terrain, first map, end to end.
        rt_gpu_further: Ray tracer on the GPU, each further map of that terrain.
        rt_cpu_new: Ray tracer on the CPU (llvm), new terrain, first map, end to end.
        rt_cpu_further: Ray tracer on the CPU, each further map.
        surrogate_gpu: Batch-1 inference on the GPU.
        surrogate_cpu: Batch-1 inference on the CPU.
        terrain_features: Geometry, LOS mask and height channels for a new terrain.
        map_inputs: B0 and the per-map channels.
        gpu_record: The GPU ray-tracer timing record (terrains, samples, provenance).
        cpu_record: The CPU ray-tracer timing record.
    """

    rt_gpu_new: float
    rt_gpu_further: float
    rt_cpu_new: float
    rt_cpu_further: float
    surrogate_gpu: float
    surrogate_cpu: float
    terrain_features: float
    map_inputs: float
    gpu_record: dict[str, Any]
    cpu_record: dict[str, Any]


def measure_timing(
    dataset: Path, split: str, run: Path, gpu_timing: Path, cpu_terrains: int, cpu_further: int
) -> Timing:
    """Time the ray tracer and the surrogate on the same machine (spec E5).

    The GPU ray-tracer record comes from `twin raytrace-timing` run through the Windows
    runner; the CPU one is measured here with the same function on llvm, which must be the
    selected variant.

    Args:
        dataset: Dataset directory.
        split: Split whose maps are timed.
        run: One training run (its model is timed).
        gpu_timing: JSON written by `twin raytrace-timing` on the GPU.
        cpu_terrains: New terrains for the CPU ray-tracer timing.
        cpu_further: Further maps per terrain for the CPU ray-tracer timing.

    Returns:
        The medians.

    Raises:
        RuntimeError: If the selected variant is not llvm or the GPU record is not cuda.
    """
    from sionna_twin_ops.backend import active_variant
    from sionna_twin_ops.crosscheck import raytracer_timing

    if active_variant() != "llvm_ad_mono_polarized":
        raise RuntimeError(f"the CPU timing needs llvm, not {active_variant()}")
    gpu_record = json.loads(gpu_timing.read_text())
    if not gpu_record["provenance"]["variant"].startswith("cuda"):
        raise RuntimeError(f"{gpu_timing} is not a GPU timing")
    cpu_record = raytracer_timing(dataset, split, cpu_terrains, cpu_further)

    lines = sorted(
        (line for line in read_manifest(dataset) if line["split"] == split),
        key=lambda line: line["file"],
    )
    terrain_times, input_times = [], []
    for line in lines[:: max(1, len(lines) // 5)][:5]:
        start = time.perf_counter()
        terrain = generate_terrain(line["terrain_id"], GRID_SPACING_M)
        features = terrain_features(terrain, Site(**line["site"]))
        terrain_times.append(time.perf_counter() - start)
        timed_terrain = line["terrain_id"]
    # Map inputs are timed on the terrain whose features were built last.
    for line in [ln for ln in lines if ln["terrain_id"] == timed_terrain][:20]:
        start = time.perf_counter()
        x, _ = map_inputs(features, line["azimuth_deg"], line["tilt_deg"])
        input_times.append(time.perf_counter() - start)

    def infer(device: torch.device) -> float:
        ((_, model),) = load_models([run], device)
        batch = torch.from_numpy(x[None]).to(device)
        times = []
        with torch.no_grad():
            for i in range(INFERENCE_WARMUP + INFERENCE_TIMED):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                model(batch)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                if i >= INFERENCE_WARMUP:  # warm-up calls settle kernels, clocks, allocators
                    times.append(time.perf_counter() - start)
        return statistics.median(times)

    return Timing(
        rt_gpu_new=gpu_record["new_terrain_first_map_median_s"],
        rt_gpu_further=gpu_record["further_map_median_s"],
        rt_cpu_new=cpu_record["new_terrain_first_map_median_s"],
        rt_cpu_further=cpu_record["further_map_median_s"],
        surrogate_gpu=infer(torch.device("cuda")),
        surrogate_cpu=infer(torch.device("cpu")),
        terrain_features=statistics.median(terrain_times),
        map_inputs=statistics.median(input_times),
        gpu_record=gpu_record,
        cpu_record=cpu_record,
    )


def reflection_note(tallies: dict[str, Tally], seeds: list[str], uncertainty: Uncertainty) -> str:
    """One sentence on the LOS reflection-dominated stratum, written from the numbers.

    Args:
        tallies: Output of `evaluate_split`.
        seeds: Surrogate method names.
        uncertainty: Output of `quoted_uncertainty`.

    Returns:
        The sentence.
    """
    bias = np.mean([error_stats(tallies[s].errors[("all", "LOS reflection")])[3] for s in seeds])
    median = np.mean(
        [error_stats(tallies[s].errors[("hilltop", "LOS reflection")])[1] for s in seeds]
    )
    fold = uncertainty.terms[("hilltop", "LOS reflection")]["fold"][0]
    direction = "underpredicts" if bias < 0 else "overpredicts"
    relation = "above" if median > fold else "not above"
    return (
        f"In LOS reflection-dominated cells the surrogate {direction} (bias {bias:+.1f} dB), "
        f"and on hilltops its median error there ({median:.2f} dB) is {relation} the fold "
        f"term's median ({fold:.2f} dB)."
    )


def _spread(values: list[float], digits: int) -> str:
    """Mean over seeds with the min-max range."""
    return f"{np.mean(values):.{digits}f} ({min(values):.{digits}f} to {max(values):.{digits}f})"


def report_markdown(
    tallies: dict[str, Tally], facts: dict[str, Any], uncertainty: Uncertainty, timing: Timing
) -> str:
    """The evaluation report (spec E1 to E6).

    Args:
        tallies: Output of `evaluate_split`.
        facts: Output of `evaluate_split`.
        uncertainty: Output of `quoted_uncertainty`.
        timing: Output of `measure_timing`.

    Returns:
        Markdown text.
    """
    seeds = [name for name in tallies if name.startswith("seed")]
    gpu_new = timing.terrain_features + timing.map_inputs + timing.surrogate_gpu
    gpu_further = timing.map_inputs + timing.surrogate_gpu
    cpu_new = timing.terrain_features + timing.map_inputs + timing.surrogate_cpu
    cpu_further = timing.map_inputs + timing.surrogate_cpu

    def e1_block(group: str) -> list[str]:
        rows = [
            "| stratum | method | mean abs | median abs | p95 abs | bias | cells |",
            "|---|---|---|---|---|---|---|",
        ]
        for stratum in STRATA:
            per_seed = [error_stats(tallies[s].errors.get((group, stratum), [])) for s in seeds]
            if per_seed[0][4] == 0:
                rows.append(f"| {stratum} | all methods | no cells | | | | 0 |")
                continue
            rows.append(
                f"| {stratum} | surrogate, {len(seeds)} seeds | "
                + " | ".join(_spread([p[i] for p in per_seed], 3) for i in range(4))
                + f" | {per_seed[0][4]} |"
            )
            for name in BASELINES:
                st = error_stats(tallies[name].errors.get((group, stratum), []))
                rows.append(
                    f"| {stratum} | {name} | {st[0]:.3f} | {st[1]:.3f} | {st[2]:.3f} | "
                    f"{st[3]:+.3f} | {st[4]} |"
                )
            if (group, stratum) in uncertainty.terms:
                t = uncertainty.terms[(group, stratum)]
                rows.append(
                    f"| {stratum} | ray tracer, fold term (quoted) | | {t['fold'][0]:.3f} | "
                    f"{t['fold'][1]:.3f} | | |"
                )
                rows.append(
                    f"| {stratum} | ray tracer, sampling floor (quoted) | | "
                    f"{t['sampling'][0]:.3f} | {t['sampling'][1]:.3f} | | |"
                )
        return rows

    lines = [
        f"# Evaluation on the {facts['split']} split",
        "",
        "Synthetic terrain, not a real place; one sector, no vegetation, buildings or "
        "interference, flat Earth. Ground truth: Sionna RT, line of sight and specular "
        f"reflection only ({'; '.join(facts['ray tracing'])}; map commits "
        f"{', '.join(facts['commits'])}); pattern: 3GPP TR 38.901. Written by `twin evaluate`.",
        "",
        f"Maps: {facts['maps']} ({facts['grid maps']} grid, {facts['off-grid maps']} off-grid) "
        f"on terrains {', '.join(str(t) for t in facts['terrains'])}. Surrogate: the "
        f"best-epoch checkpoints of {', '.join(facts['runs'])}; each surrogate figure is the "
        "mean over the seeds, with the lowest and highest seed in brackets.",
        "",
        "Errors are predicted minus traced path gain in dB, over cells where the ray tracer "
        "has power. Strata: LOS direct-dominated and LOS reflection-dominated (traced gain "
        f"at least {REFLECTION_EXCESS_DB:.0f} dB above B0, ASSUMPTION) and NLOS, by the "
        "heightmap LOS mask. B0: free space plus the antenna pattern; B1: B0 minus the "
        "Bullington diffraction loss (ITU-R P.526-16 section 4.5.1).",
        "",
        "The ray tracer's own uncertainty is quoted from `results/fold_check.md` (commit "
        f"{uncertainty.commit}; training terrains, one azimuth and tilt): the fold term (the "
        "map with the other cell diagonal) and the sampling floor (1e9 against 4e9 rays), "
        "as median and p95 of the absolute difference. The sampling floor is split only LOS "
        "and NLOS there, so both LOS strata quote the LOS value. Read the surrogate's errors "
        "against these; they are not a claim that the surrogate beats the ray tracer.",
        "",
        "## E1 and E3: path-gain error by stratum, all grid maps",
        "",
        *e1_block("all"),
        "",
        reflection_note(tallies, seeds, uncertainty),
        "",
        "## E3: path-gain error by stratum, per site class",
    ]
    for site_class in SITE_CLASSES:
        lines += ["", f"### {site_class}", "", *e1_block(site_class)]

    lines += [
        "",
        "## E3: the no-signal class",
        "",
        "The surrogate's power head (logit above 0 means the ray tracer has power) against "
        "the traced power mask, over all cells of the grid maps. B0 and B1 always predict "
        "power, so they have no no-signal class.",
        "",
        "| group | power precision | power recall | no-signal precision | no-signal recall | "
        "accuracy | cells with no signal |",
        "|---|---|---|---|---|---|---|",
    ]
    for group in GROUPS:
        stats = []
        for s in seeds:
            tp, fp, tn, fn = tallies[s].power[group]
            stats.append(
                (
                    tp / (tp + fp),
                    tp / (tp + fn),
                    tn / (tn + fn),
                    tn / (tn + fp),
                    (tp + tn) / (tp + fp + tn + fn),
                )
            )
        tp, fp, tn, fn = tallies[seeds[0]].power[group]
        lines.append(
            f"| {group} | "
            + " | ".join(_spread([st[i] * 100 for st in stats], 2) + "%" for i in range(5))
            + f" | {(tn + fp) / (tp + fp + tn + fn) * 100:.1f}% |"
        )

    lines += [
        "",
        "## E2: coverage",
        "",
        f"RSRP = {POWER_PER_RE_DBM:.2f} dBm per resource element ({SECTOR_POWER_DBM:.0f} dBm "
        f"over {RESOURCE_ELEMENTS} resource elements of a 20 MHz carrier, ASSUMPTION) plus path "
        f"gain; a cell is covered at {RSRP_THRESHOLD_DBM:.0f} dBm (ASSUMPTION), i.e. path gain "
        f"of at least {COVERED_GAIN_DB:.2f} dB. The ray tracer's covered cells need power; the "
        "surrogate's need its power head to say power; B0 and B1 always predict power. "
        "Covered-area error: predicted minus traced covered cells over traced covered cells, "
        "pooled over the grid maps. IoU: pooled, and the mean of per-map IoU.",
        "",
        "| method | covered-area error | IoU (pooled) | IoU (mean per map) |",
        "|---|---|---|---|",
    ]

    def coverage(name: str) -> tuple[float, float, float]:
        pred, true, both, either = tallies[name].coverage
        return (pred - true) / true * 100, both / either, float(np.mean(tallies[name].iou))

    cov = [coverage(s) for s in seeds]
    lines.append(
        f"| surrogate, {len(seeds)} seeds | {_spread([c[0] for c in cov], 2)}% | "
        f"{_spread([c[1] for c in cov], 4)} | {_spread([c[2] for c in cov], 4)} |"
    )
    for name in BASELINES:
        c = coverage(name)
        lines.append(f"| {name} | {c[0]:+.2f}% | {c[1]:.4f} | {c[2]:.4f} |")

    lines += ["", "## E4: off-grid tilts", ""]
    if facts["off-grid maps"] == 0:
        lines.append("Not in this split: off-grid tilts (1.5, 4.5, 7.5 deg) exist on test only.")
    else:
        lines += [
            f"The {facts['off-grid maps']} off-grid maps (tilts 1.5, 4.5 and 7.5 deg), which "
            "no model saw during training.",
            "",
            *e1_block("off-grid"),
        ]

    lines += [
        "",
        "## E5: time per map, same machine",
        "",
        f"GPU: {timing.gpu_record['provenance']['gpu']}. Medians, in seconds. The ray tracer "
        f"runs the dataset's settings ({timing.gpu_record['samples_per_tx']:.0e} rays): its "
        "new-terrain figure covers terrain generation, scene and mesh build, measurement "
        "surface and the first solve; further maps reuse the scene and time the solve only. "
        "The surrogate's new-terrain figure covers geometry, LOS mask, B0, the per-map channels "
        "and inference; further maps cover B0, the channels and inference. Both sides skip "
        f"their warm-up (kernel compilation). GPU ray tracer: terrains "
        f"{', '.join(str(t) for t in timing.gpu_record['terrains'])} through the Windows runner "
        f"(commit {timing.gpu_record['provenance']['commit'][:7]}); CPU ray tracer: terrains "
        f"{', '.join(str(t) for t in timing.cpu_record['terrains'])}, llvm, measured here.",
        "",
        "| hardware | case | ray tracer | surrogate | ray tracer / surrogate |",
        "|---|---|---|---|---|",
        *(
            f"| {hw} | {case} | {rt:.3f} | {sg:.4f} | {rt / sg:.1f} |"
            for hw, case, rt, sg in (
                ("GPU", "new terrain, first map", timing.rt_gpu_new, gpu_new),
                ("GPU", "each further map", timing.rt_gpu_further, gpu_further),
                ("CPU", "new terrain, first map", timing.rt_cpu_new, cpu_new),
                ("CPU", "each further map", timing.rt_cpu_further, cpu_further),
            )
        ),
        "",
        "Surrogate parts: geometry and LOS mask for a new terrain "
        f"{timing.terrain_features:.3f}; B0 and channels per map {timing.map_inputs:.4f}; "
        f"inference, batch 1, GPU {timing.surrogate_gpu:.4f} and CPU {timing.surrogate_cpu:.4f}.",
        "",
        "On this machine, per map, the ray tracer takes "
        f"{timing.rt_gpu_new / gpu_new:.1f} times as long as the surrogate for a new terrain's "
        f"first map and {timing.rt_gpu_further / gpu_further:.1f} times as long for each "
        "further map on the GPU; without a GPU the ratios are "
        f"{timing.rt_cpu_new / cpu_new:.1f} and {timing.rt_cpu_further / cpu_further:.1f}.",
        "",
        "## Per seed, all grid maps",
        "",
        "| seed | mean abs, all | mean abs, LOS direct | mean abs, LOS reflection | "
        "mean abs, NLOS | power accuracy |",
        "|---|---|---|---|---|---|",
    ]
    for s in seeds:
        tp, fp, tn, fn = tallies[s].power["all"]
        means = [error_stats(tallies[s].errors[("all", st)])[0] for st in STRATA]
        lines.append(
            f"| {s} | "
            + " | ".join(f"{m:.3f}" for m in means)
            + f" | {(tp + tn) / (tp + fp + tn + fn) * 100:.2f}% |"
        )
    return "\n".join(lines) + "\n"


FIGURE_AZIMUTH_DEG = 90.0
FIGURE_TILT_DEG = 6.0
ERROR_LIMIT_DB = 10.0


def evaluation_figure(dataset: Path, split: str, run: Path, caption: str, path: Path) -> None:
    """Three maps, one per site class: ray-traced, predicted and error over hillshade.

    Uses the first terrain of each class in the split at azimuth FIGURE_AZIMUTH_DEG and tilt
    FIGURE_TILT_DEG, and one training run's best checkpoint. Cells where the ray tracer (or,
    for the prediction, the power head) has no power are left clear.

    Args:
        dataset: Dataset directory.
        split: Split the terrains come from.
        run: Training run whose model is shown.
        caption: Provenance caption printed under the panels.
        path: Output image.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource

    matplotlib.rcParams["axes.unicode_minus"] = False
    lines = [line for line in read_manifest(dataset) if line["split"] == split]
    ((_, model),) = load_models([run], torch.device("cpu"))
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 11.0), layout="constrained")
    half_km = 2.56
    extent = (-half_km, half_km, -half_km, half_km)
    for row, site_class in enumerate(SITE_CLASSES):
        line = min(
            (
                ln
                for ln in lines
                if ln["site"]["site_class"] == site_class
                and ln["azimuth_deg"] == FIGURE_AZIMUTH_DEG
                and ln["tilt_deg"] == FIGURE_TILT_DEG
            ),
            key=lambda ln: ln["terrain_id"],
        )
        site = Site(**line["site"])
        features = terrain_features(generate_terrain(line["terrain_id"], GRID_SPACING_M), site)
        x, b0 = map_inputs(features, FIGURE_AZIMUTH_DEG, FIGURE_TILT_DEG)
        residual, power_f = targets(np.load(dataset / "maps" / line["file"]), b0)
        power = power_f > 0.5
        with torch.no_grad():
            out = model(torch.from_numpy(x[None])).numpy()[0]
        traced = np.where(power, residual + b0, np.nan)
        predicted_all = b0 + out[0] * RESIDUAL_SCALE_DB
        predicted = np.where(out[1] > 0, predicted_all, np.nan)
        error = np.where(power, predicted_all - (residual + b0), np.nan)
        ground = features.geometry.rx_m - 1.5
        shade = LightSource(azdeg=315, altdeg=45).hillshade(ground, vert_exag=2.0, dx=40, dy=40)
        panels = (
            (traced, "viridis", (-150.0, -60.0), "ray-traced path gain (dB)"),
            (predicted, "viridis", (-150.0, -60.0), "predicted path gain (dB)"),
            (error, "RdBu_r", (-ERROR_LIMIT_DB, ERROR_LIMIT_DB), "predicted minus traced (dB)"),
        )
        for col, (values, cmap, (low, high), label) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(shade, origin="lower", extent=extent, cmap="gray", vmin=0.0, vmax=1.0)
            image = ax.imshow(
                values, origin="lower", extent=extent, cmap=cmap, vmin=low, vmax=high, alpha=0.85
            )
            ax.plot(0, 0, "^", color="white", mec="black", ms=8)
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title(label, fontsize=9, loc="left")
            if col == 0:
                ax.set_ylabel(
                    f"{site_class}, terrain {line['terrain_id']}\ny north (km)", fontsize=8
                )
            if row == 2:
                ax.set_xlabel("x east (km)", fontsize=8)
            if row == 2 or col == 2:
                fig.colorbar(image, ax=ax, shrink=0.8).ax.tick_params(labelsize=7)
    fig.text(0.01, -0.01, caption, fontsize=7.5, color="#52514e", ha="left", va="top", wrap=True)
    fig.savefig(path, dpi=100, bbox_inches="tight", pil_kwargs={"quality": 88})
    plt.close(fig)


_BACKEND_ROW = re.compile(
    r"^\| \d+ \| [\d.e+]+ \| ([\d.]+) \| ([\d.]+) \| [\d.]+ \| \d+ \| ([\d.]+) \| ([\d.]+) \|"
)


def backend_agreement(backend_report: Path) -> tuple[float, float]:
    """Largest median and p95 of |llvm - cuda| over all rows of the committed backend check.

    Args:
        backend_report: results/backend_check.md.

    Returns:
        (largest median, largest p95), in dB, over LOS and NLOS cells of every row.

    Raises:
        ValueError: If the report has no agreement rows.
    """
    medians, p95s = [], []
    for line in backend_report.read_text().splitlines():
        if m := _BACKEND_ROW.match(line):
            medians += [float(m.group(1)), float(m.group(3))]
            p95s += [float(m.group(2)), float(m.group(4))]
    if not medians:
        raise ValueError(f"no agreement rows in {backend_report}")
    return max(medians), max(p95s)
