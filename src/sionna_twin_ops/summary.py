"""results/SUMMARY.md: the headline numbers, each read from a committed report's data and
linked to the report, with the ray tracer's own uncertainty beside the surrogate's errors and
the limitations. No number is typed here; `test_summary` checks the committed page is what
this module writes from the committed data.
"""

from pathlib import Path
from typing import Any

import numpy as np

from sionna_twin_ops.reports import read_report


def _spread(values: list[float], digits: int) -> str:
    """Mean over seeds with the min-max range."""
    return f"{np.mean(values):.{digits}f} ({min(values):.{digits}f} to {max(values):.{digits}f})"


def _link(name: str) -> str:
    return f"[{name}]({name}.md)"


def _stratum(block: list[dict[str, Any]], stratum: str) -> dict[str, Any]:
    (row,) = [r for r in block if r["stratum"] == stratum]
    return row


def summary_markdown(results: Path) -> str:
    """The one-page summary of the committed results.

    Args:
        results: The results directory (its .json reports are read).

    Returns:
        Markdown.
    """
    ev = read_report(results / "evaluation_test.md")
    solver = read_report(results / "solver_check.md")
    backend = read_report(results / "backend_check.md")
    dataset = read_report(results / "dataset_summary.md")
    training = read_report(results / "training_summary.md")
    search = read_report(results / "search_test.md")

    seeds = ev["seeds"]
    block = ev["blocks"]["all"]
    lines = [
        "# Summary",
        "",
        "Synthetic terrain generated from seeds, not a real place; one sector on a 30 m mast "
        "at 1.8 GHz over a 5.12 km map; flat Earth, no vegetation, buildings or interference. "
        "Ground truth is NVIDIA Sionna RT (line of sight and specular reflection only). Every "
        "number below is read from the linked report's data by `twin summary`; the reports "
        "give the settings, seeds and package versions.",
        "",
        "## The surrogate on held-out test terrain",
        "",
        f"Test split: {ev['facts']['maps']} maps on {len(ev['facts']['terrains'])} terrains no "
        f"model saw in training. Mean absolute path-gain error in dB over cells where the ray "
        f"tracer has power; surrogate: mean over {len(seeds)} seeds (lowest to highest). "
        "Beside it, the ray tracer's own uncertainty: the fold term (the map with the other "
        "cell diagonal) and the sampling floor (1e9 against 4e9 rays), median / p95 of the "
        f"absolute difference ({_link('evaluation_test')}, {_link('fold_check')}).",
        "",
        "| cells | surrogate | B0 (free space and pattern) | B1 (B0 with diffraction loss) | "
        "fold term, median / p95 | sampling floor, median / p95 |",
        "|---|---|---|---|---|---|",
    ]
    for stratum in ("all", "LOS direct", "LOS reflection", "NLOS"):
        row = _stratum(block, stratum)
        u = row["uncertainty"]
        truth = (
            f"{u['fold'][0]:.3f} / {u['fold'][1]:.3f} | "
            f"{u['sampling'][0]:.3f} / {u['sampling'][1]:.3f}"
            if u is not None
            else "- | -"
        )
        lines.append(
            f"| {stratum} | {_spread([s[0] for s in row['surrogate']], 3)} | "
            f"{row['baselines']['B0'][0]:.3f} | {row['baselines']['B1'][0]:.3f} | {truth} |"
        )
    reflection = _stratum(block, "LOS reflection")
    off_grid = _stratum(ev["blocks"]["off-grid"], "all")
    accuracy = [
        (tp + tn) / (tp + fp + tn + fn) * 100
        for tp, fp, tn, fn in (ev["power"]["all"][s] for s in seeds)
    ]
    cov = ev["coverage"]
    bias = [s[3] for s in reflection["surrogate"]]
    direction = "underpredicts" if np.mean(bias) < 0 else "overpredicts"
    ranked = sorted(
        ("LOS direct", "LOS reflection", "NLOS"),
        key=lambda st: -np.mean([s[0] for s in _stratum(block, st)["surrogate"]]),
    )
    lines += [
        "",
        f"- LOS reflection-dominated cells: bias {_spread(bias, 3)} dB, the surrogate "
        f"{direction} them.",
        f"- Off-grid tilts (1.5, 4.5, 7.5 deg, never trained on): mean abs "
        f"{_spread([s[0] for s in off_grid['surrogate']], 3)} dB.",
        f"- No-signal cells: the power head's accuracy over all cells is {_spread(accuracy, 2)}%.",
        f"- Coverage at {ev['rsrp_threshold_dbm']:.0f} dBm (ASSUMPTION): IoU "
        f"{_spread([cov[s][1] for s in seeds], 4)} against B0 {cov['B0'][1]:.4f} and B1 "
        f"{cov['B1'][1]:.4f} (pooled).",
        "",
        "## Tilt and power search on test terrain",
        "",
    ]
    scores = search["scores"]
    within = 0.01
    rows = [
        "| chooser | median shortfall | max shortfall | within 1 point of the optimum |",
        "|---|---|---|---|",
    ]
    for name, values in scores["shortfalls"].items():
        if name.endswith("(GPU)"):
            continue
        v = np.asarray(list(values.values()))
        rows.append(
            f"| {name} | {np.median(v):.4f} | {v.max():.4f} | "
            f"{int((v <= within).sum())} of {len(v)} |"
        )
    lines += [
        f"{len(scores['optimum'])} (terrain, azimuth) cases; the objective is the covered "
        f"fraction within {search['plan']['radius_m'] / 1000:g} km minus the covered fraction "
        "beyond it (ASSUMPTION), on a -1 to 1 scale; every chosen setting is scored with the "
        "ray tracer, as its shortfall from the ray tracer's optimum on a 1 deg tilt grid "
        f"({_link('search_test')}).",
        "",
        *rows,
        "",
        "## Speed, same machine",
        "",
    ]
    t = ev["timing"]
    gpu_new = t["terrain_features"] + t["map_inputs"] + t["surrogate_gpu"]
    gpu_further = t["map_inputs"] + t["surrogate_gpu"]
    cpu_new = t["terrain_features"] + t["map_inputs"] + t["surrogate_cpu"]
    cpu_further = t["map_inputs"] + t["surrogate_cpu"]
    rt_s = list(search["trace_meta"]["terrain_seconds"].values())
    gpu_s = [s for per in search["surrogate"]["cuda"]["seconds"].values() for s in per.values()]
    lines += [
        "Ray tracer time over surrogate time, medians "
        f"({_link('evaluation_test')}, {_link('search_test')}):",
        "",
        "| case | GPU | CPU |",
        "|---|---|---|",
        f"| one map, new terrain | {t['rt_gpu_new'] / gpu_new:.1f} | "
        f"{t['rt_cpu_new'] / cpu_new:.1f} |",
        f"| each further map | {t['rt_gpu_further'] / gpu_further:.1f} | "
        f"{t['rt_cpu_further'] / cpu_further:.1f} |",
        f"| whole search, one terrain | {np.median(rt_s) / np.median(gpu_s):.0f} | "
        "estimated only, see the report |",
        "",
        "## Behind the numbers",
        "",
    ]
    best = [m["best"] for m in training["metas"]]
    agreement = [
        row[region]
        for row in backend["agreement"]
        for region in ("los", "nlos")
        if row[region]["cells"]
    ]
    lines += [
        f"- Dataset: {dataset['maps_present']} ray-traced maps on "
        + ", ".join(f"{r['terrains']} {r['split']}" for r in dataset["splits"])
        + f" terrains ({_link('dataset_summary')}).",
        f"- Training: a U-Net of residual over B0 plus a power head; validation L1 "
        f"{', '.join(f'{b["l1_db"]:.3f}' for b in best)} dB for seeds "
        f"{', '.join(str(m['hyperparameters']['seed']) for m in training['metas'])}, best "
        f"epochs {', '.join(str(b['epoch']) for b in best)} ({_link('training_summary')}).",
        f"- Antenna tilt check (3GPP TR 38.901 element, 8 x 1 column): "
        f"{'PASS' if solver['a4_pass'] else 'FAIL'}, largest main-lobe error "
        f"{solver['lobe_error_deg']:.2f} deg ({_link('solver_check')}).",
        f"- GPU against CPU ray tracing: the largest median and p95 |difference| over every "
        f"map are {max(a['median'] for a in agreement):.3f} and "
        f"{max(a['p95'] for a in agreement):.3f} dB at the report's three decimals "
        f"({_link('backend_check')}).",
        "",
        "## Limitations",
        "",
        "- Synthetic terrain only: procedural heightmaps from seeds, one ground material, no "
        "vegetation, buildings or clutter, flat Earth.",
        "- One sector, one antenna configuration, one carrier.",
        "- The ray tracer runs line of sight and specular reflection only: no diffraction "
        "(it does not reach a mesh measurement surface in Sionna RT) and no diffuse "
        "scattering. Shadowed cells are lit by reflections alone.",
        "- The ground truth depends on how each map cell is folded into triangles: the fold "
        "term is far above the sampling floor, largest in LOS reflection-dominated and NLOS "
        f"cells ({_link('fold_check')}). Surrogate errors below it are not a claim of "
        "accuracy beyond the ray tracer's own.",
        f"- The surrogate {direction} LOS reflection-dominated cells (bias above); its "
        f"largest mean errors are in {ranked[0]} and {ranked[1]} cells, its smallest in "
        f"{ranked[2]} cells (table above).",
        "- Training overfits after the best epoch; the saved weights are the best epoch's, "
        f"chosen on validation ({_link('training_summary')}).",
        "- Search optima often sit on the edge of the feasible box (tilt 0 to 12 deg, 28 to "
        "46 dBm, ASSUMPTION): they are constrained optima, not the unconstrained ones "
        f"({_link('search_test')}).",
        "- The search objective is not smooth in tilt, so the 1 deg ray-tracer grid is a "
        "reference at that resolution; a half-degree setting can beat it "
        f"({_link('search_test')}).",
        "- The first search objective was a flawed definition: its 3 km radius reached "
        "beyond the map's half-width, so its optimum collapsed to 46 dBm and 0 deg; it is "
        "kept as superseded ([search_test_first_objective](search_test_first_objective.md)).",
        "",
    ]
    return "\n".join(lines)
