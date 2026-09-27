"""Committed reports: each command builds one JSON-safe data dict and the markdown is rendered
from it, so a report's .md and .json cannot drift (`test_reports` re-renders every committed
.json and compares it with its .md byte for byte).

Only numpy is imported here, so the report notebook and the tests read results without
torch, Mitsuba or the dataset.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

Data = dict[str, Any]


def write_report(md_path: Path, data: Data) -> str:
    """Write a report's .json and its .md rendered from it.

    Args:
        md_path: The .md path; the .json goes beside it.
        data: The report data, with its "kind".

    Returns:
        The markdown.
    """
    text = render(data)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.with_suffix(".json").write_text(json.dumps(data, indent=1) + "\n", newline="\n")
    md_path.write_text(text, newline="\n")
    return text


def render(data: Data) -> str:
    """Markdown of a report's data.

    Args:
        data: The report data, with its "kind".

    Returns:
        The markdown.

    Raises:
        ValueError: If the kind has no renderer.
    """
    kind = data["kind"]
    if kind not in RENDERERS:
        raise ValueError(f"no renderer for report kind {kind!r}")
    return RENDERERS[kind](data)


def read_report(md_path: Path) -> Data:
    """The data of a committed report.

    Args:
        md_path: The .md path (or the .json path).

    Returns:
        The data.
    """
    data: Data = json.loads(md_path.with_suffix(".json").read_text())
    return data


def site_check_markdown(data: Data) -> str:
    """The site placement check (spec S2)."""
    lines = [
        "# Site placement check (spec S2)",
        "",
        f"Synthetic terrain ids {data['first_id']} to {data['last_id']}, "
        f"{data['spacing_m']:.0f} m grid, each id placed with its own class. Percentile: "
        "midrank percentile of the site height among the vertices of its 5.12 km map. Written "
        "by `twin site-check`.",
        "",
        "| class | ids | placed | skipped | p10 | median | p90 | skipped ids |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in data["classes"]:
        if c["percentiles"]:
            p10, p50, p90 = np.percentile(c["percentiles"], [10, 50, 90])
            stats = f"{p10:.1f} | {p50:.1f} | {p90:.1f}"
        else:
            stats = "- | - | -"
        skipped = ", ".join(str(i) for i in c["skipped"]) or "none"
        lines.append(
            f"| {c['site_class']} | {c['ids']} | {len(c['percentiles'])} | {len(c['skipped'])} "
            f"| {stats} | {skipped} |"
        )
    return "\n".join(lines) + "\n"


def dataset_summary_markdown(data: Data) -> str:
    """The committed record of a dataset (spec D4)."""
    version = data["version"]
    missing = data["maps_expected"] - data["maps_present"]
    classes = data["site_classes"]
    out = [
        f"# Dataset summary ({version})",
        "",
        "Synthetic terrain, not a real place. Propagation: Sionna RT, line of sight and "
        "specular reflection only (no diffraction, no diffuse scattering); pattern: 3GPP TR "
        "38.901. One sector per map, no vegetation or buildings, flat Earth. Written by "
        "`twin dataset-summary`; the maps themselves are not in git.",
        "",
        f"Maps in the manifest: {data['maps_present']} of {data['maps_expected']} expected"
        + ("." if missing == 0 else f"; {missing} not yet traced."),
        "",
        "## Terrains and maps per split and site class",
        "",
        "| split | " + " | ".join(classes) + " | terrains | grid maps | off-grid maps |",
        "|---|" + "---|" * (len(classes) + 3),
    ]
    for row in data["splits"]:
        counts = " | ".join(str(n) for n in row["per_class"])
        out.append(
            f"| {row['split']} | {counts} | {row['terrains']} | {row['grid_maps']} | "
            f"{row['off_grid_maps']} |"
        )
    ids_by_split = {row["split"]: ", ".join(str(i) for i in row["ids"]) for row in data["splits"]}
    out += [
        "",
        "Terrain ids by split: " + "; ".join(f"{s} {ids}" for s, ids in ids_by_split.items()) + ".",
        "",
        "## Skipped terrain ids",
        "",
        "Ids with no site candidate for their class (spec S2); the walk moved on to the next "
        f"id of that class. Highest id walked: {data['highest_id']}.",
        "",
        "| site class | skipped | ids |",
        "|---|---|---|",
    ]
    for site_class in classes:
        ids = data["skipped"][site_class]
        out.append(f"| {site_class} | {len(ids)} | {', '.join(str(i) for i in ids) or 'none'} |")
    out += [
        "",
        "## No-hit share per site class",
        "",
        "Share of map cells no ray reached (the 'no signal' class of spec E3), over all maps "
        "of the class.",
        "",
        "| site class | maps | mean | min | max |",
        "|---|---|---|---|---|",
    ]
    for site_class in classes:
        nh = data["no_hit"][site_class]
        if nh["maps"]:
            out.append(
                f"| {site_class} | {nh['maps']} | {nh['mean'] * 100:.1f}% | "
                f"{nh['min'] * 100:.1f}% | {nh['max'] * 100:.1f}% |"
            )
        else:
            out.append(f"| {site_class} | 0 | - | - | - |")
    total = data["wall_s_total"]
    out += [
        "",
        "## Time and settings",
        "",
        f"Solver wall time over all maps: {total / 3600:.2f} h ({total:.0f} s; "
        f"median {data['wall_s_median']:.2f} s per map). Scene building and file "
        "writing are not included.",
        "",
        "| setting | value |",
        "|---|---|",
    ]
    if data["settings"] is not None:
        out += [f"| {k} | {v} |" for k, v in data["settings"].items()]
        out += ["", "| provenance | values seen |", "|---|---|"]
        for key, seen in data["provenance_seen"].items():
            out.append(f"| {key} | {'; '.join(seen)} |")
    return "\n".join(out) + "\n"


def training_summary_markdown(data: Data) -> str:
    """The training runs: settings, best epochs, spread across seeds."""
    metas = data["metas"]
    shared = {k: v for k, v in metas[0]["hyperparameters"].items() if k != "seed"}
    best = [meta["best"] for meta in metas]

    def spread(key: str) -> str:
        values = np.array([b[key] for b in best])
        return f"{values.mean():.3f} (min {values.min():.3f}, max {values.max():.3f})"

    lines = [
        "# Training summary",
        "",
        "Synthetic terrain. The surrogate is a U-Net predicting the ray-traced path gain as a "
        "residual over B0, plus a logit for whether the ray tracer has power in each cell "
        "(spec M1, M1b). Trained on the train split, best epoch chosen on validation; the "
        "test split is not touched here. Written by `twin training-summary`.",
        "",
        "| setting | value |",
        "|---|---|",
        *(f"| {k} | {v} |" for k, v in shared.items()),
        f"| dataset | {metas[0]['dataset']['path']} (manifest sha256 "
        f"{metas[0]['dataset']['manifest_sha256'][:16]}...) |",
        f"| maps | train {metas[0]['maps']['train']}, "
        f"validation {metas[0]['maps']['validation']} |",
        "",
        "| provenance | values seen |",
        "|---|---|",
        *(
            f"| {k} | {'; '.join(sorted({str(m['provenance'][k]) for m in metas}))} |"
            for k in metas[0]["provenance"]
        ),
        "",
        "## Best epoch per seed (validation)",
        "",
        "L1 is the mean absolute error in dB over validation cells where the ray tracer has "
        "power; power accuracy is the share of all validation cells whose power logit has "
        "the right sign. LOS cells are direct-dominated when the traced gain is less than "
        f"{data['reflection_excess_db']:.0f} dB above B0 and reflection-dominated otherwise "
        "(ASSUMPTION, the rule of `twin fold-check`).",
        "",
        "| seed | best epoch | L1 (dB) | NLOS L1 (dB) | LOS direct L1 (dB) | "
        "LOS reflection L1 (dB) | BCE | power accuracy | total loss | last epoch: L1 (dB) | "
        "last epoch: BCE | last epoch: power accuracy | peak RSS after load (MiB) | seconds |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for meta in metas:
        b = meta["best"]

        def recorded(key: str, best: dict[str, Any] = b) -> str:
            return f"{best[key]:.3f}" if key in best else "not recorded"

        last = meta["history"][-1]
        rss = meta.get("peak_rss_mib_after_load", "not recorded")
        lines.append(
            f"| {meta['hyperparameters']['seed']} | {b['epoch']} | {b['l1_db']:.3f} | "
            f"{recorded('l1_nlos_db')} | {recorded('l1_los_direct_db')} | "
            f"{recorded('l1_los_reflection_db')} | "
            f"{b['bce']:.4f} | {b['power_accuracy'] * 100:.2f}% | {b['total']:.4f} | "
            f"{last['l1_db']:.3f} (epoch {last['epoch']}) | {last['bce']:.4f} | "
            f"{last['power_accuracy'] * 100:.2f}% | {rss} | "
            f"{meta['seconds']['total']:.0f} |"
        )
    lines += [
        "",
        f"Across seeds: L1 {spread('l1_db')} dB; power accuracy {spread('power_accuracy')}; "
        f"BCE {spread('bce')}.",
        "",
        overfitting_note(metas),
    ]
    if data["context"]:
        context = data["context"]
        commits = sorted({m["provenance"]["commit"][:7] for m in context})
        augmentations = sorted({m["hyperparameters"].get("augmentation", "none") for m in context})
        lines += [
            "",
            "## Context: earlier runs, quoted",
            "",
            f"Quoted from the run records of commit {', '.join(commits)} (augmentation "
            f"{', '.join(augmentations)}), not retrained or recomputed here.",
            "",
            "| seed | best epoch | L1 (dB) | power accuracy | L1 at last epoch (dB) |",
            "|---|---|---|---|---|",
        ]
        for meta in context:
            b, last = meta["best"], meta["history"][-1]
            lines.append(
                f"| {meta['hyperparameters']['seed']} | {b['epoch']} | {b['l1_db']:.3f} | "
                f"{b['power_accuracy'] * 100:.2f}% | {last['l1_db']:.3f} (epoch {last['epoch']}) |"
            )
    return "\n".join(lines) + "\n"


def overfitting_note(metas: list[dict[str, Any]]) -> str:
    """One sentence on where the seeds peak and what happens after, from their histories.

    Args:
        metas: Run records, the first of which supplies the example numbers.

    Returns:
        The sentence.
    """
    epochs = [meta["best"]["epoch"] for meta in metas]
    first = metas[0]
    history = first["history"]
    best = first["best"]["epoch"]
    at_best, last = history[best - 1], history[-1]
    return (
        f"All {len(metas)} seeds peak at epochs {min(epochs)}-{max(epochs)} and overfit after: "
        f"for seed {first['hyperparameters']['seed']}, validation L1 goes from "
        f"{at_best['l1_db']:.3f} dB at epoch {best} to {last['l1_db']:.3f} dB at epoch "
        f"{last['epoch']}, while the training loss falls from {at_best['train_total']:.3f} to "
        f"{last['train_total']:.3f}. The saved weights are those of the best epoch."
    )


def _agreement_cells(st: dict[str, Any]) -> str:
    if st["cells"] == 0:
        return "- | - | - | 0"
    return f"{st['median']:.3f} | {st['p95']:.3f} | {st['max']:.3f} | {st['cells']}"


def backend_check_markdown(data: Data) -> str:
    """Per-cell agreement of two backends on the same maps."""
    ref_prov, cand_prov = data["reference_provenance"], data["candidate_provenance"]
    lines = [
        "# Backend check",
        "",
        "The solver-check floor maps (synthetic terrain ids "
        f"{', '.join(str(t) for t in data['terrain_ids'])}, azimuth {data['azimuth_deg']:.0f}, "
        f"tilt {data['tilt_deg']:.0f}, line of sight and specular reflection, max_depth 3) "
        "solved on two backends and compared cell by cell. Written by `twin backend-check`.",
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
    for row in data["agreement"]:
        lines.append(
            f"| {row['terrain_id']} | {row['samples']:.0e} | {_agreement_cells(row['los'])} | "
            f"{_agreement_cells(row['nlos'])} | {row['power_only_reference']} | "
            f"{row['power_only_candidate']} |"
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
    for pair in data["timing"]:
        r, c = pair["reference"], pair["candidate"]
        lines.append(
            f"| {r['terrain_id']} | {r['samples']:.0e} | {r['warm_s']:.2f} | {r['cold_s']:.2f} | "
            f"{r['repeat_identical']} | {c['warm_s']:.2f} | {c['cold_s']:.2f} | "
            f"{c['repeat_identical']} | {c['gpu_memory_mib']} |"
        )
    return "\n".join(lines) + "\n"


def symmetry_check_markdown(data: Data) -> str:
    """A site-centred crop and its 8 symmetric variants against the moved original map."""
    lines = [
        "# Symmetry check",
        "",
        f"Synthetic terrain id {data['terrain_id']}, a {data['crop_side_m']:.0f} m square "
        f"centred on its {data['site_class']} site; azimuth {data['azimuth_deg']:.0f}, tilt "
        f"{data['tilt_deg']:.0f}; line of sight and specular reflection, "
        f"{data['samples_per_tx']:.0e} rays. Each variant (k clockwise "
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
        *(f"| {k} | {v} |" for k, v in data["provenance"].items()),
        "",
        "| k | mirror | keeps fold | azimuth | LOS median | LOS p95 | LOS max | LOS cells | "
        "NLOS median | NLOS p95 | NLOS max | NLOS cells | power only in moved original | "
        "power only in traced |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in data["rows"]:
        lines.append(
            f"| {row['k']} | {row['mirror']} | {row['keeps_fold']} | "
            f"{row['azimuth_deg']:.0f} | "
            f"{_agreement_cells(row['los'])} | {_agreement_cells(row['nlos'])} | "
            f"{row['power_only_moved_original']} | {row['power_only_traced']} |"
        )
    return "\n".join(lines) + "\n"


def _median_p95(st: dict[str, Any]) -> str:
    if st["cells"] == 0:
        return "- / - (0)"
    return f"{st['median']:.3f} / {st['p95']:.3f} ({st['cells']})"


def fold_check_markdown(data: Data) -> str:
    """The fold term beside the sampling floor, on training terrains (spec N3b)."""
    lines = [
        "# Fold check",
        "",
        "Synthetic terrain ids "
        f"{', '.join(str(i) for i in data['terrain_ids'])} (training terrains, two per site "
        f"class), azimuth {data['azimuth_deg']:.0f}, tilt {data['tilt_deg']:.0f}; line of sight "
        "and specular reflection. The terrain mesh and the measurement surface split every "
        "cell into two triangles along one diagonal; the dataset uses the SW-NE one. The fold "
        f"term compares the dataset's SW-NE maps with NW-SE maps at "
        f"{data['dataset_samples']:.0e} rays; the sampling floor compares the same SW-NE maps "
        f"with {data['reference_samples']:.0e} rays. Both are absolute dB differences over "
        "cells with power in both maps, pooled over the terrains of each row. Written by "
        "`twin fold-check`.",
        "",
        "| provenance | |",
        "|---|---|",
        *(f"| {k} | {v} |" for k, v in data["provenance"].items()),
        "",
        "## Truth uncertainty: fold term beside the sampling floor",
        "",
        "| site class | term | LOS median | LOS p95 | LOS max | LOS cells | NLOS median | "
        "NLOS p95 | NLOS max | NLOS cells | power only with SW-NE | power only in the other map |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in data["truth"]:
        lines.append(
            f"| {row['group']} | {row['term']} | {_agreement_cells(row['los'])} | "
            f"{_agreement_cells(row['nlos'])} | {row['power_only_sw_ne']} | "
            f"{row['power_only_other']} |"
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
    for row in data["shadow"]:
        low, high = row["low"], row["high"]
        label = f"{low:.0f} to {high:.0f}" if high is not None else f"{low:.0f} or more"
        lines.append(f"| {row['group']} | {label} | {_agreement_cells(row['fold'])} |")
    lines += [
        "",
        "## Fold term by dominant path",
        "",
        "A cell with power in the SW-NE map is reflection-dominated when its traced gain "
        f"exceeds B0 (direct path and antenna pattern, no terrain) by "
        f"{data['reflection_excess_db']:.0f} "
        "dB or more (ASSUMPTION), and direct-dominated otherwise. Median / p95 of the fold "
        "term, and the cells compared. B0 has no terrain, so shadowed cells rarely exceed "
        "it: the NLOS split says little, and nearly all NLOS cells fall under direct.",
        "",
        "| site class | LOS direct | LOS reflection | NLOS direct | NLOS reflection | "
        "LOS cells reflection-dominated |",
        "|---|---|---|---|---|---|",
    ]
    for row in data["paths"]:
        lines.append(
            f"| {row['group']} | {_median_p95(row['los_direct'])} | "
            f"{_median_p95(row['los_reflection'])} | {_median_p95(row['nlos_direct'])} | "
            f"{_median_p95(row['nlos_reflection'])} | "
            f"{row['los_reflection_cells'] / row['los_lit_cells'] * 100:.1f}% |"
        )
    return "\n".join(lines) + "\n"


def solver_check_markdown(data: Data) -> str:
    """The solver checks (A4 tilt, S5 surface, N3 time and sampling floor)."""

    def verdict(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    def stats(d: dict[str, Any]) -> str:
        return (
            f"{d['median_db']:.1e} | {d['p95_db']:.1e} | {d['bias_db']:+.1e} | "
            f"{d['valid'] * 100:.1f}%"
        )

    far = data["far_field_range_m"]
    dataset, reference, context = (
        data["dataset_samples"],
        data["reference_samples"],
        data["context_samples"],
    )
    floor_samples = data["floor_samples"]
    floor_rows = data["floor_rows"]
    lines = [
        "# Solver check",
        "",
        "Synthetic terrain and a flat test tile, not a real place. Propagation: Sionna RT; "
        "pattern: 3GPP TR 38.901. Line of sight and specular reflection only. Written by "
        "`twin solver-check`.",
        "",
        "| provenance | |",
        "|---|---|",
        *(f"| {key} | {value} |" for key, value in data["provenance"].items()),
        "",
        "Settings of the A4 and S5 maps:",
        "",
        "| setting | value |",
        "|---|---|",
        *(f"| {key} | {value} |" for key, value in data["settings"].items()),
        "",
        f"Main-lobe measurements: free space, {data['lobe_samples']:.0e} rays, a vertical planar "
        f"map {data['lobe_distance_m']:.0f} m out on boresight with 1 m cells.",
        "",
        f"## A4 flat-plain tilt check: {verdict(data['a4_pass'])}",
        "",
        f"Criteria: main-lobe elevation within {data['lobe_tolerance_deg']} deg of the commanded "
        f"tilt; far-field median moves by at least {data['min_far_field_span_db']} dB across the "
        f"tilts. Far field (ASSUMPTION): cells {far[0]:.0f} to {far[1]:.0f} m "
        f"from the site within {data['far_field_half_angle_deg']:.0f} deg of boresight.",
        "",
        "| tilt (deg) | main-lobe elevation (deg) | error (deg) | far-field median (dB) | "
        "far-field cells hit |",
        "|---|---|---|---|---|",
    ]
    for r in data["tilt_rows"]:
        lines.append(
            f"| {r['tilt_deg']:.0f} | {r['lobe_elevation_deg']:+.2f} | "
            f"{r['lobe_elevation_deg'] + r['tilt_deg']:+.2f} | {r['far_field_median_db']:.2f} | "
            f"{r['far_field_valid'] * 100:.1f}% |"
        )
    surface = data["surface"]
    lines += [
        "",
        f"Largest lobe error {data['lobe_error_deg']:.2f} deg; far-field span "
        f"{data['far_field_span_db']:.1f} dB. The median is "
        "not monotonic in tilt: the 8-element, 0.8-wavelength column has nulls about 9 deg "
        "apart, so as the lobe tilts down the far-field ring passes through the first null "
        "and then the first side lobe.",
        "",
        "## S5 measurement-surface check",
        "",
        "Flat tile. The mesh surface at 1.5 m against a planar radio map at 1.5 m, same "
        "cells and settings; the two-seed spreads show the noise each map has on its own. "
        "Values in dB over cells with power in both maps.",
        "",
        "| comparison | median abs | p95 abs | bias | cells |",
        "|---|---|---|---|---|",
        f"| mesh surface vs planar map | {stats(surface['surface_vs_planar'])} |",
        f"| planar map, seed vs seed + 1 | {stats(surface['planar_noise'])} |",
        f"| mesh surface, seed vs seed + 1 | {stats(surface['surface_noise'])} |",
        "",
        "## Time per map and the sampling floor (N3)",
        "",
        f"Terrains {', '.join(str(r['terrain_id']) for r in floor_rows)}, azimuth "
        f"{data['check_azimuth_deg']:.0f}, tilt {data['floor_tilt_deg']:.0f}, the ruled settings "
        "(line of sight and specular reflection, max_depth 3), compared over cells with power "
        "in both, overall and split by the LOS mask (see `baselines.py`).",
        "",
        f"The floor compares the dataset's {dataset:.0e} rays with "
        f"{reference:.0e}. A larger reference is not possible: Mitsuba's sampler "
        f"wavefront is 32-bit, so one solve launches at most {data['max_samples_per_tx']} rays, "
        "and repeating solves adds nothing because the rays come from the same deterministic "
        "lattice each time (the seed does not change these maps). A 4x step understates "
        "the error against the fully converged map more than the earlier 10x step "
        f"({context:.0e} vs {dataset:.0e}) did; that step is listed after "
        "the main table as context. A lattice with a different ray count points its rays in "
        "different directions rather than adding to the old ones, so a grazing cell reached "
        "by a single ray at one count can be missed at another: that is why a cell or two "
        "can have power at the smaller count only.",
        "",
        "| terrain | site | "
        + " | ".join(f"s/map {n:.0e}" for n in floor_samples)
        + " | floor median / p95 | LOS median / p95 (cells) | NLOS median / p95 (cells) | "
        f"no-hit {dataset:.0e} | no-hit {reference:.0e} | "
        f"hit only at {reference:.0e} | hit only at {dataset:.0e} |",
        "|" + "---|" * (2 + len(floor_samples) + 7),
    ]
    for f in floor_rows:
        seconds = " | ".join(f"{t:.2f}" for t in f["seconds"])
        fl, fll, fln = f["floor"], f["floor_los"], f["floor_nlos"]
        lines.append(
            f"| {f['terrain_id']} | {f['site_class']} | {seconds} | "
            f"{fl['median_db']:.3f} / {fl['p95_db']:.3f} | "
            f"{fll['median_db']:.3f} / {fll['p95_db']:.3f} ({fll['cells']}) | "
            f"{fln['median_db']:.3f} / {fln['p95_db']:.3f} ({fln['cells']}) | "
            f"{f['no_hit'][0] * 100:.2f}% | {f['no_hit'][1] * 100:.2f}% | "
            f"{f['hit_only_at_reference']} | {f['hit_only_at_dataset']} |"
        )
    lines += [
        "",
        f"Context, {context:.0e} against {dataset:.0e}:",
        "",
        "| terrain | site | LOS median / p95 (cells) | NLOS median / p95 (cells) |",
        "|---|---|---|---|",
    ]
    for f in floor_rows:
        cl, cn = f["context_los"], f["context_nlos"]
        lines.append(
            f"| {f['terrain_id']} | {f['site_class']} | "
            f"{cl['median_db']:.3f} / {cl['p95_db']:.3f} ({cl['cells']}) | "
            f"{cn['median_db']:.3f} / {cn['p95_db']:.3f} ({cn['cells']}) |"
        )
    return "\n".join(lines) + "\n"


def reflection_note(values: dict[str, float]) -> str:
    """One sentence on the LOS reflection-dominated stratum, written from the numbers.

    Args:
        values: Output of `evaluate.reflection_values`.

    Returns:
        The sentence.
    """
    bias, median, fold = values["bias"], values["hilltop_median"], values["hilltop_fold_median"]
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


def evaluation_markdown(data: Data) -> str:
    """The evaluation report (spec E1 to E6)."""
    facts, seeds, timing = data["facts"], data["seeds"], data["timing"]
    gpu_new = timing["terrain_features"] + timing["map_inputs"] + timing["surrogate_gpu"]
    gpu_further = timing["map_inputs"] + timing["surrogate_gpu"]
    cpu_new = timing["terrain_features"] + timing["map_inputs"] + timing["surrogate_cpu"]
    cpu_further = timing["map_inputs"] + timing["surrogate_cpu"]

    def e1_block(group: str) -> list[str]:
        rows = [
            "| stratum | method | mean abs | median abs | p95 abs | bias | cells |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in data["blocks"][group]:
            stratum, per_seed = row["stratum"], row["surrogate"]
            if per_seed[0][4] == 0:
                rows.append(f"| {stratum} | all methods | no cells | | | | 0 |")
                continue
            rows.append(
                f"| {stratum} | surrogate, {len(seeds)} seeds | "
                + " | ".join(_spread([p[i] for p in per_seed], 3) for i in range(4))
                + f" | {per_seed[0][4]} |"
            )
            for name in data["baselines"]:
                st = row["baselines"][name]
                rows.append(
                    f"| {stratum} | {name} | {st[0]:.3f} | {st[1]:.3f} | {st[2]:.3f} | "
                    f"{st[3]:+.3f} | {st[4]} |"
                )
            if row["uncertainty"] is not None:
                u = row["uncertainty"]
                rows.append(
                    f"| {stratum} | ray tracer, fold term (quoted) | | {u['fold'][0]:.3f} | "
                    f"{u['fold'][1]:.3f} | | |"
                )
                rows.append(
                    f"| {stratum} | ray tracer, sampling floor (quoted) | | "
                    f"{u['sampling'][0]:.3f} | {u['sampling'][1]:.3f} | | |"
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
    ]
    if data["rescored_note"] is not None:
        lines += [data["rescored_note"], ""]
    lines += [
        f"Maps: {facts['maps']} ({facts['grid maps']} grid, {facts['off-grid maps']} off-grid) "
        f"on terrains {', '.join(str(t) for t in facts['terrains'])}. Surrogate: the "
        f"best-epoch checkpoints of {', '.join(facts['runs'])}; each surrogate figure is the "
        "mean over the seeds, with the lowest and highest seed in brackets.",
        "",
        "Errors are predicted minus traced path gain in dB, over cells where the ray tracer "
        "has power. Strata: LOS direct-dominated and LOS reflection-dominated (traced gain "
        f"at least {data['reflection_excess_db']:.0f} dB above B0, ASSUMPTION) and NLOS, by the "
        "heightmap LOS mask. B0: free space plus the antenna pattern; B1: B0 minus the "
        "Bullington diffraction loss (ITU-R P.526-16 section 4.5.1).",
        "",
        "The ray tracer's own uncertainty is quoted from `results/fold_check.md` (commit "
        f"{data['uncertainty_commit']}; training terrains, one azimuth and tilt): the fold term "
        "(the map with the other cell diagonal) and the sampling floor (1e9 against 4e9 rays), "
        "as median and p95 of the absolute difference. The sampling floor is split only LOS "
        "and NLOS there, so both LOS strata quote the LOS value. Read the surrogate's errors "
        "against these; they are not a claim that the surrogate beats the ray tracer.",
        "",
        "## E1 and E3: path-gain error by stratum, all grid maps",
        "",
        *e1_block("all"),
        "",
        reflection_note(data["reflection"]),
        "",
        "## E3: path-gain error by stratum, per site class",
    ]
    for site_class in data["site_classes"]:
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
    for group in data["groups"]:
        stats = []
        for s in seeds:
            tp, fp, tn, fn = data["power"][group][s]
            stats.append(
                (
                    tp / (tp + fp),
                    tp / (tp + fn),
                    tn / (tn + fn),
                    tn / (tn + fp),
                    (tp + tn) / (tp + fp + tn + fn),
                )
            )
        tp, fp, tn, fn = data["power"][group][seeds[0]]
        lines.append(
            f"| {group} | "
            + " | ".join(_spread([st[i] * 100 for st in stats], 2) + "%" for i in range(5))
            + f" | {(tn + fp) / (tp + fp + tn + fn) * 100:.1f}% |"
        )

    lines += [
        "",
        "## E2: coverage",
        "",
        f"RSRP = {data['power_per_re_dbm']:.2f} dBm per resource element "
        f"({data['sector_power_dbm']:.0f} dBm over {data['resource_elements']} resource elements "
        f"of a 20 MHz carrier, ASSUMPTION) plus path gain; a cell is covered at "
        f"{data['rsrp_threshold_dbm']:.0f} dBm (ASSUMPTION), i.e. path gain of at least "
        f"{data['covered_gain_db']:.2f} dB. The ray tracer's covered cells need power; the "
        "surrogate's need its power head to say power; B0 and B1 always predict power. "
        "Covered-area error: predicted minus traced covered cells over traced covered cells, "
        "pooled over the grid maps. IoU: pooled, and the mean of per-map IoU.",
        "",
        "| method | covered-area error | IoU (pooled) | IoU (mean per map) |",
        "|---|---|---|---|",
    ]
    cov = [data["coverage"][s] for s in seeds]
    lines.append(
        f"| surrogate, {len(seeds)} seeds | {_spread([c[0] for c in cov], 2)}% | "
        f"{_spread([c[1] for c in cov], 4)} | {_spread([c[2] for c in cov], 4)} |"
    )
    for name in data["baselines"]:
        c = data["coverage"][name]
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

    gpu_record = data["gpu_record"]
    lines += [
        "",
        "## E5: time per map, same machine",
        "",
        f"GPU: {gpu_record['gpu']}. Medians, in seconds. The ray tracer "
        f"runs the dataset's settings ({gpu_record['samples_per_tx']:.0e} rays): its "
        "new-terrain figure covers terrain generation, scene and mesh build, measurement "
        "surface and the first solve; further maps reuse the scene and time the solve only. "
        "The surrogate's new-terrain figure covers geometry, LOS mask, B0, the per-map channels "
        "and inference; further maps cover B0, the channels and inference. Both sides skip "
        f"their warm-up (kernel compilation). GPU ray tracer: terrains "
        f"{', '.join(str(t) for t in gpu_record['terrains'])} through the Windows runner "
        f"(commit {gpu_record['commit'][:7]}); CPU ray tracer: terrains "
        f"{', '.join(str(t) for t in data['cpu_terrains'])}, llvm, measured here.",
        "",
        "| hardware | case | ray tracer | surrogate | ray tracer / surrogate |",
        "|---|---|---|---|---|",
        *(
            f"| {hw} | {case} | {rt:.3f} | {sg:.4f} | {rt / sg:.1f} |"
            for hw, case, rt, sg in (
                ("GPU", "new terrain, first map", timing["rt_gpu_new"], gpu_new),
                ("GPU", "each further map", timing["rt_gpu_further"], gpu_further),
                ("CPU", "new terrain, first map", timing["rt_cpu_new"], cpu_new),
                ("CPU", "each further map", timing["rt_cpu_further"], cpu_further),
            )
        ),
        "",
        "Surrogate parts: geometry and LOS mask for a new terrain "
        f"{timing['terrain_features']:.3f}; B0 and channels per map {timing['map_inputs']:.4f}; "
        f"inference, batch 1, GPU {timing['surrogate_gpu']:.4f} and CPU "
        f"{timing['surrogate_cpu']:.4f}.",
        "",
        "On this machine, per map, the ray tracer takes "
        f"{timing['rt_gpu_new'] / gpu_new:.1f} times as long as the surrogate for a new "
        f"terrain's first map and {timing['rt_gpu_further'] / gpu_further:.1f} times as long "
        "for each further map on the GPU; without a GPU the ratios are "
        f"{timing['rt_cpu_new'] / cpu_new:.1f} and {timing['rt_cpu_further'] / cpu_further:.1f}.",
        "",
        "## Per seed, all grid maps",
        "",
        "| seed | mean abs, all | mean abs, LOS direct | mean abs, LOS reflection | "
        "mean abs, NLOS | power accuracy |",
        "|---|---|---|---|---|---|",
    ]
    for s in seeds:
        tp, fp, tn, fn = data["per_seed"][s]["power"]
        lines.append(
            f"| {s} | "
            + " | ".join(f"{m:.3f}" for m in data["per_seed"][s]["means"])
            + f" | {(tp + tn) / (tp + fp + tn + fn) * 100:.2f}% |"
        )
    return "\n".join(lines) + "\n"


def search_markdown(data: Data) -> str:
    """The tilt and power search report (rendered by `search.report_markdown_from_data`)."""
    from sionna_twin_ops.search import report_markdown_from_data

    return report_markdown_from_data(data)


RENDERERS: dict[str, Callable[[Data], str]] = {
    "site_check": site_check_markdown,
    "dataset_summary": dataset_summary_markdown,
    "training_summary": training_summary_markdown,
    "backend_check": backend_check_markdown,
    "search": search_markdown,
    "symmetry_check": symmetry_check_markdown,
    "fold_check": fold_check_markdown,
    "solver_check": solver_check_markdown,
    "evaluation": evaluation_markdown,
}
