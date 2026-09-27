"""Command line entry point: `twin`."""

import argparse
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

from sionna_twin_ops.backend import DEFAULT_VARIANT

if TYPE_CHECKING:
    from sionna_twin_ops.site import ClassCheck

PACKAGES = ("sionna-rt", "mitsuba", "drjit", "numpy", "torch", "matplotlib")


def environment_report() -> dict[str, object]:
    """Installed package versions, Mitsuba variants and PyTorch CUDA status."""
    import mitsuba as mi
    import torch

    cuda = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: version(name) for name in PACKAGES},
        "mitsuba_variants": sorted(mi.variants()),
        "torch_cuda_build": torch.version.cuda,
        "torch_cuda_available": cuda,
        "torch_cuda_device": torch.cuda.get_device_name(0) if cuda else None,
    }


def site_check_markdown(checks: list["ClassCheck"], ids: range, spacing_m: float) -> str:
    """Render the site check as a markdown table.

    Args:
        checks: Output of `site_check`.
        ids: The terrain ids checked.
        spacing_m: Grid spacing used.

    Returns:
        Markdown text.
    """
    import numpy as np

    lines = [
        "# Site placement check (spec S2)",
        "",
        f"Synthetic terrain ids {ids.start} to {ids.stop - 1}, {spacing_m:.0f} m grid, each id "
        "placed with its own class. Percentile: midrank percentile of the site height among "
        "the vertices of its 5.12 km map. Written by `twin site-check`.",
        "",
        "| class | ids | placed | skipped | p10 | median | p90 | skipped ids |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in checks:
        if c.percentiles:
            p10, p50, p90 = np.percentile(c.percentiles, [10, 50, 90])
            stats = f"{p10:.1f} | {p50:.1f} | {p90:.1f}"
        else:
            stats = "- | - | -"
        skipped = ", ".join(str(i) for i in c.skipped) or "none"
        lines.append(
            f"| {c.site_class} | {len(c.ids)} | {len(c.percentiles)} | {len(c.skipped)} "
            f"| {stats} | {skipped} |"
        )
    return "\n".join(lines) + "\n"


def solver_check(variant: str, seed: int, out_dir: Path) -> None:
    """Run the ray-tracer checks and write the report and the tilt figure.

    Args:
        variant: Mitsuba variant, selected before Sionna RT is imported.
        seed: Solver seed.
        out_dir: Directory for solver_check.md and figures/tilt_check.png.
    """
    from sionna_twin_ops.backend import select_variant

    select_variant(variant)
    from sionna_twin_ops import checks
    from sionna_twin_ops.figures import tilt_check_figure
    from sionna_twin_ops.provenance import provenance
    from sionna_twin_ops.solve import DEV_SAMPLES, specular_settings

    settings = specular_settings(DEV_SAMPLES, seed)
    lobe_samples = 10**8
    tilt_rows = checks.tilt_check(settings, lobe_samples)
    surface = checks.surface_check(settings, tilt_deg=6.0)
    floor_rows = checks.sampling_floor(seed)
    header = "\n".join(
        [
            "Synthetic terrain and a flat test tile, not a real place. Propagation: Sionna RT; "
            "pattern: 3GPP TR 38.901. Line of sight and specular reflection only. Written by "
            "`twin solver-check`.",
            "",
            "| provenance | |",
            "|---|---|",
            *(f"| {key} | {value} |" for key, value in provenance().items()),
        ]
    )
    report = checks.solver_check_markdown(
        tilt_rows, surface, floor_rows, settings, lobe_samples, header
    )
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "solver_check.md").write_text(report, newline="\n")
    tilt_check_figure(
        [r.tilt_deg for r in tilt_rows],
        [r.lobe_elevation_deg for r in tilt_rows],
        [r.far_field_median_db for r in tilt_rows],
        "Flat test tile, 30 m mast, 1.8 GHz, 8 x 1 column of 3GPP TR 38.901 elements, "
        f"Sionna RT {version('sionna-rt')}, {variant}. Left: free-space main lobe, error "
        "labelled. Right: median over cells 1.5 to 2.5 km out within 60 deg of boresight; "
        "the dip at 9 deg is the column's first null reaching the horizon.",
        out_dir / "figures" / "tilt_check.png",
    )
    sys.stdout.write(report)


def evaluate_command(args: argparse.Namespace) -> None:
    """Run `twin evaluate`: score a split, time both methods, write the report and figure.

    Args:
        args: Parsed arguments of the evaluate command.

    Raises:
        RuntimeError: If PyTorch sees no CUDA device (the timing needs the GPU).
    """
    import torch

    from sionna_twin_ops.backend import select_variant

    if not torch.cuda.is_available():
        raise RuntimeError("twin evaluate times the surrogate on the GPU; none is visible")
    select_variant(args.variant)
    from sionna_twin_ops import evaluate

    tallies, facts = evaluate.evaluate_split(
        args.dataset, args.split, args.runs, torch.device("cuda")
    )
    uncertainty = evaluate.quoted_uncertainty(args.fold_report)
    timing = evaluate.measure_timing(
        args.dataset,
        args.split,
        args.runs[0],
        args.gpu_timing,
        args.cpu_timing_terrains,
        args.cpu_further_maps,
    )
    median, p95 = evaluate.backend_agreement(args.backend_report)
    report = evaluate.report_markdown(tallies, facts, uncertainty, timing)
    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"evaluation_{args.split}.md").write_text(report, newline="\n")
    caption = (
        f"Synthetic terrain, {args.split} split; propagation: "
        f"{'; '.join(facts['ray tracing'])} (llvm-equivalent: median {median:.3f} and p95 "
        f"{p95:.3f} dB, backend_check.md); pattern: 3GPP TR 38.901. "
        f"Azimuth {evaluate.FIGURE_AZIMUTH_DEG:.0f} deg, tilt {evaluate.FIGURE_TILT_DEG:.0f} "
        f"deg; surrogate: {args.runs[0]}. All rows share one path-gain scale (-150 to -60 dB) "
        f"and one error scale (clipped at +-{evaluate.ERROR_LIMIT_DB:.0f} dB). Clear cells: no "
        "traced power (left, right) or no predicted power (middle)."
    )
    evaluate.evaluation_figure(
        args.dataset,
        args.split,
        args.runs[0],
        caption,
        args.out_dir / "figures" / f"evaluation_{args.split}.jpg",
    )
    sys.stdout.write(report)


def viewer_command(args: argparse.Namespace) -> None:
    """Run `twin viewer`: write the 3D viewer page for a split.

    Args:
        args: Parsed arguments of the viewer command.
    """
    from sionna_twin_ops import viewer
    from sionna_twin_ops.evaluate import ERROR_LIMIT_DB, FIGURE_AZIMUTH_DEG, FIGURE_TILT_DEG

    cases = viewer.viewer_cases(args.dataset, args.split, FIGURE_AZIMUTH_DEG, FIGURE_TILT_DEG)
    data = viewer.viewer_data(args.dataset, args.split, args.run, cases)
    caption = (
        f"Synthetic terrain, {args.split} split; propagation: {'; '.join(data['ray_tracing'])}; "
        f"LOS and specular reflection only; pattern: 3GPP TR 38.901. Surrogate: {args.run} "
        f"(seed {data['seed']}). Path-gain views share one scale (-150 to -60 dB); the error "
        f"is clipped at +-{ERROR_LIMIT_DB:.0f} dB. Grey ground: no traced power "
        "(ray-traced, error) or no predicted power (predicted). Markers: cells where the power "
        "head and the ray tracer disagree. Written by `twin viewer`."
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(viewer.viewer_html(data, caption), newline="\n")
    size_mb = args.out.stat().st_size / 1e6
    sys.stdout.write(f"wrote {args.out}: {len(cases)} cases, {size_mb:.2f} MB\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="twin", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("env", help="print package versions and PyTorch CUDA status as JSON")
    figures = commands.add_parser(
        "terrain-figures", help="draw sample hillshades and the site placement figure"
    )
    figures.add_argument("--count", type=int, required=True, help="terrains 0..count-1")
    figures.add_argument(
        "--site-terrains", type=int, nargs="+", required=True, help="one panel per terrain id"
    )
    figures.add_argument("--spacing", type=float, required=True, help="grid spacing in m")
    figures.add_argument("--out", type=Path, required=True, help="output directory")
    check = commands.add_parser(
        "site-check", help="place each id's site class; report percentiles and skips"
    )
    check.add_argument("--count", type=int, required=True, help="terrains 0..count-1")
    check.add_argument("--spacing", type=float, required=True, help="grid spacing in m")
    check.add_argument("--out", type=Path, required=True, help="markdown file to write")
    solver = commands.add_parser(
        "solver-check", help="tilt check, surface check, time per map and sampling floor"
    )
    solver.add_argument(
        "--variant", default=DEFAULT_VARIANT, help=f"Mitsuba variant (default {DEFAULT_VARIANT})"
    )
    solver.add_argument("--seed", type=int, required=True, help="solver seed")
    solver.add_argument(
        "--out-dir", type=Path, required=True, help="writes solver_check.md and figures/"
    )
    maps = commands.add_parser(
        "backend-maps", help="solve the floor maps on one backend and save them for comparison"
    )
    maps.add_argument(
        "--variant", default=DEFAULT_VARIANT, help=f"Mitsuba variant (default {DEFAULT_VARIANT})"
    )
    maps.add_argument("--samples", required=True, help="comma-separated rays per map, e.g. 1e7,1e8")
    maps.add_argument("--seed", type=int, required=True, help="solver seed")
    maps.add_argument("--out", type=Path, required=True, help="directory for maps and meta.json")
    compare = commands.add_parser(
        "backend-check", help="compare two backend-maps runs cell by cell"
    )
    compare.add_argument("--reference", type=Path, required=True, help="reference run directory")
    compare.add_argument("--candidate", type=Path, required=True, help="candidate run directory")
    compare.add_argument("--out", type=Path, required=True, help="markdown file to write")
    sweep_cmd = commands.add_parser("dataset-sweep", help="trace the dataset maps (resumable)")
    sweep_cmd.add_argument(
        "--variant", default=DEFAULT_VARIANT, help=f"Mitsuba variant (default {DEFAULT_VARIANT})"
    )
    sweep_cmd.add_argument(
        "--selection", choices=("v1", "v2"), required=True, help="dataset version"
    )
    sweep_cmd.add_argument("--out", type=Path, required=True, help="dataset directory")
    summary_cmd = commands.add_parser("dataset-summary", help="write the committed dataset summary")
    summary_cmd.add_argument("--dataset", type=Path, required=True, help="dataset directory")
    summary_cmd.add_argument(
        "--selection", choices=("v1", "v2"), required=True, help="dataset version"
    )
    summary_cmd.add_argument("--out", type=Path, required=True, help="markdown file to write")
    train_cmd = commands.add_parser("train", help="train one seed of the surrogate")
    train_cmd.add_argument("--dataset", type=Path, required=True, help="dataset directory")
    train_cmd.add_argument("--seed", type=int, required=True, help="training seed")
    train_cmd.add_argument("--epochs", type=int, required=True, help="training epochs")
    train_cmd.add_argument("--width", type=int, required=True, help="U-Net first-level width")
    train_cmd.add_argument(
        "--augmentation",
        choices=("none", "symmetry"),
        required=True,
        help="symmetry: the 8 exact map symmetries, training split only",
    )
    train_cmd.add_argument(
        "--device", choices=("cuda", "cpu"), required=True, help="explicit device, no fallback"
    )
    train_cmd.add_argument("--out", type=Path, required=True, help="run directory")
    runs_cmd = commands.add_parser("training-summary", help="summarise training runs")
    runs_cmd.add_argument("--runs", type=Path, nargs="+", required=True, help="run directories")
    runs_cmd.add_argument(
        "--context-runs",
        type=Path,
        nargs="*",
        required=True,
        help="earlier run directories to quote as context (may be empty)",
    )
    runs_cmd.add_argument("--out", type=Path, required=True, help="markdown file to write")
    sym = commands.add_parser(
        "symmetry-check", help="trace a map and its 8 symmetric variants and compare them"
    )
    sym.add_argument(
        "--variant", default=DEFAULT_VARIANT, help=f"Mitsuba variant (default {DEFAULT_VARIANT})"
    )
    sym.add_argument("--terrain-id", type=int, required=True, help="terrain id")
    sym.add_argument("--azimuth", type=float, required=True, help="boresight azimuth in deg")
    sym.add_argument("--tilt", type=float, required=True, help="electrical tilt in deg")
    sym.add_argument("--out", type=Path, required=True, help="markdown file to write")
    fold_cmd = commands.add_parser(
        "fold-check", help="fold term beside the sampling floor on training terrains"
    )
    fold_cmd.add_argument(
        "--variant", default=DEFAULT_VARIANT, help=f"Mitsuba variant (default {DEFAULT_VARIANT})"
    )
    fold_cmd.add_argument("--azimuth", type=float, required=True, help="boresight azimuth in deg")
    fold_cmd.add_argument("--tilt", type=float, required=True, help="electrical tilt in deg")
    fold_cmd.add_argument("--out", type=Path, required=True, help="markdown file to write")
    eval_cmd = commands.add_parser("evaluate", help="score the surrogate, B0 and B1 on a split")
    eval_cmd.add_argument("--dataset", type=Path, required=True, help="dataset directory")
    eval_cmd.add_argument("--split", choices=("validation", "test"), required=True, help="split")
    eval_cmd.add_argument("--runs", type=Path, nargs="+", required=True, help="training runs")
    eval_cmd.add_argument(
        "--fold-report", type=Path, required=True, help="results/fold_check.md to quote"
    )
    eval_cmd.add_argument(
        "--variant", default=DEFAULT_VARIANT, help="Mitsuba variant for the CPU ray-tracer timing"
    )
    eval_cmd.add_argument(
        "--gpu-timing", type=Path, required=True, help="JSON from `twin raytrace-timing` on the GPU"
    )
    eval_cmd.add_argument(
        "--backend-report", type=Path, required=True, help="results/backend_check.md to quote"
    )
    eval_cmd.add_argument(
        "--cpu-timing-terrains", type=int, required=True, help="new terrains timed on the CPU"
    )
    eval_cmd.add_argument(
        "--cpu-further-maps", type=int, required=True, help="further maps timed per terrain"
    )
    eval_cmd.add_argument(
        "--out-dir", type=Path, required=True, help="writes evaluation_<split>.md and figures/"
    )
    rt_time = commands.add_parser(
        "raytrace-timing", help="time the ray tracer end to end on new terrains (JSON)"
    )
    rt_time.add_argument(
        "--variant", default=DEFAULT_VARIANT, help=f"Mitsuba variant (default {DEFAULT_VARIANT})"
    )
    rt_time.add_argument("--dataset", type=Path, required=True, help="dataset directory")
    rt_time.add_argument("--split", choices=("validation", "test"), required=True, help="split")
    rt_time.add_argument("--terrains", type=int, required=True, help="new terrains to time")
    rt_time.add_argument("--further-maps", type=int, required=True, help="further maps each")
    rt_time.add_argument("--out", type=Path, required=True, help="JSON file to write")
    view_cmd = commands.add_parser("viewer", help="write the 3D viewer page (one HTML file)")
    view_cmd.add_argument("--dataset", type=Path, required=True, help="dataset directory")
    view_cmd.add_argument("--split", choices=("validation", "test"), required=True, help="split")
    view_cmd.add_argument("--run", type=Path, required=True, help="training run that predicts")
    view_cmd.add_argument("--out", type=Path, required=True, help="HTML file to write")
    args = parser.parse_args(argv)

    if args.command == "env":
        json.dump(environment_report(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    if args.command == "terrain-figures":
        from sionna_twin_ops.figures import site_placement, terrain_grid

        args.out.mkdir(parents=True, exist_ok=True)
        terrain_grid(list(range(args.count)), args.spacing, args.out / "terrain_samples.jpg")
        site_placement(args.site_terrains, args.spacing, args.out / "site_placement.jpg")
        return 0
    if args.command == "site-check":
        from sionna_twin_ops.site import site_check

        ids = range(args.count)
        report = site_check_markdown(site_check(ids, args.spacing), ids, args.spacing)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, newline="\n")
        sys.stdout.write(report)
        return 0
    if args.command == "solver-check":
        solver_check(args.variant, args.seed, args.out_dir)
        return 0
    if args.command == "backend-maps":
        from sionna_twin_ops.backend import select_variant

        select_variant(args.variant)
        from sionna_twin_ops.crosscheck import floor_maps

        floor_maps([int(float(n)) for n in args.samples.split(",")], args.seed, args.out)
        return 0
    if args.command == "backend-check":
        from sionna_twin_ops.crosscheck import compare_markdown

        report = compare_markdown(args.reference, args.candidate)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, newline="\n")
        sys.stdout.write(report)
        return 0
    if args.command == "dataset-sweep":
        from sionna_twin_ops.backend import select_variant

        select_variant(args.variant)
        from sionna_twin_ops.dataset import SOLVER_SEED, selection_for, sweep
        from sionna_twin_ops.solve import DATASET_SAMPLES

        selection = selection_for(args.selection)
        traced = sweep(args.out, selection.terrains, DATASET_SAMPLES, SOLVER_SEED)
        sys.stdout.write(f"traced {traced} maps into {args.out}\n")
        return 0
    if args.command == "dataset-summary":
        from sionna_twin_ops.dataset import summary_markdown

        report = summary_markdown(args.dataset, args.selection)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, newline="\n")
        sys.stdout.write(report)
        return 0
    if args.command == "train":
        from sionna_twin_ops.train import train

        train(
            args.dataset,
            args.seed,
            args.epochs,
            args.width,
            args.augmentation,
            args.device,
            args.out,
        )
        return 0
    if args.command == "training-summary":
        from sionna_twin_ops.train import training_summary_markdown

        report = training_summary_markdown(args.runs, args.context_runs)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, newline="\n")
        sys.stdout.write(report)
        return 0
    if args.command == "symmetry-check":
        from sionna_twin_ops.backend import select_variant

        select_variant(args.variant)
        from sionna_twin_ops.crosscheck import symmetry_check_markdown

        report = symmetry_check_markdown(args.terrain_id, args.azimuth, args.tilt)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, newline="\n")
        sys.stdout.write(report)
        return 0
    if args.command == "fold-check":
        from sionna_twin_ops.backend import select_variant

        select_variant(args.variant)
        from sionna_twin_ops.crosscheck import fold_check_markdown

        report = fold_check_markdown(args.azimuth, args.tilt)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, newline="\n")
        sys.stdout.write(report)
        return 0
    if args.command == "raytrace-timing":
        from sionna_twin_ops.backend import select_variant

        select_variant(args.variant)
        from sionna_twin_ops.crosscheck import raytracer_timing

        record = raytracer_timing(args.dataset, args.split, args.terrains, args.further_maps)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=1), newline="\n")
        sys.stdout.write(json.dumps(record, indent=1) + "\n")
        return 0
    if args.command == "evaluate":
        evaluate_command(args)
        return 0
    if args.command == "viewer":
        viewer_command(args)
        return 0
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
