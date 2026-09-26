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
    from sionna_twin_ops.solve import DEV_SAMPLES, specular_settings

    settings = specular_settings(DEV_SAMPLES, seed)
    lobe_samples = 10**8
    tilt_rows = checks.tilt_check(settings, lobe_samples)
    surface = checks.surface_check(settings, tilt_deg=6.0)
    floor_rows = checks.sampling_floor(seed)
    header = (
        f"Synthetic terrain and a flat test tile, not a real place. Propagation: Sionna RT "
        f"{version('sionna-rt')} (Mitsuba {version('mitsuba')}, variant {variant}); pattern: "
        "3GPP TR 38.901. Line of sight and specular reflection only. Written by "
        "`twin solver-check`."
    )
    report = checks.solver_check_markdown(
        tilt_rows, surface, floor_rows, settings, lobe_samples, header
    )
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "solver_check.md").write_text(report)
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
        args.out.write_text(report)
        sys.stdout.write(report)
        return 0
    if args.command == "solver-check":
        solver_check(args.variant, args.seed, args.out_dir)
        return 0
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
