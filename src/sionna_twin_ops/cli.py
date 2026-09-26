"""Command line entry point: `twin`."""

import argparse
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

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
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
