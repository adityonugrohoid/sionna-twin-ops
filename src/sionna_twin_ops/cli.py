"""Command line entry point: `twin`."""

import argparse
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="twin", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("env", help="print package versions and PyTorch CUDA status as JSON")
    figures = commands.add_parser(
        "terrain-figures", help="draw sample hillshades and the site placement figure"
    )
    figures.add_argument("--count", type=int, required=True, help="terrains 0..count-1")
    figures.add_argument("--site-terrain", type=int, required=True, help="terrain for sites")
    figures.add_argument("--spacing", type=float, required=True, help="grid spacing in m")
    figures.add_argument("--out", type=Path, required=True, help="output directory")
    args = parser.parse_args(argv)

    if args.command == "env":
        json.dump(environment_report(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    if args.command == "terrain-figures":
        from sionna_twin_ops.figures import site_placement, terrain_grid

        args.out.mkdir(parents=True, exist_ok=True)
        terrain_grid(list(range(args.count)), args.spacing, args.out / "terrain_samples.jpg")
        site_placement(args.site_terrain, args.spacing, args.out / "site_placement.jpg")
        return 0
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
