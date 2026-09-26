"""Command line entry point: `twin`."""

import argparse
import json
import platform
import sys
from importlib.metadata import version

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
    args = parser.parse_args(argv)

    if args.command == "env":
        json.dump(environment_report(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
