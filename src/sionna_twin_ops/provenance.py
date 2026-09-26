"""Where a result came from: commit, Mitsuba variant, platform, packages and GPU driver.

Recorded with every ray-traced output (repo rules 2 and 8).
"""

import os
import platform
import shutil
import subprocess
from importlib.metadata import version
from pathlib import Path

from sionna_twin_ops.backend import active_variant

PACKAGES = ("sionna-rt", "mitsuba", "drjit", "numpy")


def commit() -> str:
    """The code's commit: TWIN_COMMIT when set by the Windows runner, else git HEAD.

    A git checkout with uncommitted changes to tracked files gets a "-dirty" suffix.

    Returns:
        The commit hash.

    Raises:
        RuntimeError: If TWIN_COMMIT is unset and git cannot name the commit.
    """
    pinned = os.environ.get("TWIN_COMMIT")
    if pinned:
        return pinned
    here = Path(__file__).resolve().parent
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=here,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError("cannot name the commit: set TWIN_COMMIT or run from git") from error
    return f"{head}-dirty" if status else head


def gpu() -> str:
    """GPU name and driver version from nvidia-smi, or why they are unknown.

    Returns:
        "<name>, driver <version>", or a statement that nvidia-smi is absent.

    Raises:
        RuntimeError: If nvidia-smi exists but fails.
    """
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return "unknown: nvidia-smi not found"
    result = subprocess.run(
        [exe, "--query-gpu=name,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    name, driver = (part.strip() for part in result.stdout.strip().splitlines()[0].split(","))
    return f"{name}, driver {driver}"


def gpu_memory_used_mib() -> int | None:
    """GPU memory in use now, from nvidia-smi; None when nvidia-smi is absent.

    Returns:
        MiB in use on the first GPU, or None.
    """
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    result = subprocess.run(
        [exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout.strip().splitlines()[0])


def provenance() -> dict[str, str]:
    """Everything that identifies how a ray-traced result was made.

    Returns:
        Field name to value, in a fixed order.
    """
    return {
        "commit": commit(),
        "variant": active_variant(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        **{name: version(name) for name in PACKAGES},
        "gpu": gpu(),
    }
