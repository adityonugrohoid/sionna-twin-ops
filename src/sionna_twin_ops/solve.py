"""Solve one path-gain map with Sionna RT's radio map solver (spec rules N and S4)."""

import time
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from sionna_twin_ops.backend import active_variant
from sionna_twin_ops.sionna_rt import mi, rt
from sionna_twin_ops.site import MAP_CELLS


@dataclass(frozen=True)
class SolverSettings:
    """Everything that sets the ray tracer's output for a given scene (spec N1).

    Attributes:
        samples_per_tx: Rays launched from the transmitter.
        max_depth: Maximum interactions per path.
        los: Line-of-sight paths.
        specular_reflection: Specular reflections.
        diffuse_reflection: Diffuse reflections.
        refraction: Refraction (transmission into materials).
        diffraction: Wedge diffraction.
        edge_diffraction: Diffraction on free-floating edges.
        seed: Solver seed.
    """

    samples_per_tx: int
    max_depth: int
    los: bool
    specular_reflection: bool
    diffuse_reflection: bool
    refraction: bool
    diffraction: bool
    edge_diffraction: bool
    seed: int

    def record(self) -> dict[str, object]:
        """Settings plus the Mitsuba variant, for manifests and result captions."""
        return {"variant": active_variant(), **asdict(self)}


def precoding_vec(weights: np.ndarray) -> tuple[mi.Float, mi.Float]:
    """Weights in the (real, imaginary) form the radio map solver takes.

    Args:
        weights: Complex per-element weights.

    Returns:
        (real parts, imaginary parts).
    """
    return mi.Float(weights.real.astype(np.float32)), mi.Float(weights.imag.astype(np.float32))


DATASET_SAMPLES = 10**9  # dataset rule N7 (docs/dataset.md): dataset maps, on the GPU
DEV_SAMPLES = 10**7  # dataset rule N6 (docs/dataset.md): tests and development runs
# Mitsuba's sampler takes a 32-bit wavefront size, so one transmitter can launch at most
# 2**32 - 1 rays per solve; the Fibonacci ray lattice makes repeated solves identical.
MAX_SAMPLES_PER_TX = 2**32 - 1
REFERENCE_SAMPLES = 4 * 10**9  # the sampling-floor reference, the largest under the cap


def specular_settings(samples_per_tx: int, seed: int) -> SolverSettings:
    """The ruled solver method (3a): line of sight and specular reflection only.

    Diffraction is off because it never reaches a mesh measurement surface in Sionna RT;
    diffuse reflection is off because it only adds Monte Carlo noise in shadow; refraction
    is off because the ground is opaque.

    Args:
        samples_per_tx: Rays launched (DATASET_SAMPLES or DEV_SAMPLES).
        seed: Solver seed.

    Returns:
        The settings.

    Raises:
        ValueError: If samples_per_tx is not between 1 and MAX_SAMPLES_PER_TX.
    """
    if not 1 <= samples_per_tx <= MAX_SAMPLES_PER_TX:
        raise ValueError(
            f"samples_per_tx {samples_per_tx} outside 1..{MAX_SAMPLES_PER_TX}: Mitsuba's "
            "sampler wavefront is 32-bit"
        )
    return SolverSettings(
        samples_per_tx=samples_per_tx,
        max_depth=3,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=False,
        diffraction=False,
        edge_diffraction=False,
        seed=seed,
    )


@dataclass(frozen=True)
class MapResult:
    """One solved map.

    Attributes:
        path_gain: Linear path gain per cell, shape (128, 128), row 0 at the south edge,
            column 0 at the west edge; 0 where no ray reached the cell.
        seconds: Wall time of the solve, including reading the result back.
    """

    path_gain: NDArray[np.float64]
    seconds: float


def solve_map(
    scene: rt.Scene, surface: mi.Mesh, weights: np.ndarray, settings: SolverSettings
) -> MapResult:
    """Solve the path-gain map on a measurement surface.

    Each map cell is two triangles of the surface; its value is the mean of their path
    gains in linear power (spec S4).

    Args:
        scene: Scene with one transmitter.
        surface: Measurement surface built by `scene.measurement_surface`.
        weights: Complex per-element transmit weights (the electrical tilt).
        settings: Solver settings.

    Returns:
        The map and its wall time.

    Raises:
        ValueError: If the surface does not have 2 triangles per map cell.
    """
    if surface.face_count() != 2 * MAP_CELLS * MAP_CELLS:
        raise ValueError(f"surface has {surface.face_count()} faces, expected 2 per map cell")
    start = time.perf_counter()
    radio_map = rt.RadioMapSolver()(
        scene,
        measurement_surface=surface,
        precoding_vec=precoding_vec(weights),
        samples_per_tx=settings.samples_per_tx,
        max_depth=settings.max_depth,
        los=settings.los,
        specular_reflection=settings.specular_reflection,
        diffuse_reflection=settings.diffuse_reflection,
        refraction=settings.refraction,
        diffraction=settings.diffraction,
        edge_diffraction=settings.edge_diffraction,
        seed=settings.seed,
    )
    triangles = np.asarray(radio_map.path_gain.numpy(), dtype=np.float64)[0]
    seconds = time.perf_counter() - start
    cells = triangles.reshape(MAP_CELLS, MAP_CELLS, 2).mean(axis=2)
    return MapResult(path_gain=cells, seconds=seconds)
