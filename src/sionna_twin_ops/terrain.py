"""Procedural hilly terrain (spec rule T): power-law spectral noise, from a seed.

Coordinates: the tile is centred on the origin, x points east and y north, in metres.
`heights[i, j]` is the ground height at x = coords[j], y = coords[i]. Heights run from
0 m (the lowest vertex) to the drawn relief. Synthetic terrain, not a real place.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

BASE_SEED = 20260926
TILE_SIZE_M = 10000.0  # spec T3
GRID_SPACING_M = 40.0  # spec T3

# Per-terrain parameter ranges (spec T2). ASSUMPTION: judged by eye on the sample grid.
RELIEF_RANGE_M = (100.0, 800.0)
BETA_RANGE = (3.4, 4.0)


@dataclass(frozen=True)
class TerrainParams:
    """Parameters drawn for one terrain.

    Attributes:
        terrain_id: Terrain index; with BASE_SEED it fixes every draw.
        relief_m: Height difference between the highest and lowest vertex.
        beta: Power-law exponent of the noise power spectrum.
    """

    terrain_id: int
    relief_m: float
    beta: float


@dataclass(frozen=True)
class Terrain:
    """A heightmap on a regular grid.

    Attributes:
        params: The parameters it was generated from.
        spacing_m: Grid spacing.
        coords_m: Vertex coordinates along x and along y (the grid is square).
        heights_m: Ground heights, shape (len(coords_m), len(coords_m)), indexed [y, x].
    """

    params: TerrainParams
    spacing_m: float
    coords_m: NDArray[np.float64]
    heights_m: NDArray[np.float64]


def draw_params(terrain_id: int, rng: np.random.Generator) -> TerrainParams:
    """Draw the parameters of one terrain, in a fixed order.

    Args:
        terrain_id: Terrain index, stored in the result.
        rng: Generator seeded from (BASE_SEED, terrain_id).

    Returns:
        The drawn parameters.
    """
    return TerrainParams(
        terrain_id=terrain_id,
        relief_m=float(rng.uniform(*RELIEF_RANGE_M)),
        beta=float(rng.uniform(*BETA_RANGE)),
    )


def spectral_noise(
    n: int, spacing_m: float, beta: float, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Gaussian noise shaped to a power spectrum proportional to k**-beta.

    Args:
        n: Grid points per side.
        spacing_m: Grid spacing, used for the wavenumbers.
        beta: Power-law exponent of the power spectrum.
        rng: Source of the white noise.

    Returns:
        A zero-mean field of shape (n, n), periodic across the tile edges.
    """
    white = rng.standard_normal((n, n))
    kx = np.fft.fftfreq(n, d=spacing_m)
    k = np.hypot(kx[None, :], kx[:, None])
    amplitude = np.zeros_like(k)
    amplitude[k > 0] = k[k > 0] ** (-beta / 2.0)
    return np.real(np.fft.ifft2(np.fft.fft2(white) * amplitude))


def generate_terrain(terrain_id: int, spacing_m: float) -> Terrain:
    """Generate terrain `terrain_id` deterministically (spec T1, T2, T3).

    Args:
        terrain_id: Terrain index; non-negative.
        spacing_m: Grid spacing; must divide TILE_SIZE_M.

    Returns:
        The terrain, identical byte for byte for the same id and spacing.

    Raises:
        ValueError: If terrain_id is negative or spacing_m does not divide the tile.
    """
    if terrain_id < 0:
        raise ValueError(f"terrain_id must be non-negative, got {terrain_id}")
    cells = TILE_SIZE_M / spacing_m
    if cells != round(cells):
        raise ValueError(f"spacing {spacing_m} m does not divide the {TILE_SIZE_M} m tile")
    n = round(cells) + 1

    rng = np.random.default_rng([BASE_SEED, terrain_id])
    params = draw_params(terrain_id, rng)
    coords = np.linspace(-TILE_SIZE_M / 2.0, TILE_SIZE_M / 2.0, n)

    raw = spectral_noise(n, spacing_m, params.beta, rng)
    heights = params.relief_m * (raw - raw.min()) / (raw.max() - raw.min())
    if not np.isfinite(heights).all():
        raise ValueError(f"terrain {terrain_id}: heightmap has non-finite values")
    return Terrain(params=params, spacing_m=spacing_m, coords_m=coords, heights_m=heights)
