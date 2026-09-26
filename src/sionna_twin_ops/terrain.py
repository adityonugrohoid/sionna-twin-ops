"""Procedural hilly terrain (spec rule T): spectral noise plus ridged noise, from a seed.

Coordinates: the tile is centred on the origin, x points east and y north, in metres.
`heights[i, j]` is the ground height at x = coords[j], y = coords[i]. Heights run from
0 m (the lowest vertex) to the drawn relief. Synthetic terrain, not a real place.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

BASE_SEED = 20260926
TILE_SIZE_M = 10000.0  # spec T3
GRID_SPACING_M = 40.0  # START (spec T3); PR 3 tries 20 m

# Per-terrain parameter ranges (spec T2). ASSUMPTION: judged by eye on the sample grid.
RELIEF_RANGE_M = (100.0, 800.0)
BETA_RANGE = (3.2, 4.0)
RIDGE_WEIGHT_RANGE = (0.0, 1.0)

# Band of the field whose zero set carries the ridge crests. ASSUMPTION: not given by the
# spec; wavelengths of 1 to 4 km put crests roughly 0.5 to 2 km apart inside a 5.12 km map.
RIDGE_BAND_WAVELENGTH_M = (1000.0, 4000.0)


@dataclass(frozen=True)
class TerrainParams:
    """Parameters drawn for one terrain.

    Attributes:
        terrain_id: Terrain index; with BASE_SEED it fixes every draw.
        relief_m: Height difference between the highest and lowest vertex.
        beta: Power-law exponent of the base noise power spectrum.
        ridge_weight: Weight of the ridge term; 0 gives rolling hills only.
    """

    terrain_id: int
    relief_m: float
    beta: float
    ridge_weight: float


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
        ridge_weight=float(rng.uniform(*RIDGE_WEIGHT_RANGE)),
    )


def wavenumbers(n: int, spacing_m: float) -> NDArray[np.float64]:
    """Radial wavenumber of every FFT coefficient of an n x n grid.

    Args:
        n: Grid points per side.
        spacing_m: Grid spacing.

    Returns:
        Array of shape (n, n) in cycles per metre.
    """
    kx = np.fft.fftfreq(n, d=spacing_m)
    return np.hypot(kx[None, :], kx[:, None])


def shape_noise(white: NDArray[np.float64], amplitude: NDArray[np.float64]) -> NDArray[np.float64]:
    """Filter white noise by a spectral amplitude.

    Args:
        white: White Gaussian noise, shape (n, n).
        amplitude: Amplitude per FFT coefficient, shape (n, n); zero at k = 0.

    Returns:
        A zero-mean field of shape (n, n), periodic across the tile edges.
    """
    return np.real(np.fft.ifft2(np.fft.fft2(white) * amplitude))


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
    k = wavenumbers(n, spacing_m)
    amplitude = np.zeros_like(k)
    amplitude[k > 0] = k[k > 0] ** (-beta / 2.0)
    return shape_noise(rng.standard_normal((n, n)), amplitude)


def ridge_term(n: int, spacing_m: float, rng: np.random.Generator) -> NDArray[np.float64]:
    """Ridged noise (1 - |f|/max|f|)**2 of a band-limited field f.

    The term is 1 on the zero set of f and falls off to either side, so crests follow
    the zero set and branch where it does.

    Args:
        n: Grid points per side.
        spacing_m: Grid spacing, used for the wavenumbers.
        rng: Source of the white noise.

    Returns:
        Field of shape (n, n) with values in [0, 1].
    """
    k = wavenumbers(n, spacing_m)
    shortest, longest = RIDGE_BAND_WAVELENGTH_M
    band = (k >= 1.0 / longest) & (k <= 1.0 / shortest)
    if not band.any():
        raise ValueError(f"no FFT coefficient in the ridge band {RIDGE_BAND_WAVELENGTH_M} m")
    field = shape_noise(rng.standard_normal((n, n)), band.astype(np.float64))
    magnitude = np.abs(field)
    ridged: NDArray[np.float64] = (1.0 - magnitude / magnitude.max()) ** 2
    return ridged


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

    base = spectral_noise(n, spacing_m, params.beta, rng)
    base = (base - base.min()) / (base.max() - base.min())
    raw = base + params.ridge_weight * ridge_term(n, spacing_m, rng)
    heights = params.relief_m * (raw - raw.min()) / (raw.max() - raw.min())
    if not np.isfinite(heights).all():
        raise ValueError(f"terrain {terrain_id}: heightmap has non-finite values")
    return Terrain(params=params, spacing_m=spacing_m, coords_m=coords, heights_m=heights)
