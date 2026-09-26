"""Procedural hilly terrain (spec rule T): spectral noise plus ridge lines, from a seed.

Coordinates: the tile is centred on the origin, x points east and y north, in metres.
`heights[i, j]` is the ground height at x = coords[j], y = coords[i]. Heights run from
0 m (the lowest vertex) to the drawn relief. Synthetic terrain, not a real place.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

BASE_SEED = 20260926
TILE_SIZE_M = 8000.0
GRID_SPACING_M = 40.0  # START (spec T3); PR 3 tries 20 m

# Per-terrain parameter ranges (spec T2, START).
RELIEF_RANGE_M = (100.0, 800.0)
BETA_RANGE = (2.0, 3.2)
MAX_RIDGES = 2

# Ridge shape ranges. ASSUMPTION: generator choices, not given by the spec; judged by eye
# on the sample hillshade grid.
RIDGE_OFFSET_RANGE_M = (-2000.0, 2000.0)  # perpendicular distance of the ridge line from centre
RIDGE_WIDTH_RANGE_M = (200.0, 600.0)  # Gaussian sigma across the ridge
RIDGE_AMPLITUDE_RANGE = (0.5, 1.0)  # relative to the noise field scaled to [0, 1]


@dataclass(frozen=True)
class Ridge:
    """One straight ridge line with a Gaussian cross-section.

    Attributes:
        angle_rad: Direction of the ridge line, counter-clockwise from east.
        offset_m: Signed perpendicular distance of the line from the tile centre.
        width_m: Gaussian sigma of the cross-section.
        amplitude: Crest height relative to the noise field scaled to [0, 1].
    """

    angle_rad: float
    offset_m: float
    width_m: float
    amplitude: float


@dataclass(frozen=True)
class TerrainParams:
    """Parameters drawn for one terrain.

    Attributes:
        terrain_id: Terrain index; with BASE_SEED it fixes every draw.
        relief_m: Height difference between the highest and lowest vertex.
        beta: Power-law exponent of the noise power spectrum.
        ridges: Ridge lines added to the noise field.
    """

    terrain_id: int
    relief_m: float
    beta: float
    ridges: tuple[Ridge, ...]


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
    relief = float(rng.uniform(*RELIEF_RANGE_M))
    beta = float(rng.uniform(*BETA_RANGE))
    n_ridges = int(rng.integers(0, MAX_RIDGES + 1))
    ridges = tuple(
        Ridge(
            angle_rad=float(rng.uniform(0.0, np.pi)),
            offset_m=float(rng.uniform(*RIDGE_OFFSET_RANGE_M)),
            width_m=float(rng.uniform(*RIDGE_WIDTH_RANGE_M)),
            amplitude=float(rng.uniform(*RIDGE_AMPLITUDE_RANGE)),
        )
        for _ in range(n_ridges)
    )
    return TerrainParams(terrain_id=terrain_id, relief_m=relief, beta=beta, ridges=ridges)


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


def ridge_field(coords_m: NDArray[np.float64], ridges: tuple[Ridge, ...]) -> NDArray[np.float64]:
    """Sum of the Gaussian ridge cross-sections on the grid.

    Args:
        coords_m: Vertex coordinates along each axis.
        ridges: Ridge lines to raise.

    Returns:
        Field of shape (len(coords_m), len(coords_m)), indexed [y, x].
    """
    x, y = np.meshgrid(coords_m, coords_m)
    field = np.zeros_like(x)
    for ridge in ridges:
        # Signed distance from the line through the offset point along angle_rad.
        distance = -np.sin(ridge.angle_rad) * x + np.cos(ridge.angle_rad) * y - ridge.offset_m
        field += ridge.amplitude * np.exp(-0.5 * (distance / ridge.width_m) ** 2)
    return field


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

    noise = spectral_noise(n, spacing_m, params.beta, rng)
    unit = (noise - noise.min()) / (noise.max() - noise.min())
    raw = unit + ridge_field(coords, params.ridges)
    heights = params.relief_m * (raw - raw.min()) / (raw.max() - raw.min())
    return Terrain(params=params, spacing_m=spacing_m, coords_m=coords, heights_m=heights)
