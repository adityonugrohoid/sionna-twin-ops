"""Terrain generator (spec rule T)."""

import numpy as np
import pytest

from sionna_twin_ops.terrain import (
    BASE_SEED,
    BETA_RANGE,
    MAX_RIDGES,
    RELIEF_RANGE_M,
    generate_terrain,
    spectral_noise,
)


def test_same_id_is_identical_byte_for_byte() -> None:
    a = generate_terrain(7, 40.0)
    b = generate_terrain(7, 40.0)
    assert a.heights_m.tobytes() == b.heights_m.tobytes()
    assert a.params == b.params


def test_global_random_state_has_no_effect() -> None:
    np.random.seed(1)
    a = generate_terrain(3, 40.0)
    np.random.seed(2)
    b = generate_terrain(3, 40.0)
    assert a.heights_m.tobytes() == b.heights_m.tobytes()


def test_different_ids_differ() -> None:
    assert not np.array_equal(
        generate_terrain(0, 40.0).heights_m, generate_terrain(1, 40.0).heights_m
    )


@pytest.mark.parametrize(("spacing", "n"), [(40.0, 201), (20.0, 401)])
def test_grid_covers_the_tile(spacing: float, n: int) -> None:
    terrain = generate_terrain(0, spacing)
    assert terrain.heights_m.shape == (n, n)
    assert terrain.coords_m[0] == -4000.0
    assert terrain.coords_m[-1] == 4000.0
    assert np.allclose(np.diff(terrain.coords_m), spacing)


def test_parameters_and_heights_stay_in_range() -> None:
    ridge_counts = set()
    for terrain_id in range(60):
        terrain = generate_terrain(terrain_id, 40.0)
        p = terrain.params
        assert RELIEF_RANGE_M[0] <= p.relief_m <= RELIEF_RANGE_M[1]
        assert BETA_RANGE[0] <= p.beta <= BETA_RANGE[1]
        assert 0 <= len(p.ridges) <= MAX_RIDGES
        ridge_counts.add(len(p.ridges))
        assert terrain.heights_m.min() == 0.0
        assert terrain.heights_m.max() == pytest.approx(p.relief_m)
    assert ridge_counts == {0, 1, 2}


@pytest.mark.parametrize("beta", [2.0, 2.6, 3.2])
def test_noise_spectrum_follows_the_power_law(beta: float) -> None:
    n, spacing = 256, 40.0
    rng = np.random.default_rng([BASE_SEED, 999])
    field = spectral_noise(n, spacing, beta, rng)
    power = np.abs(np.fft.fft2(field)) ** 2
    kx = np.fft.fftfreq(n, d=spacing)
    k = np.hypot(kx[None, :], kx[:, None])
    # Log-log fit over every nonzero coefficient: log of the white-noise power has a
    # constant mean, so the fitted slope is the spectral exponent.
    nonzero = k > 0
    slope = np.polyfit(np.log(k[nonzero]), np.log(power[nonzero]), 1)[0]
    assert slope == pytest.approx(-beta, abs=0.05)


def test_invalid_arguments_raise() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        generate_terrain(-1, 40.0)
    with pytest.raises(ValueError, match="does not divide"):
        generate_terrain(0, 30.0)
