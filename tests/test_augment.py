"""The 8 exact map symmetries: transformed features equal recomputed features."""

import numpy as np
import pytest
import torch

from sionna_twin_ops.augment import (
    ALL_VARIANTS,
    VARIANTS,
    centred_crop,
    keeps_fold,
    transform_azimuth,
    transform_inputs,
    transform_raster,
    transform_terrain,
)
from sionna_twin_ops.dataset import select_terrains
from sionna_twin_ops.features import map_inputs, terrain_features
from sionna_twin_ops.site import Site
from sionna_twin_ops.terrain import Terrain, generate_terrain
from sionna_twin_ops.train import transform_batch

AZIMUTH = 45.0
TILT = 6.0


@pytest.fixture(scope="module")
def crop_and_inputs() -> tuple[Terrain, Site, np.ndarray]:
    entry = select_terrains(20).terrains[0]
    terrain = generate_terrain(entry.terrain_id, 40.0)
    crop, site = centred_crop(terrain, entry.site)
    inputs, _ = map_inputs(terrain_features(crop, site), AZIMUTH, TILT)
    return crop, site, inputs


def test_crop_reproduces_the_full_tile_features() -> None:
    entry = select_terrains(20).terrains[0]
    terrain = generate_terrain(entry.terrain_id, 40.0)
    crop, site = centred_crop(terrain, entry.site)
    full, _ = map_inputs(terrain_features(terrain, entry.site), AZIMUTH, TILT)
    cropped, _ = map_inputs(terrain_features(crop, site), AZIMUTH, TILT)
    np.testing.assert_allclose(cropped, full, atol=1e-5)


@pytest.mark.parametrize(("k", "mirror"), ALL_VARIANTS)
def test_transformed_features_equal_recomputed_features(
    crop_and_inputs: tuple[Terrain, Site, np.ndarray], k: int, mirror: bool
) -> None:
    crop, site, inputs = crop_and_inputs
    moved = transform_terrain(crop, k, mirror)
    recomputed, _ = map_inputs(
        terrain_features(moved, site),
        transform_azimuth(AZIMUTH, k, mirror),
        TILT,
    )
    np.testing.assert_allclose(transform_inputs(inputs, k, mirror), recomputed, atol=1e-5)


@pytest.mark.parametrize(("k", "mirror"), ALL_VARIANTS)
def test_torch_batch_transform_matches_numpy(
    crop_and_inputs: tuple[Terrain, Site, np.ndarray], k: int, mirror: bool
) -> None:
    _, _, inputs = crop_and_inputs
    raster = inputs[0]
    x, (r,) = transform_batch(
        torch.from_numpy(inputs[None].copy()), [torch.from_numpy(raster[None].copy())], k, mirror
    )
    np.testing.assert_array_equal(x[0].numpy(), transform_inputs(inputs, k, mirror))
    np.testing.assert_array_equal(r[0].numpy(), transform_raster(raster, k, mirror))


def test_azimuth_transform_is_a_group_action() -> None:
    assert transform_azimuth(AZIMUTH, 0, False) == AZIMUTH
    assert transform_azimuth(AZIMUTH, 1, False) == AZIMUTH + 90.0
    assert transform_azimuth(transform_azimuth(AZIMUTH, 0, True), 0, True) == AZIMUTH
    assert transform_azimuth(AZIMUTH, 4, False) == AZIMUTH
    assert len(set(ALL_VARIANTS)) == 8


def test_training_variants_are_exactly_the_fold_keeping_ones() -> None:
    # One cell's SW-NE fold on a 2 x 2 grid (row 0 south, column 0 west): its south-west
    # corner (0, 0) and its north-east corner (1, 1).
    fold = np.eye(2)
    kept = {v for v in ALL_VARIANTS if np.array_equal(transform_raster(fold, *v), fold)}
    assert kept == set(VARIANTS)
    assert all(keeps_fold(*v) == (v in VARIANTS) for v in ALL_VARIANTS)
