"""The mesh fold and distance to shadow (spec N3b)."""

import numpy as np
import pytest

from sionna_twin_ops.crosscheck import distance_to_shadow
from sionna_twin_ops.scene import DATASET_FOLD, Fold, grid_faces


def test_distance_to_shadow_matches_brute_force() -> None:
    rng = np.random.default_rng(0)
    los = rng.random((20, 20)) > 0.3
    distance = distance_to_shadow(los)
    shadow = np.argwhere(~los)
    for i, j in np.argwhere(los):
        expected = np.sqrt(((shadow - [i, j]) ** 2).sum(axis=1)).min()
        assert distance[i, j] == pytest.approx(expected)
    assert np.all(distance[~los] == 0.0)


def test_distance_to_shadow_edge_cases() -> None:
    assert np.all(np.isinf(distance_to_shadow(np.ones((3, 3), dtype=bool))))
    los = np.ones((1, 5), dtype=bool)
    los[0, 0] = False
    assert distance_to_shadow(los).tolist() == [[0.0, 1.0, 2.0, 3.0, 4.0]]


@pytest.mark.parametrize(("fold", "diagonal"), [("sw-ne", {0, 5}), ("nw-se", {1, 4})])
def test_each_fold_splits_cells_along_its_diagonal(fold: Fold, diagonal: set[int]) -> None:
    faces = grid_faces(3, 4, fold)
    # Cell (0, 0): vertices 0 (SW), 1 (SE), 4 (NW), 5 (NE); faces 0 and 1.
    assert set(faces[0]) | set(faces[1]) == {0, 1, 4, 5}
    assert set(faces[0]) & set(faces[1]) == diagonal


@pytest.mark.parametrize("fold", ["sw-ne", "nw-se"])
def test_every_triangle_is_counter_clockwise_from_above(fold: Fold) -> None:
    rows, cols = 3, 4
    xs, ys = np.meshgrid(np.arange(cols), np.arange(rows))
    xy = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(float)
    for a, b, c in grid_faces(rows, cols, fold):
        cross = (xy[b] - xy[a])[0] * (xy[c] - xy[a])[1] - (xy[b] - xy[a])[1] * (xy[c] - xy[a])[0]
        assert cross > 0


def test_the_dataset_uses_the_sw_ne_fold() -> None:
    assert DATASET_FOLD == "sw-ne"
