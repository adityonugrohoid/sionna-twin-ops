"""Features, the U-Net and one training step (spec M1, M1b, M2)."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from sionna_twin_ops.dataset import MANIFEST, map_name, select_terrains
from sionna_twin_ops.features import INPUT_CHANNELS, map_inputs, targets, terrain_features
from sionna_twin_ops.model import UNet, parameter_count
from sionna_twin_ops.reports import render
from sionna_twin_ops.terrain import generate_terrain
from sionna_twin_ops.train import losses, train, training_summary_data


def test_unet_is_under_ten_million_parameters_with_two_heads() -> None:
    model = UNet(32)
    assert parameter_count(model) < 10_000_000
    assert model(torch.zeros(1, INPUT_CHANNELS, 128, 128)).shape == (1, 2, 128, 128)


def test_targets_mark_power_and_leave_no_residual_without_it() -> None:
    gain = np.array([[1e-10, 0.0]], dtype=np.float32)
    b0 = np.array([[-95.0, -90.0]], dtype=np.float32)
    residual, power = targets(gain, b0)
    assert power.tolist() == [[1.0, 0.0]]
    assert residual[0, 0] == pytest.approx(-100.0 + 95.0, abs=1e-4)
    assert residual[0, 1] == 0.0


def test_path_gain_loss_ignores_cells_without_power() -> None:
    output = torch.zeros(1, 2, 2, 2)
    residual = torch.tensor([[[1.0, 50.0], [1.0, 50.0]]])
    power = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    l1, _, _ = losses(output, residual, power)
    assert float(l1) == pytest.approx(1.0)


def test_inputs_have_every_channel_and_finite_values() -> None:
    entry = select_terrains(20).terrains[0]
    terrain = generate_terrain(entry.terrain_id, 40.0)
    x, b0 = map_inputs(terrain_features(terrain, entry.site), 90.0, 6.0)
    assert x.shape == (INPUT_CHANNELS, 128, 128)
    assert np.isfinite(x).all()
    assert np.isfinite(b0).all()


def fake_dataset(path: Path) -> None:
    """Two terrains (one train, one validation), two maps each, random path gain."""
    rng = np.random.default_rng(0)
    selection = select_terrains(20).terrains
    entries = [next(e for e in selection if e.split == s) for s in ("train", "validation")]
    (path / "maps").mkdir(parents=True)
    lines = []
    for entry in entries:
        for tilt in (0.0, 6.0):
            name = map_name(entry.terrain_id, 90.0, tilt)
            gain = rng.uniform(1e-13, 1e-9, (128, 128)).astype(np.float32)
            gain[:40] = 0.0
            np.save(path / "maps" / name, gain)
            lines.append(
                {
                    "file": name,
                    "terrain_id": entry.terrain_id,
                    "split": entry.split,
                    "site": entry.site.__dict__,
                    "azimuth_deg": 90.0,
                    "tilt_deg": tilt,
                    "kind": "grid",
                    "settings": {},
                }
            )
    (path / MANIFEST).write_text("".join(json.dumps(line) + "\n" for line in lines))


def test_training_writes_weights_and_a_record(tmp_path: Path) -> None:
    fake_dataset(tmp_path / "data")
    record = train(tmp_path / "data", 0, 2, 8, "symmetry", "cpu", tmp_path / "run")
    assert (tmp_path / "run" / "model.pt").exists()
    assert record["maps"] == {"train": 2, "validation": 2}
    assert record["best"]["epoch"] in (1, 2)
    assert len(record["history"]) == 2
    report = render(training_summary_data([tmp_path / "run"], [tmp_path / "run"]))
    assert "| 0 | " in report
    assert "Context: earlier runs, quoted" in report
    assert record["hyperparameters"]["augmentation"] == "symmetry"
    assert "l1_nlos_db" in record["best"]
    assert "l1_los_direct_db" in record["best"]
    assert record["storage"] == {"train": "float16", "validation": "float32"}
    assert record["peak_rss_mib_after_load"] > 0


@pytest.mark.sionna
def test_float16_storage_keeps_a_real_target_within_0_05_db() -> None:
    from sionna_twin_ops.antenna import tilt_weights
    from sionna_twin_ops.scene import DATASET_FOLD, build_scene, measurement_surface
    from sionna_twin_ops.solve import solve_map, specular_settings

    entry = select_terrains(20).terrains[0]
    terrain = generate_terrain(entry.terrain_id, 40.0)
    x, b0 = map_inputs(terrain_features(terrain, entry.site), 90.0, 6.0)
    scene = build_scene(terrain, entry.site, 90.0, DATASET_FOLD)
    surface = measurement_surface(terrain, entry.site, DATASET_FOLD)
    gain = solve_map(scene, surface, tilt_weights(6.0), specular_settings(10**6, 1)).path_gain
    residual, power = targets(gain.astype(np.float32), b0)
    lit = power > 0
    assert lit.any()
    round_trip = residual.astype(np.float16).astype(np.float32)
    assert np.abs(round_trip - residual)[lit].max() < 0.05
    stored = x.astype(np.float16)
    assert np.isfinite(stored).all()
    assert np.abs(stored).max() < np.finfo(np.float16).max
    assert np.array_equal(power.astype(np.float16).astype(np.float32), power)


@pytest.mark.sionna
def test_per_terrain_loader_equals_the_cache_all_loader(tmp_path: Path) -> None:
    """The per-terrain feature loader (9e81212) must give the arrays the earlier
    cache-every-terrain loader gave; the latter is rebuilt here as the reference."""
    from sionna_twin_ops.antenna import tilt_weights
    from sionna_twin_ops.scene import DATASET_FOLD, build_scene, measurement_surface
    from sionna_twin_ops.solve import solve_map, specular_settings
    from sionna_twin_ops.train import load_split

    selection = select_terrains(20).terrains
    entries = [e for e in selection if e.split == "train"][:3]
    (tmp_path / "maps").mkdir()
    lines: list[dict[str, Any]] = []
    for entry in entries:
        terrain = generate_terrain(entry.terrain_id, 40.0)
        for azimuth, tilt in ((90.0, 6.0), (225.0, 0.0)):
            scene = build_scene(terrain, entry.site, azimuth, DATASET_FOLD)
            surface = measurement_surface(terrain, entry.site, DATASET_FOLD)
            gain = solve_map(scene, surface, tilt_weights(tilt), specular_settings(10**5, 1))
            name = map_name(entry.terrain_id, azimuth, tilt)
            np.save(tmp_path / "maps" / name, gain.path_gain.astype(np.float32))
            lines.append(
                {
                    "file": name,
                    "terrain_id": entry.terrain_id,
                    "split": "train",
                    "site": entry.site.__dict__,
                    "azimuth_deg": azimuth,
                    "tilt_deg": tilt,
                    "kind": "grid",
                    "settings": {},
                }
            )
    # Written in reverse so the loader's own sorting is exercised.
    (tmp_path / MANIFEST).write_text("".join(json.dumps(line) + "\n" for line in lines[::-1]))

    cache = {
        e.terrain_id: terrain_features(generate_terrain(e.terrain_id, 40.0), e.site)
        for e in entries
    }
    reference = sorted(lines, key=lambda line: line["file"])
    for storage in (np.float32, np.float16):
        loaded = load_split(tmp_path, "train", storage)
        assert loaded.files == tuple(line["file"] for line in reference)
        for i, line in enumerate(reference):
            x, b0 = map_inputs(cache[line["terrain_id"]], line["azimuth_deg"], line["tilt_deg"])
            residual, power = targets(np.load(tmp_path / "maps" / line["file"]), b0)
            assert np.array_equal(loaded.inputs[i].numpy(), x.astype(storage))
            assert np.array_equal(loaded.residual[i].numpy(), residual.astype(storage))
            assert np.array_equal(loaded.power[i].numpy(), power.astype(storage))
