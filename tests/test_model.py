"""Features, the U-Net and one training step (spec M1, M1b, M2)."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sionna_twin_ops.dataset import MANIFEST, map_name, select_terrains
from sionna_twin_ops.features import INPUT_CHANNELS, map_inputs, targets, terrain_features
from sionna_twin_ops.model import UNet, parameter_count
from sionna_twin_ops.terrain import generate_terrain
from sionna_twin_ops.train import losses, train, training_summary_markdown


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
    report = training_summary_markdown([tmp_path / "run"])
    assert "| 0 | " in report
    assert record["hyperparameters"]["augmentation"] == "symmetry"
    assert "l1_nlos_db" in record["best"]
