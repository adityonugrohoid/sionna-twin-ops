"""Provenance and the cross-backend comparison (spec N7, repo rules 2 and 8)."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from sionna_twin_ops.backend import DEFAULT_VARIANT
from sionna_twin_ops.crosscheck import META, TERRAIN_IDS, compare_markdown
from sionna_twin_ops.provenance import commit, provenance


def test_commit_prefers_the_runner_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TWIN_COMMIT", "abc123")
    assert commit() == "abc123"


def test_provenance_names_the_variant_and_the_gpu_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TWIN_COMMIT", "abc123")  # hermetic: no git checkout needed
    record = provenance()
    assert record["variant"] == DEFAULT_VARIANT
    assert record["gpu"]  # a GPU description, or an explicit "unknown: ..." statement
    assert record["commit"] == "abc123"


def fake_run(path: Path, variant: str, scale: float) -> None:
    rng = np.random.default_rng(0)
    path.mkdir(parents=True)
    rows = []
    for terrain_id in TERRAIN_IDS:
        gain = rng.uniform(1e-12, 1e-9, (128, 128)) * scale
        gain[:10] = 0.0
        np.save(path / f"t{terrain_id}_{10**7:.0e}.npy", gain)
        rows.append(
            {
                "terrain_id": terrain_id,
                "samples": 10**7,
                "cold_s": 1.0,
                "warm_s": 0.5,
                "repeat_identical": True,
                "gpu_memory_mib": None,
            }
        )
    meta = {
        "provenance": {"commit": "x", "variant": variant},
        "settings": {"variant": variant, "samples_per_tx": [10**7], "seed": 1},
        "azimuth_deg": 90.0,
        "tilt_deg": 6.0,
        "rows": rows,
    }
    (path / META).write_text(json.dumps(meta))


def test_identical_runs_agree_exactly(tmp_path: Path) -> None:
    fake_run(tmp_path / "a", "llvm", 1.0)
    fake_run(tmp_path / "b", "cuda", 1.0)
    report = compare_markdown(tmp_path / "a", tmp_path / "b")
    assert "| 3 | 1e+07 | 0.000 | 0.000 | 0.000 |" in report
    assert "llvm" in report and "cuda" in report


def test_a_uniform_2x_gain_reads_as_3_db(tmp_path: Path) -> None:
    fake_run(tmp_path / "a", "llvm", 1.0)
    fake_run(tmp_path / "b", "cuda", 2.0)
    report = compare_markdown(tmp_path / "a", tmp_path / "b")
    assert "| 3 | 1e+07 | 3.010 | 3.010 | 3.010 |" in report


def test_runs_with_different_settings_are_refused(tmp_path: Path) -> None:
    fake_run(tmp_path / "a", "llvm", 1.0)
    shutil.copytree(tmp_path / "a", tmp_path / "b")
    meta = json.loads((tmp_path / "b" / META).read_text())
    meta["settings"]["seed"] = 2
    (tmp_path / "b" / META).write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="differ"):
        compare_markdown(tmp_path / "a", tmp_path / "b")
