"""Variant selection is explicit, recorded and never automatic (repo rule 8)."""

import subprocess
import sys

import pytest

from sionna_twin_ops.backend import DEFAULT_VARIANT, active_variant, select_variant
from sionna_twin_ops.solve import specular_settings


def test_sionna_import_refuses_without_a_selected_variant() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import sionna_twin_ops.sionna_rt"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "no Mitsuba variant selected" in result.stderr


def test_switching_variant_in_a_process_raises() -> None:
    assert active_variant() == DEFAULT_VARIANT
    with pytest.raises(RuntimeError, match="already set"):
        select_variant("scalar_rgb")


def test_unknown_variant_raises() -> None:
    with pytest.raises(ValueError, match="not installed"):
        select_variant("no_such_variant")


def test_settings_record_carries_the_variant() -> None:
    record = specular_settings(10**7, 1).record()
    assert record["variant"] == DEFAULT_VARIANT
    assert record["diffraction"] is False
    assert record["diffuse_reflection"] is False
    assert record["refraction"] is False
    assert record["max_depth"] == 3


def test_sample_counts_beyond_the_32_bit_wavefront_are_refused() -> None:
    from sionna_twin_ops.solve import MAX_SAMPLES_PER_TX

    assert specular_settings(MAX_SAMPLES_PER_TX, 1).samples_per_tx == 2**32 - 1
    with pytest.raises(ValueError, match="32-bit"):
        specular_settings(10**10, 1)
    with pytest.raises(ValueError, match="32-bit"):
        specular_settings(0, 1)


def test_solver_check_tables_have_consistent_columns() -> None:
    from sionna_twin_ops import checks

    d = checks.DiffStats(0.1, 0.2, 0.0, 1.0, 5)
    floor = checks.FloorRow(
        3, "hilltop", (0.1,) * len(checks.FLOOR_SAMPLES), d, d, d, d, d, (0.6, 0.5), 1, 0
    )
    tilts = [checks.TiltRow(t, -t, -90.0, 1.0) for t in checks.TILTS_DEG]
    report = checks.solver_check_markdown(
        tilts, checks.SurfaceCheck(d, d, d), [floor], specular_settings(10**7, 1), 10**8, "header"
    )
    for block in report.split("\n\n"):
        rows = [line for line in block.splitlines() if line.startswith("|")]
        assert len({row.count("|") for row in rows}) <= 1, block
