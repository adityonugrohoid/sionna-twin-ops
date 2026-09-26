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
