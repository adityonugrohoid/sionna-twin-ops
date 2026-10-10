"""The 3D viewer's packing and page (spec E7)."""

import base64
import json
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sionna_twin_ops.viewer import (
    NO_POWER,
    PLOTLY_URL,
    pack_bits,
    pack_int16,
    unpack_bits,
    unpack_int16,
    viewer_html,
)


def test_int16_packing_round_trips_at_a_tenth_with_the_no_power_sentinel() -> None:
    values = np.array([[-123.44, np.nan], [0.05, 3276.7]])
    raw = zlib.decompress(base64.b64decode(pack_int16(values)))
    out = np.frombuffer(raw, dtype="<i2")
    assert out.tolist() == [-1234, NO_POWER, 0, 32767]
    with pytest.raises(ValueError, match="int16 range"):
        pack_int16(np.array([3276.8]))


def test_bit_packing_puts_the_first_cell_in_the_high_bit() -> None:
    mask = np.zeros(16, dtype=bool)
    mask[[0, 9]] = True
    raw = zlib.decompress(base64.b64decode(pack_bits(mask)))
    assert list(raw) == [0b10000000, 0b01000000]


def test_unpacking_inverts_packing_with_nan_for_no_power() -> None:
    values = np.array([-123.4, np.nan, 0.0, 3276.7])
    out = unpack_int16(pack_int16(values))
    assert np.isnan(out[1]) and out[[0, 2, 3]].tolist() == [-123.4, 0.0, 3276.7]
    mask = np.zeros(11, dtype=bool)
    mask[[0, 4, 10]] = True
    assert unpack_bits(pack_bits(mask), 11).tolist() == mask.tolist()


def test_the_still_draws_one_case_from_the_packed_data(tmp_path: Path) -> None:
    from sionna_twin_ops.figures import viewer_still

    n = 6
    ground = np.linspace(100.0, 160.0, n * n)
    traced = np.where(np.arange(n * n) % 3 == 0, np.nan, -90.0 - np.arange(n * n))
    data: dict[str, Any] = {
        "terrains": {
            "7": {
                "ground": pack_int16(ground),
                "site_class": "slope",
                "relief_m": 60,
                "site_ground_m": 130.0,
                "antenna_m": 160.0,
                "percentile": 50,
            }
        },
        "cases": [
            {
                "terrain": 7,
                "azimuth": 90.0,
                "tilt": 6.0,
                "traced": pack_int16(traced),
                "predicted": pack_int16(traced),
                "says_power": pack_bits(~np.isnan(traced)),
                "false_power": 0,
                "missed_power": 0,
            }
        ],
        "seed": 0,
        "ray_tracing": ["test settings"],
    }
    out = tmp_path / "still.jpg"
    viewer_still(data, 0, 2.0, "a caption", out)
    assert out.stat().st_size > 0
    data["cases"][0]["traced"] = pack_int16(traced[:-1])
    with pytest.raises(ValueError, match="one square grid"):
        viewer_still(data, 0, 2.0, "a caption", tmp_path / "bad.jpg")


def test_the_page_fills_every_placeholder_and_escapes_the_caption() -> None:
    data = {"terrains": {}, "cases": [], "seed": 0, "ray_tracing": []}
    page = viewer_html(data, "a < b & c")
    assert "__" not in page
    assert PLOTLY_URL in page and "@2.35.2/" in page
    assert "a &lt; b &amp; c" in page
    assert json.dumps(data, separators=(",", ":")) in page


def test_cases_cover_the_split_then_the_figure_terrains_at_other_azimuths(
    tmp_path: Path,
) -> None:
    from sionna_twin_ops.viewer import viewer_cases

    lines = [
        {"split": "test", "terrain_id": t, "site": {"site_class": c}, "azimuth_deg": a}
        for t, c in ((5, "hilltop"), (2, "slope"), (8, "valley"), (3, "hilltop"))
        for a in (0.0, 90.0)
    ]
    lines.append({"split": "train", "terrain_id": 1, "site": {"site_class": "slope"}})
    (tmp_path / "manifest.jsonl").write_text("".join(json.dumps(ln) + "\n" for ln in lines))
    cases = viewer_cases(tmp_path, "test", 90.0, 6.0)
    assert cases[:4] == [(2, 90.0, 6.0), (3, 90.0, 6.0), (5, 90.0, 6.0), (8, 90.0, 6.0)]
    extra = cases[4:]
    assert {t for t, _, _ in extra} == {3, 2, 8}
    assert all(a != 90.0 and tilt == 6.0 for _, a, tilt in extra)
    assert len(extra) == 3 * 7
