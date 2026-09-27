"""Dataset selection, configurations, the resumable sweep and its summary (rules C, D)."""

import json
from collections import Counter
from pathlib import Path

import pytest

from sionna_twin_ops.dataset import (
    MANIFEST,
    OFF_GRID_TILTS_DEG,
    SPLIT_PER_CLASS,
    TERRAINS_PER_CLASS,
    configurations,
    map_name,
    read_manifest,
    select_terrains,
    selection_for,
    summary_data,
    sweep,
)
from sionna_twin_ops.reports import render
from sionna_twin_ops.site import SITE_CLASSES, NoSiteError, place_site, site_class_for
from sionna_twin_ops.terrain import generate_terrain


def test_selection_is_balanced_per_split_and_class() -> None:
    selection = select_terrains(TERRAINS_PER_CLASS)
    counts = Counter((e.split, e.site.site_class) for e in selection.terrains)
    for split, n in SPLIT_PER_CLASS:
        for site_class in SITE_CLASSES:
            assert counts[(split, site_class)] == n
    ids = [e.terrain_id for e in selection.terrains]
    assert ids == sorted(set(ids))


def test_skipped_ids_are_exactly_those_without_a_candidate() -> None:
    selection = select_terrains(TERRAINS_PER_CLASS)
    chosen = {e.terrain_id for e in selection.terrains}
    skipped = {i for i, _ in selection.skipped}
    for terrain_id in range(max(chosen) + 1):
        site_class = site_class_for(terrain_id)
        try:
            place_site(generate_terrain(terrain_id, 40.0), site_class)
        except NoSiteError:
            assert terrain_id in skipped
        else:
            assert terrain_id not in skipped


def test_splits_follow_id_order_within_each_class() -> None:
    selection = select_terrains(TERRAINS_PER_CLASS)
    order = {split: rank for rank, (split, _) in enumerate(SPLIT_PER_CLASS)}
    for site_class in SITE_CLASSES:
        members = [e for e in selection.terrains if e.site.site_class == site_class]
        ranks = [order[e.split] for e in members]
        assert ranks == sorted(ranks)


def test_selection_rejects_quotas_that_do_not_fit_the_splits() -> None:
    with pytest.raises(ValueError, match="do not fit"):
        select_terrains(TERRAINS_PER_CLASS + 1)


def test_test_terrains_get_the_off_grid_maps() -> None:
    assert len(configurations("train")) == 40
    assert len(configurations("validation")) == 40
    test = configurations("test")
    assert len(test) == 40 + 2 * len(OFF_GRID_TILTS_DEG)
    assert {t for _, t, kind in test if kind == "off-grid"} == set(OFF_GRID_TILTS_DEG)


def test_map_names_are_unique_and_sortable() -> None:
    assert map_name(3, 45.0, 6.0) == "t003_a045_t06.0.npy"
    names = {map_name(1, a, t) for a, t, _ in configurations("test")}
    assert len(names) == len(configurations("test"))


@pytest.mark.sionna
def test_sweep_resumes_and_refuses_mixed_settings(tmp_path: Path) -> None:
    entry = next(e for e in select_terrains(TERRAINS_PER_CLASS).terrains if e.split == "train")
    assert sweep(tmp_path, (entry,), 10**5, 1) == 40
    assert sweep(tmp_path, (entry,), 10**5, 1) == 0
    lines = read_manifest(tmp_path)
    assert len(lines) == 40
    assert all((tmp_path / "maps" / line["file"]).exists() for line in lines)
    assert all(line["cells_valid"] + line["cells_no_hit"] == 128 * 128 for line in lines)
    assert {line["provenance"]["variant"] for line in lines} == {"llvm_ad_mono_polarized"}

    with pytest.raises(ValueError, match="made with"):
        sweep(tmp_path, (entry,), 2 * 10**5, 1)

    report = render(summary_data(tmp_path, "v1"))
    assert "Maps in the manifest: 40 of 2454 expected; 2414 not yet traced." in report

    (tmp_path / "maps" / lines[0]["file"]).unlink()
    with pytest.raises(ValueError, match="missing"):
        sweep(tmp_path, (entry,), 10**5, 1)


def test_summary_refuses_mixed_settings(tmp_path: Path) -> None:
    line = {
        "file": map_name(1, 0.0, 0.0),
        "settings": {"samples_per_tx": 1},
        "split": "train",
        "kind": "grid",
    }
    other = dict(line, file=map_name(1, 0.0, 3.0), settings={"samples_per_tx": 2})
    (tmp_path / MANIFEST).write_text(json.dumps(line) + "\n" + json.dumps(other) + "\n")
    with pytest.raises(ValueError, match="mixes"):
        summary_data(tmp_path, "v1")


def test_v2_adds_28_training_terrains_per_class_after_the_v1_walk() -> None:
    v1 = selection_for("v1")
    v2 = selection_for("v2")
    new = [e for e in v2.terrains if e not in v1.terrains]
    assert len(new) == 84
    assert all(e.split == "train" for e in new)
    counts = Counter(e.site.site_class for e in new)
    assert counts == {c: 28 for c in SITE_CLASSES}
    last_v1 = max([e.terrain_id for e in v1.terrains] + [i for i, _ in v1.skipped])
    assert min(e.terrain_id for e in new) > last_v1
    assert set(v1.terrains) <= set(v2.terrains)
    assert [e for e in v2.terrains if e.split != "train"] == [
        e for e in v1.terrains if e.split != "train"
    ]
    assert set(v1.skipped) <= set(v2.skipped)
