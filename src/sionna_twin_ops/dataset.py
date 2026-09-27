"""The dataset (spec rules C and D): terrain selection, splits, the resumable sweep, and
its summary.

Selection walks terrain ids upward, gives each id its own site class (spec S2), skips and
records ids with no site candidate, and stops when every class has TERRAINS_PER_CLASS
terrains; splits are taken within each class by id order (spec D2). Each map is saved
as float32 linear path gain in <dataset>/maps and gets one line in <dataset>/manifest.jsonl
once its file is in place, so an interrupted sweep resumes where it stopped (spec D3).
Synthetic terrain.
"""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sionna_twin_ops.site import (
    SITE_CLASSES,
    NoSiteError,
    Site,
    SiteClass,
    place_site,
    site_class_for,
)
from sionna_twin_ops.terrain import GRID_SPACING_M, generate_terrain

AZIMUTHS_DEG = tuple(float(a) for a in range(0, 360, 45))  # spec C2
TILTS_DEG = (0.0, 3.0, 6.0, 9.0, 12.0)  # spec C2
OFF_GRID_TILTS_DEG = (1.5, 4.5, 7.5)  # spec C3, test terrains only
OFF_GRID_AZIMUTHS_DEG = (0.0, 180.0)  # ASSUMPTION: spec C3 asks for two, without naming them
TERRAINS_PER_CLASS = 20  # spec D1: 60 terrains
SPLIT_PER_CLASS = (("train", 14), ("validation", 3), ("test", 3))  # spec D2: 42 / 9 / 9
V2_EXTRA_TRAIN_PER_CLASS = 28  # spec D5: +84 training terrains
SOLVER_SEED = 1  # the ray lattice does not depend on it; recorded all the same
MANIFEST = "manifest.jsonl"


@dataclass(frozen=True)
class TerrainEntry:
    """One selected terrain.

    Attributes:
        terrain_id: Terrain id.
        split: "train", "validation" or "test".
        site: Its site.
    """

    terrain_id: int
    split: str
    site: Site


@dataclass(frozen=True)
class Selection:
    """The terrains of the dataset and the ids skipped on the way.

    Attributes:
        terrains: Selected terrains, in id order.
        skipped: (terrain id, site class) of each id with no site candidate.
    """

    terrains: tuple[TerrainEntry, ...]
    skipped: tuple[tuple[int, SiteClass], ...]


def select_terrains(per_class: int) -> Selection:
    """Walk ids upward until every site class has `per_class` terrains (spec S2, D2).

    Args:
        per_class: Terrains wanted per class; must equal the sum of SPLIT_PER_CLASS.

    Returns:
        The selection.

    Raises:
        ValueError: If per_class does not match the split sizes.
    """
    if per_class != sum(n for _, n in SPLIT_PER_CLASS):
        raise ValueError(f"{per_class} terrains per class do not fit the splits {SPLIT_PER_CLASS}")
    found: dict[SiteClass, list[tuple[int, Site]]] = {c: [] for c in SITE_CLASSES}
    skipped: list[tuple[int, SiteClass]] = []
    terrain_id = 0
    while any(len(found[c]) < per_class for c in SITE_CLASSES):
        site_class = site_class_for(terrain_id)
        if len(found[site_class]) < per_class:
            try:
                site = place_site(generate_terrain(terrain_id, GRID_SPACING_M), site_class)
            except NoSiteError:
                skipped.append((terrain_id, site_class))
            else:
                found[site_class].append((terrain_id, site))
        terrain_id += 1
    entries = []
    for site_class in SITE_CLASSES:
        members = iter(found[site_class])
        for split, count in SPLIT_PER_CLASS:
            entries += [TerrainEntry(i, split, s) for i, s in (next(members) for _ in range(count))]
    return Selection(
        terrains=tuple(sorted(entries, key=lambda e: e.terrain_id)), skipped=tuple(skipped)
    )


def extend_selection(base: Selection, extra_per_class: int) -> Selection:
    """Continue the id walk after the base selection's last id, adding training terrains.

    Ids are walked from one past the highest id the base walk reached; each id keeps its
    own site class, ids with no site candidate are skipped and recorded, and the walk
    stops when every class has `extra_per_class` new terrains (spec D5). All new terrains
    go to the training split; validation and test are the base selection's.

    Args:
        base: The v1 selection.
        extra_per_class: New training terrains per class.

    Returns:
        The base terrains plus the new ones, and all skipped ids.
    """
    walked = max([e.terrain_id for e in base.terrains] + [i for i, _ in base.skipped])
    added: dict[SiteClass, list[TerrainEntry]] = {c: [] for c in SITE_CLASSES}
    skipped = list(base.skipped)
    terrain_id = walked + 1
    while any(len(added[c]) < extra_per_class for c in SITE_CLASSES):
        site_class = site_class_for(terrain_id)
        if len(added[site_class]) < extra_per_class:
            try:
                site = place_site(generate_terrain(terrain_id, GRID_SPACING_M), site_class)
            except NoSiteError:
                skipped.append((terrain_id, site_class))
            else:
                added[site_class].append(TerrainEntry(terrain_id, "train", site))
        terrain_id += 1
    new = [e for c in SITE_CLASSES for e in added[c]]
    return Selection(
        terrains=tuple(sorted(base.terrains + tuple(new), key=lambda e: e.terrain_id)),
        skipped=tuple(skipped),
    )


def selection_for(version: str) -> Selection:
    """The terrains of a dataset version.

    Args:
        version: "v1" (spec D1, D2) or "v2" (v1 plus spec D5).

    Returns:
        The selection.

    Raises:
        ValueError: For an unknown version.
    """
    v1 = select_terrains(TERRAINS_PER_CLASS)
    if version == "v1":
        return v1
    if version == "v2":
        return extend_selection(v1, V2_EXTRA_TRAIN_PER_CLASS)
    raise ValueError(f"unknown dataset version {version!r}")


def configurations(split: str) -> list[tuple[float, float, str]]:
    """(azimuth, tilt, kind) of every map traced for a terrain in `split` (spec C2, C3).

    Args:
        split: The terrain's split.

    Returns:
        The grid maps, then the off-grid maps for test terrains.
    """
    grid = [(a, t, "grid") for a in AZIMUTHS_DEG for t in TILTS_DEG]
    if split != "test":
        return grid
    return grid + [(a, t, "off-grid") for a in OFF_GRID_AZIMUTHS_DEG for t in OFF_GRID_TILTS_DEG]


def map_name(terrain_id: int, azimuth_deg: float, tilt_deg: float) -> str:
    """File name of one map, e.g. t003_a045_t06.0.npy.

    Args:
        terrain_id: Terrain id.
        azimuth_deg: Azimuth.
        tilt_deg: Tilt.

    Returns:
        The file name.
    """
    return f"t{terrain_id:03d}_a{azimuth_deg:03.0f}_t{tilt_deg:04.1f}.npy"


def read_manifest(dataset: Path) -> list[dict[str, Any]]:
    """All manifest lines of a dataset (none if it has no manifest yet).

    Args:
        dataset: Dataset directory.

    Returns:
        One record per finished map.
    """
    path = dataset / MANIFEST
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sweep(
    dataset: Path,
    terrains: tuple[TerrainEntry, ...],
    samples_per_tx: int,
    seed: int,
) -> int:
    """Trace every map of `terrains` that the manifest does not already hold.

    Must run after `backend.select_variant`. A map file is written under a temporary name
    and renamed into place before its manifest line is appended, so every manifest line
    points at a complete file.

    Args:
        dataset: Dataset directory (created if missing).
        terrains: Terrains to trace.
        samples_per_tx: Rays per map.
        seed: Solver seed.

    Returns:
        The number of maps traced in this call.

    Raises:
        ValueError: If the manifest holds maps made with other settings, or a manifest
            line points at a missing file.
    """
    from sionna_twin_ops.antenna import tilt_weights
    from sionna_twin_ops.provenance import provenance
    from sionna_twin_ops.scene import DATASET_FOLD, build_scene, measurement_surface
    from sionna_twin_ops.solve import solve_map, specular_settings

    settings = specular_settings(samples_per_tx, seed)
    record = settings.record()
    origin = provenance()
    maps_dir = dataset / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    for line in read_manifest(dataset):
        if line["settings"] != record:
            raise ValueError(f"{dataset} holds maps made with {line['settings']}, not {record}")
        if not (maps_dir / line["file"]).exists():
            raise ValueError(f"manifest names {line['file']} but the file is missing")
        done.add(line["file"])

    traced = 0
    with (dataset / MANIFEST).open("a", newline="\n") as manifest:
        for entry in terrains:
            todo = [
                c
                for c in configurations(entry.split)
                if map_name(entry.terrain_id, c[0], c[1]) not in done
            ]
            if not todo:
                continue
            terrain = generate_terrain(entry.terrain_id, GRID_SPACING_M)
            surface = measurement_surface(terrain, entry.site, DATASET_FOLD)
            for azimuth in sorted({a for a, _, _ in todo}):
                scene = build_scene(terrain, entry.site, azimuth, DATASET_FOLD)
                for _, tilt, kind in (c for c in todo if c[0] == azimuth):
                    result = solve_map(scene, surface, tilt_weights(tilt), settings)
                    name = map_name(entry.terrain_id, azimuth, tilt)
                    partial = maps_dir / f"{name}.partial"
                    with partial.open("wb") as handle:
                        np.save(handle, result.path_gain.astype(np.float32))
                    os.replace(partial, maps_dir / name)
                    line = {
                        "file": name,
                        "terrain_id": entry.terrain_id,
                        "terrain_params": asdict(terrain.params),
                        "split": entry.split,
                        "site": asdict(entry.site),
                        "azimuth_deg": azimuth,
                        "tilt_deg": tilt,
                        "kind": kind,
                        "settings": record,
                        "provenance": origin,
                        "wall_s": round(result.seconds, 4),
                        "cells_valid": int((result.path_gain > 0).sum()),
                        "cells_no_hit": int((result.path_gain == 0).sum()),
                    }
                    manifest.write(json.dumps(line) + "\n")
                    manifest.flush()
                    traced += 1
            print(f"terrain {entry.terrain_id} ({entry.split}): {len(todo)} maps", flush=True)
    return traced


def summary_data(dataset: Path, version: str) -> dict[str, Any]:
    """The committed record of a dataset (spec D4): counts, skips, time, no-hit shares.

    Args:
        dataset: Dataset directory.
        version: Dataset version whose selection the manifest should match.

    Returns:
        JSON-safe data for `reports.dataset_summary_markdown`.

    Raises:
        ValueError: If the manifest mixes settings or holds maps outside the selection.
    """
    lines_ = read_manifest(dataset)
    selection = selection_for(version)
    expected = {
        map_name(e.terrain_id, a, t): e
        for e in selection.terrains
        for a, t, _ in configurations(e.split)
    }
    present = {line["file"]: line for line in lines_}
    stray = sorted(set(present) - set(expected))
    if stray:
        raise ValueError(f"manifest holds {len(stray)} maps outside the selection, e.g. {stray[0]}")
    settings = {json.dumps(line["settings"], sort_keys=True) for line in lines_}
    if len(settings) > 1:
        raise ValueError("manifest mixes solver settings")

    cells = 128 * 128
    splits = [s for s, _ in SPLIT_PER_CLASS]
    rows = []
    for split in splits:
        members = [e for e in selection.terrains if e.split == split]
        maps = [line for line in present.values() if line["split"] == split]
        rows.append(
            {
                "split": split,
                "per_class": [sum(e.site.site_class == c for e in members) for c in SITE_CLASSES],
                "terrains": len(members),
                "grid_maps": sum(m["kind"] == "grid" for m in maps),
                "off_grid_maps": sum(m["kind"] == "off-grid" for m in maps),
                "ids": [e.terrain_id for e in selection.terrains if e.split == split],
            }
        )
    no_hit = {}
    for site_class in SITE_CLASSES:
        shares = [
            line["cells_no_hit"] / cells
            for line in present.values()
            if line["site"]["site_class"] == site_class
        ]
        no_hit[site_class] = (
            {
                "maps": len(shares),
                "mean": float(np.mean(shares)),
                "min": min(shares),
                "max": max(shares),
            }
            if shares
            else {"maps": 0}
        )
    wall = [line["wall_s"] for line in present.values()]
    return {
        "kind": "dataset_summary",
        "version": version,
        "maps_present": len(present),
        "maps_expected": len(expected),
        "site_classes": list(SITE_CLASSES),
        "splits": rows,
        "highest_id": max(e.terrain_id for e in selection.terrains),
        "skipped": {c: [i for i, k in selection.skipped if k == c] for c in SITE_CLASSES},
        "no_hit": no_hit,
        "wall_s_total": sum(wall),
        "wall_s_median": float(np.median(wall)) if wall else 0,
        "settings": lines_[0]["settings"] if lines_ else None,
        "provenance_seen": (
            {
                key: sorted({str(line["provenance"][key]) for line in lines_})
                for key in lines_[0]["provenance"]
            }
            if lines_
            else None
        ),
    }
