"""Cross-backend agreement: the same maps solved on two Mitsuba variants, compared.

`floor_maps` runs on one backend (after `backend.select_variant`) and saves the maps of the
solver-check floor configuration with their timings and provenance. `compare_markdown`
needs no Sionna: it loads two such runs and reports per-cell agreement, split by the LOS
mask. Synthetic terrain.
"""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sionna_twin_ops.baselines import los_mask, map_geometry
from sionna_twin_ops.site import place_site, site_class_for
from sionna_twin_ops.terrain import GRID_SPACING_M, generate_terrain

TERRAIN_IDS = (3, 1, 5)  # one hilltop, one slope, one valley (as the solver-check floor)
AZIMUTH_DEG = 90.0
TILT_DEG = 6.0
META = "meta.json"


def floor_maps(samples: list[int], seed: int, out: Path) -> dict[str, Any]:
    """Solve and save the floor maps on the selected backend.

    Each map is solved twice: the first time includes JIT compilation (cold), the second
    is the one saved and timed (warm). Whether the two are bit-identical is recorded.

    Args:
        samples: Rays per map, one run each.
        seed: Solver seed.
        out: Directory for the maps and META.

    Returns:
        The metadata written to META.
    """
    from sionna_twin_ops.antenna import tilt_weights
    from sionna_twin_ops.provenance import gpu_memory_used_mib, provenance
    from sionna_twin_ops.scene import build_scene, measurement_surface
    from sionna_twin_ops.solve import solve_map, specular_settings

    out.mkdir(parents=True, exist_ok=True)
    weights = tilt_weights(TILT_DEG)
    rows = []
    for terrain_id in TERRAIN_IDS:
        terrain = generate_terrain(terrain_id, GRID_SPACING_M)
        site = place_site(terrain, site_class_for(terrain_id))
        scene = build_scene(terrain, site, AZIMUTH_DEG)
        surface = measurement_surface(terrain, site)
        for n in samples:
            settings = specular_settings(n, seed)
            start = time.perf_counter()
            first = solve_map(scene, surface, weights, settings)
            cold = time.perf_counter() - start
            second = solve_map(scene, surface, weights, settings)
            np.save(out / f"t{terrain_id}_{n:.0e}.npy", second.path_gain)
            rows.append(
                {
                    "terrain_id": terrain_id,
                    "samples": n,
                    "cold_s": round(cold, 3),
                    "warm_s": round(second.seconds, 3),
                    "repeat_identical": bool(np.array_equal(first.path_gain, second.path_gain)),
                    "gpu_memory_mib": gpu_memory_used_mib(),
                }
            )
            print(json.dumps(rows[-1]), flush=True)
    meta = {
        "provenance": provenance(),
        "settings": specular_settings(0, seed).record() | {"samples_per_tx": samples},
        "azimuth_deg": AZIMUTH_DEG,
        "tilt_deg": TILT_DEG,
        "rows": rows,
    }
    (out / META).write_text(json.dumps(meta, indent=1))
    return meta


def load(run: Path) -> tuple[dict[str, Any], dict[tuple[int, int], NDArray[np.float64]]]:
    """A saved run: its metadata and maps keyed by (terrain id, samples).

    Args:
        run: Directory written by `floor_maps`.

    Returns:
        (metadata, maps).
    """
    meta: dict[str, Any] = json.loads((run / META).read_text())
    maps = {
        (r["terrain_id"], r["samples"]): np.load(run / f"t{r['terrain_id']}_{r['samples']:.0e}.npy")
        for r in meta["rows"]
    }
    return meta, maps


def _stats(a: NDArray[np.float64], b: NDArray[np.float64], region: NDArray[np.bool_]) -> str:
    both = region & (a > 0) & (b > 0)
    if not both.any():
        return "- | - | - | 0"
    d = np.abs(10.0 * np.log10(a[both]) - 10.0 * np.log10(b[both]))
    return f"{np.median(d):.3f} | {np.percentile(d, 95):.3f} | {d.max():.3f} | {int(both.sum())}"


def compare_markdown(reference: Path, candidate: Path) -> str:
    """Per-cell agreement of two backends on the same maps, as markdown.

    Args:
        reference: Run directory of the reference backend.
        candidate: Run directory of the backend under test.

    Returns:
        Markdown text.

    Raises:
        ValueError: If the runs differ in anything but the backend.
    """
    ref_meta, ref = load(reference)
    cand_meta, cand = load(candidate)
    for key in ("azimuth_deg", "tilt_deg"):
        if ref_meta[key] != cand_meta[key]:
            raise ValueError(f"runs differ in {key}")
    ref_settings = dict(ref_meta["settings"])
    cand_settings = dict(cand_meta["settings"])
    ref_settings.pop("variant")
    cand_settings.pop("variant")
    if ref_settings != cand_settings or set(ref) != set(cand):
        raise ValueError("runs differ in settings or in the maps they hold")

    ref_prov, cand_prov = ref_meta["provenance"], cand_meta["provenance"]
    lines = [
        "# Backend check",
        "",
        "The solver-check floor maps (synthetic terrain ids "
        f"{', '.join(str(t) for t in TERRAIN_IDS)}, azimuth {AZIMUTH_DEG:.0f}, tilt "
        f"{TILT_DEG:.0f}, line of sight and specular reflection, max_depth 3) solved on two "
        "backends and compared cell by cell. Written by `twin backend-check`.",
        "",
        "| | reference | candidate |",
        "|---|---|---|",
        *(f"| {k} | {ref_prov[k]} | {cand_prov[k]} |" for k in ref_prov),
        "",
        "## Agreement",
        "",
        "Absolute dB difference over cells with power on both backends, split by the LOS "
        "mask. The last columns count cells with power on only one backend.",
        "",
        "| terrain | samples | LOS median | LOS p95 | LOS max | LOS cells | NLOS median | "
        "NLOS p95 | NLOS max | NLOS cells | power only on reference | power only on candidate |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    los_by_terrain = {}
    for terrain_id in TERRAIN_IDS:
        terrain = generate_terrain(terrain_id, GRID_SPACING_M)
        site = place_site(terrain, site_class_for(terrain_id))
        los_by_terrain[terrain_id] = los_mask(map_geometry(terrain, site))
    for terrain_id, n in sorted(ref, key=lambda k: (TERRAIN_IDS.index(k[0]), k[1])):
        a, b = ref[(terrain_id, n)], cand[(terrain_id, n)]
        los = los_by_terrain[terrain_id]
        lines.append(
            f"| {terrain_id} | {n:.0e} | {_stats(a, b, los)} | {_stats(a, b, ~los)} | "
            f"{int(((a > 0) & (b == 0)).sum())} | {int(((a == 0) & (b > 0)).sum())} |"
        )
    lines += [
        "",
        "## Time per map",
        "",
        "Cold includes JIT compilation; warm is a second solve of the same map. Repeat "
        "identical says whether the two solves gave bit-identical maps.",
        "",
        "| terrain | samples | reference warm s | reference cold s | reference repeat identical "
        "| candidate warm s | candidate cold s | candidate repeat identical | candidate GPU MiB |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    cand_rows = {(r["terrain_id"], r["samples"]): r for r in cand_meta["rows"]}
    for r in ref_meta["rows"]:
        c = cand_rows[(r["terrain_id"], r["samples"])]
        lines.append(
            f"| {r['terrain_id']} | {r['samples']:.0e} | {r['warm_s']:.2f} | {r['cold_s']:.2f} | "
            f"{r['repeat_identical']} | {c['warm_s']:.2f} | {c['cold_s']:.2f} | "
            f"{c['repeat_identical']} | {c['gpu_memory_mib']} |"
        )
    return "\n".join(lines) + "\n"
