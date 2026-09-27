# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # A learned surrogate of a ray tracer on hilly terrain
#
# The story of the project in order, told from the committed results only: every table
# below is read from a report's data in `results/` and every picture is a committed figure.
# Nothing is traced, trained or evaluated here, so the notebook runs without the dataset,
# the GPU or the Windows runner.
#
# Terrain, sites and configurations are synthetic, generated from seeds: no real place,
# operator or network. Ground truth is NVIDIA Sionna RT; this project is not affiliated
# with NVIDIA.

# %%
from pathlib import Path

import numpy as np
from IPython.display import Image, Markdown, display

from sionna_twin_ops.reports import read_report

RESULTS = Path.cwd().parent / "results" if Path.cwd().name == "notebooks" else Path("results")
FIGURES = RESULTS / "figures"


def report(name: str) -> dict:
    """The data of a committed report."""
    return read_report(RESULTS / f"{name}.md")


def table(header: list[str], rows: list[list[object]]) -> None:
    """Show rows as a markdown table."""
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(str(c) for c in row) + " |" for row in rows),
    ]
    display(Markdown("\n".join(lines)))


def spread(values: list[float], digits: int) -> str:
    """Mean over seeds with the lowest and highest seed."""
    return f"{np.mean(values):.{digits}f} ({min(values):.{digits}f} to {max(values):.{digits}f})"


# %% [markdown]
# ## 1. Terrain and sites
#
# Each terrain is a spectral-noise heightmap drawn from a seed. Each terrain id gets one
# site class (hilltop, slope or valley) and the site is placed by rule on the map; ids with
# no qualifying spot are skipped. Source: `results/site_check.md`.

# %%
display(Image(FIGURES / "terrain_samples.jpg", width=720))
display(Image(FIGURES / "site_placement.jpg", width=720))
sites = report("site_check")
table(
    ["class", "ids", "placed", "skipped", "median map percentile"],
    [
        [
            c["site_class"],
            c["ids"],
            len(c["percentiles"]),
            len(c["skipped"]),
            f"{np.median(c['percentiles']):.1f}" if c["percentiles"] else "-",
        ]
        for c in sites["classes"]
    ],
)

# %% [markdown]
# ## 2. The ray tracer and its own uncertainty
#
# Sionna RT computes the path-gain map on a mesh measurement surface with line of sight
# and specular reflection only. Before trusting it as ground truth, the solver check tests
# the antenna tilt on a flat plain, the backend check compares GPU with CPU, and the fold
# check measures two sources of uncertainty in the truth itself: the sampling floor (more
# rays) and the fold term (the other triangulation of each map cell). Sources:
# `results/solver_check.md`, `results/backend_check.md`, `results/symmetry_check.md`,
# `results/fold_check.md`.

# %%
solver = report("solver_check")
display(Image(FIGURES / "tilt_check.png", width=720))
display(
    Markdown(
        f"Tilt check: {'PASS' if solver['a4_pass'] else 'FAIL'}; largest main-lobe error "
        f"{solver['lobe_error_deg']:.2f} deg; far-field span {solver['far_field_span_db']:.1f} dB."
    )
)
backend = report("backend_check")
agreement = [
    row[region]
    for row in backend["agreement"]
    for region in ("los", "nlos")
    if row[region]["cells"]
]
display(
    Markdown(
        f"GPU against CPU, every map and region: largest median |difference| "
        f"{max(a['median'] for a in agreement):.3f} dB, largest p95 "
        f"{max(a['p95'] for a in agreement):.3f} dB."
    )
)
symmetry = report("symmetry_check")
table(
    ["variant (k, mirror)", "keeps the fold", "LOS median |dB|", "NLOS median |dB|"],
    [
        [
            f"{r['k']}, {r['mirror']}",
            r["keeps_fold"],
            f"{r['los']['median']:.3f}",
            f"{r['nlos']['median']:.3f}",
        ]
        for r in symmetry["rows"]
    ],
)

# %%
fold = report("fold_check")
table(
    ["site class", "term", "LOS median / p95", "NLOS median / p95"],
    [
        [
            r["group"],
            r["term"],
            f"{r['los']['median']:.3f} / {r['los']['p95']:.3f}",
            f"{r['nlos']['median']:.3f} / {r['nlos']['p95']:.3f}",
        ]
        for r in fold["truth"]
    ],
)

# %% [markdown]
# ## 3. Dataset
#
# One map per terrain, azimuth and tilt, traced on the GPU and split by terrain, so the
# validation and test terrains are never seen in training. Source:
# `results/dataset_summary.md`.

# %%
dataset = report("dataset_summary")
table(
    ["split", *dataset["site_classes"], "terrains", "grid maps", "off-grid maps"],
    [
        [r["split"], *r["per_class"], r["terrains"], r["grid_maps"], r["off_grid_maps"]]
        for r in dataset["splits"]
    ],
)

# %% [markdown]
# ## 4. Model
#
# A U-Net predicts the path gain as a residual over B0 (free space and the antenna pattern)
# and a logit for whether the ray tracer has power in each cell. It is trained with the
# map's exact symmetries that keep the fold; the best epoch is chosen on validation.
# Source: `results/training_summary.md`.

# %%
training = report("training_summary")
table(
    ["seed", "best epoch", "validation L1 (dB)", "last epoch", "L1 at last epoch (dB)"],
    [
        [
            m["hyperparameters"]["seed"],
            m["best"]["epoch"],
            f"{m['best']['l1_db']:.3f}",
            m["history"][-1]["epoch"],
            f"{m['history'][-1]['l1_db']:.3f}",
        ]
        for m in training["metas"]
    ],
)

# %% [markdown]
# ## 5. Test evaluation
#
# The frozen checkpoints on the held-out test terrains, beside two baselines and the ray
# tracer's own uncertainty, per stratum: LOS cells dominated by the direct path, LOS cells
# dominated by reflections, and shadowed (NLOS) cells. Source: `results/evaluation_test.md`.

# %%
evaluation = report("evaluation_test")
seeds = evaluation["seeds"]
rows = []
for row in evaluation["blocks"]["all"]:
    u = row["uncertainty"]
    rows.append(
        [
            row["stratum"],
            spread([s[0] for s in row["surrogate"]], 3),
            f"{row['baselines']['B0'][0]:.3f}",
            f"{row['baselines']['B1'][0]:.3f}",
            f"{u['fold'][0]:.3f} / {u['fold'][1]:.3f}" if u else "-",
            f"{u['sampling'][0]:.3f} / {u['sampling'][1]:.3f}" if u else "-",
        ]
    )
table(
    ["cells", "surrogate mean abs", "B0", "B1", "fold term", "sampling floor"],
    rows,
)
coverage = evaluation["coverage"]
display(
    Markdown(
        f"Coverage IoU (pooled): surrogate {spread([coverage[s][1] for s in seeds], 4)}, "
        f"B0 {coverage['B0'][1]:.4f}, B1 {coverage['B1'][1]:.4f}."
    )
)
display(Image(FIGURES / "evaluation_test.jpg", width=720))

# %% [markdown]
# ## 6. Search
#
# Choosing tilt and power per (terrain, azimuth) with the surrogate, a rule of thumb, a
# fixed setting and the ray tracer itself. Every choice is scored with the ray tracer, as
# its shortfall from the ray tracer's optimum. The objective was designed on the
# validation terrains; the first definition was flawed and is kept as superseded. Source:
# `results/search_test.md`.

# %%
search = report("search_test")
rows = []
for name, values in search["scores"]["shortfalls"].items():
    if name.endswith("(GPU)"):
        continue
    v = np.asarray(list(values.values()))
    rows.append(
        [name, f"{np.median(v):.4f}", f"{v.max():.4f}", f"{int((v <= 0.01).sum())} of {len(v)}"]
    )
table(["chooser", "median shortfall", "max shortfall", "within 1 point"], rows)
display(Image(FIGURES / "search_test.jpg", width=720))

# %% [markdown]
# ## 7. Speed
#
# Wall time on the same machine, the ray tracer over the surrogate, per map and for a
# whole search of one terrain. Sources: `results/evaluation_test.md`,
# `results/search_test.md`.

# %%
t = evaluation["timing"]
gpu_new = t["terrain_features"] + t["map_inputs"] + t["surrogate_gpu"]
gpu_further = t["map_inputs"] + t["surrogate_gpu"]
cpu_new = t["terrain_features"] + t["map_inputs"] + t["surrogate_cpu"]
cpu_further = t["map_inputs"] + t["surrogate_cpu"]
rt_s = list(search["trace_meta"]["terrain_seconds"].values())
gpu_s = [s for per in search["surrogate"]["cuda"]["seconds"].values() for s in per.values()]
table(
    ["case", "GPU", "CPU"],
    [
        [
            "one map, new terrain",
            f"{t['rt_gpu_new'] / gpu_new:.1f}",
            f"{t['rt_cpu_new'] / cpu_new:.1f}",
        ],
        [
            "each further map",
            f"{t['rt_gpu_further'] / gpu_further:.1f}",
            f"{t['rt_cpu_further'] / cpu_further:.1f}",
        ],
        ["whole search, one terrain", f"{np.median(rt_s) / np.median(gpu_s):.0f}", "estimated"],
    ],
)
