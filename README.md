# sionna-twin-ops

A radio digital twin pipeline for one sector on synthetic hilly terrain generated from seeds
(no real place, operator or network): ray-traced ground truth with its own uncertainty
measured, a versioned dataset contract, a learned surrogate scored against two propagation
baselines on held-out terrain, and a tilt and power search that uses it. NVIDIA Sionna RT
computes path-gain maps over procedurally generated hills, slopes and valleys; a small U-Net
learns to predict those maps on terrain it has never seen. It is for radio and
machine-learning engineers who want to see how far such a surrogate can be trusted, and at
what cost. What sets it apart is that every surrogate number is shown beside two classic
propagation baselines and the ray tracer's own uncertainty, on held-out terrain.

## Quickstart

Needs [uv](https://docs.astral.sh/uv/) and Linux. The ray-tracing step runs on the CPU and
needs the LLVM shared library (`sudo apt install llvm` on Ubuntu; tested on Ubuntu 24.04 and
WSL2). No GPU and no dataset are needed.

```bash
git clone https://github.com/adityonugrohoid/sionna-twin-ops.git
cd sionna-twin-ops
uv sync --locked
uv run twin summary --results results --out /tmp/sionna-twin-ops-summary.md
uv run pytest -q --nbmake notebooks/report.ipynb
uv run twin backend-maps --samples 1e7 --seed 1 --out /tmp/sionna-twin-ops-trace
```

Measured on a 16-thread machine:

- `uv sync` installs the locked environment: it downloads about 3 GB, most of it PyTorch,
  and takes about 6 GB on disk. A cold install took 34 to 62 s in CI.
- `twin summary` rebuilds [results/SUMMARY.md](results/SUMMARY.md) from the committed report
  data in under a second.
- The nbmake line runs the report notebook end to end in about 3 s. It reads only committed
  results. To read it with its tables and figures, open it with
  `uv run --with jupyterlab jupyter lab notebooks/report.ipynb`.
- `twin backend-maps` ray-traces three terrains with Sionna RT on the CPU at 1e7 rays per
  map, in about 5 s. It writes one path-gain map per terrain as `.npy`, and its settings and
  provenance in `meta.json`.

## In plain terms

Before an antenna is installed or re-aimed, engineers want to know where its signal will
reach. On a computer there are two usual ways to predict that:

- A quick formula, from distance, antenna height and the shape of the antenna's beam. It
  is instant, but it sees the hills in between only roughly, if at all.
- A detailed simulation (ray tracing), which follows the radio waves over the terrain: which
  paths a hill blocks and which bounce off the ground. It is much closer to reality, but
  slow.

This repo tests a third way. A small AI model studies thousands of detailed simulations,
the way a student studies a teacher's worked answers, and then predicts new coverage maps
almost instantly. It is tested on hills it has never seen, against the simulation and two
quick formulas. On that unseen terrain:

- Its typical error is 1.549 dB, half or less than that of the quick formulas (3.132 and
  3.844 dB).
- Asked to choose an antenna's tilt and power, it lands within 1 point of the simulation's
  best choice (0.01 on a score from -1 to 1) in 68 to 71 of 72 cases. A common tilt rule of
  thumb aims at the same distance but cannot see the hills; it lands that close in 3.
- After the first map of a terrain, each further map of that terrain (another antenna
  direction or tilt) is 133.5 times faster than the simulation on the same GPU.

Everything here is simulated: made-up hills with no buildings or trees, and no field
measurements. It shows how far such a model can be trusted against a detailed simulation.
It is not a planning tool and makes no claim about a real network.

## How it works

1. **Terrain and sites.** Each terrain is a spectral-noise heightmap on a 40 m grid, drawn
   from its id. Each id gets a site class (hilltop, slope or valley), and the site is placed
   by rule; ids with no qualifying spot are skipped.
2. **Ground truth.** Sionna RT's radio-map solver computes the path gain on a mesh
   measurement surface 1.5 m above the ground, with line of sight and specular reflection
   only (diffraction does not reach a mesh surface in Sionna RT). The antenna is an 8 x 1
   column of 3GPP TR 38.901 elements on a 30 m mast at 1.8 GHz, tilted electrically.
3. **The ray tracer's own uncertainty.** Two terms are measured before the truth is used:
   - the sampling floor: the dataset's ray count against the largest possible;
   - the fold term: the same map with every cell split along the other diagonal.
4. **Dataset.** One map per terrain, azimuth and tilt, traced on the GPU, split by terrain.
   Validation and test terrains are never seen in training.
5. **Model.** A U-Net predicts the path gain as a residual over B0 (free space and the
   antenna pattern), plus a logit for whether the ray tracer has power in each cell. It
   trains with the four exact map symmetries that keep each cell's fold, on three seeds,
   with the best epoch chosen on validation.
6. **Evaluation.** The frozen checkpoints run once on the test terrains. The surrogate is
   compared with B0 and with B1 (B0 with a Bullington diffraction loss, ITU-R P.526-16), per
   stratum, beside the fold term and the sampling floor.
7. **Search.** For each terrain and azimuth, tilt and power are chosen by the surrogate, a
   rule of thumb, a fixed setting and the ray tracer itself. Every choice is scored with the
   ray tracer, as its shortfall from the ray tracer's optimum. The objective runs from -1 to
   1, and a point is 0.01.

Every report in `results/` is written by a `twin` command from one data record (the `.json`
beside each `.md`) and states its settings, seeds, commit and package versions. The
notebook and the summary read only those records. The rules the code and reports cite by id
("spec S2", "rule Q0b") are in [docs/dataset.md](docs/dataset.md).

## Headline results

On the held-out test terrain, from [results/SUMMARY.md](results/SUMMARY.md):

- **Path-gain error.** Mean absolute path-gain error: surrogate 1.549 dB, against 3.132 dB
  for B0 and 3.844 dB for B1. By kind of cell:
  - LOS cells dominated by the direct path: 0.838 dB, beside a fold term of median
    0.484 dB and p95 3.287 dB;
  - LOS cells dominated by reflections: 4.084 dB, beside a fold term of median 1.715 dB
    and p95 13.241 dB;
  - shadowed (NLOS) cells: 3.856 dB, beside a fold term of median 1.725 dB and p95
    9.905 dB.
- **Coverage.** Coverage IoU at -110 dBm: 0.9385 against 0.8018 for B1.
- **Search.** The surrogate's choice is within 1 point of the ray tracer's optimum in 68 to
  71 of 72 cases (by seed), against 3 for the rule of thumb and 15 for a fixed setting.
- **Speed.** On the GPU, the ray tracer takes 2.5 times as long as the surrogate for a new
  terrain's first map (most of the surrogate's time goes to the terrain's features) and
  133.5 times as long for each further map of that terrain. A whole search of one terrain
  takes 46 times as long.

## Limitations

- Synthetic terrain only: one ground material, no vegetation, buildings or clutter, flat
  Earth.
- One sector and one antenna configuration.
- Specular reflection only: no diffraction, no diffuse scattering.
- The truth depends on how map cells are folded into triangles. That fold term is far above
  the sampling floor, and it was measured on training terrains at one azimuth and tilt.
- The surrogate underpredicts reflection-dominated LOS cells and is weakest there and in
  shadow.
- Search optima often sit on the edge of the allowed tilt and power range. The objective is
  not smooth in tilt, and its first definition was flawed and is kept as superseded.

The full list, with links, is in [results/SUMMARY.md](results/SUMMARY.md).

## Rebuilding the results

The dataset, weights and run outputs are not in git. The commands that made them are `twin
dataset-sweep`, `twin train`, `twin evaluate`, `twin fold-check` and the other `twin`
subcommands (`uv run twin --help`); each report names the command that wrote it. Ray
tracing at the dataset's 1e9 rays per map ran on native Windows with the GPU, driven from
WSL by the runner in [docs/windows-gpu.md](docs/windows-gpu.md). The CPU backend agrees
with it: median and p95 differences print as 0.000 dB, though single cells differ by up to
0.483 dB ([results/backend_check.md](results/backend_check.md)). It is also much slower.

## Repository layout

```
src/sionna_twin_ops/   terrain, sites, antenna, scene and solver, baselines, dataset,
                       features, model and training, evaluation, search, reports, summary, CLI
notebooks/             report.ipynb, jupytext-paired with report.py, run in CI
results/               committed reports (.md with .json data), SUMMARY.md, figures/
tests/                 unit tests, the llvm ray-tracing tests, report and summary checks
windows/               the Windows GPU runner (setup, environment, run)
docs/                  dataset.md (the rules cited by id), windows-gpu.md
```

## Tests and CI

```bash
uv run pytest -q
uv run pytest -q --nbmake notebooks/report.ipynb
```

The tests include ray tracing on the CPU backend and a byte-for-byte check that every
committed report is its data rendered. Every push to main and every pull request runs ruff,
strict mypy, the tests, a check that the notebook is stripped, and the notebook itself, in
about 2.5 minutes.

Built on NVIDIA Sionna RT; not affiliated with NVIDIA.

## License

MIT, see [LICENSE](LICENSE).

## Author

Adityo Nugroho ([adityonugroho.com](https://adityonugroho.com)),
building with a Claude Code agentic workflow.
