# Tilt and power search on the test split

Synthetic terrain, not a real place; one sector, no vegetation, buildings or interference, flat Earth. Ray tracer: Sionna RT 2.1.0, cuda_ad_mono_polarized, 1e+09 rays, max depth 3, line of sight and specular reflection only, seed 1 (grid maps at commit 1d2e070, NVIDIA GeForce RTX 4060 Laptop GPU, driver 616.56); pattern: 3GPP TR 38.901. Surrogate: runs/model-f6f13e3-v2/seed0, runs/model-f6f13e3-v2/seed1, runs/model-f6f13e3-v2/seed2 (best epochs; torch 2.14.0; search commit 320b5ac). Written by `twin search-report`.

The objective was designed on the validation split (`results/search_validation.md`, spec Q0, Q0b, Q0c) and this split was then searched once.

Cases: 72 (terrains 52, 55, 58, 71, 72, 74, 77, 78, 81, 8 azimuths each). Variables: electrical tilt and sector power, within the feasible box: tilt 0 to 12 deg, power 28 to 46 dBm, 46 dBm being the sector maximum (both ASSUMPTION, spec Q0c). A cell is covered where RSRP reaches -110 dBm, with the sector power spread over 1200 resource elements (both ASSUMPTION, as in the evaluation).

Choosers:
- Ray tracer: tilt in 1 deg (13 traces per case), power in 1 dB (post-processing). Its best setting is the optimum every shortfall is taken from.
- Surrogate: tilt in 0.5 deg (25 inferences per case), the same powers; covered cells need the power head to say power. Choices come from the CPU pass; the GPU pass in float32 breaks some near-ties the other way, so its choices are scored too and reported separately.
- Rule of thumb: tilt = arctan(30 m / radius) + half the column's vertical half-power beamwidth (from the TR 38.901 element times the 8 x 1 array factor), at 46 dBm.
- Fixed setting, no search: 0 deg at 46 dBm.

Every chosen setting is scored with the ray tracer: settings off the 1 deg grid (half-degree tilts, the rule's tilt) were traced for the purpose. Ties go to the lower azimuth, then the lower tilt, then the lower power.

## Objective with a 1.5 km service radius

Objective Q0b: the covered fraction of the cells within 1.5 km of the site (4404 cells) minus 1 times the covered fraction of the map's cells beyond it (11980 cells), so neither region wins by size; radius and equal weights are ASSUMPTION. It runs from -1 to 1. Rule of thumb: arctan(30 m / 1.5 km) + 7.92 / 2 = 5.11 deg, at 46 dBm.

### Per (terrain, azimuth), 72 cases

Shortfall = the ray tracer's optimum on the 1 deg grid minus the chosen setting's ray-traced objective, in objective points; a half-degree choice can beat the grid and go negative. Within 1 point: shortfall at most 0.01. The objective is not smooth in tilt (the column's nulls and sidelobes sweep over distant ground, so spill beyond the radius can rise and fall within 1 deg), so the 1 deg grid's optimum is a reference with that resolution, not the true optimum; the most negative shortfalls below measure how far a half-degree setting beat it.

| chooser | median shortfall | p90 | max | within 1 point |
|---|---|---|---|---|
| rule of thumb | 0.0864 | 0.3420 | 0.5276 | 3 of 72 |
| fixed 0 deg, 46 dBm (no search) | 0.0430 | 0.3725 | 0.7337 | 15 of 72 |
| surrogate seed 0 (CPU) | 0.0000 | 0.0032 | 0.0177 | 70 of 72 |
| surrogate seed 1 (CPU) | 0.0000 | 0.0035 | 0.0192 | 68 of 72 |
| surrogate seed 2 (CPU) | 0.0000 | 0.0031 | 0.0189 | 71 of 72 |
| ray tracer, 1 deg grid | 0.0000 | 0.0000 | 0.0000 | 72 of 72 |

The GPU pass's choices differ from the CPU pass's in 11 of 216 (seed, case) pairs; scored with the ray tracer they give:

| chooser | median shortfall | p90 | max | within 1 point |
|---|---|---|---|---|
| surrogate seed 0 (GPU) | 0.0000 | 0.0032 | 0.0177 | 69 of 72 |
| surrogate seed 1 (GPU) | 0.0000 | 0.0033 | 0.0192 | 68 of 72 |
| surrogate seed 2 (GPU) | 0.0000 | 0.0035 | 0.0189 | 71 of 72 |

Most negative surrogate shortfall (a half-degree setting beating the 1 deg grid): -0.0620 per case (surrogate seed 2 (CPU), terrain 81, azimuth 270 deg) and -0.0614 over a whole terrain (surrogate seed 2 (CPU), terrain 81).

Largest surrogate shortfall: 0.0192 (surrogate seed 1 (CPU), terrain 81, azimuth 45 deg). Terrains with a case more than 1 point short: surrogate seed 0 (CPU): 81; surrogate seed 1 (CPU): 78, 81; surrogate seed 2 (CPU): 81; rule of thumb: 52, 55, 58, 71, 72, 74, 77, 78, 81; fixed setting: 52, 55, 58, 71, 72, 74, 77, 78, 81.

Where the ray tracer's optimum lies:

| optimum tilt | 0 deg | 1 deg | 2 deg | 3 deg | 4 deg | 5 deg | 6 deg | 8 deg | 9 deg | 10 deg | 11 deg | 12 deg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cases | 10 | 5 | 7 | 7 | 2 | 2 | 4 | 5 | 6 | 14 | 2 | 8 |

| optimum power | 35 dBm | 36 dBm | 37 dBm | 38 dBm | 39 dBm | 40 dBm | 41 dBm | 42 dBm | 44 dBm | 45 dBm | 46 dBm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cases | 3 | 7 | 9 | 4 | 1 | 2 | 1 | 2 | 5 | 7 | 31 |

#### Optima on the edge of the feasible box

The box (tilt 0 to 12 deg, power 28 to 46 dBm, ASSUMPTION, spec Q0c) is the feasible set, so an optimum on its edge is a constrained optimum, not a search artefact. On an edge: 37 of 72. Per site class:

| site class | cases | tilt 0 deg | tilt 12 deg | power 28 dBm | power 46 dBm | any edge |
|---|---|---|---|---|---|---|
| hilltop | 24 | 4 | 0 | 0 | 1 | 5 |
| slope | 24 | 4 | 8 | 0 | 18 | 20 |
| valley | 24 | 2 | 0 | 0 | 12 | 12 |

- hilltop: 4 of 24 at tilt 0 deg (reading: the optimum would lie at an uptilt, outside the box); 1 of 24 at power 46 dBm (reading: the optimum would use more than the sector maximum).
- slope: 18 of 24 at power 46 dBm (reading: the optimum would use more than the sector maximum); 8 of 24 at tilt 12 deg (reading: the optimum would lie at more downtilt than the box allows); 4 of 24 at tilt 0 deg (reading: the optimum would lie at an uptilt, outside the box).
- valley: 12 of 24 at power 46 dBm (reading: the optimum would use more than the sector maximum); 2 of 24 at tilt 0 deg (reading: the optimum would lie at an uptilt, outside the box).

#### By terrain

Largest shortfall over the 8 azimuths, and the number of azimuths more than 1 point short.

| terrain | site class | optimum tilt, median (deg) | rule of thumb | fixed 0 deg, 46 dBm (no search) | surrogate seed 0 (CPU) | surrogate seed 1 (CPU) | surrogate seed 2 (CPU) | ray tracer, 1 deg grid |
|---|---|---|---|---|---|---|---|---|
| 52 | slope | 12.0 | 0.1252 (8) | 0.1409 (5) | 0.0018 (0) | 0.0005 (0) | 0.0071 (0) | 0.0000 (0) |
| 55 | slope | 1.5 | 0.0450 (8) | 0.0435 (4) | 0.0023 (0) | 0.0001 (0) | 0.0009 (0) | 0.0000 (0) |
| 58 | slope | 11.0 | 0.1039 (8) | 0.0962 (7) | 0.0024 (0) | 0.0022 (0) | 0.0024 (0) | 0.0000 (0) |
| 71 | valley | 2.0 | 0.0248 (5) | 0.0551 (4) | 0.0044 (0) | 0.0082 (0) | 0.0035 (0) | 0.0000 (0) |
| 72 | hilltop | 2.0 | 0.1788 (8) | 0.0399 (7) | 0.0054 (0) | 0.0082 (0) | 0.0014 (0) | 0.0000 (0) |
| 74 | valley | 9.0 | 0.0903 (8) | 0.1210 (8) | 0.0020 (0) | 0.0017 (0) | 0.0031 (0) | 0.0000 (0) |
| 77 | valley | 3.0 | 0.1333 (8) | 0.0447 (6) | 0.0033 (0) | 0.0035 (0) | 0.0033 (0) | 0.0000 (0) |
| 78 | hilltop | 9.5 | 0.5276 (8) | 0.7337 (8) | 0.0059 (0) | 0.0113 (1) | 0.0059 (0) | 0.0000 (0) |
| 81 | hilltop | 10.0 | 0.3686 (8) | 0.3880 (8) | 0.0177 (2) | 0.0192 (3) | 0.0189 (1) | 0.0000 (0) |

### One choice per terrain over azimuth, tilt and power

Shortfall from the ray tracer's whole-grid optimum.

| terrain | ray tracer optimum (azimuth, tilt, power: objective) | surrogate seed 0 (CPU) | surrogate seed 1 (CPU) | surrogate seed 2 (CPU) |
|---|---|---|---|---|
| 52 | 90 deg, 12 deg, 46 dBm: 0.4477 | -0.0005 | -0.0005 | -0.0005 |
| 55 | 0 deg, 3 deg, 39 dBm: 0.3303 | 0.0013 | 0.0013 | 0.0017 |
| 58 | 180 deg, 10 deg, 46 dBm: 0.3209 | 0.0000 | 0.0001 | 0.0000 |
| 71 | 315 deg, 6 deg, 45 dBm: 0.5548 | -0.0005 | 0.0082 | -0.0005 |
| 72 | 180 deg, 8 deg, 36 dBm: 0.5322 | -0.0001 | -0.0001 | -0.0001 |
| 74 | 315 deg, 9 deg, 45 dBm: 0.1621 | 0.0000 | 0.0000 | 0.0028 |
| 77 | 90 deg, 2 deg, 45 dBm: 0.4951 | 0.0004 | 0.0006 | 0.0004 |
| 78 | 225 deg, 10 deg, 36 dBm: 0.5564 | -0.0301 | -0.0301 | -0.0301 |
| 81 | 315 deg, 10 deg, 44 dBm: 0.6089 | -0.0614 | -0.0614 | -0.0614 |

| chooser | median shortfall | p90 | max | within 1 point |
|---|---|---|---|---|
| surrogate seed 0 (CPU) | -0.0001 | 0.0006 | 0.0013 | 9 of 9 |
| surrogate seed 1 (CPU) | 0.0000 | 0.0027 | 0.0082 | 9 of 9 |
| surrogate seed 2 (CPU) | -0.0001 | 0.0019 | 0.0028 | 9 of 9 |

## First objective, 3 km radius: a flawed definition

The first definition (covered cells within 3 km minus covered cells beyond, raw counts, power 37 to 46 dBm, share of the optimum) is flawed: its radius is beyond the map's half-width (2.56 km), so only the corners lie outside it, there is almost no spill to avoid, and the objective rewards covering the whole map. A fixed setting then nearly ties any search. It was run once on this split; its report is kept verbatim in `search_test_first_objective.md`. Its primary table, quoted:

| chooser | median share | p10 | min | within 1% of the optimum |
|---|---|---|---|---|
| rule of thumb | 0.9549 | 0.7047 | 0.5376 | 18 of 72 |
| surrogate seed 0 (CPU) | 1.0000 | 0.9993 | 0.9969 | 72 of 72 |
| surrogate seed 1 (CPU) | 1.0000 | 0.9997 | 0.9965 | 72 of 72 |
| surrogate seed 2 (CPU) | 1.0000 | 0.9993 | 0.9966 | 72 of 72 |
| surrogate seed 0 (GPU) | 1.0000 | 0.9995 | 0.9967 | 72 of 72 |
| surrogate seed 1 (GPU) | 1.0000 | 0.9993 | 0.9965 | 72 of 72 |
| surrogate seed 2 (GPU) | 1.0000 | 0.9990 | 0.9966 | 72 of 72 |

The ray tracer's optimum uses 46 dBm in 71 of 72 cases; its tilt ranges 0 to 6 deg (median 0).

## Wall time per terrain, same machine

Seconds per terrain for the whole search (8 azimuths). Surrogate: terrain generation, terrain channels, per-map channels and 8 batched inferences of 25 tilts (200 maps), and the objectives at every power; 27 timings each (3 seeds x 9 terrains, 1.5 km objective), after one untimed warm-up. Ray tracer: terrain generation, measurement surface, one scene per azimuth and 104 solves, through the Windows runner, after one untimed solve; the objectives take milliseconds and are not in it. The CPU ray tracer was not run for this: its figure is estimated as the evaluation's measured CPU medians (`results/evaluation_test.md`), 34.699 s for a new terrain's first map plus 103 x 34.570 s.

| method | hardware | median | min | max |
|---|---|---|---|---|
| surrogate | GPU (NVIDIA GeForce RTX 4060 Laptop GPU, driver 616.56) | 2.24 | 2.04 | 2.39 |
| surrogate | CPU | 9.38 | 8.76 | 10.13 |
| ray tracer | GPU (NVIDIA GeForce RTX 4060 Laptop GPU, driver 616.56) | 103.2 | 92.3 | 124.1 |
| ray tracer | CPU (llvm), estimated | 3595 | | |

Per terrain, the ray tracer's search takes 46 times as long as the surrogate's on the GPU; without a GPU, an estimated 383 times.

## Check: search maps against the dataset

The search re-traced the dataset's grid tilts (0, 3, 6, 9, 12 deg) with the same settings: 0 of 360 maps are bit-identical; the largest difference where both have power is 7.77e-06 dB.
