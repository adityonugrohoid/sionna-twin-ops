# Tilt and power search on the validation split

Synthetic terrain, not a real place; one sector, no vegetation, buildings or interference, flat Earth. Ray tracer: Sionna RT 2.1.0, cuda_ad_mono_polarized, 1e+09 rays, max depth 3, line of sight and specular reflection only, seed 1 (grid maps at commit faa7ebe, NVIDIA GeForce RTX 4060 Laptop GPU, driver 616.56); pattern: 3GPP TR 38.901. Surrogate: runs/model-f6f13e3-v2/seed0, runs/model-f6f13e3-v2/seed1, runs/model-f6f13e3-v2/seed2 (best epochs; torch 2.14.0; search commit cc41956). Written by `twin search-report`.

This split was used to design the objective (spec Q0, Q0b, Q0c): the first definition's flaw, the area normalisation and the feasible box were settled on it before the test split was searched.

Cases: 72 (terrains 43, 46, 49, 60, 62, 63, 65, 68, 69, 8 azimuths each). Variables: electrical tilt and sector power, within the feasible box: tilt 0 to 12 deg, power 28 to 46 dBm, 46 dBm being the sector maximum (both ASSUMPTION, spec Q0c). A cell is covered where RSRP reaches -110 dBm, with the sector power spread over 1200 resource elements (both ASSUMPTION, as in the evaluation).

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
| rule of thumb | 0.0620 | 0.2519 | 0.5071 | 3 of 72 |
| fixed 0 deg, 46 dBm (no search) | 0.0626 | 0.2905 | 0.5290 | 13 of 72 |
| surrogate seed 0 (CPU) | 0.0004 | 0.0043 | 0.0083 | 72 of 72 |
| surrogate seed 1 (CPU) | 0.0000 | 0.0044 | 0.0073 | 72 of 72 |
| surrogate seed 2 (CPU) | 0.0006 | 0.0038 | 0.0338 | 71 of 72 |
| ray tracer, 1 deg grid | 0.0000 | 0.0000 | 0.0000 | 72 of 72 |

The GPU pass's choices differ from the CPU pass's in 9 of 216 (seed, case) pairs; scored with the ray tracer they give:

| chooser | median shortfall | p90 | max | within 1 point |
|---|---|---|---|---|
| surrogate seed 0 (GPU) | 0.0004 | 0.0043 | 0.0083 | 72 of 72 |
| surrogate seed 1 (GPU) | 0.0000 | 0.0044 | 0.0073 | 72 of 72 |
| surrogate seed 2 (GPU) | 0.0007 | 0.0038 | 0.0338 | 71 of 72 |

Most negative surrogate shortfall (a half-degree setting beating the 1 deg grid): -0.0394 per case (surrogate seed 2 (CPU), terrain 49, azimuth 180 deg) and -0.0027 over a whole terrain (surrogate seed 1 (GPU), terrain 69).

Largest surrogate shortfall: 0.0338 (surrogate seed 2 (CPU), terrain 49, azimuth 225 deg). Terrains with a case more than 1 point short: surrogate seed 0 (CPU): none; surrogate seed 1 (CPU): none; surrogate seed 2 (CPU): 49; rule of thumb: 43, 46, 49, 60, 62, 63, 65, 68, 69; fixed setting: 43, 46, 49, 60, 62, 63, 65, 68, 69.

Where the ray tracer's optimum lies:

| optimum tilt | 0 deg | 1 deg | 2 deg | 3 deg | 4 deg | 5 deg | 7 deg | 8 deg | 9 deg | 10 deg | 11 deg | 12 deg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cases | 18 | 2 | 4 | 1 | 3 | 1 | 5 | 7 | 10 | 6 | 5 | 10 |

| optimum power | 28 dBm | 33 dBm | 34 dBm | 35 dBm | 36 dBm | 37 dBm | 38 dBm | 39 dBm | 40 dBm | 41 dBm | 42 dBm | 44 dBm | 45 dBm | 46 dBm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cases | 2 | 1 | 1 | 5 | 9 | 8 | 10 | 4 | 1 | 3 | 1 | 1 | 6 | 20 |

#### Optima on the edge of the feasible box

The box (tilt 0 to 12 deg, power 28 to 46 dBm, ASSUMPTION, spec Q0c) is the feasible set, so an optimum on its edge is a constrained optimum, not a search artefact. On an edge: 45 of 72. Per site class:

| site class | cases | tilt 0 deg | tilt 12 deg | power 28 dBm | power 46 dBm | any edge |
|---|---|---|---|---|---|---|
| hilltop | 24 | 0 | 9 | 0 | 3 | 12 |
| slope | 24 | 0 | 1 | 2 | 11 | 14 |
| valley | 24 | 18 | 0 | 0 | 6 | 19 |

- hilltop: 9 of 24 at tilt 12 deg (reading: the optimum would lie at more downtilt than the box allows); 3 of 24 at power 46 dBm (reading: the optimum would use more than the sector maximum).
- slope: 11 of 24 at power 46 dBm (reading: the optimum would use more than the sector maximum); 2 of 24 at power 28 dBm (reading: the optimum would use less than the box's power floor); 1 of 24 at tilt 12 deg (reading: the optimum would lie at more downtilt than the box allows).
- valley: 18 of 24 at tilt 0 deg (reading: the optimum would lie at an uptilt, outside the box); 6 of 24 at power 46 dBm (reading: the optimum would use more than the sector maximum).

#### By terrain

Largest shortfall over the 8 azimuths, and the number of azimuths more than 1 point short.

| terrain | site class | optimum tilt, median (deg) | rule of thumb | fixed 0 deg, 46 dBm (no search) | surrogate seed 0 (CPU) | surrogate seed 1 (CPU) | surrogate seed 2 (CPU) | ray tracer, 1 deg grid |
|---|---|---|---|---|---|---|---|---|
| 43 | slope | 10.0 | 0.1333 (8) | 0.2120 (7) | 0.0046 (0) | 0.0046 (0) | 0.0046 (0) | 0.0000 (0) |
| 46 | slope | 8.0 | 0.1097 (8) | 0.2773 (8) | 0.0031 (0) | 0.0031 (0) | 0.0031 (0) | 0.0000 (0) |
| 49 | slope | 9.0 | 0.5071 (8) | 0.5290 (8) | 0.0083 (0) | 0.0000 (0) | 0.0338 (1) | 0.0000 (0) |
| 60 | hilltop | 12.0 | 0.1161 (8) | 0.0565 (7) | 0.0064 (0) | 0.0038 (0) | 0.0038 (0) | 0.0000 (0) |
| 62 | valley | 0.0 | 0.0475 (6) | 0.0595 (4) | 0.0042 (0) | 0.0038 (0) | 0.0050 (0) | 0.0000 (0) |
| 63 | hilltop | 9.0 | 0.1584 (7) | 0.1179 (7) | 0.0073 (0) | 0.0073 (0) | 0.0073 (0) | 0.0000 (0) |
| 65 | valley | 0.0 | 0.1041 (8) | 0.0898 (4) | 0.0050 (0) | 0.0050 (0) | 0.0050 (0) | 0.0000 (0) |
| 68 | valley | 0.0 | 0.0863 (8) | 0.1754 (6) | 0.0047 (0) | 0.0050 (0) | 0.0038 (0) | 0.0000 (0) |
| 69 | hilltop | 9.5 | 0.2286 (8) | 0.2387 (8) | 0.0037 (0) | 0.0057 (0) | 0.0022 (0) | 0.0000 (0) |

### One choice per terrain over azimuth, tilt and power

Shortfall from the ray tracer's whole-grid optimum.

| terrain | ray tracer optimum (azimuth, tilt, power: objective) | surrogate seed 0 (CPU) | surrogate seed 1 (CPU) | surrogate seed 2 (CPU) |
|---|---|---|---|---|
| 43 | 135 deg, 12 deg, 45 dBm: 0.3700 | 0.0046 | 0.0046 | 0.0046 |
| 46 | 180 deg, 8 deg, 42 dBm: 0.4625 | 0.0007 | 0.0004 | 0.0005 |
| 49 | 315 deg, 9 deg, 46 dBm: 0.6729 | 0.0000 | 0.0000 | 0.0000 |
| 60 | 135 deg, 12 deg, 36 dBm: 0.2633 | 0.0000 | 0.0000 | 0.0000 |
| 62 | 270 deg, 4 deg, 38 dBm: 0.7825 | 0.0001 | 0.0038 | -0.0001 |
| 63 | 0 deg, 12 deg, 38 dBm: 0.4729 | 0.0000 | 0.0000 | 0.0029 |
| 65 | 45 deg, 0 deg, 35 dBm: 0.3380 | 0.0000 | 0.0000 | 0.0009 |
| 68 | 315 deg, 2 deg, 35 dBm: 0.6156 | -0.0008 | 0.0021 | 0.0014 |
| 69 | 180 deg, 11 deg, 38 dBm: 0.4146 | 0.0015 | -0.0025 | -0.0025 |

| chooser | median shortfall | p90 | max | within 1 point |
|---|---|---|---|---|
| surrogate seed 0 (CPU) | 0.0000 | 0.0021 | 0.0046 | 9 of 9 |
| surrogate seed 1 (CPU) | 0.0000 | 0.0040 | 0.0046 | 9 of 9 |
| surrogate seed 2 (CPU) | 0.0005 | 0.0033 | 0.0046 | 9 of 9 |

## Wall time per terrain, same machine

Seconds per terrain for the whole search (8 azimuths). Surrogate: terrain generation, terrain channels, per-map channels and 8 batched inferences of 25 tilts (200 maps), and the objectives at every power; 27 timings each (3 seeds x 9 terrains, 1.5 km objective), after one untimed warm-up. Ray tracer: terrain generation, measurement surface, one scene per azimuth and 104 solves, through the Windows runner, after one untimed solve; the objectives take milliseconds and are not in it. The CPU ray tracer was not run for this: its figure is estimated as the evaluation's measured CPU medians (`results/evaluation_test.md`), 35.415 s for a new terrain's first map plus 103 x 35.333 s.

| method | hardware | median | min | max |
|---|---|---|---|---|
| surrogate | GPU (NVIDIA GeForce RTX 4060 Laptop GPU, driver 616.56) | 2.21 | 1.87 | 2.33 |
| surrogate | CPU | 9.06 | 8.77 | 9.39 |
| ray tracer | GPU (NVIDIA GeForce RTX 4060 Laptop GPU, driver 616.56) | 123.7 | 90.0 | 159.4 |
| ray tracer | CPU (llvm), estimated | 3675 | | |

Per terrain, the ray tracer's search takes 56 times as long as the surrogate's on the GPU; without a GPU, an estimated 406 times.

## Check: search maps against the dataset

The search re-traced the dataset's grid tilts (0, 3, 6, 9, 12 deg) with the same settings: 0 of 360 maps are bit-identical; the largest difference where both have power is 6.99e-06 dB.
