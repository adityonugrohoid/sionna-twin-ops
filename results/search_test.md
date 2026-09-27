# Tilt and power search on the test split

Synthetic terrain, not a real place; one sector, no vegetation, buildings or interference, flat Earth. Ray tracer: Sionna RT 2.1.0, cuda_ad_mono_polarized, 1e+09 rays, max depth 3, line of sight and specular reflection only, seed 1 (commit 1d2e070, NVIDIA GeForce RTX 4060 Laptop GPU, driver 616.56); pattern: 3GPP TR 38.901. Surrogate: runs/model-f6f13e3-v2/seed0, runs/model-f6f13e3-v2/seed1, runs/model-f6f13e3-v2/seed2 (best epochs; commit 9c5a628, torch 2.14.0). Written by `twin search-report`.

Cases: 72 (terrains 52, 55, 58, 71, 72, 74, 77, 78, 81, 8 azimuths each). Variables: tilt and sector power. Objective Q1: covered cells within 3 km of the site (ASSUMPTION) minus 1 (ASSUMPTION) times covered cells beyond it; a cell is covered where RSRP reaches -110 dBm, with the sector power spread over 1200 resource elements (both ASSUMPTION, as in the evaluation). The map reaches 3.6 km at its corners, so the cells beyond 3 km are the outer ring and corners only.

Choosers:
- Ray tracer: tilt 0 to 12 deg in 1 deg (13 traces per case), power 37 to 46 dBm in 1 dB (post-processing). Its best setting is the optimum every share below is taken of.
- Surrogate: tilt 0 to 12 deg in 0.5 deg (25 inferences per case), the same powers; covered cells need the power head to say power. Choices come from the CPU pass; the GPU pass in float32 breaks some near-ties the other way (choices agree: False), so its choices are scored too.
- Rule of thumb: tilt = arctan(30 m / 3 km) + half the column's vertical half-power beamwidth (7.92 deg, from the TR 38.901 element times the 8 x 1 array factor) = 4.53 deg, at 46 dBm.

Every chosen setting is scored with the ray tracer: settings off the 1 deg grid (half-degree tilts, the rule's tilt) were traced for the purpose. Ties go to the lower azimuth, then the lower tilt, then the lower power.

## Primary: per (terrain, azimuth), 72 cases

Share = the chosen setting's ray-traced objective over the ray tracer's optimum on the 1 deg grid; a half-degree choice can exceed 1.

| chooser | median share | p10 | min | within 1% of the optimum |
|---|---|---|---|---|
| rule of thumb | 0.9549 | 0.7047 | 0.5376 | 18 of 72 |
| surrogate seed 0 (CPU) | 1.0000 | 0.9993 | 0.9969 | 72 of 72 |
| surrogate seed 1 (CPU) | 1.0000 | 0.9997 | 0.9965 | 72 of 72 |
| surrogate seed 2 (CPU) | 1.0000 | 0.9993 | 0.9966 | 72 of 72 |
| surrogate seed 0 (GPU) | 1.0000 | 0.9995 | 0.9967 | 72 of 72 |
| surrogate seed 1 (GPU) | 1.0000 | 0.9993 | 0.9965 | 72 of 72 |
| surrogate seed 2 (GPU) | 1.0000 | 0.9990 | 0.9966 | 72 of 72 |

Lowest surrogate share: 0.9965 (surrogate seed 1 (CPU), terrain 78, azimuth 135 deg). No surrogate choice falls below 99% of the optimum on any terrain. The rule of thumb falls below 99% on terrains 52, 55, 58, 71, 72, 74, 77, 78; its lowest share is 0.5376.

The ray tracer's optimum uses 46 dBm in 71 of 72 cases; its tilt ranges 0 to 6 deg (median 0).

### By terrain

Lowest share over the 8 azimuths, and the number of azimuths below 99% of the optimum.

| terrain | site class | optimum tilt, median (deg) | rule of thumb | surrogate seed 0 (CPU) | surrogate seed 1 (CPU) | surrogate seed 2 (CPU) | surrogate seed 0 (GPU) | surrogate seed 1 (GPU) | surrogate seed 2 (GPU) |
|---|---|---|---|---|---|---|---|---|---|
| 52 | slope | 0.5 | 0.8270 (8) | 0.9977 (0) | 0.9988 (0) | 0.9966 (0) | 0.9977 (0) | 0.9988 (0) | 0.9966 (0) |
| 55 | slope | 0.0 | 0.7952 (8) | 1.0000 (0) | 1.0000 (0) | 1.0000 (0) | 1.0000 (0) | 1.0000 (0) | 1.0000 (0) |
| 58 | slope | 1.5 | 0.9402 (8) | 0.9969 (0) | 0.9990 (0) | 0.9983 (0) | 0.9967 (0) | 0.9990 (0) | 0.9983 (0) |
| 71 | valley | 0.0 | 0.9035 (8) | 1.0000 (0) | 1.0000 (0) | 1.0000 (0) | 1.0000 (0) | 1.0000 (0) | 1.0000 (0) |
| 72 | hilltop | 4.5 | 0.9750 (5) | 0.9992 (0) | 0.9978 (0) | 0.9978 (0) | 0.9992 (0) | 0.9978 (0) | 0.9978 (0) |
| 74 | valley | 0.0 | 0.6961 (8) | 0.9973 (0) | 1.0000 (0) | 1.0000 (0) | 0.9973 (0) | 1.0000 (0) | 1.0000 (0) |
| 77 | valley | 0.0 | 0.5376 (8) | 1.0000 (0) | 1.0000 (0) | 1.0000 (0) | 1.0000 (0) | 1.0000 (0) | 1.0000 (0) |
| 78 | hilltop | 3.0 | 0.9894 (1) | 0.9992 (0) | 0.9965 (0) | 0.9997 (0) | 0.9995 (0) | 0.9965 (0) | 0.9990 (0) |
| 81 | hilltop | 1.5 | 0.9910 (0) | 0.9994 (0) | 0.9992 (0) | 0.9997 (0) | 0.9997 (0) | 0.9991 (0) | 0.9997 (0) |

## Secondary: one choice per terrain over azimuth, tilt and power

| terrain | ray tracer optimum (azimuth, tilt, power: objective) | surrogate seed 0 (CPU) | surrogate seed 1 (CPU) | surrogate seed 2 (CPU) | surrogate seed 0 (GPU) | surrogate seed 1 (GPU) | surrogate seed 2 (GPU) |
|---|---|---|---|---|---|---|---|
| 52 | 225 deg, 0 deg, 46 dBm: 4358 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 55 | 270 deg, 0 deg, 46 dBm: 2984 | 1.0000 | 0.9977 | 1.0000 | 1.0000 | 0.9977 | 1.0000 |
| 58 | 90 deg, 0 deg, 46 dBm: 3045 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 71 | 180 deg, 0 deg, 46 dBm: 4522 | 1.0000 | 0.9993 | 1.0000 | 1.0000 | 0.9993 | 1.0000 |
| 72 | 315 deg, 2 deg, 46 dBm: 6485 | 0.9995 | 1.0000 | 1.0000 | 0.9995 | 1.0000 | 1.0000 |
| 74 | 135 deg, 0 deg, 46 dBm: 3384 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 77 | 270 deg, 0 deg, 46 dBm: 4232 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 78 | 180 deg, 5 deg, 45 dBm: 12160 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 81 | 225 deg, 2 deg, 46 dBm: 11883 | 1.0001 | 0.9999 | 0.9998 | 0.9998 | 1.0000 | 0.9998 |

| chooser | median share | p10 | min | within 1% of the optimum |
|---|---|---|---|---|
| surrogate seed 0 (CPU) | 1.0000 | 0.9999 | 0.9995 | 9 of 9 |
| surrogate seed 1 (CPU) | 1.0000 | 0.9990 | 0.9977 | 9 of 9 |
| surrogate seed 2 (CPU) | 1.0000 | 1.0000 | 0.9998 | 9 of 9 |
| surrogate seed 0 (GPU) | 1.0000 | 0.9998 | 0.9995 | 9 of 9 |
| surrogate seed 1 (GPU) | 1.0000 | 0.9990 | 0.9977 | 9 of 9 |
| surrogate seed 2 (GPU) | 1.0000 | 1.0000 | 0.9998 | 9 of 9 |

## Wall time per terrain, same machine

Seconds per terrain for the whole search (8 azimuths). Surrogate: terrain generation, terrain channels, per-map channels and 8 batched inferences of 25 tilts (200 maps), and the objectives at every power; 27 timings each (3 seeds x 9 terrains), after one untimed warm-up. Ray tracer: terrain generation, measurement surface, one scene per azimuth and 104 solves, through the Windows runner, after one untimed solve; the objectives take milliseconds and are not in it. The CPU ray tracer was not run for this: its figure is estimated as the evaluation's measured CPU medians (`results/evaluation_test.md`), 35.415 s for a new terrain's first map plus 103 x 35.333 s.

| method | hardware | median | min | max |
|---|---|---|---|---|
| surrogate | GPU (NVIDIA GeForce RTX 4060 Laptop GPU, driver 616.56) | 2.11 | 1.87 | 2.26 |
| surrogate | CPU | 9.02 | 8.69 | 9.34 |
| ray tracer | GPU (NVIDIA GeForce RTX 4060 Laptop GPU, driver 616.56) | 103.2 | 92.3 | 124.1 |
| ray tracer | CPU (llvm), estimated | 3675 | | |

Per terrain, the ray tracer's search takes 49 times as long as the surrogate's on the GPU; without a GPU, an estimated 407 times.

## Check: search maps against the dataset

The search re-traced the dataset's grid tilts (0, 3, 6, 9, 12 deg) with the same settings: 0 of 360 maps are bit-identical; the largest difference where both have power is 7.77e-06 dB.
