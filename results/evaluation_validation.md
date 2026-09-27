# Evaluation on the validation split

Synthetic terrain, not a real place; one sector, no vegetation, buildings or interference, flat Earth. Ground truth: Sionna RT, line of sight and specular reflection only (cuda_ad_mono_polarized, 1e+09 rays, Sionna RT 2.1.0; map commits eb4b064); pattern: 3GPP TR 38.901. Written by `twin evaluate`.

Re-scored at a00f22c to add machine-readable output; all non-timing numbers identical to the first run at b408fe2.

Maps: 360 (360 grid, 0 off-grid) on terrains 43, 46, 49, 60, 62, 63, 65, 68, 69. Surrogate: the best-epoch checkpoints of runs/model-f6f13e3-v2/seed0, runs/model-f6f13e3-v2/seed1, runs/model-f6f13e3-v2/seed2; each surrogate figure is the mean over the seeds, with the lowest and highest seed in brackets.

Errors are predicted minus traced path gain in dB, over cells where the ray tracer has power. Strata: LOS direct-dominated and LOS reflection-dominated (traced gain at least 3 dB above B0, ASSUMPTION) and NLOS, by the heightmap LOS mask. B0: free space plus the antenna pattern; B1: B0 minus the Bullington diffraction loss (ITU-R P.526-16 section 4.5.1).

The ray tracer's own uncertainty is quoted from `results/fold_check.md` (commit caf8c0b; training terrains, one azimuth and tilt): the fold term (the map with the other cell diagonal) and the sampling floor (1e9 against 4e9 rays), as median and p95 of the absolute difference. The sampling floor is split only LOS and NLOS there, so both LOS strata quote the LOS value. Read the surrogate's errors against these; they are not a claim that the surrogate beats the ray tracer.

## E1 and E3: path-gain error by stratum, all grid maps

| stratum | method | mean abs | median abs | p95 abs | bias | cells |
|---|---|---|---|---|---|---|
| all | surrogate, 3 seeds | 1.539 (1.499 to 1.569) | 0.677 (0.656 to 0.700) | 6.029 (5.897 to 6.097) | 0.002 (-0.132 to 0.107) | 2545400 |
| all | B0 | 3.297 | 1.389 | 14.056 | +0.702 | 2545400 |
| all | B1 | 4.304 | 1.963 | 16.445 | -3.951 | 2545400 |
| LOS direct | surrogate, 3 seeds | 0.815 (0.799 to 0.847) | 0.482 (0.469 to 0.506) | 2.567 (2.501 to 2.661) | 0.191 (0.103 to 0.270) | 1898934 |
| LOS direct | B0 | 1.249 | 0.887 | 3.069 | -0.192 | 1898934 |
| LOS direct | B1 | 2.194 | 1.196 | 8.378 | -1.903 | 1898934 |
| LOS direct | ray tracer, fold term (quoted) | | 0.484 | 3.287 | | |
| LOS direct | ray tracer, sampling floor (quoted) | | 0.006 | 0.100 | | |
| LOS reflection | surrogate, 3 seeds | 3.404 (3.229 to 3.621) | 2.026 (1.945 to 2.182) | 11.548 (10.822 to 12.282) | -2.674 (-3.058 to -2.421) | 249466 |
| LOS reflection | B0 | 7.606 | 4.676 | 22.816 | -7.606 | 249466 |
| LOS reflection | B1 | 7.768 | 4.746 | 23.437 | -7.768 | 249466 |
| LOS reflection | ray tracer, fold term (quoted) | | 1.715 | 13.241 | | |
| LOS reflection | ray tracer, sampling floor (quoted) | | 0.006 | 0.100 | | |
| NLOS | surrogate, 3 seeds | 3.829 (3.755 to 3.896) | 2.598 (2.490 to 2.665) | 11.834 (11.589 to 12.040) | 0.775 (0.583 to 0.991) | 397000 |
| NLOS | B0 | 10.383 | 8.471 | 24.566 | +10.202 | 397000 |
| NLOS | B1 | 12.219 | 11.370 | 23.771 | -11.347 | 397000 |
| NLOS | ray tracer, fold term (quoted) | | 1.725 | 9.905 | | |
| NLOS | ray tracer, sampling floor (quoted) | | 0.019 | 0.667 | | |

In LOS reflection-dominated cells the surrogate underpredicts (bias -2.7 dB), and on hilltops its median error there (2.76 dB) is above the fold term's median (1.03 dB).

## E3: path-gain error by stratum, per site class

### hilltop

| stratum | method | mean abs | median abs | p95 abs | bias | cells |
|---|---|---|---|---|---|---|
| all | surrogate, 3 seeds | 1.539 (1.507 to 1.572) | 0.589 (0.575 to 0.608) | 6.281 (6.148 to 6.387) | 0.181 (0.131 to 0.272) | 852640 |
| all | B0 | 3.532 | 1.002 | 15.811 | +2.230 | 852640 |
| all | B1 | 4.678 | 1.140 | 19.117 | -4.264 | 852640 |
| LOS direct | surrogate, 3 seeds | 0.692 (0.680 to 0.712) | 0.397 (0.389 to 0.411) | 2.253 (2.210 to 2.315) | 0.097 (0.067 to 0.154) | 621794 |
| LOS direct | B0 | 1.021 | 0.666 | 3.002 | +0.073 | 621794 |
| LOS direct | B1 | 1.569 | 0.720 | 7.241 | -1.200 | 621794 |
| LOS direct | ray tracer, fold term (quoted) | | 0.446 | 2.792 | | |
| LOS direct | ray tracer, sampling floor (quoted) | | 0.008 | 0.115 | | |
| LOS reflection | surrogate, 3 seeds | 3.598 (3.461 to 3.855) | 2.757 (2.658 to 2.956) | 9.863 (9.512 to 10.532) | -2.966 (-3.361 to -2.661) | 26366 |
| LOS reflection | B0 | 9.220 | 6.509 | 23.020 | -9.220 | 26366 |
| LOS reflection | B1 | 9.582 | 6.783 | 24.123 | -9.582 | 26366 |
| LOS reflection | ray tracer, fold term (quoted) | | 1.030 | 3.224 | | |
| LOS reflection | ray tracer, sampling floor (quoted) | | 0.008 | 0.115 | | |
| NLOS | surrogate, 3 seeds | 3.849 (3.768 to 3.942) | 2.584 (2.465 to 2.679) | 11.888 (11.747 to 11.963) | 0.842 (0.756 to 1.008) | 204480 |
| NLOS | B0 | 10.432 | 8.465 | 24.883 | +10.266 | 204480 |
| NLOS | B1 | 13.498 | 12.892 | 24.771 | -12.896 | 204480 |
| NLOS | ray tracer, fold term (quoted) | | 1.444 | 8.849 | | |
| NLOS | ray tracer, sampling floor (quoted) | | 0.027 | 0.900 | | |

### slope

| stratum | method | mean abs | median abs | p95 abs | bias | cells |
|---|---|---|---|---|---|---|
| all | surrogate, 3 seeds | 1.490 (1.449 to 1.519) | 0.708 (0.690 to 0.731) | 5.618 (5.523 to 5.688) | -0.092 (-0.245 to -0.010) | 1044000 |
| all | B0 | 3.082 | 1.607 | 12.091 | -0.452 | 1044000 |
| all | B1 | 3.954 | 2.169 | 14.461 | -3.678 | 1044000 |
| LOS direct | surrogate, 3 seeds | 0.883 (0.858 to 0.916) | 0.527 (0.508 to 0.555) | 2.769 (2.677 to 2.856) | 0.257 (0.148 to 0.319) | 787735 |
| LOS direct | B0 | 1.358 | 1.045 | 3.035 | -0.396 | 787735 |
| LOS direct | B1 | 2.238 | 1.467 | 8.145 | -1.970 | 787735 |
| LOS direct | ray tracer, fold term (quoted) | | 0.427 | 2.735 | | |
| LOS direct | ray tracer, sampling floor (quoted) | | 0.004 | 0.074 | | |
| LOS reflection | surrogate, 3 seeds | 3.087 (2.902 to 3.268) | 1.703 (1.609 to 1.832) | 10.954 (10.134 to 11.680) | -2.286 (-2.609 to -1.990) | 158585 |
| LOS reflection | B0 | 7.211 | 4.395 | 22.864 | -7.211 | 158585 |
| LOS reflection | B1 | 7.305 | 4.445 | 23.302 | -7.305 | 158585 |
| LOS reflection | ray tracer, fold term (quoted) | | 1.399 | 5.051 | | |
| LOS reflection | ray tracer, sampling floor (quoted) | | 0.004 | 0.074 | | |
| NLOS | surrogate, 3 seeds | 3.793 (3.726 to 3.836) | 2.614 (2.520 to 2.724) | 11.563 (11.166 to 11.987) | 0.655 (0.423 to 0.880) | 97680 |
| NLOS | B0 | 10.280 | 8.377 | 24.243 | +10.073 | 97680 |
| NLOS | B1 | 12.353 | 11.586 | 23.500 | -11.558 | 97680 |
| NLOS | ray tracer, fold term (quoted) | | 1.927 | 9.112 | | |
| NLOS | ray tracer, sampling floor (quoted) | | 0.010 | 0.338 | | |

### valley

| stratum | method | mean abs | median abs | p95 abs | bias | cells |
|---|---|---|---|---|---|---|
| all | surrogate, 3 seeds | 1.618 (1.570 to 1.644) | 0.740 (0.706 to 0.765) | 6.329 (6.201 to 6.460) | -0.084 (-0.294 to 0.093) | 648760 |
| all | B0 | 3.334 | 1.544 | 14.141 | +0.552 | 648760 |
| all | B1 | 4.375 | 2.468 | 12.668 | -3.978 | 648760 |
| LOS direct | surrogate, 3 seeds | 0.863 (0.834 to 0.907) | 0.538 (0.510 to 0.573) | 2.624 (2.531 to 2.767) | 0.205 (0.072 to 0.337) | 489405 |
| LOS direct | B0 | 1.364 | 1.090 | 3.186 | -0.201 | 489405 |
| LOS direct | B1 | 2.917 | 1.720 | 9.151 | -2.687 | 489405 |
| LOS direct | ray tracer, fold term (quoted) | | 0.591 | 4.821 | | |
| LOS direct | ray tracer, sampling floor (quoted) | | 0.006 | 0.084 | | |
| LOS reflection | surrogate, 3 seeds | 4.103 (3.940 to 4.391) | 2.534 (2.401 to 2.723) | 13.424 (12.746 to 14.088) | -3.508 (-4.036 to -3.195) | 64515 |
| LOS reflection | B0 | 7.916 | 4.938 | 22.528 | -7.916 | 64515 |
| LOS reflection | B1 | 8.167 | 5.066 | 23.458 | -8.167 | 64515 |
| LOS reflection | ray tracer, fold term (quoted) | | 4.083 | 18.456 | | |
| LOS reflection | ray tracer, sampling floor (quoted) | | 0.006 | 0.084 | | |
| NLOS | surrogate, 3 seeds | 3.823 (3.756 to 3.861) | 2.615 (2.515 to 2.734) | 11.942 (11.567 to 12.224) | 0.754 (0.363 to 1.071) | 94840 |
| NLOS | B0 | 10.384 | 8.565 | 24.182 | +10.198 | 94840 |
| NLOS | B1 | 9.321 | 8.717 | 20.195 | -7.788 | 94840 |
| NLOS | ray tracer, fold term (quoted) | | 1.844 | 10.696 | | |
| NLOS | ray tracer, sampling floor (quoted) | | 0.019 | 0.648 | | |

## E3: the no-signal class

The surrogate's power head (logit above 0 means the ray tracer has power) against the traced power mask, over all cells of the grid maps. B0 and B1 always predict power, so they have no no-signal class.

| group | power precision | power recall | no-signal precision | no-signal recall | accuracy | cells with no signal |
|---|---|---|---|---|---|---|
| all | 97.75 (97.57 to 97.92)% | 97.77 (97.61 to 97.90)% | 98.30 (98.19 to 98.41)% | 98.29 (98.15 to 98.43)% | 98.06 (98.00 to 98.12)% | 56.8% |
| hilltop | 96.77 (96.47 to 97.01)% | 96.54 (96.41 to 96.61)% | 97.35 (97.27 to 97.40)% | 97.53 (97.29 to 97.72)% | 97.10 (96.99 to 97.16)% | 56.6% |
| slope | 98.64 (98.59 to 98.73)% | 98.68 (98.58 to 98.82)% | 98.51 (98.40 to 98.66)% | 98.46 (98.40 to 98.56)% | 98.58 (98.53 to 98.64)% | 46.9% |
| valley | 97.58 (97.39 to 97.82)% | 97.90 (97.60 to 98.15)% | 98.97 (98.82 to 99.09)% | 98.80 (98.71 to 98.93)% | 98.51 (98.46 to 98.57)% | 67.0% |

## E2: coverage

RSRP = 12.21 dBm per resource element (43 dBm over 1200 resource elements of a 20 MHz carrier, ASSUMPTION) plus path gain; a cell is covered at -110 dBm (ASSUMPTION), i.e. path gain of at least -122.21 dB. The ray tracer's covered cells need power; the surrogate's need its power head to say power; B0 and B1 always predict power. Covered-area error: predicted minus traced covered cells over traced covered cells, pooled over the grid maps. IoU: pooled, and the mean of per-map IoU.

| method | covered-area error | IoU (pooled) | IoU (mean per map) |
|---|---|---|---|
| surrogate, 3 seeds | 0.26 (-0.28 to 0.63)% | 0.9353 (0.9336 to 0.9368) | 0.9272 (0.9253 to 0.9288) |
| B0 | +130.94% | 0.4127 | 0.4135 |
| B1 | -2.31% | 0.8026 | 0.7953 |

## E4: off-grid tilts

Not in this split: off-grid tilts (1.5, 4.5, 7.5 deg) exist on test only.

## E5: time per map, same machine

GPU: NVIDIA GeForce RTX 4060 Laptop GPU, driver 616.56. Medians, in seconds. The ray tracer runs the dataset's settings (1e+09 rays): its new-terrain figure covers terrain generation, scene and mesh build, measurement surface and the first solve; further maps reuse the scene and time the solve only. The surrogate's new-terrain figure covers geometry, LOS mask, B0, the per-map channels and inference; further maps cover B0, the channels and inference. Both sides skip their warm-up (kernel compilation). GPU ray tracer: terrains 43, 46, 49 through the Windows runner (commit caf8c0b); CPU ray tracer: terrains 43, 46, 49, llvm, measured here.

| hardware | case | ray tracer | surrogate | ray tracer / surrogate |
|---|---|---|---|---|
| GPU | new terrain, first map | 1.417 | 0.4402 | 3.2 |
| GPU | each further map | 1.403 | 0.0083 | 169.7 |
| CPU | new terrain, first map | 41.702 | 0.4653 | 89.6 |
| CPU | each further map | 41.088 | 0.0334 | 1231.6 |

Surrogate parts: geometry and LOS mask for a new terrain 0.432; B0 and channels per map 0.0046; inference, batch 1, GPU 0.0037 and CPU 0.0288.

On this machine, per map, the ray tracer takes 3.2 times as long as the surrogate for a new terrain's first map and 169.7 times as long for each further map on the GPU; without a GPU the ratios are 89.6 and 1231.6.

## Per seed, all grid maps

| seed | mean abs, all | mean abs, LOS direct | mean abs, LOS reflection | mean abs, NLOS | power accuracy |
|---|---|---|---|---|---|
| seed 0 | 1.549 | 0.799 | 3.621 | 3.836 | 98.07% |
| seed 1 | 1.569 | 0.847 | 3.362 | 3.896 | 98.00% |
| seed 2 | 1.499 | 0.800 | 3.229 | 3.755 | 98.12% |
