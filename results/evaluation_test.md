# Evaluation on the test split

Synthetic terrain, not a real place; one sector, no vegetation, buildings or interference, flat Earth. Ground truth: Sionna RT, line of sight and specular reflection only (cuda_ad_mono_polarized, 1e+09 rays, Sionna RT 2.1.0; map commits eb4b064); pattern: 3GPP TR 38.901. Written by `twin evaluate`.

Maps: 414 (360 grid, 54 off-grid) on terrains 52, 55, 58, 71, 72, 74, 77, 78, 81. Surrogate: the best-epoch checkpoints of runs/model-f6f13e3-v2/seed0, runs/model-f6f13e3-v2/seed1, runs/model-f6f13e3-v2/seed2; each surrogate figure is the mean over the seeds, with the lowest and highest seed in brackets.

Errors are predicted minus traced path gain in dB, over cells where the ray tracer has power. Strata: LOS direct-dominated and LOS reflection-dominated (traced gain at least 3 dB above B0, ASSUMPTION) and NLOS, by the heightmap LOS mask. B0: free space plus the antenna pattern; B1: B0 minus the Bullington diffraction loss (ITU-R P.526-16 section 4.5.1).

The ray tracer's own uncertainty is quoted from `results/fold_check.md` (commit 73018e6; training terrains, one azimuth and tilt): the fold term (the map with the other cell diagonal) and the sampling floor (1e9 against 4e9 rays), as median and p95 of the absolute difference. The sampling floor is split only LOS and NLOS there, so both LOS strata quote the LOS value. Read the surrogate's errors against these; they are not a claim that the surrogate beats the ray tracer.

## E1 and E3: path-gain error by stratum, all grid maps

| stratum | method | mean abs | median abs | p95 abs | bias | cells |
|---|---|---|---|---|---|---|
| all | surrogate, 3 seeds | 1.549 (1.524 to 1.584) | 0.663 (0.645 to 0.698) | 6.305 (6.275 to 6.336) | 0.111 (-0.011 to 0.237) | 2359640 |
| all | B0 | 3.132 | 1.275 | 13.308 | +0.541 | 2359640 |
| all | B1 | 3.844 | 1.744 | 13.001 | -3.483 | 2359640 |
| LOS direct | surrogate, 3 seeds | 0.838 (0.810 to 0.887) | 0.492 (0.470 to 0.526) | 2.690 (2.585 to 2.851) | 0.333 (0.243 to 0.428) | 1819331 |
| LOS direct | B0 | 1.236 | 0.921 | 3.044 | -0.173 | 1819331 |
| LOS direct | B1 | 2.200 | 1.195 | 8.203 | -1.946 | 1819331 |
| LOS direct | ray tracer, fold term (quoted) | | 0.484 | 3.287 | | |
| LOS direct | ray tracer, sampling floor (quoted) | | 0.006 | 0.100 | | |
| LOS reflection | surrogate, 3 seeds | 4.084 (3.968 to 4.200) | 2.887 (2.766 to 2.998) | 11.378 (11.215 to 11.584) | -3.017 (-3.271 to -2.804) | 207869 |
| LOS reflection | B0 | 8.410 | 5.856 | 22.745 | -8.410 | 207869 |
| LOS reflection | B1 | 8.610 | 6.028 | 23.099 | -8.610 | 207869 |
| LOS reflection | ray tracer, fold term (quoted) | | 1.715 | 13.241 | | |
| LOS reflection | ray tracer, sampling floor (quoted) | | 0.006 | 0.100 | | |
| NLOS | surrogate, 3 seeds | 3.856 (3.793 to 3.907) | 2.524 (2.442 to 2.582) | 12.428 (12.245 to 12.751) | 0.856 (0.636 to 1.092) | 332440 |
| NLOS | B0 | 10.207 | 8.174 | 25.090 | +10.043 | 332440 |
| NLOS | B1 | 9.857 | 9.518 | 19.059 | -8.689 | 332440 |
| NLOS | ray tracer, fold term (quoted) | | 1.725 | 9.905 | | |
| NLOS | ray tracer, sampling floor (quoted) | | 0.019 | 0.667 | | |

In LOS reflection-dominated cells the surrogate underpredicts (bias -3.0 dB), and on hilltops its median error there (2.67 dB) is above the fold term's median (1.03 dB).

## E3: path-gain error by stratum, per site class

### hilltop

| stratum | method | mean abs | median abs | p95 abs | bias | cells |
|---|---|---|---|---|---|---|
| all | surrogate, 3 seeds | 1.434 (1.407 to 1.471) | 0.636 (0.619 to 0.668) | 5.838 (5.794 to 5.868) | 0.085 (-0.019 to 0.185) | 1363840 |
| all | B0 | 2.786 | 1.283 | 11.312 | -0.224 | 1363840 |
| all | B1 | 3.394 | 1.713 | 10.578 | -3.065 | 1363840 |
| LOS direct | surrogate, 3 seeds | 0.865 (0.838 to 0.914) | 0.499 (0.481 to 0.531) | 2.821 (2.732 to 2.977) | 0.384 (0.295 to 0.467) | 1102130 |
| LOS direct | B0 | 1.293 | 1.004 | 2.998 | -0.288 | 1102130 |
| LOS direct | B1 | 2.222 | 1.305 | 8.109 | -1.967 | 1102130 |
| LOS direct | ray tracer, fold term (quoted) | | 0.446 | 2.792 | | |
| LOS direct | ray tracer, sampling floor (quoted) | | 0.008 | 0.115 | | |
| LOS reflection | surrogate, 3 seeds | 3.903 (3.823 to 3.989) | 2.670 (2.558 to 2.786) | 10.987 (10.926 to 11.081) | -2.909 (-3.031 to -2.772) | 143150 |
| LOS reflection | B0 | 8.223 | 5.700 | 22.193 | -8.223 | 143150 |
| LOS reflection | B1 | 8.339 | 5.816 | 22.369 | -8.339 | 143150 |
| LOS reflection | ray tracer, fold term (quoted) | | 1.030 | 3.224 | | |
| LOS reflection | ray tracer, sampling floor (quoted) | | 0.008 | 0.115 | | |
| NLOS | surrogate, 3 seeds | 3.750 (3.695 to 3.811) | 2.467 (2.391 to 2.518) | 12.247 (12.070 to 12.578) | 0.920 (0.702 to 1.132) | 118560 |
| NLOS | B0 | 10.100 | 8.240 | 24.284 | +10.031 | 118560 |
| NLOS | B1 | 8.311 | 8.343 | 15.843 | -6.905 | 118560 |
| NLOS | ray tracer, fold term (quoted) | | 1.444 | 8.849 | | |
| NLOS | ray tracer, sampling floor (quoted) | | 0.027 | 0.900 | | |

### slope

| stratum | method | mean abs | median abs | p95 abs | bias | cells |
|---|---|---|---|---|---|---|
| all | surrogate, 3 seeds | 1.657 (1.632 to 1.682) | 0.654 (0.640 to 0.681) | 6.806 (6.769 to 6.860) | 0.096 (-0.041 to 0.251) | 488120 |
| all | B0 | 3.600 | 1.170 | 15.863 | +1.648 | 488120 |
| all | B1 | 4.518 | 1.531 | 16.793 | -4.145 | 488120 |
| LOS direct | surrogate, 3 seeds | 0.733 (0.706 to 0.774) | 0.435 (0.415 to 0.472) | 2.295 (2.214 to 2.414) | 0.176 (0.110 to 0.270) | 348455 |
| LOS direct | B0 | 1.069 | 0.722 | 2.945 | -0.041 | 348455 |
| LOS direct | B1 | 1.882 | 0.859 | 7.997 | -1.628 | 348455 |
| LOS direct | ray tracer, fold term (quoted) | | 0.427 | 2.735 | | |
| LOS direct | ray tracer, sampling floor (quoted) | | 0.004 | 0.074 | | |
| LOS reflection | surrogate, 3 seeds | 4.231 (4.020 to 4.435) | 3.114 (2.990 to 3.214) | 11.409 (10.816 to 12.201) | -3.018 (-3.578 to -2.596) | 30585 |
| LOS reflection | B0 | 8.869 | 6.158 | 24.080 | -8.869 | 30585 |
| LOS reflection | B1 | 9.181 | 6.375 | 24.670 | -9.181 | 30585 |
| LOS reflection | ray tracer, fold term (quoted) | | 1.399 | 5.051 | | |
| LOS reflection | ray tracer, sampling floor (quoted) | | 0.004 | 0.074 | | |
| NLOS | surrogate, 3 seeds | 3.885 (3.814 to 3.928) | 2.521 (2.417 to 2.608) | 12.539 (12.307 to 12.851) | 0.715 (0.466 to 0.990) | 109080 |
| NLOS | B0 | 10.207 | 8.039 | 25.712 | +9.993 | 109080 |
| NLOS | B1 | 11.632 | 11.209 | 21.332 | -10.773 | 109080 |
| NLOS | ray tracer, fold term (quoted) | | 1.927 | 9.112 | | |
| NLOS | ray tracer, sampling floor (quoted) | | 0.010 | 0.338 | | |

### valley

| stratum | method | mean abs | median abs | p95 abs | bias | cells |
|---|---|---|---|---|---|---|
| all | surrogate, 3 seeds | 1.754 (1.733 to 1.793) | 0.758 (0.727 to 0.807) | 7.096 (7.060 to 7.123) | 0.197 (0.037 to 0.363) | 507680 |
| all | B0 | 3.612 | 1.335 | 15.427 | +1.529 | 507680 |
| all | B1 | 4.404 | 2.103 | 14.038 | -3.970 | 507680 |
| LOS direct | surrogate, 3 seeds | 0.857 (0.810 to 0.914) | 0.527 (0.495 to 0.572) | 2.695 (2.518 to 2.889) | 0.328 (0.212 to 0.461) | 368746 |
| LOS direct | B0 | 1.224 | 0.872 | 3.327 | +0.045 | 368746 |
| LOS direct | B1 | 2.436 | 1.204 | 8.554 | -2.185 | 368746 |
| LOS direct | ray tracer, fold term (quoted) | | 0.591 | 4.821 | | |
| LOS direct | ray tracer, sampling floor (quoted) | | 0.006 | 0.084 | | |
| LOS reflection | surrogate, 3 seeds | 4.714 (4.527 to 4.874) | 3.446 (3.312 to 3.553) | 12.855 (12.435 to 13.189) | -3.469 (-4.002 to -3.128) | 34134 |
| LOS reflection | B0 | 8.786 | 6.252 | 23.240 | -8.786 | 34134 |
| LOS reflection | B1 | 9.232 | 6.622 | 23.864 | -9.232 | 34134 |
| LOS reflection | ray tracer, fold term (quoted) | | 4.083 | 18.456 | | |
| LOS reflection | ray tracer, sampling floor (quoted) | | 0.006 | 0.084 | | |
| NLOS | surrogate, 3 seeds | 3.944 (3.880 to 3.995) | 2.594 (2.528 to 2.665) | 12.496 (12.220 to 12.800) | 0.929 (0.739 to 1.153) | 104800 |
| NLOS | B0 | 10.328 | 8.220 | 25.453 | +10.110 | 104800 |
| NLOS | B1 | 9.757 | 9.566 | 18.395 | -8.538 | 104800 |
| NLOS | ray tracer, fold term (quoted) | | 1.844 | 10.696 | | |
| NLOS | ray tracer, sampling floor (quoted) | | 0.019 | 0.648 | | |

## E3: the no-signal class

The surrogate's power head (logit above 0 means the ray tracer has power) against the traced power mask, over all cells of the grid maps. B0 and B1 always predict power, so they have no no-signal class.

| group | power precision | power recall | no-signal precision | no-signal recall | accuracy | cells with no signal |
|---|---|---|---|---|---|---|
| all | 97.92 (97.80 to 98.10)% | 97.99 (97.87 to 98.16)% | 98.66 (98.58 to 98.77)% | 98.61 (98.53 to 98.74)% | 98.36 (98.31 to 98.39)% | 60.0% |
| hilltop | 98.68 (98.63 to 98.78)% | 98.70 (98.63 to 98.84)% | 97.06 (96.91 to 97.36)% | 97.02 (96.89 to 97.24)% | 98.19 (98.11 to 98.24)% | 30.6% |
| slope | 96.82 (96.61 to 97.09)% | 96.88 (96.68 to 97.11)% | 98.97 (98.90 to 99.04)% | 98.95 (98.87 to 99.04)% | 98.44 (98.42 to 98.45)% | 75.2% |
| valley | 96.91 (96.73 to 97.26)% | 97.12 (96.97 to 97.36)% | 99.00 (98.95 to 99.08)% | 98.92 (98.86 to 99.05)% | 98.46 (98.39 to 98.51)% | 74.2% |

## E2: coverage

RSRP = 12.21 dBm per resource element (43 dBm over 1200 resource elements of a 20 MHz carrier, ASSUMPTION) plus path gain; a cell is covered at -110 dBm (ASSUMPTION), i.e. path gain of at least -122.21 dB. The ray tracer's covered cells need power; the surrogate's need its power head to say power; B0 and B1 always predict power. Covered-area error: predicted minus traced covered cells over traced covered cells, pooled over the grid maps. IoU: pooled, and the mean of per-map IoU.

| method | covered-area error | IoU (pooled) | IoU (mean per map) |
|---|---|---|---|
| surrogate, 3 seeds | 0.48 (-0.06 to 0.83)% | 0.9385 (0.9369 to 0.9394) | 0.9274 (0.9258 to 0.9281) |
| B0 | +139.68% | 0.3977 | 0.3866 |
| B1 | -0.11% | 0.8018 | 0.7805 |

## E4: off-grid tilts

The 54 off-grid maps (tilts 1.5, 4.5 and 7.5 deg), which no model saw during training.

| stratum | method | mean abs | median abs | p95 abs | bias | cells |
|---|---|---|---|---|---|---|
| all | surrogate, 3 seeds | 1.336 (1.312 to 1.371) | 0.593 (0.577 to 0.622) | 5.122 (5.084 to 5.171) | 0.168 (0.044 to 0.291) | 353946 |
| all | B0 | 2.799 | 1.200 | 11.907 | +0.873 | 353946 |
| all | B1 | 3.512 | 1.585 | 12.010 | -3.151 | 353946 |
| LOS direct | surrogate, 3 seeds | 0.766 (0.740 to 0.811) | 0.464 (0.445 to 0.494) | 2.434 (2.340 to 2.586) | 0.250 (0.159 to 0.341) | 284558 |
| LOS direct | B0 | 1.230 | 0.924 | 2.999 | -0.198 | 284558 |
| LOS direct | B1 | 2.156 | 1.182 | 8.120 | -1.906 | 284558 |
| LOS reflection | surrogate, 3 seeds | 3.284 (3.133 to 3.435) | 2.214 (2.080 to 2.354) | 10.282 (10.031 to 10.674) | -2.668 (-3.032 to -2.328) | 19522 |
| LOS reflection | B0 | 6.842 | 4.488 | 19.682 | -6.842 | 19522 |
| LOS reflection | B1 | 7.046 | 4.562 | 20.390 | -7.046 | 19522 |
| NLOS | surrogate, 3 seeds | 3.822 (3.762 to 3.875) | 2.514 (2.435 to 2.569) | 12.288 (12.085 to 12.582) | 0.809 (0.596 to 1.030) | 49866 |
| NLOS | B0 | 10.171 | 8.130 | 24.970 | +10.005 | 49866 |
| NLOS | B1 | 9.865 | 9.511 | 18.979 | -8.727 | 49866 |

## E5: time per map, same machine

GPU: NVIDIA GeForce RTX 4060 Laptop GPU, driver 616.56. Medians, in seconds. The ray tracer runs the dataset's settings (1e+09 rays): its new-terrain figure covers terrain generation, scene and mesh build, measurement surface and the first solve; further maps reuse the scene and time the solve only. The surrogate's new-terrain figure covers geometry, LOS mask, B0, the per-map channels and inference; further maps cover B0, the channels and inference. Both sides skip their warm-up (kernel compilation). GPU ray tracer: terrains 52, 55, 58 through the Windows runner (commit c5de482); CPU ray tracer: terrains 52, 55, 58, llvm, measured here.

| hardware | case | ray tracer | surrogate | ray tracer / surrogate |
|---|---|---|---|---|
| GPU | new terrain, first map | 1.066 | 0.4038 | 2.6 |
| GPU | each further map | 1.033 | 0.0080 | 129.3 |
| CPU | new terrain, first map | 35.415 | 0.4295 | 82.5 |
| CPU | each further map | 35.333 | 0.0336 | 1050.9 |

Surrogate parts: geometry and LOS mask for a new terrain 0.396; B0 and channels per map 0.0045; inference, batch 1, GPU 0.0035 and CPU 0.0292.

On this machine, per map, the ray tracer takes 2.6 times as long as the surrogate for a new terrain's first map and 129.3 times as long for each further map on the GPU; without a GPU the ratios are 82.5 and 1050.9.

## Per seed, all grid maps

| seed | mean abs, all | mean abs, LOS direct | mean abs, LOS reflection | mean abs, NLOS | power accuracy |
|---|---|---|---|---|---|
| seed 0 | 1.539 | 0.810 | 4.200 | 3.867 | 98.39% |
| seed 1 | 1.584 | 0.887 | 3.968 | 3.907 | 98.31% |
| seed 2 | 1.524 | 0.817 | 4.085 | 3.793 | 98.38% |
