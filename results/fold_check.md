# Fold check

Synthetic terrain ids 3, 6, 1, 4, 5, 8 (training terrains, two per site class), azimuth 90, tilt 6; line of sight and specular reflection. The terrain mesh and the measurement surface split every cell into two triangles along one diagonal; the dataset uses the SW-NE one. The fold term compares the dataset's SW-NE maps with NW-SE maps at 1e+09 rays; the sampling floor compares the same SW-NE maps with 4e+09 rays. Both are absolute dB differences over cells with power in both maps, pooled over the terrains of each row. Written by `twin fold-check`.

| provenance | |
|---|---|
| commit | 73018e685c42afe8cc4b833bf12ffb00e68ea51a |
| variant | cuda_ad_mono_polarized |
| platform | Windows-11-10.0.26200-SP0 |
| python | 3.13.12 |
| sionna-rt | 2.1.0 |
| mitsuba | 3.9.1 |
| drjit | 1.5.0 |
| numpy | 2.5.3 |
| gpu | NVIDIA GeForce RTX 4060 Laptop GPU, driver 616.56 |

## Truth uncertainty: fold term beside the sampling floor

| site class | term | LOS median | LOS p95 | LOS max | LOS cells | NLOS median | NLOS p95 | NLOS max | NLOS cells | power only with SW-NE | power only in the other map |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hilltop | fold | 0.492 | 2.885 | 14.958 | 17751 | 1.444 | 8.849 | 26.248 | 1456 | 94 | 107 |
| hilltop | sampling | 0.008 | 0.115 | 14.044 | 17758 | 0.027 | 0.900 | 6.021 | 1541 | 2 | 20 |
| slope | fold | 0.449 | 2.906 | 20.069 | 4264 | 1.927 | 9.112 | 25.365 | 1197 | 78 | 68 |
| slope | sampling | 0.004 | 0.074 | 2.079 | 4264 | 0.010 | 0.338 | 6.021 | 1273 | 2 | 11 |
| valley | fold | 0.765 | 9.037 | 37.549 | 11642 | 1.844 | 10.696 | 31.778 | 2490 | 209 | 116 |
| valley | sampling | 0.006 | 0.084 | 14.440 | 11652 | 0.019 | 0.648 | 19.296 | 2688 | 1 | 26 |
| all | fold | 0.558 | 4.570 | 37.549 | 33657 | 1.725 | 9.905 | 31.778 | 5143 | 381 | 291 |
| all | sampling | 0.006 | 0.100 | 14.440 | 33674 | 0.019 | 0.667 | 19.296 | 5502 | 5 | 57 |

## Fold term on LOS cells by distance to the nearest shadow cell

Distance is Euclidean, in cells (40 m), from a LOS cell to the nearest non-LOS cell.

| site class | distance (cells) | median | p95 | max | cells |
|---|---|---|---|---|---|
| hilltop | 1 to 2 | 0.529 | 3.456 | 14.958 | 3402 |
| hilltop | 2 to 4 | 0.422 | 2.701 | 11.728 | 5017 |
| hilltop | 4 or more | 0.519 | 2.695 | 10.353 | 9332 |
| slope | 1 to 2 | 0.598 | 3.451 | 20.069 | 1991 |
| slope | 2 to 4 | 0.359 | 2.234 | 8.762 | 1673 |
| slope | 4 or more | 0.325 | 2.333 | 6.822 | 600 |
| valley | 1 to 2 | 0.823 | 7.351 | 34.697 | 4752 |
| valley | 2 to 4 | 0.703 | 9.993 | 37.549 | 4421 |
| valley | 4 or more | 0.794 | 10.196 | 29.431 | 2469 |
| all | 1 to 2 | 0.670 | 4.970 | 34.697 | 10145 |
| all | 2 to 4 | 0.490 | 5.024 | 37.549 | 11111 |
| all | 4 or more | 0.541 | 3.843 | 29.431 | 12401 |

## Fold term by dominant path

A cell with power in the SW-NE map is reflection-dominated when its traced gain exceeds B0 (direct path and antenna pattern, no terrain) by 3 dB or more (ASSUMPTION), and direct-dominated otherwise. Median / p95 of the fold term, and the cells compared. B0 has no terrain, so shadowed cells rarely exceed it: the NLOS split says little, and nearly all NLOS cells fall under direct.

| site class | LOS direct | LOS reflection | NLOS direct | NLOS reflection | LOS cells reflection-dominated |
|---|---|---|---|---|---|
| hilltop | 0.446 / 2.792 (15685) | 1.030 / 3.224 (2066) | 1.444 / 8.849 (1456) | - / - (0) | 11.6% |
| slope | 0.427 / 2.735 (4081) | 1.399 / 5.051 (183) | 1.928 / 9.063 (1190) | 1.764 / 9.528 (7) | 4.3% |
| valley | 0.591 / 4.821 (9644) | 4.083 / 18.456 (1998) | 1.800 / 10.403 (2454) | 8.523 / 21.462 (36) | 17.1% |
| all | 0.484 / 3.287 (29410) | 1.715 / 13.241 (4247) | 1.718 / 9.685 (5100) | 7.091 / 20.011 (43) | 12.6% |
