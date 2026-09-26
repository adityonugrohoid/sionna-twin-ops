# Solver check

Synthetic terrain and a flat test tile, not a real place. Propagation: Sionna RT; pattern: 3GPP TR 38.901. Line of sight and specular reflection only. Written by `twin solver-check`.

| provenance | |
|---|---|
| commit | c4246c0ecc992e0ce33e6f5f066e59d69ff8f8d5 |
| variant | cuda_ad_mono_polarized |
| platform | Windows-11-10.0.26200-SP0 |
| python | 3.13.12 |
| sionna-rt | 2.1.0 |
| mitsuba | 3.9.1 |
| drjit | 1.5.0 |
| numpy | 2.5.3 |
| gpu | NVIDIA GeForce RTX 4060 Laptop GPU, driver 616.56 |

Settings of the A4 and S5 maps:

| setting | value |
|---|---|
| variant | cuda_ad_mono_polarized |
| samples_per_tx | 10000000 |
| max_depth | 3 |
| los | True |
| specular_reflection | True |
| diffuse_reflection | False |
| refraction | False |
| diffraction | False |
| edge_diffraction | False |
| seed | 1 |

Main-lobe measurements: free space, 1e+08 rays, a vertical planar map 200 m out on boresight with 1 m cells.

## A4 flat-plain tilt check: PASS

Criteria: main-lobe elevation within 1.0 deg of the commanded tilt; far-field median moves by at least 6.0 dB across the tilts. Far field (ASSUMPTION): cells 1500 to 2500 m from the site within 60 deg of boresight.

| tilt (deg) | main-lobe elevation (deg) | error (deg) | far-field median (dB) | far-field cells hit |
|---|---|---|---|---|
| 0 | +0.02 | +0.02 | -87.04 | 100.0% |
| 3 | -2.93 | +0.07 | -87.75 | 100.0% |
| 6 | -5.86 | +0.14 | -92.25 | 100.0% |
| 9 | -8.79 | +0.21 | -107.07 | 100.0% |
| 12 | -11.67 | +0.33 | -101.72 | 100.0% |

Largest lobe error 0.33 deg; far-field span 20.0 dB. The median is not monotonic in tilt: the 8-element, 0.8-wavelength column has nulls about 9 deg apart, so as the lobe tilts down the far-field ring passes through the first null and then the first side lobe.

## S5 measurement-surface check

Flat tile. The mesh surface at 1.5 m against a planar radio map at 1.5 m, same cells and settings; the two-seed spreads show the noise each map has on its own. Values in dB over cells with power in both maps.

| comparison | median abs | p95 abs | bias | cells |
|---|---|---|---|---|
| mesh surface vs planar map | 2.2e-07 | 1.3e-06 | +4.3e-06 | 100.0% |
| planar map, seed vs seed + 1 | 0.0e+00 | 5.1e-07 | -1.1e-09 | 100.0% |
| mesh surface, seed vs seed + 1 | 0.0e+00 | 3.2e-07 | +3.4e-10 | 100.0% |

## Time per map and the sampling floor (N3)

Terrains 3, 1, 5, azimuth 90, tilt 6, the ruled settings (line of sight and specular reflection, max_depth 3), compared over cells with power in both, overall and split by the LOS mask (see `baselines.py`).

The floor compares the dataset's 1e+09 rays with 4e+09. A larger reference is not possible: Mitsuba's sampler wavefront is 32-bit, so one solve launches at most 4294967295 rays, and repeating solves adds nothing because the rays come from the same deterministic lattice each time (the seed does not change these maps). A 4x step understates the error against the fully converged map more than the earlier 10x step (1e+08 vs 1e+09) did; that step is listed after the main table as context. A lattice with a different ray count points its rays in different directions rather than adding to the old ones, so a grazing cell reached by a single ray at one count can be missed at another: that is why a cell or two can have power at the smaller count only.

| terrain | site | s/map 1e+07 | s/map 1e+08 | s/map 1e+09 | s/map 4e+09 | floor median / p95 | LOS median / p95 (cells) | NLOS median / p95 (cells) | no-hit 1e+09 | no-hit 4e+09 | hit only at 4e+09 | hit only at 1e+09 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | hilltop | 0.01 | 0.10 | 1.02 | 4.07 | 0.005 / 0.091 | 0.004 / 0.059 (5609) | 0.016 / 0.587 (921) | 60.14% | 60.10% | 7 | 1 |
| 1 | slope | 0.01 | 0.10 | 1.00 | 4.04 | 0.005 / 0.114 | 0.004 / 0.066 (2236) | 0.012 / 0.396 (645) | 82.41% | 82.39% | 4 | 1 |
| 5 | valley | 0.01 | 0.08 | 0.85 | 3.41 | 0.006 / 0.138 | 0.005 / 0.083 (6112) | 0.019 / 0.825 (1015) | 56.50% | 56.46% | 7 | 0 |

Context, 1e+08 against 1e+09:

| terrain | site | LOS median / p95 (cells) | NLOS median / p95 (cells) |
|---|---|---|---|
| 3 | hilltop | 0.019 / 0.314 (5608) | 0.094 / 1.874 (900) |
| 1 | slope | 0.017 / 0.238 (2236) | 0.062 / 1.477 (628) |
| 5 | valley | 0.030 / 0.391 (6111) | 0.111 / 1.743 (970) |
