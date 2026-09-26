# Solver check

Synthetic terrain and a flat test tile, not a real place. Propagation: Sionna RT; pattern: 3GPP TR 38.901. Line of sight and specular reflection only. Written by `twin solver-check`.

| provenance | |
|---|---|
| commit | 474693ab941ae645936633955c27bbeb9ed03ee2 |
| variant | llvm_ad_mono_polarized |
| platform | Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39 |
| python | 3.13.12 |
| sionna-rt | 2.1.0 |
| mitsuba | 3.9.1 |
| drjit | 1.5.0 |
| numpy | 2.5.3 |
| gpu | NVIDIA GeForce RTX 4060 Laptop GPU, driver 616.56 |

Settings of the A4 and S5 maps:

| setting | value |
|---|---|
| variant | llvm_ad_mono_polarized |
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
| mesh surface vs planar map | 1.9e-07 | 6.4e-07 | +4.2e-06 | 100.0% |
| planar map, seed vs seed + 1 | 0.0e+00 | 0.0e+00 | +3.3e-09 | 100.0% |
| mesh surface, seed vs seed + 1 | 0.0e+00 | 0.0e+00 | +6.9e-10 | 100.0% |

## Time per map and the sampling floor (N3)

Terrains 3, 1, 5, azimuth 90, tilt 6, the ruled settings (line of sight and specular reflection, max_depth 3). The floor compares 1e8 with 1e9 samples over cells with power in both, overall and split by the LOS mask (see `baselines.py`). Rays are launched on a fixed lattice, so the seed does not change these maps; the sample count does.

| terrain | site | s/map 1e7 | s/map 1e8 | s/map 1e9 | floor median / p95 | LOS median / p95 (cells) | NLOS median / p95 (cells) | no-hit 1e8 | no-hit 1e9 | hit only at 1e9 | hit only at 1e8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | hilltop | 0.3 | 3.0 | 30.2 | 0.023 / 0.484 | 0.019 / 0.314 (5608) | 0.094 / 1.874 (900) | 60.28% | 60.14% | 23 | 0 |
| 1 | slope | 0.4 | 3.5 | 35.0 | 0.021 / 0.448 | 0.016 / 0.236 (2236) | 0.061 / 1.477 (628) | 82.52% | 82.41% | 18 | 0 |
| 5 | valley | 0.3 | 3.1 | 29.4 | 0.035 / 0.581 | 0.030 / 0.391 (6111) | 0.109 / 1.721 (970) | 56.78% | 56.50% | 46 | 0 |
