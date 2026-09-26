# Dataset summary

Synthetic terrain, not a real place. Propagation: Sionna RT, line of sight and specular reflection only (no diffraction, no diffuse scattering); pattern: 3GPP TR 38.901. One sector per map, no vegetation or buildings, flat Earth. Written by `twin dataset-summary`; the maps themselves are not in git.

Maps in the manifest: 2454 of 2454 expected.

## Terrains and maps per split and site class

| split | hilltop | slope | valley | terrains | grid maps | off-grid maps |
|---|---|---|---|---|---|---|
| train | 14 | 14 | 14 | 42 | 1680 | 0 |
| validation | 3 | 3 | 3 | 9 | 360 | 0 |
| test | 3 | 3 | 3 | 9 | 360 | 54 |

Terrain ids by split: train 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 40, 44, 45, 47, 50, 51, 53, 54, 57; validation 43, 46, 49, 60, 62, 63, 65, 68, 69; test 52, 55, 58, 71, 72, 74, 77, 78, 81.

## Skipped terrain ids

Ids with no site candidate for their class (spec S2); the walk moved on to the next id of that class. Highest id walked: 81.

| site class | skipped | ids |
|---|---|---|
| hilltop | 8 | 0, 12, 15, 39, 42, 48, 66, 75 |
| slope | 0 | none |
| valley | 6 | 2, 26, 35, 41, 56, 59 |

## No-hit share per site class

Share of map cells no ray reached (the 'no signal' class of spec E3), over all maps of the class.

| site class | maps | mean | min | max |
|---|---|---|---|---|
| hilltop | 818 | 46.4% | 17.7% | 71.6% |
| slope | 818 | 64.3% | 9.9% | 94.2% |
| valley | 818 | 72.6% | 48.6% | 95.6% |

## Time and settings

Solver wall time over all maps: 0.75 h (2710 s; median 1.05 s per map). Scene building and file writing are not included.

| setting | value |
|---|---|
| variant | cuda_ad_mono_polarized |
| samples_per_tx | 1000000000 |
| max_depth | 3 |
| los | True |
| specular_reflection | True |
| diffuse_reflection | False |
| refraction | False |
| diffraction | False |
| edge_diffraction | False |
| seed | 1 |

| provenance | values seen |
|---|---|
| commit | eb4b06482429e56ca3e93fa1625316b2fb107d7e |
| variant | cuda_ad_mono_polarized |
| platform | Windows-11-10.0.26200-SP0 |
| python | 3.13.12 |
| sionna-rt | 2.1.0 |
| mitsuba | 3.9.1 |
| drjit | 1.5.0 |
| numpy | 2.5.3 |
| gpu | NVIDIA GeForce RTX 4060 Laptop GPU, driver 616.56 |
