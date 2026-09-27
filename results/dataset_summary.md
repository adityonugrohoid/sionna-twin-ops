# Dataset summary (v2)

Synthetic terrain, not a real place. Propagation: Sionna RT, line of sight and specular reflection only (no diffraction, no diffuse scattering); pattern: 3GPP TR 38.901. One sector per map, no vegetation or buildings, flat Earth. Written by `twin dataset-summary`; the maps themselves are not in git.

Maps in the manifest: 5814 of 5814 expected.

## Terrains and maps per split and site class

| split | hilltop | slope | valley | terrains | grid maps | off-grid maps |
|---|---|---|---|---|---|---|
| train | 42 | 42 | 42 | 126 | 5040 | 0 |
| validation | 3 | 3 | 3 | 9 | 360 | 0 |
| test | 3 | 3 | 3 | 9 | 360 | 54 |

Terrain ids by split: train 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 40, 44, 45, 47, 50, 51, 53, 54, 57, 82, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 103, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 118, 119, 120, 121, 123, 124, 126, 127, 128, 130, 131, 132, 133, 134, 136, 139, 141, 142, 143, 144, 145, 146, 148, 149, 150, 151, 152, 154, 155, 156, 157, 159, 160, 163, 165, 167, 168, 170, 171, 174, 176, 177, 182, 185, 189, 191, 192, 194, 197, 203, 206, 207, 209, 210; validation 43, 46, 49, 60, 62, 63, 65, 68, 69; test 52, 55, 58, 71, 72, 74, 77, 78, 81.

## Skipped terrain ids

Ids with no site candidate for their class (spec S2); the walk moved on to the next id of that class. Highest id walked: 210.

| site class | skipped | ids |
|---|---|---|
| hilltop | 23 | 0, 12, 15, 39, 42, 48, 66, 75, 102, 117, 129, 135, 138, 147, 153, 162, 180, 183, 186, 195, 198, 201, 204 |
| slope | 0 | none |
| valley | 21 | 2, 26, 35, 41, 56, 59, 83, 104, 110, 116, 122, 125, 137, 140, 158, 161, 164, 173, 179, 188, 200 |

## No-hit share per site class

Share of map cells no ray reached (the 'no signal' class of spec E3), over all maps of the class.

| site class | maps | mean | min | max |
|---|---|---|---|---|
| hilltop | 1938 | 44.0% | 12.2% | 71.6% |
| slope | 1938 | 62.6% | 9.9% | 96.2% |
| valley | 1938 | 71.9% | 15.8% | 95.6% |

## Time and settings

Solver wall time over all maps: 1.81 h (6527 s; median 1.09 s per map). Scene building and file writing are not included.

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
| commit | d2467cfb3341c0f6c88450b86eb1fbc093076707; eb4b06482429e56ca3e93fa1625316b2fb107d7e |
| variant | cuda_ad_mono_polarized |
| platform | Windows-11-10.0.26200-SP0 |
| python | 3.13.12 |
| sionna-rt | 2.1.0 |
| mitsuba | 3.9.1 |
| drjit | 1.5.0 |
| numpy | 2.5.3 |
| gpu | NVIDIA GeForce RTX 4060 Laptop GPU, driver 616.56 |
