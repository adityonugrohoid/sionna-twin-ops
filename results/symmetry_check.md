# Symmetry check

Synthetic terrain id 3, a 6960 m square centred on its hilltop site; azimuth 45, tilt 6; line of sight and specular reflection, 1e+09 rays. Each variant (k clockwise quarter turns after an optional east-west mirror) moves the terrain and the azimuth together and is traced afresh; its map is compared with the original map moved the same way. Written by `twin symmetry-check`.

The terrain mesh and the measurement surface split every cell along its SW-NE diagonal. Variants that keep that fold (the identity, the half turn and the two diagonal mirrors) move the traced surface exactly and should agree within the sampling floor. Quarter turns and axis mirrors flip every cell's fold: the vertex heights are the same but the surface between them is not, so those maps differ by more than the floor. Training augments with the four fold-keeping variants only (spec M2b).

| provenance | |
|---|---|
| commit | cc4199d2bfd652b6261b1b15d7fb93ccf7da24f7 |
| variant | cuda_ad_mono_polarized |
| platform | Windows-11-10.0.26200-SP0 |
| python | 3.13.12 |
| sionna-rt | 2.1.0 |
| mitsuba | 3.9.1 |
| drjit | 1.5.0 |
| numpy | 2.5.3 |
| gpu | NVIDIA GeForce RTX 4060 Laptop GPU, driver 616.56 |

| k | mirror | keeps fold | azimuth | LOS median | LOS p95 | LOS max | LOS cells | NLOS median | NLOS p95 | NLOS max | NLOS cells | power only in moved original | power only in traced |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | False | True | 45 | 0.000 | 0.000 | 0.000 | 5609 | 0.000 | 0.000 | 0.000 | 934 | 0 | 0 |
| 1 | False | False | 135 | 0.338 | 2.225 | 12.956 | 5608 | 1.565 | 8.945 | 25.409 | 890 | 45 | 65 |
| 2 | False | True | 225 | 0.004 | 0.069 | 5.378 | 5609 | 0.016 | 0.571 | 9.742 | 930 | 4 | 5 |
| 3 | False | False | 315 | 0.339 | 2.244 | 12.982 | 5608 | 1.575 | 9.235 | 25.410 | 893 | 42 | 66 |
| 0 | True | False | 315 | 0.337 | 2.204 | 13.071 | 5608 | 1.557 | 9.215 | 26.199 | 891 | 44 | 65 |
| 1 | True | True | 45 | 0.004 | 0.075 | 2.642 | 5609 | 0.017 | 0.651 | 4.772 | 931 | 3 | 4 |
| 2 | True | False | 135 | 0.339 | 2.212 | 12.986 | 5608 | 1.597 | 9.415 | 25.407 | 891 | 44 | 63 |
| 3 | True | True | 225 | 0.004 | 0.068 | 12.892 | 5609 | 0.015 | 0.528 | 5.820 | 928 | 6 | 6 |
