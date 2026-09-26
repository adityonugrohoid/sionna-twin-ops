# Backend check

The solver-check floor maps (synthetic terrain ids 3, 1, 5, azimuth 90, tilt 6, line of sight and specular reflection, max_depth 3) solved on two backends and compared cell by cell. Written by `twin backend-check`.

| | reference | candidate |
|---|---|---|
| commit | 474693ab941ae645936633955c27bbeb9ed03ee2 | 474693ab941ae645936633955c27bbeb9ed03ee2 |
| variant | llvm_ad_mono_polarized | cuda_ad_mono_polarized |
| platform | Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39 | Windows-11-10.0.26200-SP0 |
| python | 3.13.12 | 3.13.12 |
| sionna-rt | 2.1.0 | 2.1.0 |
| mitsuba | 3.9.1 | 3.9.1 |
| drjit | 1.5.0 | 1.5.0 |
| numpy | 2.5.3 | 2.5.3 |
| gpu | NVIDIA GeForce RTX 4060 Laptop GPU, driver 616.56 | NVIDIA GeForce RTX 4060 Laptop GPU, driver 616.56 |

## Agreement

Absolute dB difference over cells with power on both backends, split by the LOS mask. The last columns count cells with power on only one backend.

| terrain | samples | LOS median | LOS p95 | LOS max | LOS cells | NLOS median | NLOS p95 | NLOS max | NLOS cells | power only on reference | power only on candidate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 1e+07 | 0.000 | 0.000 | 0.483 | 5602 | 0.000 | 0.000 | 0.001 | 801 | 0 | 0 |
| 3 | 1e+08 | 0.000 | 0.000 | 0.083 | 5608 | 0.000 | 0.000 | 0.227 | 900 | 0 | 0 |
| 1 | 1e+07 | 0.000 | 0.000 | 0.452 | 2233 | 0.000 | 0.000 | 0.000 | 564 | 0 | 0 |
| 1 | 1e+08 | 0.000 | 0.000 | 0.273 | 2236 | 0.000 | 0.000 | 0.046 | 628 | 0 | 0 |
| 5 | 1e+07 | 0.000 | 0.000 | 0.360 | 6095 | 0.000 | 0.000 | 0.000 | 844 | 0 | 0 |
| 5 | 1e+08 | 0.000 | 0.000 | 0.035 | 6111 | 0.000 | 0.000 | 0.071 | 970 | 0 | 0 |

## Time per map

Cold includes JIT compilation; warm is a second solve of the same map. Repeat identical says whether the two solves gave bit-identical maps.

| terrain | samples | reference warm s | reference cold s | reference repeat identical | candidate warm s | candidate cold s | candidate repeat identical | candidate GPU MiB |
|---|---|---|---|---|---|---|---|---|
| 3 | 1e+07 | 0.34 | 0.35 | False | 0.10 | 0.38 | False | 835 |
| 3 | 1e+08 | 2.91 | 2.89 | False | 0.12 | 0.29 | False | 835 |
| 1 | 1e+07 | 0.35 | 0.36 | False | 0.01 | 0.02 | False | 867 |
| 1 | 1e+08 | 3.29 | 3.29 | False | 0.10 | 0.10 | False | 867 |
| 5 | 1e+07 | 0.31 | 0.32 | False | 0.01 | 0.01 | False | 867 |
| 5 | 1e+08 | 2.87 | 2.86 | False | 0.09 | 0.08 | False | 867 |
