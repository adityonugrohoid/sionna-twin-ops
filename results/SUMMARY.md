# Summary

Synthetic terrain generated from seeds, not a real place; one sector on a 30 m mast at 1.8 GHz over a 5.12 km map; flat Earth, no vegetation, buildings or interference. Ground truth is NVIDIA Sionna RT (line of sight and specular reflection only). Every number below is read from the linked report's data by `twin summary`; the reports give the settings, seeds and package versions.

## The surrogate on held-out test terrain

Test split: 414 maps on 9 terrains no model saw in training. Mean absolute path-gain error in dB over cells where the ray tracer has power; surrogate: mean over 3 seeds (lowest to highest). Beside it, the ray tracer's own uncertainty: the fold term (the map with the other cell diagonal) and the sampling floor (1e9 against 4e9 rays), median / p95 of the absolute difference ([evaluation_test](evaluation_test.md), [fold_check](fold_check.md)).

| cells | surrogate | B0 (free space and pattern) | B1 (B0 with diffraction loss) | fold term, median / p95 | sampling floor, median / p95 |
|---|---|---|---|---|---|
| all | 1.549 (1.524 to 1.584) | 3.132 | 3.844 | - | - |
| LOS direct | 0.838 (0.810 to 0.887) | 1.236 | 2.200 | 0.484 / 3.287 | 0.006 / 0.100 |
| LOS reflection | 4.084 (3.968 to 4.200) | 8.410 | 8.610 | 1.715 / 13.241 | 0.006 / 0.100 |
| NLOS | 3.856 (3.793 to 3.907) | 10.207 | 9.857 | 1.725 / 9.905 | 0.019 / 0.667 |

The sampling floor is measured split LOS / NLOS only, so both LOS rows show the LOS floor.

- LOS reflection-dominated cells: bias -3.017 (-3.271 to -2.804) dB, the surrogate underpredicts them.
- Off-grid tilts (1.5, 4.5, 7.5 deg, never trained on): mean abs 1.336 (1.312 to 1.371) dB.
- No-signal cells: the power head's accuracy over all cells is 98.36 (98.31 to 98.39)%.
- Coverage at -110 dBm (ASSUMPTION): IoU 0.9385 (0.9369 to 0.9394) against B0 0.3977 and B1 0.8018 (pooled).

## Tilt and power search on test terrain

72 (terrain, azimuth) cases; the objective is the covered fraction within 1.5 km minus the covered fraction beyond it (ASSUMPTION), on a -1 to 1 scale; every chosen setting is scored with the ray tracer, as its shortfall from the ray tracer's optimum on a 1 deg tilt grid ([search_test](search_test.md)).

| chooser | median shortfall | max shortfall | within 1 point of the optimum |
|---|---|---|---|
| rule of thumb | 0.0864 | 0.5276 | 3 of 72 |
| fixed 0 deg, 46 dBm (no search) | 0.0430 | 0.7337 | 15 of 72 |
| surrogate seed 0 (CPU) | 0.0000 | 0.0177 | 70 of 72 |
| surrogate seed 1 (CPU) | 0.0000 | 0.0192 | 68 of 72 |
| surrogate seed 2 (CPU) | 0.0000 | 0.0189 | 71 of 72 |
| ray tracer, 1 deg grid | 0.0000 | 0.0000 | 72 of 72 |

## Speed, same machine

Ray tracer time over surrogate time, medians ([evaluation_test](evaluation_test.md), [search_test](search_test.md)):

| case | GPU | CPU |
|---|---|---|
| one map, new terrain | 2.5 | 75.0 |
| each further map | 133.5 | 1038.3 |
| whole search, one terrain | 46 | estimated only, see the report |

## Behind the numbers

- Dataset: 5814 ray-traced maps on 126 train, 9 validation, 9 test terrains ([dataset_summary](dataset_summary.md)).
- Training: a U-Net of residual over B0 plus a power head; validation L1 1.549, 1.569, 1.499 dB for seeds 0, 1, 2, best epochs 13, 10, 14 ([training_summary](training_summary.md)).
- Antenna tilt check (3GPP TR 38.901 element, 8 x 1 column): PASS, largest main-lobe error 0.33 deg ([solver_check](solver_check.md)).
- GPU against CPU ray tracing: the largest median and p95 |difference| over every map are 0.000 and 0.000 dB at the report's three decimals ([backend_check](backend_check.md)).

## Limitations

- Synthetic terrain only: procedural heightmaps from seeds, one ground material, no vegetation, buildings or clutter, flat Earth.
- One sector, one antenna configuration, one carrier.
- The ray tracer runs line of sight and specular reflection only: no diffraction (it does not reach a mesh measurement surface in Sionna RT) and no diffuse scattering. Shadowed cells are lit by reflections alone.
- The ground truth depends on how each map cell is folded into triangles: the fold term is far above the sampling floor, largest in LOS reflection-dominated and NLOS cells ([fold_check](fold_check.md)). It is quoted from 6 training terrains at one azimuth and tilt (azimuth 90 deg, tilt 6 deg), not measured on the test split. Surrogate errors below it are not a claim of accuracy beyond the ray tracer's own.
- The surrogate underpredicts LOS reflection-dominated cells (bias above); its largest mean errors are in LOS reflection and NLOS cells, its smallest in LOS direct cells (table above).
- Training overfits after the best epoch; the saved weights are the best epoch's, chosen on validation ([training_summary](training_summary.md)).
- Search optima often sit on the edge of the feasible box (tilt 0 to 12 deg, 28 to 46 dBm, ASSUMPTION): they are constrained optima, not the unconstrained ones ([search_test](search_test.md)).
- The search objective is not smooth in tilt, so the 1 deg ray-tracer grid is a reference at that resolution; a half-degree setting can beat it ([search_test](search_test.md)).
- The search objective was revised twice on the validation split before the final objective's single test run: first because its radius reached beyond the map, then because raw cell counts let the larger spill region outweigh the service area; the final objective compares area fractions ([search_validation](search_validation.md)). The first definition had already run once on test (next item).
- The first search objective was a flawed definition: its 3 km radius reached beyond the map's half-width, so its optimum collapsed to 46 dBm and 0 deg; it is kept as superseded ([search_test_first_objective](search_test_first_objective.md)).
