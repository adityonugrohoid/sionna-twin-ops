# Dataset and measurement rules

The rules the code, tests and reports cite by id ("spec S2", "rule Q0b"). Each rule is stated
as the project runs it now. Constants that are choices rather than facts are labelled
ASSUMPTION; physical references are cited by document, edition and clause. Where a rule was
revised, a short history line says what changed and what measurement caused it.

Terrain, sites and configurations are synthetic, generated from seeds: no real place,
operator or network. The claim is only what the held-out measurements show.

## T: terrain generator

### T1 Determinism

Each terrain comes from `np.random.default_rng([BASE_SEED, terrain_id])`, with no clock and
no global random state: the same id gives the same terrain, byte for byte.

### T2 Heightmap

Spectral synthesis on a regular grid: Gaussian noise shaped by a power-law spectrum with
exponent `beta`, rescaled to [0, relief]. Ridges and valleys come from the spectrum itself.
Per terrain, relief is drawn from 100 to 800 m and `beta` from 3.4 to 4.0 (ASSUMPTION ranges,
judged by eye on sample grids).

History: an added ridge term was removed. Straight ridges read as berms, and ridged noise
`(1 - |n|)**2` gave an even maze of crests. Below `beta` 3.4, grid-scale roughness read as
synthetic.

### T3 Extent

The tile is 10 x 10 km on a 40 m grid.

### T4 Material

One ITU material, `medium_dry_ground`, for the whole surface (ASSUMPTION). No vegetation,
clutter or buildings.

### T5 Flat Earth

Earth curvature is ignored (about 1.5 m drop at 5 km with a 4/3 Earth radius).

### T6 Tile edge

No walls or skirts at the tile edge: the surface ends, and rays that leave the tile are
lost, as into open sky.

## S: site and map geometry

### S1 Site

One site and one sector per map, the antenna 30 m above local ground (ASSUMPTION).

### S2 Site class and placement

The site class cycles with the terrain id: ("hilltop", "slope", "valley")[terrain_id % 3].
Candidates lie inside the central 3 x 3 km of the tile, and the height percentile is the
midrank percentile among the vertices of the site's own map.

- Hilltop: the highest point of a 200 m radius disc around it, percentile at least 85.
- Valley: the lowest point of such a disc, percentile at most 15.
- Slope: percentile 35 to 65 and a gradient of at least 0.05 m per m (ASSUMPTION), measured
  over a 200 m baseline.

Among qualifying candidates, the one nearest the window centre is chosen. A terrain with no
candidate for its class is skipped; the dataset walks ids in order and stops each class at
its quota, so the classes stay balanced. `results/site_check.md` reports each class's
percentiles and skips.

History: the slope gradient was first taken over one grid step; it is now taken over a 200 m
baseline. Picking the strongest extreme pulled sites to the window edge, so the rule now picks
the qualifier nearest the centre.

### S3 Map

128 x 128 cells of 40 m (5.12 x 5.12 km), centred on the site. The terrain beyond the map
leaves a guard band of at least 0.94 km on every side.

### S4 Measurement surface

A grid mesh over the map area, following the terrain 1.5 m above the ground, with two
triangles per cell. A cell's value is the mean of its two triangles in linear power. It is a
mesh of its own, not the terrain mesh.

### S5 Surface check

The measurement surface must not block or reflect rays. On a flat tile, the mesh surface and
a planar radio map, both at 1.5 m, agree within the noise (`results/solver_check.md`).

## A: sector antenna

### A1 Carrier

1.8 GHz (ASSUMPTION: a common rural band).

### A2 Array

A `PlanarArray` of 8 rows x 1 column with 0.8 wavelength vertical spacing and vertical
polarization. The element pattern is 3GPP TR 38.901 Table 7.3-1. The element count and
spacing are ASSUMPTION: TR 38.901 does not fix a macro panel size.

### A3 Electrical tilt

Downtilt is a phase taper across the rows, passed as `precoding_vec` and computed from the
array geometry. There is no mechanical tilt.

### A4 Flat-plain tilt check

On a flat tile, the far-field median path gain must move by at least 6 dB across tilts 0 to
12 deg, and the main-lobe elevation must follow the commanded tilt within 1 deg. The result
is in `results/solver_check.md`.

## C: configurations

### C1 Power is not ray-traced

The solver's output is path gain, which does not depend on transmit power. RSRP is the power
per resource element plus the path gain, computed afterwards.

### C2 Traced grid

Per terrain: 8 azimuths (0 to 315 deg in 45 deg steps, clockwise from true north) x 5
electrical tilts (0, 3, 6, 9, 12 deg), 40 maps.

### C3 Off-grid tilts

Test terrains also get tilts of 1.5, 4.5 and 7.5 deg at azimuths 0 and 180 deg (the two
azimuths are ASSUMPTION), to measure interpolation.

## N: solver settings and the truth's uncertainty

### N1 Recorded settings

Fixed per dataset and recorded with every map: the Mitsuba variant, `samples_per_tx`,
`max_depth`, `los`, specular, diffuse, refraction, diffraction, edge diffraction and the seed.

### N2 Cost

Seconds per map, and the share of cells no ray reaches, are measured per sample count
(`results/solver_check.md`, `results/dataset_summary.md`).

### N3 Sampling floor

Sionna RT's radio-map solver launches rays on a deterministic Fibonacci lattice, so for
specular-only maps the seed changes nothing. The sampling floor is the difference between
the dataset's 1e9 rays per map and 4e9 rays (the largest count possible, since one solve
launches at most 2^32 - 1 rays), over cells with power in both maps. It is reported as the
median and p95 in dB, split LOS and NLOS. No surrogate error below it is claimed as
meaningful.

History: the floor was first measured as 1e8 against 1e9 rays, when the dataset was to use
1e8. It moved to 1e9 against 4e9 when the dataset moved to 1e9 on the GPU (N7). It was
defined as a two-seed spread until the lattice showed the seed has no effect.

### N3b Fold term

The ground truth also depends on how each map cell is split into two triangles. The dataset
uses the SW-NE diagonal; the fold term compares its maps with the NW-SE maps.
`results/fold_check.md` reports it split LOS and NLOS, by distance to shadow and by
dominant path. Results quote it beside the sampling floor as the truth's own uncertainty.

History: added when the symmetry check showed that transforms which flip the fold change
the traced map by far more than the sampling floor (see M2b). On six training terrains, the
fold term's p95 is 4.570 dB in LOS and 9.905 dB in NLOS, against a sampling floor of
0.100 and 0.667 dB (`results/fold_check.md`).

### N4 Diffraction

Diffraction is off. Sionna RT's diffraction does not reach a mesh measurement surface, so
shadowed cells are lit only by reflections.

### N5 Budget

The sweep must fit in a few hours of wall time on one machine. N7 settles the backend and
ray count that make this possible (`results/dataset_summary.md` gives the time).

### N6 Method

Sionna RT's `RadioMapSolver` on the S4 mesh surface, with line of sight and specular
reflection only (diffuse reflection, refraction and diffraction off) and `max_depth` 3. The
Mitsuba variant is an explicit option, recorded in every output and never auto-detected.
Tests and development runs use 1e7 rays per map.

### N7 Dataset backend and ray count

Dataset maps are traced at 1e9 rays per map with `cuda_ad_mono_polarized` on native
Windows, driven from WSL (`docs/windows-gpu.md`). The CPU variant `llvm_ad_mono_polarized`
stays the reference.

The two backends agree: median and p95 differences print as 0.000 dB, single cells differ by
up to 0.483 dB, and the same cells have power (`results/backend_check.md`). At 1e9 rays a
further map takes about 1.07 s on the GPU against 34.57 s on the CPU
(`results/evaluation_test.md`).

## D: dataset

### D1 Size

The dataset is 144 terrains (126 train, 9 validation, 9 test) and 5,814 maps, including the
test terrains' off-grid maps (`results/dataset_summary.md`).

### D2 Split

The split is by terrain id, never by map, with site classes balanced within each split.
Validation and test terrains are never seen in training.

### D3 Manifest

Each map has a manifest record: terrain id and parameters, site, azimuth, tilt, solver
settings, variant, wall time, package versions and cell counts (valid and no-hit). The sweep
is resumable: a finished map is never recomputed.

### D4 Storage

Maps live in `data/`, which is not in git. The committed record is
`results/dataset_summary.md`.

### D5 Version 2

Version 2 adds 84 training terrains (28 per class) by continuing the id walk past the first
version's ids, with the same generator and settings. Validation and test terrains are
unchanged.

History: the first version had 42 training terrains. With it, validation L1 was 1.796 to
1.841 dB at the best epoch and 2.350 to 2.502 dB by epoch 50 (`results/training_summary.md`,
context). More terrains were preferred to traced rotations (M2b): they cost the same GPU time
and give independent landscapes.

## B: baselines

Every surrogate number is shown next to both baselines.

### B0 Geometry

Free-space path loss plus the array's pattern gain at the commanded tilt, towards each
cell. No terrain.

### B1 Terrain profile

B0 plus the diffraction loss along the terrain profile from the site to each cell, by the
Bullington method of ITU-R P.526-16, clause 4.5.1 (flat Earth per T5), with J(nu) from
clause 4.1.

History: the Deygout construction was the first choice. It exists only in superseded
editions (P.526-11 clause 4.4.2), so the edition in force is used instead.

A LOS mask per cell, computed from the heightmap, splits the evaluation strata and is a
model input.

## M: model

### M1 Network

A PyTorch U-Net of width 32 with 7,767,170 parameters. The inputs per cell are:
- height relative to the antenna;
- log distance;
- sin and cos of the horizontal angle off boresight;
- elevation angle from the antenna;
- the LOS mask;
- B0 in dB.

It outputs the path gain in dB as a residual over B0.

### M1b Power head

A second output: a per-cell logit for "the ray tracer has power here", trained with binary
cross-entropy on all cells. It is scored as the no-signal class of E3, and the path-gain
head is scored only where the truth has power.

### M2 Loss

The path-gain loss (L1 in dB) covers only cells where the ray-traced power is above zero,
the same set as the power head's positive class.

History: first defined as cells above the noise floor. No floor threshold was ever set, so
"valid" became "power above zero".

### M2b Augmentation

Training uses the 4 map symmetries that keep each cell's fold diagonal: the identity, the
half turn and the two diagonal mirrors. Rasters and azimuth are transformed together.

History: all 8 symmetries were first proposed. Quarter turns and axis mirrors flip every
cell's fold and change the traced map (LOS p95 2.204 to 2.244 dB, NLOS p95 8.945 to
9.415 dB). The fold-keeping variants agree within the floor (LOS p95 at most 0.075 dB,
NLOS 0.651 dB) (`results/symmetry_check.md`).

### M3 Seeds

Three training seeds; results report the mean and the spread across seeds.

### M4 Hardware and weights

Training runs on the GPU under WSL2. Weights live in `runs/`, which is not in git; the
committed results do not need them.

## E: evaluation, on the test terrains only

The test split is used once, by `twin evaluate`.

### E1 Error

Per-cell error in dB, predicted minus traced, over cells where the ray tracer has power:
mean absolute, median absolute, p95 absolute and bias.

### E2 Coverage

A cell is covered where RSRP reaches -110 dBm (ASSUMPTION), with 43 dBm sector power spread
over the 1,200 resource elements of a 20 MHz carrier (ASSUMPTION). Reported as the
covered-area error in percent and the IoU of the covered masks.

### E3 Strata and the no-signal class

E1 is split by site class and by stratum. The strata are LOS direct-dominated cells, LOS
reflection-dominated cells (traced gain at least 3 dB above B0, ASSUMPTION) and NLOS cells.
Cells where the ray tracer has no power form a class of their own, scored with precision
and recall. The shadow is never hidden in one average.

### E4 Off-grid tilts

The C3 maps are reported separately.

### E5 Timing

Seconds per map on the same machine, for the ray tracer and for the surrogate (batch 1),
on the GPU and on the CPU. Both for a new terrain and for each further map of it.

### E6 Table and figure

One table: surrogate against B0, B1, the fold term and the sampling floor. One figure: three
test cases, ray-traced, predicted and error, over hillshaded relief. Every figure carries a
provenance caption.

### E7 3D viewer

`twin viewer` writes one self-contained HTML page that drapes test maps over their terrain
in 3D. It shows the ray-traced, predicted and error views, the site mast and the beam, and
the vertical exaggeration can be changed (`results/viewer_test.html`).

## Q: tilt and power search, on the test terrains

### Q1 Objective

The covered fraction of the cells within 1.5 km of the site, minus the covered fraction of
the map's cells beyond 1.5 km, at the E2 RSRP threshold. It runs from -1 to 1. The radius and
the equal weight of the two fractions are ASSUMPTION; the spill beyond the radius is a proxy
for interference to neighbours.

### Q2 Choosers

Variables: electrical tilt and sector power per (terrain, azimuth), and additionally the
azimuth for one choice per terrain. The choosers are:
- the ray tracer: tilt 0 to 12 deg in 1 deg steps, 13 traces per case;
- the surrogate: tilt 0 to 12 deg in 0.5 deg steps;
- a rule of thumb: tilt = arctan(antenna height / radius) plus half the column's vertical
  half-power beamwidth, at 46 dBm;
- a fixed setting with no search: 0 deg at 46 dBm.

Power runs from 28 to 46 dBm in 1 dB steps on every side (C1: power is post-processing).

### Q3 Scoring

Every chosen setting is scored with the ray tracer. The score is the shortfall: the ray
tracer's optimum on the 1 deg grid minus the chosen setting's objective, in objective
points. "Within 1 point" means a shortfall of at most 0.01. The report gives the median, p90
and maximum shortfall and the count within 1 point, per chooser, with wall time per
terrain (`results/search_test.md`).

### Q0 The first objective

The first objective counted covered cells within 3 km minus covered cells beyond it, on a
power grid of 37 to 46 dBm, scored as a share of the optimum. The map's half-width is
2.56 km, so only the corners lay beyond the radius, and the optimum collapsed to full power
and low tilt (46 dBm in 71 of 72 test cases). That definition cannot show what a search
buys. Its single test run is kept as superseded (`results/search_test_first_objective.md`),
and the radius became 1.5 km.

### Q0b Area fractions

With raw cell counts and a 1.5 km radius, the region beyond the radius is about 2.7 times
the service disc, so the optimum ran to the lowest power and the highest tilt of the grid
on the validation terrains. The objective became the difference of the two area fractions
(Q1), the power grid widened to 28 to 46 dBm, and the score became the shortfall (Q3). The
objective was designed on the validation terrains only.

### Q0c The feasible box

On the validation terrains, 45 of 72 optima sit on the edge of the tilt and power ranges,
while the rule of thumb (3 of 72) and the fixed setting (13 of 72) stay far from the
optimum (`results/search_validation.md`), so the objective is not degenerate. The box is
the feasible set: electrical tilt 0 to 12 deg (the surrogate's training range) and power
28 to 46 dBm, 46 dBm being the sector maximum (both ASSUMPTION). An optimum on its edge is a
constrained optimum, and the reports give edge counts per site class. The test terrains
were then searched once under this definition.
