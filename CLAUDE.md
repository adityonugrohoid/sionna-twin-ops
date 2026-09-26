# CLAUDE.md - sionna-twin-ops

A learned surrogate of a ray tracer on hilly terrain: Sionna RT computes
path-gain maps for one sector over procedurally generated terrain, and a
small neural network learns to predict them on terrain it has never seen.
`README.md` is the human overview; `docs/dataset.md` holds the dataset and
measurement rules once written.

## Rules for this repo

1. Terrain, sites and configurations are synthetic, generated from seeds.
   Every result says so. No real place, operator or network.
2. Every published number comes from committed code and states its
   settings: solver variant, samples, depth, enabled effects, seeds and
   package versions. No number from memory.
3. Surrogate results are always shown beside the baselines and the ray
   tracer's own noise floor, on held-out terrain. Shadowed cells are
   reported separately, never hidden in one average.
4. Physical constants are cited (document, edition, clause) or labelled
   ASSUMPTION. Antenna patterns come from 3GPP TR 38.901, never invented.
5. Sionna code and scenes are never copied into the repo; the installed
   package is used at run time. Built on NVIDIA Sionna RT, not affiliated
   with NVIDIA; no NVIDIA logo.
6. No AI-RAN, live network, PHY-loop or planning-tool claims.
7. Generated data, weights and run outputs stay in `data/` and `runs/`,
   never committed. `results/` holds chosen summaries and small figures.
8. Code fails loudly: no silent fallbacks (a backend or device change is
   explicit and recorded).
9. Personal repo: no employer or client name, branding or detail. No
   commit subject says "house template" or "standardize".
