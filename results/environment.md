# Environment record

Recorded 2026-09-26 with `uv run twin env` on the development machine.
Exact versions of every package, direct and transitive, are in `uv.lock`.

## Machine

- Windows 11, WSL2 kernel 6.6.87.2, x86_64, 16 CPU threads
- NVIDIA GeForce RTX 4060 Laptop GPU, Windows driver 616.56

## Direct dependencies

Each was the latest release on PyPI on 2026-09-26.

| Package    | Version |
|------------|---------|
| Python     | 3.13.12 (uv managed, pinned in `.python-version`) |
| sionna-rt  | 2.1.0   |
| mitsuba    | 3.9.1 (pinned by sionna-rt) |
| drjit      | 1.5.0 (pinned by sionna-rt) |
| numpy      | 2.5.3   |
| torch      | 2.14.0 (CUDA 13.0 build) |
| matplotlib | 3.11.2  |
| ruff       | 0.16.9  |
| mypy       | 2.3.1   |
| pytest     | 9.1.1   |
| pre-commit | 4.6.2   |

## Backends

- Ray tracing: Mitsuba variant `llvm_ad_mono_polarized` on the CPU, set
  explicitly before `sionna.rt` is imported. Left alone, Sionna tries a
  `cuda` variant first and falls back to llvm without saying so; the
  project never relies on that fallback. The `cuda` variants are
  installed but not used: under WSL2, `load_scene()` on
  `cuda_ad_mono_polarized` raises "Could not initialize OptiX!"
  (checked 2026-09-26 with the versions above).
- PyTorch: `torch.cuda.is_available()` is `True` under WSL2 and the
  RTX 4060 is visible. Training runs on the GPU. PyTorch CUDA does not
  depend on OptiX.

## Smoke test

`tests/test_smoke.py::test_flat_tile_solves_on_llvm` builds a flat
2 x 2 km square of `medium_dry_ground` in memory, places an isotropic
transmitter 30 m above it at 1.8 GHz and solves a 20 x 20 planar radio
map (samples_per_tx 1e5, max_depth 1, seed 1) on llvm. It checks that
every cell is finite, more than 90 percent of cells are hit, and the
centre cell has more gain than the corner. The whole suite runs in
about 4 s.
