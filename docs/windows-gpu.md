# Ray tracing on the GPU from WSL

Under WSL2, Mitsuba's `cuda` variants cannot load a scene (OptiX is not available there),
so ray tracing in WSL runs on the CPU (`llvm_ad_mono_polarized`). On native Windows the
same GPU runs `cuda_ad_mono_polarized` about 30 times faster. This page describes the
runner that lets a WSL session run `twin` commands on Windows, against an exact commit.
The CPU variant stays the reference; `twin backend-check` compares the two.

No admin rights are needed, and nothing is written outside the runner folder.

## Set up a runner folder (once)

From WSL:

```bash
powershell.exe -NoProfile -ExecutionPolicy Bypass \
    -File "$(wslpath -w windows/setup.ps1)" -Root 'C:\Users\<you>\dev\sionna-twin-ops-gpu'
```

`setup.ps1` downloads uv 0.10.2 (the version that wrote `uv.lock`) from its GitHub
release, checks the zip's SHA-256 against the value pinned in the script, and installs a
uv-managed CPython 3.13.12. `windows/env.ps1` keeps every uv location (cache, Python
installs, Python bin directory, tools) inside the folder. The execution policy bypass
applies to that one PowerShell process only.

## Run a command

```bash
windows/twin-win.sh 'C:\Users\<you>\dev\sionna-twin-ops-gpu' <commit> -- <twin arguments>
```

For example, the GPU half of the backend check:

```bash
windows/twin-win.sh 'C:\Users\<you>\dev\sionna-twin-ops-gpu' HEAD -- \
    backend-maps --variant cuda_ad_mono_polarized --samples 1e7,1e8 --seed 1 \
    --out 'C:\Users\<you>\dev\sionna-twin-ops-gpu\out\<commit>\backend-cuda'
```

What happens:

1. The commit is resolved in this repository; uncommitted changes never reach Windows.
2. `git archive` of that commit is written once to `<root>\code\<12-character sha>`, with a
   `.twin-commit` marker. An existing snapshot is checked against the marker, never
   rewritten.
3. The snapshot's own `windows\run.ps1` builds `<root>\venvs\<sha>` from that commit's
   `uv.lock` on first use. torch, triton, `nvidia-*` and `cuda-*` are left out (no model
   training on Windows; the list printed on each build says exactly what was dropped), and
   the project is installed without dependencies. The venv is marked ready only when the
   build finishes; an unfinished one stops the next run with a message.
4. `twin` runs with `TWIN_COMMIT` set, so every output records the commit it came from.

Paths in the twin arguments are Windows paths. The arguments travel to PowerShell as one
base64-encoded JSON array, because PowerShell would otherwise read tokens such as `--out`
as its own parameters.

On Windows, each run prints `jitc_llvm_init(): LLVM API initialization failed`. That is
expected: the LLVM backend is not installed there and is not used; the `cuda` variant is.

## Check the backends agree

```bash
uv run twin backend-maps --variant llvm_ad_mono_polarized --samples 1e7,1e8 --seed 1 \
    --out runs/backend/<commit>/llvm
uv run twin backend-check --reference runs/backend/<commit>/llvm \
    --candidate /mnt/c/Users/<you>/dev/sionna-twin-ops-gpu/out/<commit>/backend-cuda \
    --out results/backend_check.md
```

The report lists both runs' provenance (commit, variant, platform, package versions, GPU
and driver), the per-cell agreement split by the LOS mask, and the time per map.

## Clean up

A runner folder holds only generated files: the uv binary, Python, the package cache,
snapshots, venvs and outputs. Removing one is a manual decision; no script deletes it.
