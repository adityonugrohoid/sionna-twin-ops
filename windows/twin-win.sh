#!/usr/bin/env bash
# Run a `twin` command on native Windows (GPU) from WSL, against a snapshot of one commit.
#
# Usage: windows/twin-win.sh <runner root, Windows path> <commit> -- <twin arguments...>
# Example:
#   windows/twin-win.sh 'C:\Users\me\dev\sionna-twin-ops-gpu' HEAD -- \
#       backend-maps --variant cuda_ad_mono_polarized --samples 1e7,1e8 --seed 1 \
#       --out 'C:\Users\me\dev\sionna-twin-ops-gpu\out\backend\cuda'
#
# The commit must exist in this repository; uncommitted changes never reach Windows.
# The snapshot is `git archive` of that commit, written once to <root>\code\<12-char sha>
# and never modified. Paths in the twin arguments are Windows paths. See docs/windows-gpu.md.
set -euo pipefail

usage() {
    echo "usage: $0 <runner root (Windows path)> <commit> -- <twin arguments...>" >&2
    exit 2
}
[[ $# -ge 3 && $3 == "--" ]] || usage
root_win=$1
commit=$2
shift 3

repo=$(git rev-parse --show-toplevel)
sha=$(git -C "$repo" rev-parse --verify "${commit}^{commit}")
short=${sha:0:12}
root_wsl=$(wslpath -u "$root_win")
if [[ ! -x "$root_wsl/tools/uv/uv.exe" ]]; then
    echo "no runner at $root_win: run windows/setup.ps1 first" >&2
    exit 1
fi

code="$root_wsl/code/$short"
if [[ -e "$code" ]]; then
    if [[ "$(tr -d '\r\n' < "$code/.twin-commit")" != "$sha" ]]; then
        echo "snapshot $code exists but is not commit $sha" >&2
        exit 1
    fi
else
    mkdir -p "$code.tmp"
    git -C "$repo" archive "$sha" | tar -x -C "$code.tmp"
    printf '%s\n' "$sha" > "$code.tmp/.twin-commit"
    mv "$code.tmp" "$code"
fi

exec powershell.exe -NoProfile -ExecutionPolicy Bypass \
    -File "$(wslpath -w "$code/windows/run.ps1")" -Root "$root_win" -Commit "$short" "$@"
