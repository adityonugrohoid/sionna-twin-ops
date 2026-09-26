# Run `twin` from a commit snapshot inside a runner folder, on Windows.
#
# Called by windows/twin-win.sh from WSL; it lives inside the snapshot it runs:
#   powershell -NoProfile -ExecutionPolicy Bypass -File <Root>\code\<commit>\windows\run.ps1 `
#       -Root <Root> -Commit <commit> -TwinArgsBase64 <base64 of a JSON array of strings>
#
# The twin arguments travel as base64 JSON because PowerShell would otherwise bind tokens
# such as --out to its own common parameters.
#
# The snapshot's own uv.lock sets the packages. The first run for a commit builds its venv
# at <Root>\venvs\<commit>; torch, triton, nvidia-* and cuda-* are left out (no training
# here) and the project is installed without dependencies. See docs/windows-gpu.md.

param(
    [Parameter(Mandatory = $true)][string]$Root,
    [Parameter(Mandatory = $true)][string]$Commit,
    [Parameter(Mandatory = $true)][string]$TwinArgsBase64
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$PythonVersion = '3.13.12'
$Excluded = '^(torch|triton|nvidia-|cuda-)'

$Root = [System.IO.Path]::GetFullPath($Root)
. (Join-Path $PSScriptRoot 'env.ps1') -Root $Root
if (-not (Test-Path $UvExe)) { throw "no uv at ${UvExe}; run setup.ps1 first" }

$code = Join-Path $Root "code\$Commit"
$marker = Join-Path $code '.twin-commit'
if (-not (Test-Path $marker)) { throw "snapshot $code has no .twin-commit marker" }
$fullSha = (Get-Content $marker -Raw).Trim()
if (-not $fullSha.StartsWith($Commit)) { throw "snapshot marker $fullSha does not match $Commit" }

$venv = Join-Path $Root "venvs\$Commit"
$ready = Join-Path $venv '.twin-ready'
if ((Test-Path $venv) -and -not (Test-Path $ready)) {
    throw "venv $venv exists but was not finished; remove it and run again"
}
if (-not (Test-Path $venv)) {
    # Built in place (uv's launchers record absolute paths), then marked ready.
    & $UvExe venv --python $PythonVersion $venv
    if ($LASTEXITCODE -ne 0) { throw "uv venv failed ($LASTEXITCODE)" }
    $all = & $UvExe export --locked --no-dev --no-hashes --no-emit-project --project $code
    if ($LASTEXITCODE -ne 0) { throw "uv export failed ($LASTEXITCODE)" }
    $kept = $all | Where-Object { $_ -notmatch $Excluded }
    $dropped = $all | Where-Object { $_ -match $Excluded }
    Write-Host ("left out of the Windows venv: " + (($dropped | ForEach-Object { ($_ -split '[ =;]')[0] }) -join ', '))
    $reqs = Join-Path $Root "venvs\$Commit.requirements.txt"
    $kept | Set-Content -Encoding ascii $reqs
    $python = Join-Path $venv 'Scripts\python.exe'
    & $UvExe pip install --python $python -r $reqs
    if ($LASTEXITCODE -ne 0) { throw "uv pip install failed ($LASTEXITCODE)" }
    & $UvExe pip install --python $python --no-deps -e $code
    if ($LASTEXITCODE -ne 0) { throw "project install failed ($LASTEXITCODE)" }
    Set-Content -Path $ready -Value $fullSha -Encoding ascii
}

$json = [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($TwinArgsBase64))
# Windows PowerShell 5.1 emits a JSON array as one object; the cast unrolls it.
$TwinArgs = [string[]](ConvertFrom-Json $json)
Write-Host ("twin " + ($TwinArgs -join ' '))

$env:TWIN_COMMIT = $fullSha
& (Join-Path $venv 'Scripts\python.exe') -m sionna_twin_ops.cli @TwinArgs
exit $LASTEXITCODE
