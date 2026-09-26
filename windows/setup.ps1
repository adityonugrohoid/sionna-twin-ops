# Bootstrap a Windows GPU runner folder: pinned uv and a uv-managed Python, nothing else.
#
# Usage (Windows PowerShell 5.1, no admin):
#   powershell -NoProfile -ExecutionPolicy Bypass -File setup.ps1 -Root C:\path\to\runner
#
# Everything is written inside -Root. uv is downloaded from its GitHub release and checked
# against the pinned sha256 before use. Per-commit code and environments are made later by
# run.ps1. See docs/windows-gpu.md.

param(
    [Parameter(Mandatory = $true)][string]$Root
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$UvVersion = '0.10.2'  # the uv that wrote uv.lock
$UvZipSha256 = '493ebbe0e06128d6ee4905e1ed5e2a433fb0f7cfc08b0eaca9fab4ca76778ae1'
$PythonVersion = '3.13.12'  # .python-version resolves to this build

$Root = [System.IO.Path]::GetFullPath($Root)
New-Item -ItemType Directory -Force -Path $Root | Out-Null
. (Join-Path $PSScriptRoot 'env.ps1') -Root $Root

$uvDir = Join-Path $Root 'tools\uv'
if (-not (Test-Path $UvExe)) {
    New-Item -ItemType Directory -Force -Path $uvDir | Out-Null
    $zip = Join-Path $Root 'tools\uv.zip'
    $url = "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip"
    Write-Host "downloading $url"
    Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
    $hash = (Get-FileHash -Algorithm SHA256 -Path $zip).Hash.ToLowerInvariant()
    if ($hash -ne $UvZipSha256) {
        Remove-Item $zip
        throw "uv zip sha256 mismatch: got $hash, expected $UvZipSha256"
    }
    Expand-Archive -Path $zip -DestinationPath $uvDir -Force
    Remove-Item $zip
}
$reported = (& $UvExe --version).Trim()
if ($reported -notlike "uv $UvVersion*") { throw "unexpected uv: $reported" }
Write-Host "uv ok: $reported"

& $UvExe python install $PythonVersion
if ($LASTEXITCODE -ne 0) { throw "uv python install failed ($LASTEXITCODE)" }
$python = (& $UvExe python find $PythonVersion).Trim()
if (-not $python.StartsWith($Root, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Python resolved outside the runner folder: $python"
}
Write-Host "python ok: $python"
Write-Host "runner ready at $Root"
