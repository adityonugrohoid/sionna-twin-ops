# Process-scoped environment for a runner folder: every uv location stays inside -Root.
# Dot-source it: . .\env.ps1 -Root C:\path\to\runner

param(
    [Parameter(Mandatory = $true)][string]$Root
)

$env:UV_CACHE_DIR = Join-Path $Root 'cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $Root 'python'
$env:UV_TOOL_DIR = Join-Path $Root 'tools\uvtools'
# Without this, `uv python install` puts python3.13.exe in the user's .local\bin.
$env:UV_PYTHON_BIN_DIR = Join-Path $Root 'tools\pybin'
$env:UV_PYTHON_PREFERENCE = 'only-managed'
$env:UV_NO_CONFIG = '1'
$UvExe = Join-Path $Root 'tools\uv\uv.exe'
