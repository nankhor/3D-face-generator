# FaceGen installer for Windows. Run from the repository folder:
#   powershell -ExecutionPolicy Bypass -File installer\install.ps1
# Options are passed on to installer\install.py (for example -- --yes --skip-meshroom).

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "FaceGen: preparing Python"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv (the Python manager FaceGen uses) from https://astral.sh/uv"
    powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

uv python install 3.12
if (-not (Test-Path ".venv")) {
    uv venv --python 3.12 .venv
}
uv pip install --python .venv\Scripts\python.exe -e ".[heic]"

& .venv\Scripts\python.exe installer\install.py @args
exit $LASTEXITCODE
