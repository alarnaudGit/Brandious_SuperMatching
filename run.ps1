$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$activate = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $activate)) {
  Write-Host "Ambiente virtual não encontrado. Rode primeiro: .\setup.ps1"
  exit 1
}

& $activate
python run.py

