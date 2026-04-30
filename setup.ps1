$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
  throw "Python não encontrado no PATH. Instale o Python e reinicie o terminal."
}

if (-not (Test-Path ".venv")) {
  python -m venv .venv
}

$activate = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $activate)) {
  throw "Ambiente virtual parece incompleto. Apague a pasta .venv e rode novamente."
}

& $activate
python -m pip install -U pip
pip install -r requirements.txt

Write-Host "OK. Ambiente pronto."
Write-Host "Para rodar: .\run.ps1"

