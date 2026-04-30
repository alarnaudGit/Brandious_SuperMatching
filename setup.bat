@echo off
setlocal

cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo Python nao encontrado no PATH. Instale o Python e reinicie o terminal.
  exit /b 1
)

if not exist ".venv\" (
  python -m venv .venv
)

call ".venv\Scripts\activate.bat"
python -m pip install -U pip
pip install -r requirements.txt

echo OK. Ambiente pronto.
echo Para rodar: run.bat

