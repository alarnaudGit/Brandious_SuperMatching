@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
  echo Ambiente virtual nao encontrado. Rode primeiro: setup.bat
  exit /b 1
)

call ".venv\Scripts\activate.bat"
python run.py

