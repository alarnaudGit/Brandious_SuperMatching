@echo off
setlocal
cd /d "%~dp0"

REM Launcher Windows: executa o app imediatamente (Streamlit com fallback Flask).
if exist "%~dp0.venv\Scripts\python.exe" (
  "%~dp0.venv\Scripts\python.exe" "%~dp0executar_app.py"
) else (
  python "%~dp0executar_app.py"
)

