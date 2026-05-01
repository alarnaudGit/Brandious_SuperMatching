"""
Launcher único do Brandious SuperMatching.

Uso (sem parâmetros):
  python executar_app.py

Prioridade:
  1) Streamlit UI (`streamlit_app.py`)
  2) Fallback Flask legado (`run.py`)
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import webbrowser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _is_port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            return False
        return True


def _pick_free_port(host: str, preferred: int, *, span: int = 50) -> int:
    if _is_port_free(host, preferred):
        return preferred
    for p in range(preferred + 1, preferred + span + 1):
        if _is_port_free(host, p):
            return p
    return preferred


def _run_streamlit() -> int:
    app_path = PROJECT_ROOT / "streamlit_app.py"
    if not app_path.exists():
        return 2

    host = "127.0.0.1"
    port = _pick_free_port(host, 8501)
    url = f"http://{host}:{port}/"

    # Ajuda o Streamlit a encontrar imports relativos ao projeto.
    env = dict(os.environ)
    extra = str(PROJECT_ROOT)
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = extra + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = extra

    # Importante: não abrir o navegador aqui.
    # O Streamlit já abre uma guia automaticamente; abrir aqui duplicaria.

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        host,
        "--server.port",
        str(port),
        "--browser.gatherUsageStats",
        "false",
    ]

    print(f"Iniciando Streamlit em {url}")
    print("Dica: para encerrar, use Ctrl+C neste terminal.")
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    return int(proc.returncode or 0)


def _run_flask_legacy() -> int:
    legacy_path = PROJECT_ROOT / "run.py"
    if not legacy_path.exists():
        print("Nenhum app encontrado: faltam `streamlit_app.py` e `run.py`.")
        return 2

    url = "http://127.0.0.1:5000/"
    try:
        webbrowser.open(url, new=2)
    except Exception:
        pass

    print(
        "Streamlit não disponível. Iniciando Flask legado em http://127.0.0.1:5000/"
    )
    print("Dica: para encerrar, use Ctrl+C neste terminal.")
    proc = subprocess.run([sys.executable, str(legacy_path)], cwd=str(PROJECT_ROOT))
    return int(proc.returncode or 0)


def main() -> int:
    try:
        import streamlit  # noqa: F401
    except Exception:
        return _run_flask_legacy()
    else:
        rc = _run_streamlit()
        if rc == 2:
            return _run_flask_legacy()
        return rc


if __name__ == "__main__":
    raise SystemExit(main())

