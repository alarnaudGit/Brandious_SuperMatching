"""Persistencia local do ultimo enriquecimento (matriz + preprocessor) para reuso apos reinicio.

Grava em ``<project>/artifacts/last_enrichment_session/`` (pasta ja ignorada pelo .gitignore).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .preprocessor import FeaturePreprocessor

logger = logging.getLogger(__name__)

CACHE_SCHEMA_VERSION = 1
CACHE_SUBDIR = Path("artifacts") / "last_enrichment_session"
PREPROC_FILENAME = "preprocessor.joblib"
MATRIX_FILENAME = "feature_matrix.npz"
MATRIX_KEY = "feature_matrix"
MANIFEST_FILENAME = "manifest.json"


class EnrichmentCacheError(Exception):
    """Falha ao carregar ou validar cache de enriquecimento."""

    def __init__(
        self,
        message: str,
        *,
        user_message: str | None = None,
    ) -> None:
        super().__init__(message)
        self.user_message = user_message or message


def cache_dir(project_root: Path) -> Path:
    return project_root / CACHE_SUBDIR


def manifest_path(project_root: Path) -> Path:
    return cache_dir(project_root) / MANIFEST_FILENAME


def cache_summary_line(project_root: Path) -> str | None:
    """Uma linha para UI; None se nao houver manifest valido."""
    mp = manifest_path(project_root)
    if not mp.is_file():
        return None
    try:
        manifest = json.loads(mp.read_text(encoding="utf-8"))
        n_rows = int(manifest.get("n_rows", 0))
        n_feat = int(manifest.get("n_features", 0))
        dh = str(manifest.get("dataset_hash", ""))[:16]
        saved = str(manifest.get("saved_at_iso", ""))[:19]
        tail = f" | gravado em {saved}" if saved else ""
        return (
            f"Cache local: **{n_rows:,}** linhas x **{n_feat}** features | "
            f"hash dataset `{dh}...`{tail}"
        )
    except Exception:  # noqa: BLE001
        return None


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def save_last_enrichment(
    project_root: Path,
    df: pd.DataFrame,
    *,
    dataset_hash: str,
    preprocessor: FeaturePreprocessor,
    X: np.ndarray,
    preproc_cfg_signature: dict[str, Any],
    app_version: str,
) -> Path:
    """Grava preprocessor, matriz e manifest. Devolve diretorio do cache."""
    root = cache_dir(project_root)
    root.mkdir(parents=True, exist_ok=True)
    preproc_path = root / PREPROC_FILENAME
    matrix_path = root / MATRIX_FILENAME
    manifest_json = root / MANIFEST_FILENAME

    fnames = list(preprocessor.feature_names_ordered)
    x = np.asarray(X, dtype=np.float32)
    if x.shape[0] != len(df):
        raise ValueError(
            f"save_last_enrichment: len(df)={len(df)} != X.shape[0]={x.shape[0]}",
        )
    if x.shape[1] != len(fnames):
        raise ValueError(
            f"save_last_enrichment: X.shape[1]={x.shape[1]} != len(features)={len(fnames)}",
        )

    preprocessor.save(preproc_path)
    np.savez_compressed(matrix_path, **{MATRIX_KEY: x})

    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "dataset_hash": str(dataset_hash),
        "n_rows": int(len(df)),
        "n_features": int(x.shape[1]),
        "feature_names": fnames,
        "preproc_cfg_signature": _json_safe(dict(preproc_cfg_signature)),
        "app_version": str(app_version),
    }
    manifest["saved_at_iso"] = datetime.now(tz=timezone.utc).isoformat()
    manifest_json.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Cache de enriquecimento gravado em %s", root)
    return root


def load_last_enrichment(
    project_root: Path,
    df: pd.DataFrame,
    *,
    dataset_hash: str,
) -> tuple[FeaturePreprocessor, np.ndarray, list[str], dict[str, Any]]:
    """Carrega cache se compativel com o dataset atual. Levanta EnrichmentCacheError."""
    root = cache_dir(project_root)
    mp = root / MANIFEST_FILENAME
    mx = root / MATRIX_FILENAME
    pp = root / PREPROC_FILENAME

    if not mp.is_file() or not mx.is_file() or not pp.is_file():
        raise EnrichmentCacheError(
            "Arquivos de cache incompletos.",
            user_message=(
                "Nao ha cache local completo de enriquecimento. "
                "Execute **Gerar features** na aba 2."
            ),
        )

    try:
        manifest = json.loads(mp.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EnrichmentCacheError(
            f"Manifest invalido: {exc}",
            user_message=(
                "O arquivo de manifesto do cache esta corrompido. "
                "Apague a pasta `artifacts/last_enrichment_session` ou rode "
                "**Gerar features** de novo."
            ),
        ) from exc

    if int(manifest.get("schema_version", 0)) != CACHE_SCHEMA_VERSION:
        raise EnrichmentCacheError(
            "schema_version incompativel.",
            user_message=(
                "O cache local foi gerado por uma versao mais antiga ou diferente "
                "do aplicativo. Gere o enriquecimento novamente na **aba 2**."
            ),
        )

    mh = str(manifest.get("dataset_hash", ""))
    if mh != str(dataset_hash):
        raise EnrichmentCacheError(
            "dataset_hash diferente.",
            user_message=(
                "O cache local pertence a **outro arquivo de dados** "
                "(hash SHA-256 diferente do Excel atual). "
                "Para treinar com este dataset, use **Gerar features** na aba 2."
            ),
        )

    n_rows_m = int(manifest.get("n_rows", -1))
    if n_rows_m != len(df):
        raise EnrichmentCacheError(
            f"n_rows manifest={n_rows_m} vs len(df)={len(df)}",
            user_message=(
                f"Incompatibilidade: o cache tem **{n_rows_m:,}** linhas, mas o "
                f"dataset carregado tem **{len(df):,}**. "
                "Recarregue o mesmo arquivo usado no enriquecimento ou regenere as features."
            ),
        )

    try:
        raw = np.load(mx, allow_pickle=False)
        if MATRIX_KEY not in raw.files:
            raise EnrichmentCacheError(
                f"npz sem chave {MATRIX_KEY!r}",
                user_message=(
                    "Arquivo de matriz do cache esta corrompido ou incompleto. "
                    "Regenere o enriquecimento na aba 2."
                ),
            )
        X = np.asarray(raw[MATRIX_KEY], dtype=np.float32)
    except EnrichmentCacheError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise EnrichmentCacheError(
            f"Falha ao ler matriz: {exc}",
            user_message=(
                "Nao foi possivel ler a matriz do cache (arquivo danificado). "
                "Regenere o enriquecimento na aba 2."
            ),
        ) from exc

    n_feat_m = int(manifest.get("n_features", -1))
    if X.shape != (n_rows_m, n_feat_m):
        raise EnrichmentCacheError(
            f"shape X={X.shape} esperado=({n_rows_m},{n_feat_m})",
            user_message=(
                "A matriz gravada no cache nao coincide com o manifesto (dimensoes). "
                "Regenere o enriquecimento na aba 2."
            ),
        )

    try:
        preprocessor = FeaturePreprocessor.load(pp)
    except Exception as exc:  # noqa: BLE001
        raise EnrichmentCacheError(
            f"Falha ao carregar preprocessor: {exc}",
            user_message=(
                "O arquivo do preprocessador no cache esta corrompido ou incompativel. "
                "Regenere o enriquecimento na aba 2."
            ),
        ) from exc

    fn_manifest = manifest.get("feature_names")
    if not isinstance(fn_manifest, list):
        raise EnrichmentCacheError(
            "feature_names invalidos no manifest.",
            user_message="Manifesto do cache invalido. Regenere o enriquecimento na aba 2.",
        )
    fn_loaded = list(preprocessor.feature_names_ordered)
    if fn_loaded != fn_manifest:
        raise EnrichmentCacheError(
            "feature_names preprocessor != manifest",
            user_message=(
                "Ordem ou lista de features do preprocessador nao bate com o manifesto "
                "(codigo ou cache misturados). Regenere o enriquecimento na aba 2."
            ),
        )

    sig = manifest.get("preproc_cfg_signature")
    if not isinstance(sig, dict):
        sig = {}

    return preprocessor, X, fn_loaded, dict(sig)


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "EnrichmentCacheError",
    "cache_dir",
    "cache_summary_line",
    "load_last_enrichment",
    "manifest_path",
    "save_last_enrichment",
]
