"""Carregamento, validacao, EDA e hashing do dataset de marcas (INPI)."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


REQUIRED_COLUMNS = [
    "marca_monitorada",
    "marca_colidente",
    "classe_marca_monitorada",
    "classe_marca_colidente",
    "especificacao_monitorado",
    "especificacao_colidente",
]

# Coluna real no Excel (confirmada pelo usuario):
LABEL_COLUMN_RAW = "Rótulo (1=manter, 0=outro)"
LABEL_COLUMN_CANON = "label"

# Aliases aceitos (toleramos pequenas variacoes/encoding)
LABEL_ALIASES = (
    "Rótulo (1=manter, 0=outro)",
    "Rotulo (1=manter, 0=outro)",
    "Rótulo (1=semelhante, 0=não semelhante)",
    "Rotulo (1=semelhante, 0=nao semelhante)",
    "label",
    "rotulo",
    "Rótulo",
)


@dataclass
class DatasetReport:
    """Resumo estatistico do dataset apos carregamento."""

    n_rows: int
    n_rows_dropped_null_brands: int
    n_pos: int
    n_neg: int
    pos_rate: float
    same_class_share: float
    pos_rate_same_class: float
    pos_rate_diff_class: float
    null_counts: dict[str, int]
    brand_len_stats: dict[str, dict[str, float]]
    spec_len_stats: dict[str, dict[str, float]]
    classes_top: list[tuple[Any, int]]
    dataset_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_rows_dropped_null_brands": self.n_rows_dropped_null_brands,
            "n_pos": self.n_pos,
            "n_neg": self.n_neg,
            "pos_rate": self.pos_rate,
            "same_class_share": self.same_class_share,
            "pos_rate_same_class": self.pos_rate_same_class,
            "pos_rate_diff_class": self.pos_rate_diff_class,
            "null_counts": self.null_counts,
            "brand_len_stats": self.brand_len_stats,
            "spec_len_stats": self.spec_len_stats,
            "classes_top": [list(t) for t in self.classes_top],
            "dataset_hash": self.dataset_hash,
        }


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Calcula SHA-256 de um arquivo binario, em chunks."""
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _resolve_label_column(df: pd.DataFrame) -> str:
    for alias in LABEL_ALIASES:
        if alias in df.columns:
            return alias
    candidates = [c for c in df.columns if "tulo" in c.lower() or "label" in c.lower()]
    if candidates:
        return candidates[0]
    raise ValueError(
        f"Coluna de label nao encontrada. Esperado um de: {LABEL_ALIASES}. "
        f"Colunas disponiveis: {list(df.columns)}"
    )


def validate_columns(df: pd.DataFrame) -> str:
    """Garante que as colunas obrigatorias existem; devolve nome real do label."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Colunas obrigatorias ausentes: {missing}. "
            f"Colunas disponiveis: {list(df.columns)}"
        )
    return _resolve_label_column(df)


def clean_and_normalize(df: pd.DataFrame, label_col_raw: str) -> tuple[pd.DataFrame, int]:
    """Trata nulos e renomeia label para `label`. Retorna (df, n_dropped)."""
    df = df.copy()

    n_before = len(df)
    mask_brands_ok = (
        df["marca_monitorada"].notna()
        & df["marca_colidente"].notna()
        & (df["marca_monitorada"].astype(str).str.strip() != "")
        & (df["marca_colidente"].astype(str).str.strip() != "")
    )
    n_dropped = int((~mask_brands_ok).sum())
    if n_dropped:
        logger.warning("Removidas %d linhas com marca_monitorada/colidente nula ou vazia.", n_dropped)
    df = df.loc[mask_brands_ok].reset_index(drop=True)

    df["classe_marca_monitorada"] = (
        pd.to_numeric(df["classe_marca_monitorada"], errors="coerce").fillna(-1).astype(int)
    )
    df["classe_marca_colidente"] = (
        pd.to_numeric(df["classe_marca_colidente"], errors="coerce").fillna(-1).astype(int)
    )

    df["especificacao_monitorado"] = df["especificacao_monitorado"].fillna("").astype(str)
    df["especificacao_colidente"] = df["especificacao_colidente"].fillna("").astype(str)

    df["marca_monitorada"] = df["marca_monitorada"].astype(str).str.strip()
    df["marca_colidente"] = df["marca_colidente"].astype(str).str.strip()

    df[LABEL_COLUMN_CANON] = pd.to_numeric(df[label_col_raw], errors="coerce").fillna(0).astype(int).clip(0, 1)
    if label_col_raw != LABEL_COLUMN_CANON and label_col_raw in df.columns:
        df = df.drop(columns=[label_col_raw])

    n_after = len(df)
    assert n_after + n_dropped == n_before
    return df, n_dropped


def _len_stats(s: pd.Series) -> dict[str, float]:
    arr = s.astype(str).str.len()
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p25": float(arr.quantile(0.25)),
        "p50": float(arr.quantile(0.50)),
        "p75": float(arr.quantile(0.75)),
        "max": float(arr.max()),
    }


def compute_eda(df: pd.DataFrame, dataset_hash: str, n_dropped: int) -> DatasetReport:
    label = df[LABEL_COLUMN_CANON]
    n_pos = int(label.sum())
    n_neg = int(len(df) - n_pos)
    pos_rate = float(n_pos / len(df)) if len(df) else 0.0

    same_class = df["classe_marca_monitorada"] == df["classe_marca_colidente"]
    same_class_share = float(same_class.mean()) if len(df) else 0.0
    pos_rate_same = float(label[same_class].mean()) if same_class.any() else 0.0
    pos_rate_diff = float(label[~same_class].mean()) if (~same_class).any() else 0.0

    null_counts = {c: int(df[c].isna().sum()) for c in REQUIRED_COLUMNS}

    brand_len_stats = {
        "marca_monitorada": _len_stats(df["marca_monitorada"]),
        "marca_colidente": _len_stats(df["marca_colidente"]),
    }
    spec_len_stats = {
        "especificacao_monitorado": _len_stats(df["especificacao_monitorado"]),
        "especificacao_colidente": _len_stats(df["especificacao_colidente"]),
    }

    classes_all = pd.concat([df["classe_marca_monitorada"], df["classe_marca_colidente"]])
    top = classes_all.value_counts().head(15)
    classes_top = [(int(idx), int(cnt)) for idx, cnt in top.items()]

    return DatasetReport(
        n_rows=len(df),
        n_rows_dropped_null_brands=n_dropped,
        n_pos=n_pos,
        n_neg=n_neg,
        pos_rate=pos_rate,
        same_class_share=same_class_share,
        pos_rate_same_class=pos_rate_same,
        pos_rate_diff_class=pos_rate_diff,
        null_counts=null_counts,
        brand_len_stats=brand_len_stats,
        spec_len_stats=spec_len_stats,
        classes_top=classes_top,
        dataset_hash=dataset_hash,
    )


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, DatasetReport]:
    """Carrega e valida o Excel; devolve (df_normalizado, report)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {p}")

    dataset_hash = file_sha256(p)
    df_raw = pd.read_excel(p)

    label_col_raw = validate_columns(df_raw)
    df, n_dropped = clean_and_normalize(df_raw, label_col_raw)
    report = compute_eda(df, dataset_hash, n_dropped)

    logger.info(
        "Dataset carregado: %d linhas, %d positivos (%.2f%%), hash=%s...",
        report.n_rows,
        report.n_pos,
        report.pos_rate * 100,
        dataset_hash[:12],
    )
    return df, report


def load_dataframe_from_bytes(data: bytes, file_name: str = "uploaded.xlsx") -> tuple[pd.DataFrame, DatasetReport]:
    """Carrega de bytes em memoria (uso direto via Streamlit upload)."""
    h = hashlib.sha256(data).hexdigest()
    import io

    df_raw = pd.read_excel(io.BytesIO(data))
    label_col_raw = validate_columns(df_raw)
    df, n_dropped = clean_and_normalize(df_raw, label_col_raw)
    report = compute_eda(df, h, n_dropped)
    return df, report


__all__ = [
    "REQUIRED_COLUMNS",
    "LABEL_COLUMN_RAW",
    "LABEL_COLUMN_CANON",
    "DatasetReport",
    "file_sha256",
    "validate_columns",
    "clean_and_normalize",
    "compute_eda",
    "load_dataset",
    "load_dataframe_from_bytes",
]
