"""Geracao dos artefatos finais: JSON do modelo, model.pt, preprocessor.pkl, parquet, xlsx enriquecido."""
from __future__ import annotations

import base64
import io
import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from . import __version__
from .data import DatasetReport, LABEL_COLUMN_CANON
from .model.dataset import BalancingConfig, SplitConfig
from .model.evaluate import EvalMetrics, predict_scores
from .model.mlp import BrandSimilarityMLP, MLPConfig
from .model.train import TrainConfig, TrainResult
from .pipeline.preprocessor import FeaturePreprocessor, _safe_progress

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state_dict_to_b64(state: dict[str, torch.Tensor]) -> str:
    buf = io.BytesIO()
    torch.save(state, buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))
    return obj


def build_model_config_dict(
    *,
    mlp_config: Any,
    preprocessor: FeaturePreprocessor,
    train_config: TrainConfig,
    balancing: BalancingConfig,
    split_config: SplitConfig,
    threshold_optimal: float,
    threshold_policy: dict[str, Any],
    metrics_train: EvalMetrics,
    metrics_val: EvalMetrics,
    metrics_test: EvalMetrics,
    train_result: TrainResult,
    dataset_report: DatasetReport,
    state_dict: dict[str, torch.Tensor],
    embedding_used: bool,
    architecture_bagging: dict[str, Any] | None = None,
    logreg_bagging: dict[str, Any] | None = None,
    hybrid_bagging: dict[str, Any] | None = None,
) -> dict[str, Any]:
    arch_dict = (
        mlp_config.to_dict() if hasattr(mlp_config, "to_dict") else dict(mlp_config)
    )
    cfg = {
        "version": __version__,
        "timestamp": _utc_now_iso(),
        "dataset_hash": dataset_report.dataset_hash,
        "label_column_source": "Rótulo (1=manter, 0=outro)",
        "label_column_canon": LABEL_COLUMN_CANON,
        "architecture": arch_dict,
        "feature_names_ordered": list(preprocessor.feature_names_ordered),
        "preprocessing": {
            "config": preprocessor.config.to_dict(),
            "preprocessor_pickle": "preprocessor.pkl",
            "embedding_used": bool(embedding_used),
            "top_classes": list(preprocessor.top_classes),
        },
        "balancing": balancing.to_dict() | {"pos_weight_used": float(train_result.pos_weight_used)},
        "split": split_config.to_dict(),
        "training": train_config.to_dict() | {
            "best_epoch": train_result.best_epoch,
            "best_pr_auc_val": train_result.best_pr_auc_val,
            "n_train_after_balancing": train_result.n_train_after_balancing,
        },
        "threshold_optimal": float(threshold_optimal),
        "threshold_policy": threshold_policy,
        "metrics": {
            "train": metrics_train.to_dict(),
            "val": metrics_val.to_dict(),
            "test": metrics_test.to_dict(),
        },
        "history": train_result.history,
        "dataset_report": dataset_report.to_dict(),
        "state_dict_b64": _state_dict_to_b64(state_dict),
    }
    if architecture_bagging:
        cfg["architecture_bagging"] = architecture_bagging
    if logreg_bagging:
        cfg["logreg_bagging"] = logreg_bagging
    if hybrid_bagging:
        cfg["hybrid_bagging"] = hybrid_bagging
    return _to_jsonable(cfg)


def save_model_config(cfg: dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    logger.info("JSON do modelo salvo em %s", p)
    return p


def save_state_dict(state_dict: dict[str, torch.Tensor], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, p)
    logger.info("model.pt salvo em %s", p)
    return p


def save_enriched_dataframe(
    df_original: pd.DataFrame,
    feature_matrix: np.ndarray,
    feature_names: list[str],
    score_nn: np.ndarray,
    score_heuristic: np.ndarray,
    threshold: float,
    out_xlsx: str | Path,
    out_parquet: str | Path | None = None,
    label_col: str | None = None,
    version: str = __version__,
    *,
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple[Path, Path | None]:
    """Salva o dataframe enriquecido em Excel e Parquet."""
    cb = progress_callback
    _safe_progress(cb, 0.05, "A copiar dados base...")
    df = df_original.copy().reset_index(drop=True)
    _safe_progress(cb, 0.18, "A montar DataFrame de features...")
    feat_df = pd.DataFrame(feature_matrix, columns=feature_names)
    _safe_progress(cb, 0.32, "A unir colunas (original + features)...")
    enriched = pd.concat([df, feat_df], axis=1)
    _safe_progress(cb, 0.42, "A adicionar scores e metadados...")
    enriched["score_heuristico_ofta"] = score_heuristic
    enriched["score_nn"] = score_nn
    enriched["classe_prevista"] = (score_nn >= threshold).astype(int)
    enriched["threshold_usado"] = float(threshold)
    enriched["timestamp"] = _utc_now_iso()
    enriched["versao_modelo"] = version

    if label_col and label_col in enriched.columns:
        cols = [c for c in enriched.columns if c != label_col] + [label_col]
        enriched = enriched[cols]

    out_xlsx = Path(out_xlsx)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    _safe_progress(
        cb, 0.48,
        f"A gravar Excel ({len(enriched):,} linhas x {enriched.shape[1]} colunas) — pode demorar...",
    )
    enriched.to_excel(out_xlsx, index=False)
    logger.info("Planilha enriquecida salva em %s (%d linhas, %d colunas)", out_xlsx, len(enriched), enriched.shape[1])
    _safe_progress(cb, 0.88, "Excel gravado.")

    parquet_path: Path | None = None
    if out_parquet is not None:
        parquet_path = Path(out_parquet)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        _safe_progress(cb, 0.92, "A gravar Parquet...")
        enriched.to_parquet(parquet_path, index=False)
        logger.info("Parquet enriquecido salvo em %s", parquet_path)

    _safe_progress(cb, 1.0, "Visao enriquecida gravada.")
    return out_xlsx, parquet_path


__all__ = [
    "build_model_config_dict",
    "save_model_config",
    "save_state_dict",
    "save_enriched_dataframe",
]
