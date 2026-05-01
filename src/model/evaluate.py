"""Avaliacao do modelo: ROC-AUC, PR-AUC, F1, recall + tuning de threshold otimizado p/ recall."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .mlp import BrandSimilarityMLP

logger = logging.getLogger(__name__)


@dataclass
class EvalMetrics:
    roc_auc: float
    pr_auc: float
    f1: float
    precision: float
    recall: float
    threshold: float
    confusion: list[list[int]]
    n_pos: int
    n_neg: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "roc_auc": float(self.roc_auc),
            "pr_auc": float(self.pr_auc),
            "f1": float(self.f1),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "threshold": float(self.threshold),
            "confusion": self.confusion,
            "n_pos": self.n_pos,
            "n_neg": self.n_neg,
        }


def default_device() -> str:
    """Escolhe 'cuda' se dispon\u00edvel, sen\u00e3o 'cpu'."""
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


@torch.no_grad()
def predict_scores(
    model: BrandSimilarityMLP,
    X: np.ndarray,
    batch_size: int = 1024,
    device: str | None = None,
) -> np.ndarray:
    """Devolve scores 0-1 (sigmoid dos logits). Usa GPU por defeito quando dispon\u00edvel."""
    if device is None:
        device = default_device()
    model.eval()
    model.to(device)
    n = X.shape[0]
    out = np.zeros(n, dtype=np.float32)
    for i in range(0, n, batch_size):
        chunk = torch.from_numpy(X[i : i + batch_size].astype(np.float32, copy=False)).to(device)
        logits = model(chunk).view(-1)
        out[i : i + batch_size] = torch.sigmoid(logits).cpu().numpy()
    return out


def compute_metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> EvalMetrics:
    """Metricas para um threshold fixo. ROC/PR-AUC nao dependem do threshold."""
    if len(np.unique(y_true)) < 2:
        roc = 0.5
        prauc = float(y_true.mean())
    else:
        roc = float(roc_auc_score(y_true, y_score))
        prauc = float(average_precision_score(y_true, y_score))

    y_pred = (y_score >= threshold).astype(int)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    pre = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return EvalMetrics(
        roc_auc=roc,
        pr_auc=prauc,
        f1=f1,
        precision=pre,
        recall=rec,
        threshold=float(threshold),
        confusion=cm,
        n_pos=int((y_true == 1).sum()),
        n_neg=int((y_true == 0).sum()),
    )


def find_optimal_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    recall_floor: float = 0.85,
    grid_step: float = 0.01,
) -> tuple[float, dict[str, Any]]:
    """Maximiza F1 sob `recall >= recall_floor`. Se nenhum threshold atinge o piso,
    devolve o threshold com maior recall (politica conservadora -> menos falso negativo).
    """
    grid = np.arange(0.05, 0.951, grid_step)
    rows: list[tuple[float, float, float, float]] = []
    for thr in grid:
        y_pred = (y_score >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        pre = precision_score(y_true, y_pred, zero_division=0)
        rows.append((float(thr), float(f1), float(rec), float(pre)))

    eligible = [r for r in rows if r[2] >= recall_floor]
    if eligible:
        best = max(eligible, key=lambda r: (r[1], r[2]))
        policy = f"max_f1_with_recall>={recall_floor}"
    else:
        best = max(rows, key=lambda r: (r[2], r[1]))
        policy = "max_recall_fallback"
    thr_opt = best[0]
    info = {
        "policy": policy,
        "recall_floor": float(recall_floor),
        "f1_at_thr": float(best[1]),
        "recall_at_thr": float(best[2]),
        "precision_at_thr": float(best[3]),
        "scanned": len(rows),
    }
    logger.info(
        "Threshold otimo (%s): %.3f -> F1=%.3f, Recall=%.3f, Precision=%.3f",
        policy, thr_opt, info["f1_at_thr"], info["recall_at_thr"], info["precision_at_thr"],
    )
    return thr_opt, info


def roc_curve_data(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, list[float]]:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc(fpr, tpr))}


def pr_curve_data(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, list[float]]:
    pre, rec, _ = precision_recall_curve(y_true, y_score)
    return {"precision": pre.tolist(), "recall": rec.tolist(), "auc": float(average_precision_score(y_true, y_score))}


def recall_at_precision(y_true: np.ndarray, y_score: np.ndarray, target_precision: float = 0.9) -> float:
    """Maior recall observado para precision >= target_precision (util p/ comparar com OFTA)."""
    pre, rec, _ = precision_recall_curve(y_true, y_score)
    mask = pre >= target_precision
    if not mask.any():
        return 0.0
    return float(rec[mask].max())


__all__ = [
    "EvalMetrics",
    "predict_scores",
    "compute_metrics_at_threshold",
    "find_optimal_threshold",
    "roc_curve_data",
    "pr_curve_data",
    "recall_at_precision",
]
