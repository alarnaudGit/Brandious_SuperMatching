"""Features de classe Nice (sem peso manual - apenas one-hot top-15 + relacao)."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_TOP_K = 15


def fit_top_k_classes(
    classes_a: Iterable[int],
    classes_b: Iterable[int],
    k: int = DEFAULT_TOP_K,
) -> list[int]:
    """Determina top-K classes Nice mais frequentes (uniao dos dois lados)."""
    series = pd.concat([pd.Series(list(classes_a)), pd.Series(list(classes_b))])
    series = series[series >= 0]
    return [int(c) for c in series.value_counts().head(k).index.tolist()]


def class_feature_names(top_classes: list[int]) -> list[str]:
    cols: list[str] = ["cls_same", "cls_diff_abs", "cls_a_known", "cls_b_known"]
    for c in top_classes:
        cols.append(f"cls_a_top_{c}")
    cols.append("cls_a_top_other")
    for c in top_classes:
        cols.append(f"cls_b_top_{c}")
    cols.append("cls_b_top_other")
    return cols


def build_class_features_row(
    cls_a: int,
    cls_b: int,
    top_classes: list[int],
) -> dict[str, float]:
    """Features de classe para um unico par."""
    out: dict[str, float] = {
        "cls_same": float(cls_a == cls_b and cls_a >= 0),
        "cls_diff_abs": float(abs(cls_a - cls_b)) if cls_a >= 0 and cls_b >= 0 else 0.0,
        "cls_a_known": float(cls_a >= 0),
        "cls_b_known": float(cls_b >= 0),
    }
    matched_a = False
    for c in top_classes:
        flag = float(cls_a == c)
        out[f"cls_a_top_{c}"] = flag
        if flag:
            matched_a = True
    out["cls_a_top_other"] = float(cls_a >= 0 and not matched_a)

    matched_b = False
    for c in top_classes:
        flag = float(cls_b == c)
        out[f"cls_b_top_{c}"] = flag
        if flag:
            matched_b = True
    out["cls_b_top_other"] = float(cls_b >= 0 and not matched_b)
    return out


def build_class_features_matrix(
    classes_a: np.ndarray,
    classes_b: np.ndarray,
    top_classes: list[int],
) -> tuple[np.ndarray, list[str]]:
    """Versao vetorizada para o builder em batch."""
    cols = class_feature_names(top_classes)
    n = len(classes_a)
    X = np.zeros((n, len(cols)), dtype=np.float32)

    same = (classes_a == classes_b) & (classes_a >= 0)
    diff_abs = np.where((classes_a >= 0) & (classes_b >= 0), np.abs(classes_a - classes_b), 0.0)
    X[:, 0] = same.astype(np.float32)
    X[:, 1] = diff_abs.astype(np.float32)
    X[:, 2] = (classes_a >= 0).astype(np.float32)
    X[:, 3] = (classes_b >= 0).astype(np.float32)

    base_idx = 4
    matched_a = np.zeros(n, dtype=bool)
    for j, c in enumerate(top_classes):
        col = base_idx + j
        flag = (classes_a == c)
        X[:, col] = flag.astype(np.float32)
        matched_a |= flag
    X[:, base_idx + len(top_classes)] = ((classes_a >= 0) & ~matched_a).astype(np.float32)

    base_idx_b = base_idx + len(top_classes) + 1
    matched_b = np.zeros(n, dtype=bool)
    for j, c in enumerate(top_classes):
        col = base_idx_b + j
        flag = (classes_b == c)
        X[:, col] = flag.astype(np.float32)
        matched_b |= flag
    X[:, base_idx_b + len(top_classes)] = ((classes_b >= 0) & ~matched_b).astype(np.float32)

    return X, cols


__all__ = [
    "DEFAULT_TOP_K",
    "fit_top_k_classes",
    "class_feature_names",
    "build_class_features_row",
    "build_class_features_matrix",
]
