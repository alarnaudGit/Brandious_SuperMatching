"""Explicabilidade: permutation importance (global) + integrated gradients (por par)."""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import torch

from .evaluate import predict_scores
from .mlp import BrandSimilarityMLP

logger = logging.getLogger(__name__)


def permutation_importance(
    model: BrandSimilarityMLP,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    metric: str = "pr_auc",
    n_repeats: int = 3,
    seed: int = 42,
    device: str = "cpu",
) -> list[dict[str, float]]:
    """Quanto piora `metric` quando embaralhamos cada coluna. Retorna lista ordenada."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    rng = np.random.default_rng(seed)
    base_scores = predict_scores(model, X, device=device)
    if metric == "pr_auc":
        base = float(average_precision_score(y, base_scores))
    elif metric == "roc_auc":
        base = float(roc_auc_score(y, base_scores))
    else:
        raise ValueError(f"metric desconhecida: {metric}")

    n_feats = X.shape[1]
    importances: list[float] = []
    for j in range(n_feats):
        deltas: list[float] = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            rng.shuffle(X_perm[:, j])
            scores = predict_scores(model, X_perm, device=device)
            if metric == "pr_auc":
                m = float(average_precision_score(y, scores))
            else:
                m = float(roc_auc_score(y, scores))
            deltas.append(base - m)
        importances.append(float(np.mean(deltas)))

    rows = [
        {"feature": str(name), "importance": float(imp)}
        for name, imp in zip(feature_names, importances)
    ]
    rows.sort(key=lambda r: r["importance"], reverse=True)
    return rows


def integrated_gradients_for_row(
    model: BrandSimilarityMLP,
    x: np.ndarray,
    feature_names: Sequence[str],
    baseline: np.ndarray | None = None,
    n_steps: int = 50,
    device: str = "cpu",
) -> list[dict[str, float]]:
    """Atribui contribuicao de cada feature para o logit deste par especifico.

    Usa captum se disponivel; cai em IG manual caso contrario.
    """
    model.eval()
    model.to(device)
    x_t = torch.tensor(x, dtype=torch.float32, device=device).view(1, -1)
    if baseline is None:
        baseline_t = torch.zeros_like(x_t)
    else:
        baseline_t = torch.tensor(baseline, dtype=torch.float32, device=device).view(1, -1)

    try:
        from captum.attr import IntegratedGradients

        ig = IntegratedGradients(model)
        attributions, _ = ig.attribute(
            inputs=x_t,
            baselines=baseline_t,
            n_steps=n_steps,
            return_convergence_delta=True,
        )
        attr = attributions.detach().cpu().numpy().reshape(-1)
    except Exception as exc:
        logger.warning("captum nao disponivel ou falhou (%s); usando IG manual.", exc)
        diff = (x_t - baseline_t)
        grads_total = torch.zeros_like(x_t)
        for k in range(1, n_steps + 1):
            alpha = k / n_steps
            xk = baseline_t + alpha * diff
            xk.requires_grad_(True)
            out = model(xk)
            grads = torch.autograd.grad(out.sum(), xk)[0]
            grads_total += grads
        attr = (diff * grads_total / n_steps).detach().cpu().numpy().reshape(-1)

    rows = [
        {"feature": str(n), "value": float(x[i]), "contribution": float(attr[i])}
        for i, n in enumerate(feature_names)
    ]
    rows.sort(key=lambda r: abs(r["contribution"]), reverse=True)
    return rows


__all__ = [
    "permutation_importance",
    "integrated_gradients_for_row",
]
