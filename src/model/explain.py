"""Explicabilidade: permutation importance (global) + integrated gradients (por par).

Inclui suporte a bagging hibrido (MLPs + LogRegs + Florestas):
- `predict_hybrid_bagging` agrega scores das tres listas por media simples.
- `permutation_importance_hybrid` usa o score agregado como base.
- `integrated_gradients_hybrid_row` combina IG das MLPs (captum), a
  contribuicao analitica das LogRegs (`coef_ * (x - baseline)`, equivalente
  a IG num modelo linear) e a contribuicao TreeSHAP das Florestas. Como cada
  fonte vive em uma escala propria (logit para MLP/LogReg, probabilidade para
  TreeSHAP), as contribuicoes sao normalizadas por modelo (`/ sum(|c|)`)
  antes de tirar media; isso preserva sinal/ranking e torna as escalas
  comparaveis.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from .evaluate import default_device, predict_scores

logger = logging.getLogger(__name__)


def __getattr__(name: str) -> Any:
    """Lazy import: evita dependencia circular e caches antigos sem o simbolo."""
    if name == "permutation_importance_stacking_context":
        from .stacking_meta import (
            permutation_importance_stacking_context as _permutation_importance_stacking_context,
        )

        return _permutation_importance_stacking_context
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def permutation_importance(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    metric: str = "pr_auc",
    n_repeats: int = 3,
    seed: int = 42,
    device: str | None = None,
) -> list[dict[str, float]]:
    """Quanto piora `metric` quando embaralhamos cada coluna. Retorna lista ordenada."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    if device is None:
        device = default_device()
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


def permutation_importance_arch_bagging(
    members: Sequence[Any],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    metric: str = "pr_auc",
    n_repeats: int = 3,
    seed: int = 42,
    device: str | None = None,
) -> list[dict[str, float]]:
    """Permutation importance em relacao ao score agregado (media dos K modelos)."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    from .arch_bagging import predict_architecture_bagging

    if device is None:
        device = default_device()
    rng = np.random.default_rng(seed)
    base_scores = predict_architecture_bagging(
        list(members), X, device=device, apply_calibration=True,
    )
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
            scores = predict_architecture_bagging(
                list(members), X_perm, device=device, apply_calibration=True,
            )
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
    model: nn.Module,
    x: np.ndarray,
    feature_names: Sequence[str],
    baseline: np.ndarray | None = None,
    n_steps: int = 50,
    device: str | None = None,
) -> list[dict[str, float]]:
    """Atribui contribuicao de cada feature para o logit deste par especifico.

    Usa captum se disponivel; cai em IG manual caso contrario.
    """
    if device is None:
        device = default_device()
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


def integrated_gradients_arch_bagging_row(
    members: Sequence[Any],
    x: np.ndarray,
    feature_names: Sequence[str],
    baseline: np.ndarray | None = None,
    n_steps: int = 50,
    device: str | None = None,
) -> list[dict[str, float]]:
    """Media das contribuicoes IG de cada modelo (mesmo tratamento que o score final)."""
    if not members:
        raise ValueError("integrated_gradients_arch_bagging_row: members vazio.")
    if device is None:
        device = default_device()
    acc: defaultdict[str, float] = defaultdict(float)
    vals = {str(n): float(x[i]) for i, n in enumerate(feature_names)}
    for m in members:
        mod = getattr(m, "model", m)
        rows = integrated_gradients_for_row(
            mod, x, feature_names, baseline=baseline, n_steps=n_steps, device=device,
        )
        for r in rows:
            acc[str(r["feature"])] += float(r["contribution"])
    n = len(members)
    out = [
        {"feature": str(fn), "value": vals[str(fn)], "contribution": acc[str(fn)] / n}
        for fn in feature_names
    ]
    out.sort(key=lambda r: abs(r["contribution"]), reverse=True)
    return out


# =============================================================================
# Bagging hibrido: MLPs + LogRegs + Florestas
# =============================================================================

def predict_hybrid_bagging(
    mlp_members: Sequence[Any] | None,
    logreg_members: Sequence[Any] | None,
    X: np.ndarray,
    *,
    forest_members: Sequence[Any] | None = None,
    device: str | None = None,
    apply_calibration: bool = True,
) -> np.ndarray:
    """Score final do bagging hibrido = media de TODAS as probabilidades
    (MLPs + LogRegs + Florestas) calibradas Platt por modelo.
    """
    from .arch_bagging import predict_architecture_bagging_components
    from .logreg_bagging import predict_logreg_bagging_components
    from .forest_bagging import predict_forest_bagging_components

    has_mlp = bool(mlp_members)
    has_lr = bool(logreg_members)
    has_rf = bool(forest_members)
    if not (has_mlp or has_lr or has_rf):
        raise ValueError(
            "predict_hybrid_bagging: sem MLPs, LogRegs nem Florestas."
        )

    parts: list[np.ndarray] = []
    if has_mlp:
        parts.append(
            predict_architecture_bagging_components(
                list(mlp_members), X, device=device,
                apply_calibration=apply_calibration,
            )
        )
    if has_lr:
        parts.append(
            predict_logreg_bagging_components(
                list(logreg_members), X,
                apply_calibration=apply_calibration,
            )
        )
    if has_rf:
        parts.append(
            predict_forest_bagging_components(
                list(forest_members), X,
                apply_calibration=apply_calibration,
            )
        )
    comp = np.concatenate(parts, axis=1)
    return comp.mean(axis=1).astype(np.float32)


def predict_hybrid_bagging_components(
    mlp_members: Sequence[Any] | None,
    logreg_members: Sequence[Any] | None,
    X: np.ndarray,
    *,
    forest_members: Sequence[Any] | None = None,
    device: str | None = None,
    apply_calibration: bool = True,
) -> tuple[np.ndarray, list[dict[str, str]]]:
    """Retorna (matriz_(N, K_total), info_membros).

    `info_membros` traz `{"key": ..., "kind": "mlp"|"logreg"|"rf"|"extratrees"}`
    na mesma ordem das colunas da matriz (MLPs primeiro, depois LogRegs,
    depois Florestas).
    """
    from .arch_bagging import predict_architecture_bagging_components
    from .logreg_bagging import predict_logreg_bagging_components
    from .forest_bagging import predict_forest_bagging_components

    has_mlp = bool(mlp_members)
    has_lr = bool(logreg_members)
    has_rf = bool(forest_members)
    if not (has_mlp or has_lr or has_rf):
        raise ValueError(
            "predict_hybrid_bagging_components: sem MLPs, LogRegs nem Florestas."
        )

    parts: list[np.ndarray] = []
    info: list[dict[str, str]] = []
    if has_mlp:
        parts.append(
            predict_architecture_bagging_components(
                list(mlp_members), X, device=device,
                apply_calibration=apply_calibration,
            )
        )
        for m in mlp_members:
            info.append({"key": str(m.key), "kind": "mlp"})
    if has_lr:
        parts.append(
            predict_logreg_bagging_components(
                list(logreg_members), X,
                apply_calibration=apply_calibration,
            )
        )
        for m in logreg_members:
            info.append({"key": str(m.key), "kind": "logreg"})
    if has_rf:
        parts.append(
            predict_forest_bagging_components(
                list(forest_members), X,
                apply_calibration=apply_calibration,
            )
        )
        for m in forest_members:
            fam = str(getattr(m, "family", "rf") or "rf")
            kind = "extratrees" if fam == "extratrees" else "rf"
            info.append({"key": str(m.key), "kind": kind})

    comp = np.concatenate(parts, axis=1)
    return comp, info


def permutation_importance_hybrid(
    mlp_members: Sequence[Any] | None,
    logreg_members: Sequence[Any] | None,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    *,
    forest_members: Sequence[Any] | None = None,
    metric: str = "pr_auc",
    n_repeats: int = 3,
    seed: int = 42,
    device: str | None = None,
) -> list[dict[str, float]]:
    """Permutation importance sobre o score agregado do bagging hibrido."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    if device is None:
        device = default_device()
    rng = np.random.default_rng(seed)

    base_scores = predict_hybrid_bagging(
        mlp_members, logreg_members, X,
        forest_members=forest_members,
        device=device, apply_calibration=True,
    )
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
            scores = predict_hybrid_bagging(
                mlp_members, logreg_members, X_perm,
                forest_members=forest_members,
                device=device, apply_calibration=True,
            )
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


def _logreg_ig_contributions(
    member: Any,
    x: np.ndarray,
    feature_names: Sequence[str],
    baseline: np.ndarray | None = None,
) -> dict[str, float]:
    """Contribuicao Integrated-Gradients exata para um LogReg (modelo linear).

    Para um modelo linear y = sigma(w.x + b), IG entre baseline x0 e x para
    a feature i e' exatamente coef_i * (x_i - x0_i) (em escala de logit).
    Como o IG das MLPs do bagging tambem e' calculado em escala de logit
    (saida pre-sigmoid), a media com elas e' bem definida.
    """
    coef = np.asarray(member.coef_, dtype=np.float32).reshape(-1)
    x_arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if baseline is None:
        base_arr = np.zeros_like(x_arr)
    else:
        base_arr = np.asarray(baseline, dtype=np.float32).reshape(-1)
    if coef.shape[0] != x_arr.shape[0]:
        # robustez: nao quebra se houver discrepancia (so retorna zero)
        logger.warning(
            "LogReg coef shape %s != x shape %s; IG analitico zerado.",
            coef.shape, x_arr.shape,
        )
        return {str(n): 0.0 for n in feature_names}
    contrib = (coef * (x_arr - base_arr)).astype(np.float64)
    return {str(n): float(contrib[i]) for i, n in enumerate(feature_names)}


def _forest_treeshap_contributions(
    member: Any,
    x: np.ndarray,
    feature_names: Sequence[str],
) -> dict[str, float]:
    """Contribuicao por feature via TreeSHAP para um membro Floresta.

    Retorna em escala de probabilidade (saida de `predict_proba` da classe 1).
    A normalizacao por `sum(|c|)` e' aplicada em
    `integrated_gradients_hybrid_row` antes de promediar com as demais
    fontes (que vivem em escala de logit).
    """
    try:
        import shap
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "shap nao disponivel (%s); contribuicoes da Floresta zeradas.",
            exc,
        )
        return {str(n): 0.0 for n in feature_names}

    x_arr = np.asarray(x, dtype=np.float32).reshape(1, -1)
    try:
        expl = shap.TreeExplainer(member.model)
        sv = expl.shap_values(x_arr)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "TreeSHAP falhou para %s (%s); contribuicoes zeradas.",
            getattr(member, "key", "?"), exc,
        )
        return {str(n): 0.0 for n in feature_names}

    arr = np.asarray(sv)
    # Variantes de retorno do shap.TreeExplainer.shap_values:
    # - lista [classe0, classe1] -> usar classe 1
    # - ndarray (1, n_feats) -> uso direto
    # - ndarray (1, n_feats, n_classes) -> selecionar classe 1
    if isinstance(sv, list) and len(sv) >= 2:
        arr = np.asarray(sv[1])
    if arr.ndim == 3:
        arr = arr[..., -1]
    arr = arr.reshape(-1)

    if arr.shape[0] != len(feature_names):
        logger.warning(
            "TreeSHAP shape %s != n_features %d para %s; zerando.",
            arr.shape, len(feature_names), getattr(member, "key", "?"),
        )
        return {str(n): 0.0 for n in feature_names}

    return {str(n): float(arr[i]) for i, n in enumerate(feature_names)}


def _normalize_contrib_dict(
    contrib: dict[str, float],
    feature_names: Sequence[str],
) -> dict[str, float]:
    """Normaliza um vetor de contribuicoes por sum(|c|) (preserva sinal).

    Mantem escala comparavel entre famílias (logit MLP/LogReg vs probabilidade
    TreeSHAP). Usa fallback de 1e-12 para evitar divisao por zero.
    """
    vals = np.asarray(
        [float(contrib.get(str(fn), 0.0)) for fn in feature_names],
        dtype=np.float64,
    )
    s = float(np.sum(np.abs(vals)))
    if s < 1e-12:
        return {str(fn): 0.0 for fn in feature_names}
    norm = vals / s
    return {str(fn): float(norm[i]) for i, fn in enumerate(feature_names)}


def integrated_gradients_hybrid_row(
    mlp_members: Sequence[Any] | None,
    logreg_members: Sequence[Any] | None,
    x: np.ndarray,
    feature_names: Sequence[str],
    *,
    forest_members: Sequence[Any] | None = None,
    baseline: np.ndarray | None = None,
    n_steps: int = 50,
    device: str | None = None,
) -> list[dict[str, float]]:
    """IG agregado: media das contribuicoes de TODOS os modelos.

    - MLPs: `integrated_gradients_for_row` por modelo (Captum, escala logit).
    - LogRegs: contribuicao linear analitica `coef * (x - baseline)` (logit).
    - Florestas: TreeSHAP via `shap.TreeExplainer` (escala probabilidade).

    Cada vetor de contribuicoes e' normalizado por `sum(|c|)` antes da media,
    para que escalas heterogeneas (logit vs probabilidade) sejam comparaveis;
    o sinal e o ranking sao preservados.
    """
    has_mlp = bool(mlp_members)
    has_lr = bool(logreg_members)
    has_rf = bool(forest_members)
    if not (has_mlp or has_lr or has_rf):
        raise ValueError(
            "integrated_gradients_hybrid_row: sem MLPs, LogRegs nem Florestas."
        )

    if device is None:
        device = default_device()

    acc: defaultdict[str, float] = defaultdict(float)
    vals = {str(n): float(np.asarray(x).reshape(-1)[i])
            for i, n in enumerate(feature_names)}

    if has_mlp:
        for m in mlp_members:
            mod = getattr(m, "model", m)
            rows = integrated_gradients_for_row(
                mod, x, feature_names,
                baseline=baseline, n_steps=n_steps, device=device,
            )
            raw = {str(r["feature"]): float(r["contribution"]) for r in rows}
            norm = _normalize_contrib_dict(raw, feature_names)
            for k, v in norm.items():
                acc[k] += float(v)

    if has_lr:
        for m in logreg_members:
            raw = _logreg_ig_contributions(
                m, x, feature_names, baseline=baseline,
            )
            norm = _normalize_contrib_dict(raw, feature_names)
            for k, v in norm.items():
                acc[k] += float(v)

    if has_rf:
        for m in forest_members:
            raw = _forest_treeshap_contributions(m, x, feature_names)
            norm = _normalize_contrib_dict(raw, feature_names)
            for k, v in norm.items():
                acc[k] += float(v)

    n_total = (
        (len(mlp_members) if has_mlp else 0)
        + (len(logreg_members) if has_lr else 0)
        + (len(forest_members) if has_rf else 0)
    )
    n_total = max(1, n_total)
    out = [
        {"feature": str(fn),
         "value": vals[str(fn)],
         "contribution": acc[str(fn)] / n_total}
        for fn in feature_names
    ]
    out.sort(key=lambda r: abs(r["contribution"]), reverse=True)
    return out


__all__ = [
    "permutation_importance",
    "permutation_importance_arch_bagging",
    "permutation_importance_hybrid",
    "permutation_importance_stacking_context",
    "integrated_gradients_for_row",
    "integrated_gradients_arch_bagging_row",
    "integrated_gradients_hybrid_row",
    "predict_hybrid_bagging",
    "predict_hybrid_bagging_components",
]
