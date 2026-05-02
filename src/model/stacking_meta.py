"""Stacking: meta-MLP sobre [features escaladas | preds calibradas dos membros base].

Treino da meta-MLP apenas no conjunto de validacao (ver plano do projeto).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from .evaluate import default_device
from .mlp import BrandSimilarityMLP, MLPConfig

logger = logging.getLogger(__name__)


@dataclass
class MetaStackingTrainConfig:
    hidden_dims: list[int]
    dropout: float = 0.25
    epochs: int = 80
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 12
    batch_size: int = 512
    seed: int = 42


def stacking_prediction_columns_meta(
    mlp_members: Sequence[Any] | None,
    logreg_members: Sequence[Any] | None,
    forest_members: Sequence[Any] | None,
) -> list[dict[str, str]]:
    """Ordem alinhada a `predict_hybrid_bagging_components`: MLP, LogReg, RF."""
    info: list[dict[str, str]] = []
    for m in mlp_members or []:
        info.append({"kind": "mlp", "key": str(m.key)})
    for m in logreg_members or []:
        info.append({"kind": "logreg", "key": str(m.key)})
    for m in forest_members or []:
        fam = str(getattr(m, "family", "rf") or "rf")
        kind = "extratrees" if fam == "extratrees" else "rf"
        info.append({"kind": kind, "key": str(m.key)})
    return info


def build_meta_design_matrix(
    X_scaled: np.ndarray,
    mlp_members: Sequence[Any] | None,
    logreg_members: Sequence[Any] | None,
    forest_members: Sequence[Any] | None,
    *,
    device: str | None = None,
    apply_calibration: bool = True,
) -> tuple[np.ndarray, list[dict[str, str]]]:
    """Concatena X_scaled (N, F) com matriz (N, K) de probabilidades calibradas."""
    from .arch_bagging import predict_architecture_bagging_components
    from .forest_bagging import predict_forest_bagging_components
    from .logreg_bagging import predict_logreg_bagging_components

    if device is None:
        device = default_device()

    parts: list[np.ndarray] = []
    if mlp_members:
        parts.append(
            predict_architecture_bagging_components(
                list(mlp_members), X_scaled, device=device,
                apply_calibration=apply_calibration,
            )
        )
    if logreg_members:
        parts.append(
            predict_logreg_bagging_components(
                list(logreg_members), X_scaled,
                apply_calibration=apply_calibration,
            )
        )
    if forest_members:
        parts.append(
            predict_forest_bagging_components(
                list(forest_members), X_scaled,
                apply_calibration=apply_calibration,
            )
        )
    if not parts:
        raise ValueError("build_meta_design_matrix: nenhum membro base disponivel.")

    P = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
    Xf = np.asarray(X_scaled, dtype=np.float32)
    Z = np.hstack([Xf, P]).astype(np.float32, copy=False)
    meta = stacking_prediction_columns_meta(
        mlp_members, logreg_members, forest_members,
    )
    if Z.shape[1] != Xf.shape[1] + len(meta):
        raise RuntimeError("Dimensao Z inconsistente com meta colunas.")
    return Z, meta


def predict_meta_stacking_scores(
    model: nn.Module,
    Z: np.ndarray,
    *,
    device: str | None = None,
    batch_size: int = 4096,
) -> np.ndarray:
    """Probabilidade final via sigmoid(logit) da meta-MLP."""
    if device is None:
        device = default_device()
    model.eval()
    model.to(device)
    n = Z.shape[0]
    out = np.zeros(n, dtype=np.float32)
    Z = np.asarray(Z, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            zb = torch.tensor(Z[start:end], dtype=torch.float32, device=device)
            logits = model(zb)
            out[start:end] = torch.sigmoid(logits).reshape(-1).cpu().numpy()
    return out


def train_meta_stacking_mlp(
    Z_val: np.ndarray,
    y_val: np.ndarray,
    *,
    config: MetaStackingTrainConfig,
    device: str | None = None,
) -> tuple[BrandSimilarityMLP, dict[str, float]]:
    """Treina BrandSimilarityMLP binaria em Z_val; early stopping por PR-AUC."""
    if device is None:
        device = default_device()

    rng = np.random.default_rng(int(config.seed))
    torch.manual_seed(int(config.seed))

    n = len(y_val)
    if n < 10:
        logger.warning("Meta-stacking: poucas linhas no val (%d).", n)

    Z = np.asarray(Z_val, dtype=np.float32)
    y = np.asarray(y_val, dtype=np.float32).reshape(-1)
    input_dim = int(Z.shape[1])

    model = BrandSimilarityMLP(
        MLPConfig(
            input_dim=input_dim,
            hidden_dims=list(config.hidden_dims),
            dropout=float(config.dropout),
            use_batchnorm=True,
            activation="relu",
        )
    ).to(device)

    pos = float(y.sum())
    neg = float(n - pos)
    pos_weight = torch.tensor(
        [neg / max(pos, 1.0)], dtype=torch.float32, device=device,
    )
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = AdamW(
        model.parameters(),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )

    ds = TensorDataset(
        torch.tensor(Z, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    loader = DataLoader(
        ds,
        batch_size=min(int(config.batch_size), max(1, n)),
        shuffle=True,
        drop_last=False,
    )

    best_state: dict[str, Any] | None = None
    best_pr = -1.0
    patience_left = int(config.early_stopping_patience)

    for epoch in range(1, int(config.epochs) + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device).view(-1, 1)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            full_logits = model(torch.tensor(Z, dtype=torch.float32, device=device))
            pr_full = torch.sigmoid(full_logits).reshape(-1).cpu().numpy()
        try:
            pr_auc = float(average_precision_score(y_val, pr_full))
        except Exception:  # noqa: BLE001
            pr_auc = 0.0

        if pr_auc > best_pr + 1e-7:
            best_pr = pr_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = int(config.early_stopping_patience)
        else:
            patience_left -= 1

        logger.info(
            "Meta-stacking epoch %d/%d val PR-AUC=%.5f best=%.5f",
            epoch, config.epochs, pr_auc, best_pr,
        )
        if patience_left <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"best_pr_auc_val": float(best_pr)}


def permutation_importance_stacking_context(
    meta_model: Any,
    mlp_members: Sequence[Any] | None,
    logreg_members: Sequence[Any] | None,
    forest_members: Sequence[Any] | None,
    X: np.ndarray,
    y: np.ndarray,
    context_dim: int,
    feature_names: Sequence[str],
    metric: str = "pr_auc",
    n_repeats: int = 3,
    seed: int = 42,
    device: str | None = None,
) -> list[dict[str, float]]:
    """Permutation importance com stacking: embaralha apenas colunas de contexto (X).

    As colunas de predicoes dos modelos base permanecem coerentes com X permutado.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    if device is None:
        device = default_device()
    rng = np.random.default_rng(seed)
    cdim = int(context_dim)
    if cdim > X.shape[1]:
        cdim = X.shape[1]

    Z0, _ = build_meta_design_matrix(
        X, mlp_members, logreg_members, forest_members, device=device,
    )
    base_scores = predict_meta_stacking_scores(meta_model, Z0, device=device)

    if metric == "pr_auc":
        base = float(average_precision_score(y, base_scores))
    elif metric == "roc_auc":
        base = float(roc_auc_score(y, base_scores))
    else:
        raise ValueError(f"metric desconhecida: {metric}")

    importances: list[float] = []
    for j in range(cdim):
        deltas: list[float] = []
        for _ in range(n_repeats):
            X_perm = np.array(X, copy=True)
            rng.shuffle(X_perm[:, j])
            Zp, _ = build_meta_design_matrix(
                X_perm,
                mlp_members,
                logreg_members,
                forest_members,
                device=device,
            )
            scores = predict_meta_stacking_scores(meta_model, Zp, device=device)
            if metric == "pr_auc":
                m = float(average_precision_score(y, scores))
            else:
                m = float(roc_auc_score(y, scores))
            deltas.append(base - m)
        importances.append(float(np.mean(deltas)))

    rows = [
        {"feature": str(name), "importance": float(imp)}
        for name, imp in zip(feature_names[:cdim], importances)
    ]
    rows.sort(key=lambda r: r["importance"], reverse=True)
    return rows


def save_meta_stacking_bundle(
    model: nn.Module,
    column_meta: list[dict[str, str]],
    context_dim: int,
    dir_path: str | Path,
) -> None:
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    cfg = model.config if hasattr(model, "config") else model.cfg  # type: ignore[attr-defined]
    payload = {
        "schema": "meta_stacking_v1",
        "context_dim": int(context_dim),
        "n_pred_cols": len(column_meta),
        "pred_columns": list(column_meta),
        "mlp_config": cfg.to_dict(),
    }
    (dir_path / "meta_stacking.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    torch.save(model.state_dict(), dir_path / "meta_stacking.pt")
    logger.info("Meta-stacking salvo em %s", dir_path)


def load_meta_stacking_bundle(
    dir_path: str | Path,
    *,
    map_location: str | None = None,
) -> tuple[BrandSimilarityMLP, dict[str, Any]]:
    dir_path = Path(dir_path)
    js_path = dir_path / "meta_stacking.json"
    pt_path = dir_path / "meta_stacking.pt"
    if not js_path.exists() or not pt_path.exists():
        raise FileNotFoundError(f"meta_stacking ausente em {dir_path}")
    meta = json.loads(js_path.read_text(encoding="utf-8"))
    if meta.get("schema") != "meta_stacking_v1":
        logger.warning("schema meta_stacking inesperado: %s", meta.get("schema"))
    mcfg = MLPConfig.from_dict(meta["mlp_config"])
    model = BrandSimilarityMLP(mcfg)
    loc = map_location or "cpu"
    state = torch.load(pt_path, map_location=loc)
    model.load_state_dict(state)
    model.eval()
    return model, meta


__all__ = [
    "MetaStackingTrainConfig",
    "stacking_prediction_columns_meta",
    "build_meta_design_matrix",
    "predict_meta_stacking_scores",
    "permutation_importance_stacking_context",
    "train_meta_stacking_mlp",
    "save_meta_stacking_bundle",
    "load_meta_stacking_bundle",
]
