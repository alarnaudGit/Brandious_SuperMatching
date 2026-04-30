"""Loop de treino: AdamW + ReduceLROnPlateau + early stopping no PR-AUC.

Aceita callback por epoca (`on_epoch_end`) que recebe um dict com metricas, util
para Streamlit atualizar graficos em tempo real.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .dataset import (
    BalancingConfig,
    apply_undersampling_oversampling,
    compute_pos_weight,
    make_loader,
)
from .evaluate import compute_metrics_at_threshold, predict_scores
from .mlp import BrandSimilarityMLP, MLPConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10
    scheduler_patience: int = 4
    scheduler_factor: float = 0.5
    min_lr: float = 1e-6
    grad_clip: float = 1.0
    device: str = "auto"
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "early_stopping_patience": self.early_stopping_patience,
            "scheduler_patience": self.scheduler_patience,
            "scheduler_factor": self.scheduler_factor,
            "min_lr": self.min_lr,
            "grad_clip": self.grad_clip,
            "device": self.device,
            "seed": self.seed,
        }


@dataclass
class TrainResult:
    best_state_dict: dict[str, torch.Tensor]
    best_epoch: int
    best_pr_auc_val: float
    history: list[dict[str, float]] = field(default_factory=list)
    pos_weight_used: float = 1.0
    n_train_after_balancing: int = 0


def _resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    mlp_config: MLPConfig,
    train_config: TrainConfig,
    balancing: BalancingConfig,
    on_epoch_end: Callable[[dict[str, Any]], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[BrandSimilarityMLP, TrainResult]:
    """Treina a MLP. Devolve (modelo_com_melhores_pesos, TrainResult)."""
    device = _resolve_device(train_config.device)
    torch.manual_seed(train_config.seed)
    np.random.seed(train_config.seed)

    Xb, yb = apply_undersampling_oversampling(X_train, y_train, balancing)

    pos_weight = compute_pos_weight(yb, balancing) if balancing.use_class_weight else 1.0
    logger.info("pos_weight efetivo: %.3f", pos_weight)

    model = BrandSimilarityMLP(mlp_config).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=train_config.scheduler_patience,
        factor=train_config.scheduler_factor,
        min_lr=train_config.min_lr,
    )

    train_loader = make_loader(Xb, yb, batch_size=train_config.batch_size, shuffle=True)

    best_pr_auc = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    bad_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, train_config.epochs + 1):
        if should_stop and should_stop():
            logger.info("Treino interrompido pelo usuario na epoca %d.", epoch)
            break

        model.train()
        running_loss = 0.0
        n_seen = 0
        for xb, yb_b in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb_b = yb_b.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb_b)
            loss.backward()
            if train_config.grad_clip and train_config.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            optimizer.step()
            running_loss += float(loss.item()) * xb.size(0)
            n_seen += xb.size(0)
        train_loss = running_loss / max(n_seen, 1)

        train_scores = predict_scores(model, Xb, device=device)
        val_scores = predict_scores(model, X_val, device=device)

        train_metrics = compute_metrics_at_threshold(yb, train_scores, threshold=0.5)
        val_metrics = compute_metrics_at_threshold(y_val, val_scores, threshold=0.5)

        scheduler.step(val_metrics.pr_auc)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_info = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_pr_auc": train_metrics.pr_auc,
            "train_recall@0.5": train_metrics.recall,
            "train_f1@0.5": train_metrics.f1,
            "val_pr_auc": val_metrics.pr_auc,
            "val_roc_auc": val_metrics.roc_auc,
            "val_recall@0.5": val_metrics.recall,
            "val_f1@0.5": val_metrics.f1,
            "lr": float(current_lr),
        }
        history.append(epoch_info)
        logger.info(
            "Epoca %d/%d | loss=%.4f | val_PRAUC=%.4f val_ROC=%.4f val_F1=%.3f val_Rec=%.3f | lr=%.2e",
            epoch, train_config.epochs,
            train_loss, val_metrics.pr_auc, val_metrics.roc_auc, val_metrics.f1, val_metrics.recall, current_lr,
        )

        if on_epoch_end is not None:
            try:
                on_epoch_end(epoch_info)
            except Exception as exc:
                logger.warning("on_epoch_end raised: %s", exc)

        if val_metrics.pr_auc > best_pr_auc + 1e-6:
            best_pr_auc = val_metrics.pr_auc
            best_epoch = epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= train_config.early_stopping_patience:
                logger.info("Early stopping em epoca %d (paciencia %d).", epoch, train_config.early_stopping_patience)
                break

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        best_epoch = max(1, len(history))
        best_pr_auc = float(history[-1]["val_pr_auc"]) if history else 0.0

    model.load_state_dict(best_state)
    model.eval()

    result = TrainResult(
        best_state_dict=best_state,
        best_epoch=best_epoch,
        best_pr_auc_val=float(best_pr_auc),
        history=history,
        pos_weight_used=float(pos_weight),
        n_train_after_balancing=int(len(yb)),
    )
    return model, result


__all__ = ["TrainConfig", "TrainResult", "train_model"]
