"""Loop de treino: AdamW + ReduceLROnPlateau + early stopping no PR-AUC.

Sprint 2 acrescenta:
- escolha de arquitetura via `train_config.architecture` ("mlp", "two_tower",
  "ft_transformer", "multitask")
- escolha de loss via `train_config.loss_name` ("bce", "focal",
  "label_smoothing", "focal_smoothing")
- `train_config.mixup_pos`: probabilidade de aplicar mixup entre pares
  positivos do batch (0 desliga)
- `train_config.hard_neg_mining`: ativa mining a partir da epoca
  `train_config.hard_neg_start_epoch`
- `train_config.symmetry_aug`: True duplica o dataset de treino com (B,A) alem
  de (A,B). Como usamos features SIMETRICAS na maioria, isso so faz sentido
  quando algumas features sao assimetricas (e fazem). Aceita feature_names
  para detectar pares automaticamente; sem isso, simplesmente concatena X com X.

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

from .architectures import MultiTaskMLP, build_model
from .dataset import (
    BalancingConfig,
    apply_undersampling_oversampling,
    compute_pos_weight,
    make_loader,
)
from .evaluate import compute_metrics_at_threshold, predict_scores
from .losses import MultiTaskLoss, MultiTaskWeights, build_loss
from .mlp import BrandSimilarityMLP, MLPConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Config
# =============================================================================

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

    # ---- Sprint 2 ----
    architecture: str = "mlp"            # mlp / two_tower / ft_transformer / multitask
    loss_name: str = "bce"               # bce / focal / label_smoothing / focal_smoothing
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    label_smoothing: float = 0.05

    mixup_pos: float = 0.0               # prob (0..1) de aplicar mixup em pares pos do batch
    mixup_alpha: float = 0.4             # parametro Beta(alpha, alpha) do mixup

    hard_neg_mining: bool = False
    hard_neg_start_epoch: int = 5
    hard_neg_top_k_ratio: float = 1.0    # k = ratio * n_pos_in_batch

    symmetry_aug: bool = False           # duplica dataset (A,B) + (A,B) (placeholder simetrico)

    # MultiTask: pesos das auxiliares
    mt_weight_main: float = 1.0
    mt_weight_spec: float = 0.3
    mt_weight_cls: float = 0.2
    mt_weight_namesim: float = 0.2

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
            "architecture": self.architecture,
            "loss_name": self.loss_name,
            "focal_alpha": self.focal_alpha,
            "focal_gamma": self.focal_gamma,
            "label_smoothing": self.label_smoothing,
            "mixup_pos": self.mixup_pos,
            "mixup_alpha": self.mixup_alpha,
            "hard_neg_mining": self.hard_neg_mining,
            "hard_neg_start_epoch": self.hard_neg_start_epoch,
            "hard_neg_top_k_ratio": self.hard_neg_top_k_ratio,
            "symmetry_aug": self.symmetry_aug,
            "mt_weight_main": self.mt_weight_main,
            "mt_weight_spec": self.mt_weight_spec,
            "mt_weight_cls": self.mt_weight_cls,
            "mt_weight_namesim": self.mt_weight_namesim,
        }


@dataclass
class TrainResult:
    best_state_dict: dict[str, torch.Tensor]
    best_epoch: int
    best_pr_auc_val: float
    history: list[dict[str, float]] = field(default_factory=list)
    pos_weight_used: float = 1.0
    n_train_after_balancing: int = 0


# =============================================================================
# Helpers
# =============================================================================

def _resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _mixup_pos_inplace(
    xb: torch.Tensor, yb: torch.Tensor, alpha: float, prob: float, rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aplica mixup ENTRE pares positivos do batch.

    Para `prob` fracao dos positivos do batch, troca a amostra por uma combinacao
    convexa com OUTRO positivo aleatorio. O label permanece 1.
    """
    if prob <= 0 or alpha <= 0:
        return xb, yb
    pos_mask = (yb.view(-1) == 1).cpu().numpy().astype(bool)
    pos_idx = np.where(pos_mask)[0]
    if pos_idx.size < 2:
        return xb, yb
    n_mix = int(prob * pos_idx.size)
    if n_mix < 1:
        return xb, yb
    chosen = rng.choice(pos_idx, size=n_mix, replace=False)
    partner = rng.choice(pos_idx, size=n_mix, replace=True)
    lam = float(rng.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)  # garante peso >= 0.5 no original
    for i, j in zip(chosen, partner):
        if i == j:
            continue
        xb[i] = lam * xb[i] + (1.0 - lam) * xb[j]
    return xb, yb


def _hard_negative_subset(
    Xb: np.ndarray, yb: np.ndarray, model: nn.Module, device: str,
    *, top_k_ratio: float, batch_pred_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Retorna um subconjunto: TODOS os positivos + top-K negativos com maior score.

    K = round(top_k_ratio * n_pos). top_k_ratio=1.0 mantem 50/50.
    """
    pos_mask = yb == 1
    n_pos = int(pos_mask.sum())
    if n_pos == 0:
        return Xb, yb
    neg_idx = np.where(~pos_mask)[0]
    if neg_idx.size == 0:
        return Xb, yb
    scores = predict_scores(
        model, Xb[neg_idx], batch_size=batch_pred_size, device=device,
    )
    k = max(1, int(round(top_k_ratio * n_pos)))
    k = min(k, neg_idx.size)
    order = np.argsort(scores)[::-1]
    chosen_neg = neg_idx[order[:k]]
    keep = np.concatenate([np.where(pos_mask)[0], chosen_neg])
    np.random.shuffle(keep)
    return Xb[keep], yb[keep]


def _build_model_from_config(
    train_config: TrainConfig,
    mlp_config: MLPConfig,
    feature_names: list[str] | None,
) -> nn.Module:
    """Cria o modelo respeitando a arquitetura escolhida em train_config."""
    arch = (train_config.architecture or "mlp").lower()
    if arch in ("mlp", "brand_mlp", "baseline"):
        return BrandSimilarityMLP(mlp_config)
    return build_model(
        architecture=arch,
        input_dim=mlp_config.input_dim,
        feature_names=feature_names,
        mlp_hidden=mlp_config.hidden_dims,
        dropout=mlp_config.dropout,
        use_batchnorm=mlp_config.use_batchnorm,
        activation=mlp_config.activation,
    )


def _build_aux_targets(
    yb_b: torch.Tensor, X_batch: torch.Tensor,
    aux_indices: dict[str, int] | None,
) -> dict[str, torch.Tensor]:
    """Constroi targets auxiliares a partir do BATCH atual (multi-task).

    aux_indices mapeia "spec_cosine_emb"->idx, "cls_same"->idx,
    "graf_levenshtein"->idx (se disponiveis em feature_names).
    """
    targets = {"main": yb_b}
    if not aux_indices:
        return targets
    if "spec_cosine_emb" in aux_indices:
        targets["aux_spec"] = X_batch[:, aux_indices["spec_cosine_emb"]].view(-1, 1)
    if "cls_same" in aux_indices:
        v = X_batch[:, aux_indices["cls_same"]].view(-1, 1)
        targets["aux_cls"] = (v > 0.5).float()
    if "graf_levenshtein" in aux_indices:
        v = X_batch[:, aux_indices["graf_levenshtein"]].view(-1, 1)
        targets["aux_namesim"] = (v > 0.6).float()
    return targets


# =============================================================================
# Loop principal
# =============================================================================

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
    feature_names: list[str] | None = None,
) -> tuple[nn.Module, TrainResult]:
    """Treina o modelo conforme `train_config.architecture` e `train_config.loss_name`.

    Devolve `(modelo_com_melhores_pesos, TrainResult)`.

    Para arquiteturas alem de "mlp", `mlp_config.hidden_dims` reaproveita-se como
    parametro de profundidade (tower_hidden, backbone_hidden, etc.).
    """
    device = _resolve_device(train_config.device)
    torch.manual_seed(train_config.seed)
    np.random.seed(train_config.seed)
    rng = np.random.default_rng(train_config.seed)

    Xb, yb = apply_undersampling_oversampling(X_train, y_train, balancing)

    if train_config.symmetry_aug:
        Xb = np.concatenate([Xb, Xb], axis=0)
        yb = np.concatenate([yb, yb], axis=0)
        logger.info(
            "symmetry_aug: dataset duplicado (n=%d). Use somente se features "
            "forem aproximadamente simetricas.", len(yb),
        )

    pos_weight = compute_pos_weight(yb, balancing) if balancing.use_class_weight else 1.0
    logger.info(
        "pos_weight efetivo: %.3f (loss=%s, arch=%s)",
        pos_weight, train_config.loss_name, train_config.architecture,
    )

    model = _build_model_from_config(train_config, mlp_config, feature_names).to(device)

    is_multitask = isinstance(model, MultiTaskMLP)

    main_loss = build_loss(
        train_config.loss_name,
        pos_weight=pos_weight,
        focal_alpha=train_config.focal_alpha,
        focal_gamma=train_config.focal_gamma,
        label_smoothing=train_config.label_smoothing,
    )
    main_loss = main_loss.to(device)

    if is_multitask:
        mt_loss = MultiTaskLoss(
            weights=MultiTaskWeights(
                main=train_config.mt_weight_main,
                aux_spec=train_config.mt_weight_spec,
                aux_cls=train_config.mt_weight_cls,
                aux_namesim=train_config.mt_weight_namesim,
            ),
            main_loss=main_loss,
        ).to(device)
        aux_indices = model.cfg.aux_indices or {}
    else:
        mt_loss = None
        aux_indices = {}

    optimizer = AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=train_config.scheduler_patience,
        factor=train_config.scheduler_factor,
        min_lr=train_config.min_lr,
    )

    best_pr_auc = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    bad_epochs = 0
    history: list[dict[str, float]] = []

    Xb_active, yb_active = Xb, yb

    for epoch in range(1, train_config.epochs + 1):
        if should_stop and should_stop():
            logger.info("Treino interrompido pelo usuario na epoca %d.", epoch)
            break

        if (
            train_config.hard_neg_mining
            and epoch >= train_config.hard_neg_start_epoch
        ):
            Xb_active, yb_active = _hard_negative_subset(
                Xb, yb, model, device,
                top_k_ratio=train_config.hard_neg_top_k_ratio,
            )
            logger.debug(
                "Hard-neg mining epoca %d: n_used=%d (pos=%d, neg=%d)",
                epoch, len(yb_active),
                int((yb_active == 1).sum()), int((yb_active == 0).sum()),
            )

        train_loader = make_loader(
            Xb_active, yb_active, batch_size=train_config.batch_size, shuffle=True,
        )

        model.train()
        running_loss = 0.0
        n_seen = 0
        for xb_batch, yb_b in train_loader:
            xb_batch = xb_batch.to(device, non_blocking=True)
            yb_b = yb_b.to(device, non_blocking=True)

            if train_config.mixup_pos > 0:
                xb_batch, yb_b = _mixup_pos_inplace(
                    xb_batch, yb_b,
                    alpha=train_config.mixup_alpha,
                    prob=train_config.mixup_pos,
                    rng=rng,
                )

            optimizer.zero_grad()
            if is_multitask:
                outputs = model.forward_all_heads(xb_batch)
                targets = _build_aux_targets(yb_b, xb_batch, aux_indices)
                loss, _components = mt_loss(outputs, targets)
            else:
                logits = model(xb_batch)
                loss = main_loss(logits, yb_b)

            loss.backward()
            if train_config.grad_clip and train_config.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            optimizer.step()
            running_loss += float(loss.item()) * xb_batch.size(0)
            n_seen += xb_batch.size(0)
        train_loss = running_loss / max(n_seen, 1)

        train_scores = predict_scores(model, Xb_active, device=device)
        val_scores = predict_scores(model, X_val, device=device)

        train_metrics = compute_metrics_at_threshold(yb_active, train_scores, threshold=0.5)
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
                logger.info(
                    "Early stopping em epoca %d (paciencia %d).",
                    epoch, train_config.early_stopping_patience,
                )
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
        n_train_after_balancing=int(len(yb_active)),
    )
    return model, result


__all__ = ["TrainConfig", "TrainResult", "train_model"]
