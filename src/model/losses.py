"""Funcoes de perda para Sprint 2 - foco em minority class.

- `FocalLoss`: down-weight automatico de exemplos faceis (gamma=2).
- `LabelSmoothingBCE`: BCE com label smoothing (~0.05) para reduzir
  overconfidence em positivos faceis.
- `FocalLossWithLabelSmoothing`: combina os dois.
- `MultiTaskLoss`: combina BCE+Focal na head principal e MSE/BCE nas
  auxiliares (para o `MultiTaskMLP`).

Todas trabalham com **logits** (mais estavel numericamente que com sigmoid
explicita). Use `nn.BCEWithLogitsLoss` para o caso simples.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


# =============================================================================
# Focal Loss
# =============================================================================

class FocalLoss(nn.Module):
    """Focal loss binaria estavel sobre logits.

    L = -alpha * (1 - p_t)^gamma * log(p_t)

    onde:
        p_t = p (se y=1) ou (1-p) (se y=0)
        alpha aplicado SO no termo positivo, (1-alpha) no termo negativo
        (forma classica do paper original Lin et al. 2017).

    `pos_weight` pode multiplicar o termo positivo adicionalmente
    (combina com alpha). Util quando a base e muito desbalanceada e
    alpha=0.25 nao chega.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        pos_weight: float | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor,
    ) -> torch.Tensor:
        targets = targets.float().view_as(logits)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none",
        )
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        modulating = (1.0 - p_t).pow(self.gamma)
        alpha_factor = self.alpha * targets + (1.0 - self.alpha) * (1 - targets)
        loss = alpha_factor * modulating * bce
        if self.pos_weight is not None:
            loss = loss * (1.0 + (self.pos_weight - 1.0) * targets)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# =============================================================================
# Label Smoothing BCE
# =============================================================================

class LabelSmoothingBCE(nn.Module):
    """BCE com label smoothing (suavizacao do alvo).

    Substitui y=1 por (1-eps) e y=0 por eps. Aceita pos_weight.
    """

    def __init__(
        self,
        smoothing: float = 0.05,
        pos_weight: float | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.smoothing = float(smoothing)
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor,
    ) -> torch.Tensor:
        targets = targets.float().view_as(logits)
        eps = self.smoothing
        smooth_targets = targets * (1.0 - eps) + eps * (1.0 - targets)
        if self.pos_weight is not None:
            pw = torch.tensor(
                [self.pos_weight], device=logits.device, dtype=logits.dtype,
            )
            loss = F.binary_cross_entropy_with_logits(
                logits, smooth_targets, pos_weight=pw, reduction=self.reduction,
            )
        else:
            loss = F.binary_cross_entropy_with_logits(
                logits, smooth_targets, reduction=self.reduction,
            )
        return loss


# =============================================================================
# Focal + Label Smoothing combinados
# =============================================================================

class FocalLossWithLabelSmoothing(nn.Module):
    """Focal aplicado em alvos suavizados (eps=0.05 default)."""

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        smoothing: float = 0.05,
        pos_weight: float | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor,
    ) -> torch.Tensor:
        targets = targets.float().view_as(logits)
        eps = self.smoothing
        smooth_targets = targets * (1.0 - eps) + eps * (1.0 - targets)
        bce = F.binary_cross_entropy_with_logits(
            logits, smooth_targets, reduction="none",
        )
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        modulating = (1.0 - p_t).pow(self.gamma)
        alpha_factor = self.alpha * targets + (1.0 - self.alpha) * (1 - targets)
        loss = alpha_factor * modulating * bce
        if self.pos_weight is not None:
            loss = loss * (1.0 + (self.pos_weight - 1.0) * targets)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# =============================================================================
# Multi-Task Loss
# =============================================================================

@dataclass
class MultiTaskWeights:
    """Pesos das 4 perdas (main, spec, cls, namesim)."""
    main: float = 1.0
    aux_spec: float = 0.3
    aux_cls: float = 0.2
    aux_namesim: float = 0.2


class MultiTaskLoss(nn.Module):
    """Loss combinada para `MultiTaskMLP`.

    A entrada e o dict retornado por `MultiTaskMLP.forward_all_heads(x)` e
    um dict de targets `{main, aux_spec, aux_cls, aux_namesim}`. Aux que
    nao for fornecido nao contribui para a loss.
    """

    def __init__(
        self,
        weights: MultiTaskWeights | None = None,
        main_loss: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.weights = weights or MultiTaskWeights()
        self.main_loss = main_loss or nn.BCEWithLogitsLoss()
        self.bce_aux = nn.BCEWithLogitsLoss()
        self.mse_aux = nn.MSELoss()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        components: dict[str, float] = {}
        total = self.main_loss(
            outputs["main"], targets["main"].float().view_as(outputs["main"]),
        )
        components["main"] = float(total.item())

        if "aux_spec" in targets and self.weights.aux_spec > 0:
            spec_pred = torch.sigmoid(outputs["aux_spec"])
            tgt = targets["aux_spec"].float().view_as(spec_pred)
            l_spec = self.mse_aux(spec_pred, tgt)
            total = total + self.weights.aux_spec * l_spec
            components["aux_spec"] = float(l_spec.item())

        if "aux_cls" in targets and self.weights.aux_cls > 0:
            l_cls = self.bce_aux(
                outputs["aux_cls"],
                targets["aux_cls"].float().view_as(outputs["aux_cls"]),
            )
            total = total + self.weights.aux_cls * l_cls
            components["aux_cls"] = float(l_cls.item())

        if "aux_namesim" in targets and self.weights.aux_namesim > 0:
            l_ns = self.bce_aux(
                outputs["aux_namesim"],
                targets["aux_namesim"].float().view_as(outputs["aux_namesim"]),
            )
            total = total + self.weights.aux_namesim * l_ns
            components["aux_namesim"] = float(l_ns.item())

        return total, components


# =============================================================================
# Factory
# =============================================================================

def build_loss(
    name: str,
    *,
    pos_weight: float | None = None,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.05,
) -> nn.Module:
    """Constroi a loss a partir do nome."""
    n = name.lower()
    if n in ("bce", "bce_with_logits", "default"):
        if pos_weight is not None:
            return nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([float(pos_weight)]),
            )
        return nn.BCEWithLogitsLoss()
    if n in ("focal",):
        return FocalLoss(
            alpha=focal_alpha, gamma=focal_gamma, pos_weight=pos_weight,
        )
    if n in ("label_smoothing", "label_smoothing_bce", "ls_bce"):
        return LabelSmoothingBCE(
            smoothing=label_smoothing, pos_weight=pos_weight,
        )
    if n in ("focal_smoothing", "focal_label_smoothing", "focal_ls"):
        return FocalLossWithLabelSmoothing(
            alpha=focal_alpha, gamma=focal_gamma,
            smoothing=label_smoothing, pos_weight=pos_weight,
        )
    raise ValueError(f"Loss desconhecida: {name}")


__all__ = [
    "FocalLoss",
    "LabelSmoothingBCE",
    "FocalLossWithLabelSmoothing",
    "MultiTaskLoss",
    "MultiTaskWeights",
    "build_loss",
]
