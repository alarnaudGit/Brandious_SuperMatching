"""Dataset PyTorch + balanceamento (under/over/class_weight) + split estratificado."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)


@dataclass
class BalancingConfig:
    """Estrategia combinada de balanceamento.

    A ordem aplicada e: undersample_neg -> oversample_pos -> class_weight (loss).
    Sao todos opcionais e configuraveis via Streamlit.

    `training_balance`:
        - ``equal`` (padrao): apos undersample dos negativos, replica positivos com
          reposicao ate ``len(neg) == len(pos)`` no conjunto de treino (50/50).
        - ``legacy``: mantem ``oversample_pos_factor`` sobre os positivos originais.
    """

    undersample_neg_ratio: float = 3.0
    oversample_pos_factor: float = 2.0
    training_balance: str = "equal"  # "equal" | "legacy"
    use_class_weight: bool = True
    pos_weight_override: float | None = None
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return {
            "undersample_neg_ratio": self.undersample_neg_ratio,
            "oversample_pos_factor": self.oversample_pos_factor,
            "training_balance": self.training_balance,
            "use_class_weight": self.use_class_weight,
            "pos_weight_override": self.pos_weight_override,
            "seed": self.seed,
        }


@dataclass
class SplitConfig:
    test_size: float = 0.15
    val_size: float = 0.15
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return {"test_size": self.test_size, "val_size": self.val_size, "seed": self.seed}


class BrandPairDataset(Dataset):
    """Dataset simples sobre matriz de features pre-computada."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X.astype(np.float32, copy=False))
        self.y = torch.from_numpy(y.astype(np.float32, copy=False)).view(-1, 1)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def stratified_split(
    X: np.ndarray, y: np.ndarray, cfg: SplitConfig
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    """Split 70/15/15 estratificado por label."""
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.seed, stratify=y
    )
    val_rel = cfg.val_size / (1.0 - cfg.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_rel, random_state=cfg.seed, stratify=y_tv
    )
    logger.info(
        "Split -> train=%d (%.1f%% pos), val=%d (%.1f%% pos), test=%d (%.1f%% pos)",
        len(y_train), 100 * y_train.mean(),
        len(y_val), 100 * y_val.mean(),
        len(y_test), 100 * y_test.mean(),
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def apply_undersampling_oversampling(
    X: np.ndarray, y: np.ndarray, cfg: BalancingConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Aplica undersample dos negativos e oversample dos positivos no TREINO."""
    rng = np.random.default_rng(cfg.seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    if n_pos == 0:
        logger.warning("Sem positivos no treino; balanceamento ignorado.")
        return X, y

    target_neg = int(min(n_neg, max(n_pos * cfg.undersample_neg_ratio, n_pos)))
    if target_neg < n_neg:
        chosen_neg = rng.choice(neg_idx, size=target_neg, replace=False)
    else:
        chosen_neg = neg_idx

    balance = getattr(cfg, "training_balance", "equal") or "equal"
    if balance == "equal":
        n_pos_target = int(len(chosen_neg))
        chosen_pos = rng.choice(pos_idx, size=n_pos_target, replace=True)
        logger.info(
            "Balanceamento 50/50: pos %d originais -> %d (reposicao), neg %d->%d",
            n_pos, len(chosen_pos), n_neg, len(chosen_neg),
        )
    elif cfg.oversample_pos_factor and cfg.oversample_pos_factor > 1.0:
        n_pos_target = int(round(n_pos * cfg.oversample_pos_factor))
        chosen_pos = rng.choice(pos_idx, size=n_pos_target, replace=True)
        logger.info(
            "Balanceamento legado: pos %d->%d, neg %d->%d (ratio_neg/pos=%.2f, oversample_x=%.2f)",
            n_pos, len(chosen_pos), n_neg, len(chosen_neg),
            cfg.undersample_neg_ratio, cfg.oversample_pos_factor,
        )
    else:
        chosen_pos = pos_idx
        logger.info(
            "Balanceamento (sem oversample extra): pos %d, neg %d->%d",
            n_pos, n_neg, len(chosen_neg),
        )

    final_idx = np.concatenate([chosen_pos, chosen_neg])
    rng.shuffle(final_idx)
    Xb, yb = X[final_idx], y[final_idx]
    logger.info(
        "Treino balanceado final: %d linhas (%.1f%% positivos)",
        len(yb), 100.0 * float(yb.mean()),
    )
    return Xb, yb


def compute_pos_weight(y: np.ndarray, cfg: BalancingConfig) -> float:
    """Calcula pos_weight (n_neg/n_pos) ja considerando override do usuario."""
    if cfg.pos_weight_override is not None:
        return float(cfg.pos_weight_override)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        return 1.0
    return float(n_neg / n_pos)


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader:
    ds = BrandPairDataset(X, y)
    if sampler is not None:
        return DataLoader(ds, batch_size=batch_size, sampler=sampler, drop_last=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


__all__ = [
    "BalancingConfig",
    "SplitConfig",
    "BrandPairDataset",
    "stratified_split",
    "apply_undersampling_oversampling",
    "compute_pos_weight",
    "make_loader",
]
