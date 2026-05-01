"""Ensemble por media simples de N modelos treinados em seeds diferentes.

Cada membro reusa a MESMA arquitetura, MESMA loss e MESMOS hiperparametros,
mudando apenas o `seed` em `TrainConfig` e `BalancingConfig`. Cada saida e
calibrada individualmente (Platt) sobre val_X/y antes da media final.

Uso tipico:

    members = train_ensemble(
        X_train, y_train, X_val, y_val,
        mlp_config, train_config, balancing,
        seeds=[42, 1337, 2025],
        feature_names=names,
    )
    test_scores = predict_ensemble(members, X_test)
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import torch
from torch import nn

from .calibration import PlattCalibrator
from .dataset import BalancingConfig
from .evaluate import predict_scores
from .mlp import MLPConfig
from .train import TrainConfig, TrainResult, train_model

logger = logging.getLogger(__name__)


@dataclass
class EnsembleMember:
    """Pacote modelo + seed + calibrador de cada membro do ensemble."""
    seed: int
    model: nn.Module
    calibrator: PlattCalibrator
    history: list[dict[str, float]] = field(default_factory=list)
    best_pr_auc_val: float = 0.0
    best_epoch: int = 0


def train_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    mlp_config: MLPConfig,
    train_config: TrainConfig,
    balancing: BalancingConfig,
    seeds: list[int] | None = None,
    feature_names: list[str] | None = None,
    on_epoch_end: Callable[[dict[str, Any]], None] | None = None,
    on_member_end: Callable[[int, "EnsembleMember"], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    calibrate: bool = True,
) -> list[EnsembleMember]:
    """Treina N membros (um por seed) e retorna lista pronta para inferencia."""
    seeds = seeds or [train_config.seed]
    out: list[EnsembleMember] = []
    for k, s in enumerate(seeds, start=1):
        logger.info("Ensemble: treinando membro %d/%d (seed=%d)...", k, len(seeds), s)
        tcfg = copy.deepcopy(train_config)
        tcfg.seed = int(s)
        bcfg = copy.deepcopy(balancing)
        bcfg.seed = int(s)

        model, result = train_model(
            X_train, y_train, X_val, y_val,
            mlp_config, tcfg, bcfg,
            on_epoch_end=on_epoch_end,
            should_stop=should_stop,
            feature_names=feature_names,
        )

        if calibrate:
            val_scores = predict_scores(
                model, X_val, device=tcfg.device if tcfg.device != "auto" else None,
            ) if tcfg.device != "auto" else predict_scores(model, X_val)
            calibrator = PlattCalibrator().fit(val_scores, y_val)
        else:
            calibrator = PlattCalibrator()

        member = EnsembleMember(
            seed=int(s),
            model=model,
            calibrator=calibrator,
            history=list(result.history),
            best_pr_auc_val=float(result.best_pr_auc_val),
            best_epoch=int(result.best_epoch),
        )
        out.append(member)
        if on_member_end is not None:
            try:
                on_member_end(k, member)
            except Exception as exc:
                logger.warning("on_member_end raised: %s", exc)
        if should_stop and should_stop():
            logger.info("Ensemble interrompido pelo usuario apos %d/%d membros.", k, len(seeds))
            break

    return out


def predict_ensemble(
    members: list[EnsembleMember],
    X: np.ndarray,
    device: str = "cpu",
    apply_calibration: bool = True,
) -> np.ndarray:
    """Media (aritmetica) das saidas calibradas de cada membro."""
    if not members:
        raise ValueError("predict_ensemble: lista vazia.")
    accum = np.zeros(X.shape[0], dtype=np.float64)
    for m in members:
        s = predict_scores(m.model, X, device=device)
        if apply_calibration and m.calibrator.fitted:
            s = m.calibrator.transform(s)
        accum += s.astype(np.float64)
    return (accum / len(members)).astype(np.float32)


def save_ensemble(members: list[EnsembleMember], dir_path: str | Path) -> None:
    """Persiste cada membro em `<dir>/member_<seed>.pt` + `<dir>/member_<seed>_cal.pkl`.

    Tambem grava `index.json` com a ordem e PR-AUC de cada membro para auditoria.
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    index = []
    for m in members:
        model_path = dir_path / f"member_{m.seed}.pt"
        cal_path = dir_path / f"member_{m.seed}_cal.json"
        torch.save(
            {
                "state_dict": {k: v.cpu() for k, v in m.model.state_dict().items()},
                "model_class": m.model.__class__.__name__,
            },
            model_path,
        )
        m.calibrator.save(cal_path)
        index.append({
            "seed": m.seed,
            "model_file": model_path.name,
            "calibrator_file": cal_path.name,
            "best_pr_auc_val": m.best_pr_auc_val,
            "best_epoch": m.best_epoch,
            "model_class": m.model.__class__.__name__,
        })
    joblib.dump(index, dir_path / "ensemble_index.pkl")
    (dir_path / "ensemble_index.json").write_text(
        __import__("json").dumps(index, indent=2), encoding="utf-8",
    )
    logger.info("Ensemble salvo em %s (%d membros)", dir_path, len(members))


__all__ = [
    "EnsembleMember",
    "train_ensemble",
    "predict_ensemble",
    "save_ensemble",
]
