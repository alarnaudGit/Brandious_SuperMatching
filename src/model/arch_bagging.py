"""Bagging por arquitetura: treina MLP, Two-Tower, FT-Transformer e Multi-Task.

Todos recebem a mesma matriz de features; na inferencia o score final e' a
**media aritmetica** dos quatro scores (apos calibracao Platt por modelo, opcional).

O conjunto e tratado como **um unico modelo** para metricas e explicabilidade:
- `predict_architecture_bagging` -> score agregado
- `predict_architecture_bagging_components` -> matriz (N, 4) com cada score
"""
from __future__ import annotations

import copy
import json
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
from .train import TrainConfig, train_model

logger = logging.getLogger(__name__)

# Ordem fixa dos quatro modelos no bagging
ARCH_BAGGING_KEYS: tuple[str, ...] = ("mlp", "two_tower", "ft_transformer", "multitask")


@dataclass
class ArchBaggingMember:
    """Um dos quatro modelos do bagging por arquitetura."""
    key: str
    model: nn.Module
    calibrator: PlattCalibrator
    arch_dict: dict[str, Any]
    best_epoch: int = 0
    best_pr_auc_val: float = 0.0
    history: list[dict[str, float]] = field(default_factory=list)


def _arch_dict_from_model(model: nn.Module) -> dict[str, Any]:
    if hasattr(model, "cfg") and hasattr(model.cfg, "to_dict"):
        return model.cfg.to_dict()
    return model.config.to_dict()


def mlp_config_for_architecture(key: str, input_dim: int) -> MLPConfig:
    """Hiperparametros alinhados ao A/B (`train_ab_compare.VARIANTS`)."""
    k = key.lower()
    if k == "multitask":
        return MLPConfig(
            input_dim=input_dim,
            hidden_dims=[256, 128, 64],
            dropout=0.45,
            use_batchnorm=True,
            activation="relu",
        )
    if k == "two_tower":
        return MLPConfig(
            input_dim=input_dim,
            hidden_dims=[128, 64],
            dropout=0.30,
            use_batchnorm=True,
            activation="relu",
        )
    if k == "ft_transformer":
        return MLPConfig(
            input_dim=input_dim,
            hidden_dims=[64, 32],
            dropout=0.10,
            use_batchnorm=True,
            activation="relu",
        )
    return MLPConfig(
        input_dim=input_dim,
        hidden_dims=[128, 64, 32],
        dropout=0.30,
        use_batchnorm=True,
        activation="relu",
    )


def train_architecture_bagging(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    train_config: TrainConfig,
    balancing: BalancingConfig,
    feature_names: list[str],
    *,
    calibrate: bool = True,
    on_variant_start: Callable[[str, int, int], None] | None = None,
    on_epoch_end: Callable[[dict[str, Any]], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> list[ArchBaggingMember]:
    """Treina as 4 arquiteturas em sequencia; cada uma com calibrador proprio no val."""
    input_dim = X_train.shape[1]
    out: list[ArchBaggingMember] = []
    total = len(ARCH_BAGGING_KEYS)
    for i, key in enumerate(ARCH_BAGGING_KEYS, start=1):
        if should_stop and should_stop():
            logger.info("Bagging por arquitetura interrompido apos %d/%d.", i - 1, total)
            break
        if on_variant_start:
            try:
                on_variant_start(key, i, total)
            except Exception as exc:  # noqa: BLE001
                logger.warning("on_variant_start: %s", exc)
        logger.info(
            "Arch bagging: treinando variante %d/%d (%s)...", i, total, key,
        )
        mlp_cfg = mlp_config_for_architecture(key, input_dim)
        tcfg = copy.deepcopy(train_config)
        tcfg.architecture = key
        model, result = train_model(
            X_train, y_train, X_val, y_val,
            mlp_cfg, tcfg, balancing,
            feature_names=feature_names,
            on_epoch_end=on_epoch_end,
            should_stop=should_stop,
        )
        val_raw = predict_scores(model, X_val)
        if calibrate:
            cal = PlattCalibrator().fit(val_raw, y_val)
        else:
            cal = PlattCalibrator()
        out.append(
            ArchBaggingMember(
                key=key,
                model=model,
                calibrator=cal,
                arch_dict=_arch_dict_from_model(model),
                best_epoch=int(result.best_epoch),
                best_pr_auc_val=float(result.best_pr_auc_val),
                history=list(result.history),
            )
        )
    return out


def predict_architecture_bagging_components(
    members: list[ArchBaggingMember],
    X: np.ndarray,
    device: str = "cpu",
    apply_calibration: bool = True,
) -> np.ndarray:
    """Matriz (N, K) com score de cada modelo (K = len(members))."""
    if not members:
        raise ValueError("predict_architecture_bagging_components: lista vazia.")
    n = X.shape[0]
    k = len(members)
    mat = np.zeros((n, k), dtype=np.float32)
    for j, m in enumerate(members):
        s = predict_scores(m.model, X, device=device)
        if apply_calibration and m.calibrator.fitted:
            s = m.calibrator.transform(s)
        mat[:, j] = s.astype(np.float32, copy=False)
    return mat


def predict_architecture_bagging(
    members: list[ArchBaggingMember],
    X: np.ndarray,
    device: str = "cpu",
    apply_calibration: bool = True,
) -> np.ndarray:
    """Media dos scores (bagging) — score final unico por linha."""
    comp = predict_architecture_bagging_components(
        members, X, device=device, apply_calibration=apply_calibration,
    )
    return comp.mean(axis=1).astype(np.float32)


def save_architecture_bagging(members: list[ArchBaggingMember], dir_path: str | Path) -> None:
    """Persiste os 4 modelos + index.json com kind=architecture_bagging."""
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    index_members: list[dict[str, Any]] = []
    for m in members:
        safe = m.key.replace("/", "_")
        wpath = dir_path / f"{safe}.pt"
        cal_path = dir_path / f"{safe}_cal.json"
        torch.save(
            {"state_dict": {k: v.cpu() for k, v in m.model.state_dict().items()}},
            wpath,
        )
        m.calibrator.save(cal_path)
        index_members.append({
            "key": m.key,
            "model_file": wpath.name,
            "calibrator_file": cal_path.name,
            "architecture": m.arch_dict,
            "best_epoch": m.best_epoch,
            "best_pr_auc_val": m.best_pr_auc_val,
        })
    payload = {
        "kind": "architecture_bagging",
        "members": index_members,
    }
    (dir_path / "index.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    joblib.dump(payload, dir_path / "index.pkl")
    logger.info("Architecture bagging salvo em %s (%d modelos)", dir_path, len(members))


def load_architecture_bagging(
    dir_path: str | Path,
    map_location: str = "cpu",
) -> list[ArchBaggingMember]:
    """Carrega os 4 modelos a partir de index.json."""
    from ..pipeline.inference import _build_model_from_arch_json

    dir_path = Path(dir_path)
    idx_path = dir_path / "index.json"
    if not idx_path.exists():
        raise FileNotFoundError(f"index.json nao encontrado em {dir_path}")
    payload = json.loads(idx_path.read_text(encoding="utf-8"))
    if payload.get("kind") != "architecture_bagging":
        raise ValueError(f"index.json nao e' architecture_bagging: {payload.get('kind')}")
    out: list[ArchBaggingMember] = []
    for meta in payload["members"]:
        arch = meta["architecture"]
        model = _build_model_from_arch_json(arch)
        wpath = dir_path / meta["model_file"]
        state = torch.load(wpath, map_location=map_location)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        model.eval()
        cal_path = dir_path / meta["calibrator_file"]
        cal = PlattCalibrator.load(cal_path) if cal_path.exists() else PlattCalibrator()
        out.append(
            ArchBaggingMember(
                key=str(meta["key"]),
                model=model,
                calibrator=cal,
                arch_dict=dict(arch),
                best_epoch=int(meta.get("best_epoch", 0)),
                best_pr_auc_val=float(meta.get("best_pr_auc_val", 0.0)),
            )
        )
    logger.info("Architecture bagging carregado: %d modelos de %s", len(out), dir_path)
    return out


__all__ = [
    "ARCH_BAGGING_KEYS",
    "ArchBaggingMember",
    "mlp_config_for_architecture",
    "train_architecture_bagging",
    "predict_architecture_bagging",
    "predict_architecture_bagging_components",
    "save_architecture_bagging",
    "load_architecture_bagging",
]
