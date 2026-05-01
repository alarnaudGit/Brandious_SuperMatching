"""Bagging de N MLPs com **arquiteturas aleatorias**.

Cada membro do bagging:
- e' uma `BrandSimilarityMLP` cuja arquitetura (numero/tamanho de camadas,
  dropout, ativacao, batchnorm) e' amostrada por um `MLPRandomRanges`;
- usa um seed proprio (`base_seed + i*1000`) para tornar o ensemble real;
- recebe overrides de otimizacao tambem aleatorios (lr, weight_decay,
  batch_size) dentro de ranges sensatos para o problema;
- e' treinado com `train_model` no mesmo split (train/val) e calibrado por
  Platt scaling no conjunto de validacao.

Na inferencia o score final e a **media aritmetica** dos K scores
calibrados (ver `predict_random_mlp_bagging`).

O pacote no disco continua usando `kind = "architecture_bagging"` no
`index.json` (compatibilidade com `inference.py` / `load_architecture_bagging`).
As chaves dos membros agora sao `mlp_01..mlp_NN`.
"""
from __future__ import annotations

import copy
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import joblib
import numpy as np
import torch
from torch import nn

from .calibration import PlattCalibrator
from .dataset import BalancingConfig
from .evaluate import default_device, predict_scores
from .mlp import MLPConfig
from .train import TrainConfig, train_model

logger = logging.getLogger(__name__)


# =============================================================================
# Ranges aleatorios (configuraveis via UI)
# =============================================================================

@dataclass
class MLPRandomRanges:
    """Distribuicoes para sortear cada hiperparametro de uma MLP."""

    n_layers_choices: Sequence[int] = (2, 3, 4)
    layer_size_choices: Sequence[int] = (32, 64, 96, 128, 192, 256, 384, 512)
    monotonic_decreasing: bool = True
    dropout_min: float = 0.10
    dropout_max: float = 0.50
    activation_choices: Sequence[str] = ("relu", "gelu", "leakyrelu")
    batchnorm_prob: float = 0.7

    lr_log_min: float = 1e-4
    lr_log_max: float = 3e-3
    wd_zero_prob: float = 0.30
    wd_log_min: float = 1e-5
    wd_log_max: float = 1e-3
    batch_size_choices: Sequence[int] = (128, 256, 512)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_layers_choices": list(self.n_layers_choices),
            "layer_size_choices": list(self.layer_size_choices),
            "monotonic_decreasing": bool(self.monotonic_decreasing),
            "dropout_min": float(self.dropout_min),
            "dropout_max": float(self.dropout_max),
            "activation_choices": list(self.activation_choices),
            "batchnorm_prob": float(self.batchnorm_prob),
            "lr_log_min": float(self.lr_log_min),
            "lr_log_max": float(self.lr_log_max),
            "wd_zero_prob": float(self.wd_zero_prob),
            "wd_log_min": float(self.wd_log_min),
            "wd_log_max": float(self.wd_log_max),
            "batch_size_choices": list(self.batch_size_choices),
        }


@dataclass
class SampledMLPSpec:
    """Resultado de um sorteio de hiperparametros para uma MLP do bagging."""

    mlp_config: MLPConfig
    lr: float
    weight_decay: float
    batch_size: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "mlp_config": self.mlp_config.to_dict(),
            "lr": float(self.lr),
            "weight_decay": float(self.weight_decay),
            "batch_size": int(self.batch_size),
            "seed": int(self.seed),
        }


def _sample_log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    if lo <= 0 or hi <= 0 or hi < lo:
        return float(lo)
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    return float(math.exp(rng.uniform(log_lo, log_hi)))


def sample_random_mlp(
    input_dim: int,
    ranges: MLPRandomRanges,
    *,
    seed: int,
) -> SampledMLPSpec:
    """Sorteia uma arquitetura + hiperparametros de otimizacao.

    Camadas sao sorteadas dentro de `layer_size_choices`. Se
    `monotonic_decreasing=True`, garantimos que cada camada subsequente nao
    tem MAIS neuronios que a anterior (pratica comum em MLPs tabulares).
    """
    rng = np.random.default_rng(seed)

    n_layers = int(rng.choice(list(ranges.n_layers_choices)))
    sizes_pool = sorted(set(int(s) for s in ranges.layer_size_choices), reverse=True)
    if not sizes_pool:
        sizes_pool = [128, 64, 32]

    if ranges.monotonic_decreasing:
        first = int(rng.choice(sizes_pool))
        hidden: list[int] = [first]
        for _ in range(1, n_layers):
            allowed = [s for s in sizes_pool if s <= hidden[-1]]
            if not allowed:
                allowed = [hidden[-1]]
            hidden.append(int(rng.choice(allowed)))
    else:
        hidden = [int(rng.choice(sizes_pool)) for _ in range(n_layers)]

    dropout = float(rng.uniform(ranges.dropout_min, ranges.dropout_max))
    activation = str(rng.choice(list(ranges.activation_choices)))
    use_bn = bool(rng.random() < ranges.batchnorm_prob)

    lr = _sample_log_uniform(rng, ranges.lr_log_min, ranges.lr_log_max)
    if rng.random() < ranges.wd_zero_prob:
        weight_decay = 0.0
    else:
        weight_decay = _sample_log_uniform(rng, ranges.wd_log_min, ranges.wd_log_max)
    batch_size = int(rng.choice(list(ranges.batch_size_choices)))

    mlp_cfg = MLPConfig(
        input_dim=int(input_dim),
        hidden_dims=hidden,
        dropout=float(round(dropout, 4)),
        use_batchnorm=use_bn,
        activation=activation,
    )
    return SampledMLPSpec(
        mlp_config=mlp_cfg,
        lr=float(lr),
        weight_decay=float(weight_decay),
        batch_size=int(batch_size),
        seed=int(seed),
    )


# =============================================================================
# Membro do bagging
# =============================================================================

@dataclass
class ArchBaggingMember:
    """Um membro do bagging (no novo esquema, sempre uma MLP aleatoria)."""

    key: str
    model: nn.Module
    calibrator: PlattCalibrator
    arch_dict: dict[str, Any]
    best_epoch: int = 0
    best_pr_auc_val: float = 0.0
    history: list[dict[str, float]] = field(default_factory=list)
    sampled_spec: dict[str, Any] | None = None


def _arch_dict_from_model(model: nn.Module) -> dict[str, Any]:
    if hasattr(model, "cfg") and hasattr(model.cfg, "to_dict"):
        return model.cfg.to_dict()
    return model.config.to_dict()


# =============================================================================
# Treino do bagging
# =============================================================================

def _member_key(i: int, total: int) -> str:
    width = max(2, len(str(total)))
    return f"mlp_{i:0{width}d}"


def train_random_mlp_bagging(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    train_config: TrainConfig,
    balancing: BalancingConfig,
    feature_names: list[str],
    *,
    n_models: int,
    ranges: MLPRandomRanges,
    base_seed: int,
    calibrate: bool = True,
    on_variant_start: Callable[[str, int, int], None] | None = None,
    on_epoch_end: Callable[[dict[str, Any]], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> list[ArchBaggingMember]:
    """Treina N MLPs aleatorias e devolve a lista de membros calibrados."""
    n_models = max(3, min(100, int(n_models)))
    input_dim = int(X_train.shape[1])
    out: list[ArchBaggingMember] = []

    for i in range(1, n_models + 1):
        if should_stop and should_stop():
            logger.info(
                "Bagging interrompido apos %d/%d MLPs.", i - 1, n_models,
            )
            break

        key = _member_key(i, n_models)
        member_seed = int(base_seed) + i * 1000
        spec = sample_random_mlp(input_dim, ranges, seed=member_seed)

        if on_variant_start:
            try:
                on_variant_start(key, i, n_models)
            except Exception as exc:  # noqa: BLE001
                logger.warning("on_variant_start: %s", exc)

        logger.info(
            "Bagging MLP %s (%d/%d): hidden=%s dropout=%.2f act=%s bn=%s "
            "lr=%.4g wd=%.4g batch=%d seed=%d",
            key, i, n_models, spec.mlp_config.hidden_dims, spec.mlp_config.dropout,
            spec.mlp_config.activation, spec.mlp_config.use_batchnorm,
            spec.lr, spec.weight_decay, spec.batch_size, spec.seed,
        )

        tcfg = copy.deepcopy(train_config)
        tcfg.architecture = "mlp"
        tcfg.seed = int(spec.seed)
        tcfg.lr = float(spec.lr)
        tcfg.weight_decay = float(spec.weight_decay)
        tcfg.batch_size = int(spec.batch_size)

        model, result = train_model(
            X_train, y_train, X_val, y_val,
            spec.mlp_config, tcfg, balancing,
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
                sampled_spec=spec.to_dict(),
            )
        )
    return out


# Backwards-compatible alias (a UI antiga usava esse nome; mantemos para
# nao quebrar callers que ainda chamam `train_architecture_bagging`).
def train_architecture_bagging(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    train_config: TrainConfig,
    balancing: BalancingConfig,
    feature_names: list[str],
    *,
    n_models: int = 3,
    ranges: MLPRandomRanges | None = None,
    base_seed: int | None = None,
    calibrate: bool = True,
    on_variant_start: Callable[[str, int, int], None] | None = None,
    on_epoch_end: Callable[[dict[str, Any]], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> list[ArchBaggingMember]:
    """Compat: encaminha para `train_random_mlp_bagging` com defaults."""
    return train_random_mlp_bagging(
        X_train, y_train, X_val, y_val,
        train_config, balancing, feature_names,
        n_models=n_models,
        ranges=ranges or MLPRandomRanges(),
        base_seed=int(base_seed if base_seed is not None else train_config.seed),
        calibrate=calibrate,
        on_variant_start=on_variant_start,
        on_epoch_end=on_epoch_end,
        should_stop=should_stop,
    )


# =============================================================================
# Inferencia
# =============================================================================

def predict_architecture_bagging_components(
    members: Sequence[ArchBaggingMember],
    X: np.ndarray,
    device: str | None = None,
    apply_calibration: bool = True,
) -> np.ndarray:
    """Matriz (N, K) com score de cada modelo (K = len(members))."""
    if not members:
        raise ValueError("predict_architecture_bagging_components: lista vazia.")
    if device is None:
        device = default_device()
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
    members: Sequence[ArchBaggingMember],
    X: np.ndarray,
    device: str | None = None,
    apply_calibration: bool = True,
) -> np.ndarray:
    """Score final por linha = media dos K scores calibrados."""
    comp = predict_architecture_bagging_components(
        members, X, device=device, apply_calibration=apply_calibration,
    )
    return comp.mean(axis=1).astype(np.float32)


# Aliases convenientes / mais semanticos para o novo esquema
predict_random_mlp_bagging = predict_architecture_bagging
predict_random_mlp_bagging_components = predict_architecture_bagging_components


# =============================================================================
# Persistencia
# =============================================================================

def save_architecture_bagging(
    members: Sequence[ArchBaggingMember],
    dir_path: str | Path,
) -> None:
    """Persiste todos os membros + index.json com kind=architecture_bagging.

    Mantemos o `kind` legado para nao quebrar `inference.py` (carregadores
    existentes). O `key` agora sera `mlp_01..mlp_NN`.
    """
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
        index_members.append(
            {
                "key": m.key,
                "model_file": wpath.name,
                "calibrator_file": cal_path.name,
                "architecture": m.arch_dict,
                "best_epoch": m.best_epoch,
                "best_pr_auc_val": m.best_pr_auc_val,
                "sampled_spec": m.sampled_spec,
            }
        )
    payload = {
        "kind": "architecture_bagging",
        "schema": "random_mlp_bagging_v1",
        "members": index_members,
    }
    (dir_path / "index.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    joblib.dump(payload, dir_path / "index.pkl")
    logger.info(
        "Random-MLP bagging salvo em %s (%d modelos)", dir_path, len(members),
    )


def load_architecture_bagging(
    dir_path: str | Path,
    map_location: str = "cpu",
) -> list[ArchBaggingMember]:
    """Carrega o bagging a partir de index.json.

    Funciona com runs antigos (4 arquiteturas) E com o novo esquema de
    N MLPs aleatorias, porque ambos usam `kind=architecture_bagging`.
    """
    from ..pipeline.inference import _build_model_from_arch_json

    dir_path = Path(dir_path)
    idx_path = dir_path / "index.json"
    if not idx_path.exists():
        raise FileNotFoundError(f"index.json nao encontrado em {dir_path}")
    payload = json.loads(idx_path.read_text(encoding="utf-8"))
    if payload.get("kind") != "architecture_bagging":
        raise ValueError(
            f"index.json nao e' architecture_bagging: {payload.get('kind')}"
        )
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
        cal = (
            PlattCalibrator.load(cal_path) if cal_path.exists() else PlattCalibrator()
        )
        out.append(
            ArchBaggingMember(
                key=str(meta["key"]),
                model=model,
                calibrator=cal,
                arch_dict=dict(arch),
                best_epoch=int(meta.get("best_epoch", 0)),
                best_pr_auc_val=float(meta.get("best_pr_auc_val", 0.0)),
                sampled_spec=meta.get("sampled_spec"),
            )
        )
    logger.info(
        "Bagging carregado: %d modelos de %s (schema=%s)",
        len(out),
        dir_path,
        payload.get("schema", "legacy_4arch"),
    )
    return out


__all__ = [
    "ArchBaggingMember",
    "MLPRandomRanges",
    "SampledMLPSpec",
    "sample_random_mlp",
    "train_random_mlp_bagging",
    "train_architecture_bagging",
    "predict_architecture_bagging",
    "predict_architecture_bagging_components",
    "predict_random_mlp_bagging",
    "predict_random_mlp_bagging_components",
    "save_architecture_bagging",
    "load_architecture_bagging",
]
