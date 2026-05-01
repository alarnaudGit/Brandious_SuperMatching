"""Bagging de N regress\u00f5es log\u00edsticas com hiperpar\u00e2metros aleatorios.

Cada membro:
- e' um `sklearn.linear_model.LogisticRegression` cujo C, penalty e
  class_weight sao sorteados por um `LogRegRandomRanges` (espelho do
  `MLPRandomRanges` em `arch_bagging.py`);
- usa um seed proprio (`base_seed + i*1000 + 1`) para garantir diversidade;
- pode opcionalmente fazer bootstrap (amostragem com reposicao do conjunto
  ja' balanceado) para diversificar mais o ensemble;
- e' calibrado por Platt scaling no conjunto de validacao (igual MLPs).

Na inferencia o score por modelo e' `predict_proba(X)[:,1]` (eventualmente
calibrado). O score final do bagging e' a media dos K scores; ver helpers
`predict_logreg_bagging` / `predict_logreg_bagging_components` ou a versao
hibrida em `src/model/explain.py` que combina MLPs + LogRegs.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from .calibration import PlattCalibrator
from .dataset import BalancingConfig, apply_undersampling_oversampling

logger = logging.getLogger(__name__)


# =============================================================================
# Ranges aleatorios
# =============================================================================

@dataclass
class LogRegRandomRanges:
    """Distribuicoes para sortear hiperparametros de cada LogReg do bagging."""

    c_log_min: float = 1e-2
    c_log_max: float = 10.0
    penalty_choices: Sequence[str] = ("l2", "l1")
    class_weight_choices: Sequence[str | None] = (None, "balanced")
    solver: str = "liblinear"
    max_iter: int = 2000
    bootstrap_train: bool = True
    bootstrap_fraction: float = 1.0  # tamanho da amostra bootstrap relativo ao set balanceado

    def to_dict(self) -> dict[str, Any]:
        return {
            "c_log_min": float(self.c_log_min),
            "c_log_max": float(self.c_log_max),
            "penalty_choices": list(self.penalty_choices),
            "class_weight_choices": [
                cw if cw is not None else "none"
                for cw in self.class_weight_choices
            ],
            "solver": str(self.solver),
            "max_iter": int(self.max_iter),
            "bootstrap_train": bool(self.bootstrap_train),
            "bootstrap_fraction": float(self.bootstrap_fraction),
        }


@dataclass
class SampledLogRegSpec:
    """Resultado do sorteio de hiperparametros de uma LogReg do bagging."""

    C: float
    penalty: str
    class_weight: str | None
    solver: str
    max_iter: int
    seed: int
    bootstrap: bool
    bootstrap_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "C": float(self.C),
            "penalty": str(self.penalty),
            "class_weight": (
                str(self.class_weight) if self.class_weight is not None else None
            ),
            "solver": str(self.solver),
            "max_iter": int(self.max_iter),
            "seed": int(self.seed),
            "bootstrap": bool(self.bootstrap),
            "bootstrap_fraction": float(self.bootstrap_fraction),
        }


def _sample_log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    if lo <= 0 or hi <= 0 or hi < lo:
        return float(lo)
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    return float(math.exp(rng.uniform(log_lo, log_hi)))


def sample_random_logreg(
    ranges: LogRegRandomRanges,
    *,
    seed: int,
) -> SampledLogRegSpec:
    """Sorteia C, penalty e class_weight para uma LogReg do bagging."""
    rng = np.random.default_rng(seed)

    C = _sample_log_uniform(rng, ranges.c_log_min, ranges.c_log_max)

    penalty_pool = list(ranges.penalty_choices) or ["l2"]
    penalty = str(rng.choice(penalty_pool))

    cw_pool: list[str | None] = []
    for cw in ranges.class_weight_choices:
        if cw is None or (isinstance(cw, str) and cw.lower() == "none"):
            cw_pool.append(None)
        else:
            cw_pool.append(str(cw))
    if not cw_pool:
        cw_pool = [None]
    cw_idx = int(rng.integers(0, len(cw_pool)))
    class_weight = cw_pool[cw_idx]

    return SampledLogRegSpec(
        C=float(C),
        penalty=str(penalty),
        class_weight=class_weight,
        solver=str(ranges.solver),
        max_iter=int(ranges.max_iter),
        seed=int(seed),
        bootstrap=bool(ranges.bootstrap_train),
        bootstrap_fraction=float(ranges.bootstrap_fraction),
    )


# =============================================================================
# Membro do bagging
# =============================================================================

@dataclass
class LogRegBaggingMember:
    """Um membro LogReg do bagging hibrido."""

    key: str
    model: LogisticRegression
    calibrator: PlattCalibrator
    coef_: np.ndarray
    intercept_: float
    sampled_spec: dict[str, Any] | None = None
    feature_names: list[str] = field(default_factory=list)
    train_metric_val: float = 0.0


# =============================================================================
# Treino do bagging
# =============================================================================

def _member_key(i: int, total: int) -> str:
    width = max(2, len(str(total)))
    return f"logreg_{i:0{width}d}"


def _scores_proba(model: LogisticRegression, X: np.ndarray) -> np.ndarray:
    """Devolve P(y=1|X) usando predict_proba (sempre disponivel para LogReg)."""
    p = model.predict_proba(X.astype(np.float32, copy=False))[:, 1]
    return p.astype(np.float32, copy=False)


def train_random_logreg_bagging(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    balancing: BalancingConfig,
    feature_names: list[str],
    *,
    n_models: int,
    ranges: LogRegRandomRanges,
    base_seed: int,
    calibrate: bool = True,
    on_variant_start: Callable[[str, int, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> list[LogRegBaggingMember]:
    """Treina N LogRegs aleatorias e devolve a lista de membros calibrados."""
    n_models = max(1, min(100, int(n_models)))
    out: list[LogRegBaggingMember] = []

    feature_names_list = list(feature_names) if feature_names else []

    for i in range(1, n_models + 1):
        if should_stop and should_stop():
            logger.info(
                "Bagging LogReg interrompido apos %d/%d modelos.",
                i - 1, n_models,
            )
            break

        key = _member_key(i, n_models)
        member_seed = int(base_seed) + i * 1000 + 1
        spec = sample_random_logreg(ranges, seed=member_seed)

        if on_variant_start:
            try:
                on_variant_start(key, i, n_models)
            except Exception as exc:  # noqa: BLE001
                logger.warning("on_variant_start (logreg): %s", exc)

        # Balanceamento (mesma rotina das MLPs); seed proprio para diversidade
        bal_local = BalancingConfig(
            undersample_neg_ratio=balancing.undersample_neg_ratio,
            oversample_pos_factor=balancing.oversample_pos_factor,
            training_balance=balancing.training_balance,
            use_class_weight=balancing.use_class_weight,
            pos_weight_override=balancing.pos_weight_override,
            seed=int(member_seed),
        )
        Xb, yb = apply_undersampling_oversampling(X_train, y_train, bal_local)

        if spec.bootstrap:
            rng = np.random.default_rng(member_seed)
            n_boot = int(round(spec.bootstrap_fraction * len(yb)))
            if n_boot < 10:
                n_boot = len(yb)
            idx = rng.integers(0, len(yb), size=n_boot)
            Xb = Xb[idx]
            yb = yb[idx]

        # Garante que as duas classes existem na amostra (LogReg.fit exige >=2 classes).
        unique = np.unique(yb)
        if len(unique) < 2:
            logger.warning(
                "Membro %s: amostra com 1 classe apos balanceamento; "
                "pulando este modelo.", key,
            )
            continue

        logger.info(
            "Bagging LogReg %s (%d/%d): C=%.4g penalty=%s cw=%s "
            "solver=%s max_iter=%d bootstrap=%s seed=%d",
            key, i, n_models, spec.C, spec.penalty, spec.class_weight,
            spec.solver, spec.max_iter, spec.bootstrap, spec.seed,
        )

        model = LogisticRegression(
            C=float(spec.C),
            penalty=str(spec.penalty),
            class_weight=spec.class_weight,
            solver=str(spec.solver),
            max_iter=int(spec.max_iter),
            random_state=int(spec.seed),
            n_jobs=None,
        )
        try:
            model.fit(Xb.astype(np.float32, copy=False), yb.astype(np.int32, copy=False))
        except Exception as exc:
            logger.warning(
                "Falha ao treinar LogReg %s (%s); pulando este membro.",
                key, exc,
            )
            continue

        val_raw = _scores_proba(model, X_val)
        if calibrate:
            cal = PlattCalibrator().fit(val_raw, y_val)
        else:
            cal = PlattCalibrator()

        try:
            val_for_metric = cal.transform(val_raw) if cal.fitted else val_raw
            from sklearn.metrics import average_precision_score
            metric_val = float(average_precision_score(y_val, val_for_metric))
        except Exception:  # noqa: BLE001
            metric_val = 0.0

        out.append(
            LogRegBaggingMember(
                key=key,
                model=model,
                calibrator=cal,
                coef_=np.asarray(model.coef_, dtype=np.float32).reshape(-1),
                intercept_=float(np.asarray(model.intercept_).reshape(-1)[0]),
                sampled_spec=spec.to_dict(),
                feature_names=list(feature_names_list),
                train_metric_val=metric_val,
            )
        )
    return out


# =============================================================================
# Inferencia
# =============================================================================

def predict_logreg_bagging_components(
    members: Sequence[LogRegBaggingMember],
    X: np.ndarray,
    apply_calibration: bool = True,
) -> np.ndarray:
    """Matriz (N, K) com score de cada LogReg (K = len(members))."""
    if not members:
        raise ValueError("predict_logreg_bagging_components: lista vazia.")
    n = X.shape[0]
    k = len(members)
    mat = np.zeros((n, k), dtype=np.float32)
    for j, m in enumerate(members):
        s = _scores_proba(m.model, X)
        if apply_calibration and m.calibrator.fitted:
            s = m.calibrator.transform(s)
        mat[:, j] = s.astype(np.float32, copy=False)
    return mat


def predict_logreg_bagging(
    members: Sequence[LogRegBaggingMember],
    X: np.ndarray,
    apply_calibration: bool = True,
) -> np.ndarray:
    """Score final por linha = media dos K scores calibrados."""
    comp = predict_logreg_bagging_components(
        members, X, apply_calibration=apply_calibration,
    )
    return comp.mean(axis=1).astype(np.float32)


# =============================================================================
# Persistencia
# =============================================================================

def save_logreg_bagging(
    members: Sequence[LogRegBaggingMember],
    dir_path: str | Path,
) -> None:
    """Persiste todos os LogRegs + index.json com kind=logreg_bagging."""
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    index_members: list[dict[str, Any]] = []
    for m in members:
        safe = m.key.replace("/", "_")
        wpath = dir_path / f"{safe}.joblib"
        cal_path = dir_path / f"{safe}_cal.json"
        joblib.dump(m.model, wpath)
        m.calibrator.save(cal_path)
        index_members.append(
            {
                "key": m.key,
                "model_file": wpath.name,
                "calibrator_file": cal_path.name,
                "sampled_spec": m.sampled_spec,
                "train_metric_val": float(m.train_metric_val),
            }
        )
    payload = {
        "kind": "logreg_bagging",
        "schema": "random_logreg_bagging_v1",
        "members": index_members,
    }
    (dir_path / "index.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    joblib.dump(payload, dir_path / "index.pkl")
    logger.info(
        "LogReg bagging salvo em %s (%d modelos)",
        dir_path, len(members),
    )


def load_logreg_bagging(
    dir_path: str | Path,
) -> list[LogRegBaggingMember]:
    """Carrega o bagging de LogRegs a partir de index.json."""
    dir_path = Path(dir_path)
    idx_path = dir_path / "index.json"
    if not idx_path.exists():
        raise FileNotFoundError(f"index.json nao encontrado em {dir_path}")
    payload = json.loads(idx_path.read_text(encoding="utf-8"))
    if payload.get("kind") != "logreg_bagging":
        raise ValueError(
            f"index.json nao e' logreg_bagging: {payload.get('kind')}"
        )
    out: list[LogRegBaggingMember] = []
    for meta in payload["members"]:
        wpath = dir_path / meta["model_file"]
        model: LogisticRegression = joblib.load(wpath)
        cal_path = dir_path / meta["calibrator_file"]
        cal = (
            PlattCalibrator.load(cal_path)
            if cal_path.exists() else PlattCalibrator()
        )
        coef = np.asarray(model.coef_, dtype=np.float32).reshape(-1)
        intercept = float(np.asarray(model.intercept_).reshape(-1)[0])
        out.append(
            LogRegBaggingMember(
                key=str(meta["key"]),
                model=model,
                calibrator=cal,
                coef_=coef,
                intercept_=intercept,
                sampled_spec=meta.get("sampled_spec"),
                train_metric_val=float(meta.get("train_metric_val", 0.0)),
            )
        )
    logger.info(
        "LogReg bagging carregado: %d modelos de %s (schema=%s)",
        len(out), dir_path, payload.get("schema", "unknown"),
    )
    return out


__all__ = [
    "LogRegRandomRanges",
    "SampledLogRegSpec",
    "LogRegBaggingMember",
    "sample_random_logreg",
    "train_random_logreg_bagging",
    "predict_logreg_bagging",
    "predict_logreg_bagging_components",
    "save_logreg_bagging",
    "load_logreg_bagging",
]
