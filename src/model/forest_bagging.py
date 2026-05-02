"""Bagging de N florestas (RandomForest + ExtraTrees) com hiperparametros aleatorios.

Cada membro:
- e' um `sklearn.ensemble.RandomForestClassifier` ou
  `sklearn.ensemble.ExtraTreesClassifier`, sorteado a partir de
  `family_choices` em `ForestRandomRanges`;
- usa um seed proprio (`base_seed + i*1000 + 1`) para garantir diversidade;
- pode opcionalmente fazer bootstrap externo do conjunto ja' balanceado
  (alem do bootstrap interno do sklearn) para diversificar mais o ensemble;
- e' calibrado por Platt scaling no conjunto de validacao (igual MLPs e
  LogRegs).

Na inferencia o score por modelo e' `predict_proba(X)[:,1]` (eventualmente
calibrado). O score final do bagging e' a media dos K scores; ver helpers
`predict_forest_bagging` / `predict_forest_bagging_components` ou a versao
hibrida em `src/model/explain.py` que combina MLPs + LogRegs + Florestas.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence, Union

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from .calibration import PlattCalibrator
from .dataset import BalancingConfig, apply_undersampling_oversampling

logger = logging.getLogger(__name__)


ForestModel = Union[RandomForestClassifier, ExtraTreesClassifier]


# =============================================================================
# Ranges aleatorios
# =============================================================================

@dataclass
class ForestRandomRanges:
    """Distribuicoes para sortear hiperparametros de cada Floresta do bagging."""

    family_choices: Sequence[str] = ("rf", "extratrees")
    n_estimators_min: int = 100
    n_estimators_max: int = 400
    # `None` (sem limite) e profundidades comuns para arvores
    max_depth_choices: Sequence[Any] = (None, 8, 16, 24)
    min_samples_leaf_min: int = 1
    min_samples_leaf_max: int = 8
    # `max_features` aceita string ("sqrt", "log2") ou float em (0, 1]
    max_features_choices: Sequence[Any] = ("sqrt", "log2", 0.5)
    class_weight_choices: Sequence[Any] = (None, "balanced", "balanced_subsample")
    # `max_samples_fraction` so afeta o RF (com bootstrap interno);
    # ExtraTrees ignora porque nao usa bootstrap interno.
    max_samples_fraction: float = 1.0
    # Bootstrap externo (sobre o conjunto ja' balanceado) para diversificar.
    bootstrap_train: bool = False
    bootstrap_fraction: float = 1.0
    n_jobs: int = -1

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_choices": list(self.family_choices),
            "n_estimators_min": int(self.n_estimators_min),
            "n_estimators_max": int(self.n_estimators_max),
            "max_depth_choices": [
                ("none" if d is None else int(d)) for d in self.max_depth_choices
            ],
            "min_samples_leaf_min": int(self.min_samples_leaf_min),
            "min_samples_leaf_max": int(self.min_samples_leaf_max),
            "max_features_choices": [
                (str(f) if isinstance(f, str) else float(f))
                for f in self.max_features_choices
            ],
            "class_weight_choices": [
                (cw if cw is not None else "none")
                for cw in self.class_weight_choices
            ],
            "max_samples_fraction": float(self.max_samples_fraction),
            "bootstrap_train": bool(self.bootstrap_train),
            "bootstrap_fraction": float(self.bootstrap_fraction),
            "n_jobs": int(self.n_jobs),
        }


@dataclass
class SampledForestSpec:
    """Resultado do sorteio de hiperparametros de uma Floresta do bagging."""

    family: str
    n_estimators: int
    max_depth: int | None
    min_samples_leaf: int
    max_features: Any
    class_weight: str | None
    max_samples: float | None
    seed: int
    bootstrap: bool
    bootstrap_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": str(self.family),
            "n_estimators": int(self.n_estimators),
            "max_depth": (None if self.max_depth is None else int(self.max_depth)),
            "min_samples_leaf": int(self.min_samples_leaf),
            "max_features": (
                str(self.max_features)
                if isinstance(self.max_features, str)
                else float(self.max_features)
            ),
            "class_weight": (
                str(self.class_weight) if self.class_weight is not None else None
            ),
            "max_samples": (
                None if self.max_samples is None else float(self.max_samples)
            ),
            "seed": int(self.seed),
            "bootstrap": bool(self.bootstrap),
            "bootstrap_fraction": float(self.bootstrap_fraction),
        }


def _normalize_max_depth_choice(d: Any) -> int | None:
    if d is None:
        return None
    if isinstance(d, str):
        if d.strip().lower() in {"none", "null", ""}:
            return None
        try:
            return int(d)
        except Exception:  # noqa: BLE001
            return None
    try:
        return int(d)
    except Exception:  # noqa: BLE001
        return None


def _normalize_class_weight_choice(cw: Any) -> str | None:
    if cw is None:
        return None
    if isinstance(cw, str):
        if cw.strip().lower() in {"none", "null", ""}:
            return None
        return cw
    return str(cw)


def _normalize_max_features_choice(f: Any) -> Any:
    if isinstance(f, str):
        return f
    try:
        v = float(f)
        if v <= 0:
            return "sqrt"
        return min(1.0, v)
    except Exception:  # noqa: BLE001
        return "sqrt"


def sample_random_forest(
    ranges: ForestRandomRanges,
    *,
    seed: int,
) -> SampledForestSpec:
    """Sorteia hiperparametros para uma Floresta do bagging."""
    rng = np.random.default_rng(seed)

    fam_pool = list(ranges.family_choices) or ["rf"]
    family = str(rng.choice(fam_pool))

    n_lo = max(10, int(ranges.n_estimators_min))
    n_hi = max(n_lo, int(ranges.n_estimators_max))
    n_estimators = int(rng.integers(n_lo, n_hi + 1))

    md_pool: list[int | None] = [
        _normalize_max_depth_choice(d) for d in ranges.max_depth_choices
    ]
    if not md_pool:
        md_pool = [None]
    max_depth = md_pool[int(rng.integers(0, len(md_pool)))]

    msl_lo = max(1, int(ranges.min_samples_leaf_min))
    msl_hi = max(msl_lo, int(ranges.min_samples_leaf_max))
    min_samples_leaf = int(rng.integers(msl_lo, msl_hi + 1))

    mf_pool = [
        _normalize_max_features_choice(f) for f in ranges.max_features_choices
    ] or ["sqrt"]
    max_features = mf_pool[int(rng.integers(0, len(mf_pool)))]

    cw_pool: list[str | None] = [
        _normalize_class_weight_choice(cw) for cw in ranges.class_weight_choices
    ]
    if not cw_pool:
        cw_pool = [None]
    class_weight = cw_pool[int(rng.integers(0, len(cw_pool)))]

    max_samples: float | None
    if family == "rf":
        msf = float(ranges.max_samples_fraction)
        if 0.0 < msf < 1.0:
            max_samples = msf
        else:
            max_samples = None
    else:
        # ExtraTrees nao usa bootstrap interno, max_samples nao se aplica.
        max_samples = None

    return SampledForestSpec(
        family=family,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        max_samples=max_samples,
        seed=int(seed),
        bootstrap=bool(ranges.bootstrap_train),
        bootstrap_fraction=float(ranges.bootstrap_fraction),
    )


# =============================================================================
# Membro do bagging
# =============================================================================

@dataclass
class ForestBaggingMember:
    """Um membro Floresta (RF/ExtraTrees) do bagging hibrido."""

    key: str
    family: str
    model: ForestModel
    calibrator: PlattCalibrator
    feature_importances_: np.ndarray
    sampled_spec: dict[str, Any] | None = None
    feature_names: list[str] = field(default_factory=list)
    train_metric_val: float = 0.0


# =============================================================================
# Treino do bagging
# =============================================================================

def _member_key(family: str, i: int, total: int) -> str:
    width = max(2, len(str(total)))
    prefix = "rf" if family == "rf" else "et"
    return f"{prefix}_{i:0{width}d}"


def _scores_proba(model: ForestModel, X: np.ndarray) -> np.ndarray:
    """Devolve P(y=1|X) usando predict_proba (sempre disponivel para floresta)."""
    p = model.predict_proba(X.astype(np.float32, copy=False))[:, 1]
    return p.astype(np.float32, copy=False)


def _build_forest_model(
    spec: SampledForestSpec,
    *,
    n_jobs: int,
) -> ForestModel:
    common = dict(
        n_estimators=int(spec.n_estimators),
        max_depth=spec.max_depth,
        min_samples_leaf=int(spec.min_samples_leaf),
        max_features=spec.max_features,
        class_weight=spec.class_weight,
        random_state=int(spec.seed),
        n_jobs=int(n_jobs),
    )
    if spec.family == "rf":
        return RandomForestClassifier(
            bootstrap=True,
            max_samples=spec.max_samples,
            **common,
        )
    return ExtraTreesClassifier(
        bootstrap=False,
        **common,
    )


def train_random_forest_bagging(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    balancing: BalancingConfig,
    feature_names: list[str],
    *,
    n_models: int,
    ranges: ForestRandomRanges,
    base_seed: int,
    calibrate: bool = True,
    on_variant_start: Callable[[str, str, int, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> list[ForestBaggingMember]:
    """Treina N Florestas aleatorias e devolve a lista de membros calibrados.

    `on_variant_start(key, family, i, n)`: callback para refletir progresso
    na UI (chama com a familia sorteada para a iteracao atual).
    """
    n_models = max(1, min(100, int(n_models)))
    out: list[ForestBaggingMember] = []

    feature_names_list = list(feature_names) if feature_names else []

    for i in range(1, n_models + 1):
        if should_stop and should_stop():
            logger.info(
                "Bagging Floresta interrompido apos %d/%d modelos.",
                i - 1, n_models,
            )
            break

        member_seed = int(base_seed) + i * 1000 + 1
        spec = sample_random_forest(ranges, seed=member_seed)
        key = _member_key(spec.family, i, n_models)

        if on_variant_start:
            try:
                on_variant_start(key, spec.family, i, n_models)
            except Exception as exc:  # noqa: BLE001
                logger.warning("on_variant_start (forest): %s", exc)

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

        # Garante que as duas classes existem na amostra (.fit exige >=2 classes).
        unique = np.unique(yb)
        if len(unique) < 2:
            logger.warning(
                "Membro %s: amostra com 1 classe apos balanceamento; "
                "pulando este modelo.", key,
            )
            continue

        logger.info(
            "Bagging Floresta %s (%d/%d): family=%s n_estimators=%d "
            "max_depth=%s min_samples_leaf=%d max_features=%s cw=%s "
            "max_samples=%s bootstrap=%s seed=%d",
            key, i, n_models, spec.family, spec.n_estimators, spec.max_depth,
            spec.min_samples_leaf, spec.max_features, spec.class_weight,
            spec.max_samples, spec.bootstrap, spec.seed,
        )

        model = _build_forest_model(spec, n_jobs=int(ranges.n_jobs))
        try:
            model.fit(
                Xb.astype(np.float32, copy=False),
                yb.astype(np.int32, copy=False),
            )
        except Exception as exc:
            logger.warning(
                "Falha ao treinar Floresta %s (%s); pulando este membro.",
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

        fi = np.asarray(model.feature_importances_, dtype=np.float32).reshape(-1)

        out.append(
            ForestBaggingMember(
                key=key,
                family=str(spec.family),
                model=model,
                calibrator=cal,
                feature_importances_=fi,
                sampled_spec=spec.to_dict(),
                feature_names=list(feature_names_list),
                train_metric_val=metric_val,
            )
        )
    return out


# =============================================================================
# Inferencia
# =============================================================================

def predict_forest_bagging_components(
    members: Sequence[ForestBaggingMember],
    X: np.ndarray,
    apply_calibration: bool = True,
) -> np.ndarray:
    """Matriz (N, K) com score de cada Floresta (K = len(members))."""
    if not members:
        raise ValueError("predict_forest_bagging_components: lista vazia.")
    n = X.shape[0]
    k = len(members)
    mat = np.zeros((n, k), dtype=np.float32)
    for j, m in enumerate(members):
        s = _scores_proba(m.model, X)
        if apply_calibration and m.calibrator.fitted:
            s = m.calibrator.transform(s)
        mat[:, j] = s.astype(np.float32, copy=False)
    return mat


def predict_forest_bagging(
    members: Sequence[ForestBaggingMember],
    X: np.ndarray,
    apply_calibration: bool = True,
) -> np.ndarray:
    """Score final por linha = media dos K scores calibrados."""
    comp = predict_forest_bagging_components(
        members, X, apply_calibration=apply_calibration,
    )
    return comp.mean(axis=1).astype(np.float32)


# =============================================================================
# Persistencia
# =============================================================================

def save_forest_bagging(
    members: Sequence[ForestBaggingMember],
    dir_path: str | Path,
) -> None:
    """Persiste todas as Florestas + index.json com kind=forest_bagging."""
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
                "family": m.family,
                "model_file": wpath.name,
                "calibrator_file": cal_path.name,
                "sampled_spec": m.sampled_spec,
                "train_metric_val": float(m.train_metric_val),
            }
        )
    payload = {
        "kind": "forest_bagging",
        "schema": "random_forest_bagging_v1",
        "members": index_members,
    }
    (dir_path / "index.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    joblib.dump(payload, dir_path / "index.pkl")
    logger.info(
        "Forest bagging salvo em %s (%d modelos)",
        dir_path, len(members),
    )


def load_forest_bagging(
    dir_path: str | Path,
) -> list[ForestBaggingMember]:
    """Carrega o bagging de Florestas a partir de index.json."""
    dir_path = Path(dir_path)
    idx_path = dir_path / "index.json"
    if not idx_path.exists():
        raise FileNotFoundError(f"index.json nao encontrado em {dir_path}")
    payload = json.loads(idx_path.read_text(encoding="utf-8"))
    if payload.get("kind") != "forest_bagging":
        raise ValueError(
            f"index.json nao e' forest_bagging: {payload.get('kind')}"
        )
    out: list[ForestBaggingMember] = []
    for meta in payload["members"]:
        wpath = dir_path / meta["model_file"]
        model: ForestModel = joblib.load(wpath)
        cal_path = dir_path / meta["calibrator_file"]
        cal = (
            PlattCalibrator.load(cal_path)
            if cal_path.exists() else PlattCalibrator()
        )
        fi = np.asarray(
            getattr(model, "feature_importances_", np.zeros(0)),
            dtype=np.float32,
        ).reshape(-1)
        family = str(meta.get("family", "rf"))
        out.append(
            ForestBaggingMember(
                key=str(meta["key"]),
                family=family,
                model=model,
                calibrator=cal,
                feature_importances_=fi,
                sampled_spec=meta.get("sampled_spec"),
                train_metric_val=float(meta.get("train_metric_val", 0.0)),
            )
        )
    logger.info(
        "Forest bagging carregado: %d modelos de %s (schema=%s)",
        len(out), dir_path, payload.get("schema", "unknown"),
    )
    return out


__all__ = [
    "ForestRandomRanges",
    "SampledForestSpec",
    "ForestBaggingMember",
    "ForestModel",
    "sample_random_forest",
    "train_random_forest_bagging",
    "predict_forest_bagging",
    "predict_forest_bagging_components",
    "save_forest_bagging",
    "load_forest_bagging",
]
