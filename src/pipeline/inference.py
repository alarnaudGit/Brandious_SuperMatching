"""Inferencia em producao: carrega artefatos e produz score 0-1 para um par.

Sprint 2:
- aceita arquiteturas alternativas (two_tower, ft_transformer, multitask) via
  `config_json["architecture"]["type"]` se presente
- aplica `PlattCalibrator` quando o arquivo opcional `calibrator.json|.pkl`
  estiver disponivel ao lado dos artefatos
- aceita ensemble de N membros via diretorio `ensemble/` com index_json
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..features.builder import build_features_for_row
from ..features.specs import preprocess_spec, row_cosine
from ..model.architectures import (
    FTTransformerConfig, FTTransformer,
    MultiTaskConfig, MultiTaskMLP,
    TwoTowerConfig, TwoTowerCrossAttention,
)
from ..model.arch_bagging import (
    predict_architecture_bagging,
    predict_architecture_bagging_components,
)
from ..model.calibration import PlattCalibrator
from ..model.evaluate import predict_scores
from ..model.mlp import BrandSimilarityMLP, MLPConfig
from .preprocessor import FeaturePreprocessor

logger = logging.getLogger(__name__)


# =============================================================================
# Reconstrucao de modelo a partir do JSON
# =============================================================================

def _build_model_from_arch_json(arch: dict[str, Any]) -> nn.Module:
    """Reconstroi a arquitetura usada no treino a partir do dict salvo."""
    t = str(arch.get("type", "mlp")).lower()
    if t in ("mlp", "brand_mlp", "baseline", "") or "input_dim" in arch and "type" not in arch:
        return BrandSimilarityMLP(MLPConfig.from_dict(arch))
    if t == "two_tower":
        cfg = TwoTowerConfig(
            name_dim=int(arch["name_dim"]),
            ctx_dim=int(arch["ctx_dim"]),
            name_idx=list(arch["name_idx"]),
            ctx_idx=list(arch["ctx_idx"]),
            tower_hidden=list(arch.get("tower_hidden", [128, 64])),
            embed_dim=int(arch.get("embed_dim", 64)),
            n_heads=int(arch.get("n_heads", 4)),
            dropout=float(arch.get("dropout", 0.3)),
            use_batchnorm=bool(arch.get("use_batchnorm", True)),
            head_hidden=list(arch.get("head_hidden", [128, 64])),
        )
        return TwoTowerCrossAttention(cfg)
    if t == "ft_transformer":
        cfg = FTTransformerConfig(
            input_dim=int(arch["input_dim"]),
            embed_dim=int(arch.get("embed_dim", 16)),
            n_layers=int(arch.get("n_layers", 2)),
            n_heads=int(arch.get("n_heads", 4)),
            ffn_dim=int(arch.get("ffn_dim", 64)),
            dropout=float(arch.get("dropout", 0.1)),
        )
        return FTTransformer(cfg)
    if t == "multitask":
        cfg = MultiTaskConfig(
            input_dim=int(arch["input_dim"]),
            backbone_hidden=list(arch.get("backbone_hidden", [256, 128, 64])),
            dropout=float(arch.get("dropout", 0.45)),
            use_batchnorm=bool(arch.get("use_batchnorm", True)),
            aux_indices=arch.get("aux_indices"),
        )
        return MultiTaskMLP(cfg)
    raise ValueError(f"architecture type desconhecida: {t}")


# =============================================================================
# Artefatos
# =============================================================================

@dataclass
class InferenceArtifacts:
    model: nn.Module
    preprocessor: FeaturePreprocessor
    threshold: float
    feature_names_ordered: list[str]
    config_json: dict[str, Any]
    calibrator: PlattCalibrator | None = None
    ensemble_models: list[nn.Module] = field(default_factory=list)
    ensemble_calibrators: list[PlattCalibrator] = field(default_factory=list)
    # Bagging das 4 arquiteturas (MLP + Two-Tower + FT-Transformer + Multi-Task)
    arch_bagging_members: list[Any] | None = None


def load_artifacts(
    json_path: str | Path,
    preprocessor_path: str | Path,
    model_path: str | Path | None = None,
    map_location: str = "cpu",
    calibrator_path: str | Path | None = None,
    ensemble_dir: str | Path | None = None,
    arch_bagging_dir: str | Path | None = None,
) -> InferenceArtifacts:
    """Carrega config_json + preprocessor + modelo (singleton ou ensemble) + calibrador.

    Se `arch_bagging_dir` apontar para pasta com `index.json` kind=architecture_bagging,
    carrega os 4 modelos e o score final e' a media (ver `predict_architecture_bagging`).
    """
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    preproc = FeaturePreprocessor.load(preprocessor_path)

    arch = cfg["architecture"]
    model = _build_model_from_arch_json(arch)

    if model_path is not None and Path(model_path).exists():
        state = torch.load(Path(model_path), map_location=map_location)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
    else:
        import base64
        import io
        raw = base64.b64decode(cfg["state_dict_b64"])
        state = torch.load(io.BytesIO(raw), map_location=map_location)
    model.load_state_dict(state)
    model.eval()

    calibrator: PlattCalibrator | None = None
    if calibrator_path is not None and Path(calibrator_path).exists():
        try:
            calibrator = PlattCalibrator.load(calibrator_path)
            logger.info("Calibrador Platt carregado de %s", calibrator_path)
        except Exception as exc:
            logger.warning("Falha ao carregar calibrador (%s): %s", calibrator_path, exc)

    ens_models: list[nn.Module] = []
    ens_calibrators: list[PlattCalibrator] = []
    arch_bagging_members: list[Any] | None = None

    if arch_bagging_dir is not None and Path(arch_bagging_dir).exists():
        from ..model.arch_bagging import load_architecture_bagging

        arch_bagging_members = load_architecture_bagging(
            Path(arch_bagging_dir), map_location=map_location,
        )
        logger.info(
            "Bagging por arquitetura carregado de %s (%d modelos)",
            arch_bagging_dir, len(arch_bagging_members),
        )
    elif ensemble_dir is not None and Path(ensemble_dir).exists():
        idx = Path(ensemble_dir) / "index.json"
        if idx.exists():
            try:
                meta = json.loads(idx.read_text(encoding="utf-8"))
                if meta.get("kind") == "architecture_bagging":
                    from ..model.arch_bagging import load_architecture_bagging

                    arch_bagging_members = load_architecture_bagging(
                        Path(ensemble_dir), map_location=map_location,
                    )
                else:
                    ens_models, ens_calibrators = _load_ensemble(
                        Path(ensemble_dir), arch, map_location=map_location,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Falha ao ler ensemble_dir %s: %s", ensemble_dir, exc)
                ens_models, ens_calibrators = _load_ensemble(
                    Path(ensemble_dir), arch, map_location=map_location,
                )
        else:
            ens_models, ens_calibrators = _load_ensemble(
                Path(ensemble_dir), arch, map_location=map_location,
            )
        if arch_bagging_members is None and ens_models:
            logger.info(
                "Ensemble (seeds) carregado de %s (%d membros)", ensemble_dir, len(ens_models),
            )

    return InferenceArtifacts(
        model=model,
        preprocessor=preproc,
        threshold=float(cfg.get("threshold_optimal", 0.5)),
        feature_names_ordered=list(preproc.feature_names_ordered),
        config_json=cfg,
        calibrator=calibrator,
        ensemble_models=ens_models,
        ensemble_calibrators=ens_calibrators,
        arch_bagging_members=arch_bagging_members,
    )


def _load_ensemble(
    ensemble_dir: Path,
    arch: dict[str, Any],
    map_location: str = "cpu",
) -> tuple[list[nn.Module], list[PlattCalibrator]]:
    index_path = ensemble_dir / "ensemble_index.json"
    if not index_path.exists():
        return [], []
    members_meta = json.loads(index_path.read_text(encoding="utf-8"))
    models: list[nn.Module] = []
    calibrators: list[PlattCalibrator] = []
    for meta in members_meta:
        m = _build_model_from_arch_json(arch)
        path = ensemble_dir / meta["model_file"]
        state = torch.load(path, map_location=map_location)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        m.load_state_dict(state)
        m.eval()
        models.append(m)
        cal_path = ensemble_dir / meta["calibrator_file"]
        if cal_path.exists():
            calibrators.append(PlattCalibrator.load(cal_path))
        else:
            calibrators.append(PlattCalibrator())
    return models, calibrators


# =============================================================================
# Score (singleton ou ensemble + calibracao)
# =============================================================================

def _final_score(artifacts: InferenceArtifacts, X: np.ndarray) -> np.ndarray:
    """Aplica bagging 4-arquiteturas, ensemble por seeds, ou modelo unico."""
    if artifacts.arch_bagging_members:
        return predict_architecture_bagging(
            artifacts.arch_bagging_members, X, apply_calibration=True,
        )
    if artifacts.ensemble_models:
        accum = np.zeros(X.shape[0], dtype=np.float64)
        for m, cal in zip(artifacts.ensemble_models, artifacts.ensemble_calibrators):
            s = predict_scores(m, X)
            if cal is not None and cal.fitted:
                s = cal.transform(s)
            accum += s.astype(np.float64)
        scores = (accum / len(artifacts.ensemble_models)).astype(np.float32)
    else:
        scores = predict_scores(artifacts.model, X)
        if artifacts.calibrator is not None and artifacts.calibrator.fitted:
            scores = artifacts.calibrator.transform(scores)
    return scores


def score_pair(
    artifacts: InferenceArtifacts,
    marca_a: str,
    marca_b: str,
    classe_a: int = -1,
    classe_b: int = -1,
    spec_a: str = "",
    spec_b: str = "",
) -> dict[str, Any]:
    """Score completo para um par (uso em producao e no teste manual do Streamlit)."""
    df = pd.DataFrame(
        {
            "marca_monitorada": [marca_a],
            "marca_colidente": [marca_b],
            "classe_marca_monitorada": [int(classe_a) if classe_a is not None else -1],
            "classe_marca_colidente": [int(classe_b) if classe_b is not None else -1],
            "especificacao_monitorado": [str(spec_a) if spec_a else ""],
            "especificacao_colidente": [str(spec_b) if spec_b else ""],
        }
    )
    X = artifacts.preprocessor.transform(df, scale=True)
    score = float(_final_score(artifacts, X)[0])
    classe_prevista = int(score >= artifacts.threshold)
    out: dict[str, Any] = {
        "score_nn": score,
        "threshold": artifacts.threshold,
        "classe_prevista": classe_prevista,
        "x_unscaled": None,
        "x_scaled": X[0].tolist(),
        "calibration_applied": bool(
            artifacts.calibrator and artifacts.calibrator.fitted
        ) or bool(artifacts.ensemble_models) or bool(artifacts.arch_bagging_members),
        "ensemble_size": len(artifacts.ensemble_models),
        "architecture_bagging": bool(artifacts.arch_bagging_members),
    }
    if artifacts.arch_bagging_members:
        comp = predict_architecture_bagging_components(
            artifacts.arch_bagging_members, X, apply_calibration=True,
        )[0]
        out["score_by_architecture"] = {
            m.key: float(comp[j])
            for j, m in enumerate(artifacts.arch_bagging_members)
        }
    return out


def score_batch(
    artifacts: InferenceArtifacts,
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Devolve (scores, classes_previstas) para um dataframe ja com as 6 colunas."""
    X = artifacts.preprocessor.transform(df, scale=True)
    scores = _final_score(artifacts, X)
    classes = (scores >= artifacts.threshold).astype(int)
    return scores, classes


__all__ = [
    "InferenceArtifacts",
    "load_artifacts",
    "score_pair",
    "score_batch",
]
