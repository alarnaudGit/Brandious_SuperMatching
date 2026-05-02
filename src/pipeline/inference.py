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
from ..model.explain import (
    predict_hybrid_bagging,
    predict_hybrid_bagging_components,
)
from ..model.forest_bagging import load_forest_bagging
from ..model.stacking_meta import load_meta_stacking_bundle
from ..model.logreg_bagging import load_logreg_bagging
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
    # Bagging de N MLPs aleatorias
    arch_bagging_members: list[Any] | None = None
    # Bagging de N regressoes logisticas (modo hibrido com arch_bagging)
    logreg_bagging_members: list[Any] | None = None
    # Bagging de N florestas RF/ExtraTrees (modo hibrido)
    forest_bagging_members: list[Any] | None = None
    meta_stacking_model: nn.Module | None = None


def load_artifacts(
    json_path: str | Path,
    preprocessor_path: str | Path,
    model_path: str | Path | None = None,
    map_location: str = "cpu",
    calibrator_path: str | Path | None = None,
    ensemble_dir: str | Path | None = None,
    arch_bagging_dir: str | Path | None = None,
    logreg_bagging_dir: str | Path | None = None,
    forest_bagging_dir: str | Path | None = None,
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
    logreg_bagging_members: list[Any] | None = None
    forest_bagging_members: list[Any] | None = None

    def _resolve_sibling(name: str) -> Path | None:
        for sibling_anchor in (
            arch_bagging_dir, ensemble_dir, model_path, json_path,
        ):
            if sibling_anchor is None:
                continue
            anchor = Path(sibling_anchor)
            base = anchor if anchor.is_dir() else anchor.parent
            cand1 = base.parent / name
            if cand1.exists():
                return cand1
            cand2 = base / name
            if cand2.exists():
                return cand2
        return None

    # Auto-detect logreg_bagging_dir como pasta irma das demais (se nao passada).
    if logreg_bagging_dir is None:
        logreg_bagging_dir = _resolve_sibling("ensemble_logreg")

    if logreg_bagging_dir is not None and Path(logreg_bagging_dir).exists():
        try:
            logreg_bagging_members = load_logreg_bagging(
                Path(logreg_bagging_dir),
            )
            logger.info(
                "LogReg bagging carregado de %s (%d modelos)",
                logreg_bagging_dir, len(logreg_bagging_members),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Falha ao carregar logreg_bagging_dir %s: %s",
                logreg_bagging_dir, exc,
            )

    # Auto-detect forest_bagging_dir como pasta irma das demais (se nao passada).
    if forest_bagging_dir is None:
        forest_bagging_dir = _resolve_sibling("ensemble_forest")

    if forest_bagging_dir is not None and Path(forest_bagging_dir).exists():
        try:
            forest_bagging_members = load_forest_bagging(
                Path(forest_bagging_dir),
            )
            logger.info(
                "Forest bagging carregado de %s (%d modelos)",
                forest_bagging_dir, len(forest_bagging_members),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Falha ao carregar forest_bagging_dir %s: %s",
                forest_bagging_dir, exc,
            )

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

    meta_stacking_model: nn.Module | None = None
    ms_dir = _resolve_sibling("meta_stacking")
    ms_conf = cfg.get("meta_stacking") or {}
    if ms_conf.get("enabled") and ms_dir is not None and Path(ms_dir).exists():
        try:
            meta_stacking_model, _ms_meta = load_meta_stacking_bundle(
                Path(ms_dir), map_location=map_location,
            )
            logger.info("Meta-stacking carregado de %s", ms_dir)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Falha ao carregar meta_stacking: %s", exc)

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
        logreg_bagging_members=logreg_bagging_members,
        forest_bagging_members=forest_bagging_members,
        meta_stacking_model=meta_stacking_model,
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
    """Aplica meta-stacking, bagging hibrido, ensemble legado ou modelo unico."""
    if getattr(artifacts, "meta_stacking_model", None) is not None:
        from ..model.stacking_meta import (
            build_meta_design_matrix,
            predict_meta_stacking_scores,
        )

        Z, _ = build_meta_design_matrix(
            X,
            artifacts.arch_bagging_members,
            artifacts.logreg_bagging_members,
            artifacts.forest_bagging_members,
        )
        return predict_meta_stacking_scores(
            artifacts.meta_stacking_model, Z,
        )
    if (
        artifacts.arch_bagging_members
        or artifacts.logreg_bagging_members
        or artifacts.forest_bagging_members
    ):
        return predict_hybrid_bagging(
            artifacts.arch_bagging_members,
            artifacts.logreg_bagging_members,
            X,
            forest_members=artifacts.forest_bagging_members or None,
            apply_calibration=True,
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
        ) or bool(artifacts.ensemble_models) or bool(
            artifacts.arch_bagging_members
        ) or bool(artifacts.logreg_bagging_members) or bool(
            artifacts.forest_bagging_members
        ),
        "ensemble_size": len(artifacts.ensemble_models),
        "architecture_bagging": bool(artifacts.arch_bagging_members),
        "logreg_bagging": bool(artifacts.logreg_bagging_members),
        "forest_bagging": bool(artifacts.forest_bagging_members),
        "hybrid_bagging": bool(
            artifacts.arch_bagging_members
            or artifacts.logreg_bagging_members
            or artifacts.forest_bagging_members
        ),
        "meta_stacking": bool(getattr(artifacts, "meta_stacking_model", None)),
    }
    if (
        artifacts.arch_bagging_members
        or artifacts.logreg_bagging_members
        or artifacts.forest_bagging_members
    ):
        comp_mat, comp_info = predict_hybrid_bagging_components(
            artifacts.arch_bagging_members,
            artifacts.logreg_bagging_members,
            X,
            forest_members=artifacts.forest_bagging_members or None,
            apply_calibration=True,
        )
        comp_row = comp_mat[0]
        out["score_by_member"] = [
            {
                "kind": str(info["kind"]),
                "key": str(info["key"]),
                "score": float(comp_row[j]),
            }
            for j, info in enumerate(comp_info)
        ]
        # compat: mantem score_by_architecture (apenas MLPs) para callers antigos
        if artifacts.arch_bagging_members:
            mlp_scores: dict[str, float] = {}
            for j, info in enumerate(comp_info):
                if info["kind"] == "mlp":
                    mlp_scores[str(info["key"])] = float(comp_row[j])
            out["score_by_architecture"] = mlp_scores
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
