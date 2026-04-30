"""Inferencia em producao: carrega artefatos e produz score 0-1 para um par."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..features.specs import preprocess_spec, row_cosine
from ..features.builder import build_features_for_row
from ..model.evaluate import predict_scores
from ..model.mlp import BrandSimilarityMLP, MLPConfig
from .preprocessor import FeaturePreprocessor

logger = logging.getLogger(__name__)


@dataclass
class InferenceArtifacts:
    model: BrandSimilarityMLP
    preprocessor: FeaturePreprocessor
    threshold: float
    feature_names_ordered: list[str]
    config_json: dict[str, Any]


def load_artifacts(
    json_path: str | Path,
    preprocessor_path: str | Path,
    model_path: str | Path | None = None,
    map_location: str = "cpu",
) -> InferenceArtifacts:
    """Carrega config_json + preprocessor.pkl e reconstroi a MLP.

    Se `model_path` (.pt) for fornecido, ele tem prioridade. Caso contrario,
    usa o `state_dict_b64` embutido no JSON.
    """
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    preproc = FeaturePreprocessor.load(preprocessor_path)

    arch = cfg["architecture"]
    mlp_cfg = MLPConfig.from_dict(arch)
    model = BrandSimilarityMLP(mlp_cfg)

    if model_path is not None and Path(model_path).exists():
        state = torch.load(Path(model_path), map_location=map_location)
    else:
        import base64
        import io

        raw = base64.b64decode(cfg["state_dict_b64"])
        state = torch.load(io.BytesIO(raw), map_location=map_location)
    model.load_state_dict(state)
    model.eval()

    return InferenceArtifacts(
        model=model,
        preprocessor=preproc,
        threshold=float(cfg.get("threshold_optimal", 0.5)),
        feature_names_ordered=list(preproc.feature_names_ordered),
        config_json=cfg,
    )


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
    score = float(predict_scores(artifacts.model, X)[0])
    classe_prevista = int(score >= artifacts.threshold)
    return {
        "score_nn": score,
        "threshold": artifacts.threshold,
        "classe_prevista": classe_prevista,
        "x_unscaled": None,
        "x_scaled": X[0].tolist(),
    }


def score_batch(
    artifacts: InferenceArtifacts,
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Devolve (scores, classes_previstas) para um dataframe ja com as 6 colunas."""
    X = artifacts.preprocessor.transform(df, scale=True)
    scores = predict_scores(artifacts.model, X)
    classes = (scores >= artifacts.threshold).astype(int)
    return scores, classes


__all__ = [
    "InferenceArtifacts",
    "load_artifacts",
    "score_pair",
    "score_batch",
]
