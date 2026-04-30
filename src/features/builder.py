"""Orquestrador de features por par (marca, especificacao, classe).

Funcoes:
- `build_features_for_row`: gera dict completo de features para uma linha (sem TF-IDF/emb).
  Usado pelo modo de inferencia/teste manual quando o preprocessor ja esta treinado.

- O builder em batch que materializa todas as features em uma matriz `(N, F)` esta
  no proprio preprocessor (`src/pipeline/preprocessor.py`), porque depende de
  vectorizers TF-IDF/embedding ajustados.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from .classes import build_class_features_row, class_feature_names
from .interactions import build_interaction_features, interaction_feature_names
from .nominal import build_nominal_features, nominal_feature_names
from .specs import (
    activity_match_score,
    lexical_features,
    preprocess_spec,
    row_cosine,
    spec_feature_names,
)


def build_static_feature_names(top_classes: list[int]) -> list[str]:
    """Lista canonica de nomes de features 'estaticas' (independentes de TF-IDF/emb).

    Inclui: nominal + spec_lex + spec_activity + class + interactions.
    Nao inclui: spec_cosine_tfidf_word, spec_cosine_tfidf_char, spec_cosine_emb
                (esses sao acrescentados depois pelo preprocessor).
    """
    nominal = nominal_feature_names()
    cls = class_feature_names(top_classes)

    lex_sample = lexical_features("a b c", "a c d", {"a": 1.0})
    act_sample = activity_match_score("a", "a")
    spec_lex_act = list(lex_sample.keys()) + list(act_sample.keys())

    inter = interaction_feature_names()

    return nominal + spec_lex_act + cls + inter


def canonical_feature_order(top_classes: list[int]) -> list[str]:
    """Ordem canonica COMPLETA (estaticas + cosines TF-IDF/emb)."""
    static = build_static_feature_names(top_classes)
    cosines = ["spec_cosine_tfidf_word", "spec_cosine_tfidf_char", "spec_cosine_emb"]
    return static + cosines


def build_features_for_row(
    marca_a: str,
    marca_b: str,
    spec_a_raw: str,
    spec_b_raw: str,
    cls_a: int,
    cls_b: int,
    top_classes: list[int],
    spec_cosines: Mapping[str, float] | None = None,
    idf_lookup: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Constroi vetor de features (dict) para um par.

    `spec_cosines` carrega as 3 features dependentes do preprocessor:
        spec_cosine_tfidf_word, spec_cosine_tfidf_char, spec_cosine_emb.
    Se nao for fornecido, sao preenchidas com 0.0.
    """
    nominal = build_nominal_features(marca_a, marca_b)

    a_stem = preprocess_spec(spec_a_raw)
    b_stem = preprocess_spec(spec_b_raw)
    spec_lex = lexical_features(a_stem, b_stem, dict(idf_lookup) if idf_lookup else None)
    spec_act = activity_match_score(a_stem, b_stem)
    spec = {**spec_lex, **spec_act}

    cls = build_class_features_row(int(cls_a), int(cls_b), top_classes)

    cosine_block = {
        "spec_cosine_tfidf_word": 0.0,
        "spec_cosine_tfidf_char": 0.0,
        "spec_cosine_emb": 0.0,
    }
    if spec_cosines:
        for k in cosine_block:
            if k in spec_cosines:
                cosine_block[k] = float(spec_cosines[k])
    spec_for_inter = {**spec, **cosine_block}

    inter = build_interaction_features(nominal, spec_for_inter, cls)

    feats: dict[str, float] = {}
    feats.update(nominal)
    feats.update(spec)
    feats.update(cls)
    feats.update(inter)
    feats.update(cosine_block)
    return feats


__all__ = [
    "build_static_feature_names",
    "canonical_feature_order",
    "build_features_for_row",
]
