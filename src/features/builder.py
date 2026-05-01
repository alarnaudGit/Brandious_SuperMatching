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

from .brand_embed import brand_emb_feature_names
from .class_priors import ClassPairPrior
from .classes import build_class_features_row, class_feature_names
from .extra_advanced import (
    build_extra_advanced_features,
    extra_advanced_feature_names,
)
from .extra_nominal import (
    build_extra_nominal_features,
    extra_nominal_feature_names,
)
from .generics import GenericTokenDetector
from .interactions import build_interaction_features, interaction_feature_names
from .nominal import build_nominal_features, nominal_feature_names
from .spec_decomp import (
    item_match_features,
    preprocess_spec_items,
)
from .specs import (
    activity_match_score,
    lexical_features,
    preprocess_spec,
)


def build_static_feature_names(top_classes: list[int]) -> list[str]:
    """Lista canonica de nomes de features 'estaticas' (independentes de TF-IDF/emb).

    Inclui: nominal + spec_lex + spec_activity + class + interactions + extra_nominal.
    Nao inclui: spec_cosine_*, brand_emb_*, spec_item_*, cls_pair_* (esses dependem
    de modelos/estimativas ajustadas).
    """
    nominal = nominal_feature_names()
    cls = class_feature_names(top_classes)

    lex_sample = lexical_features("a b c", "a c d", {"a": 1.0})
    act_sample = activity_match_score("a", "a")
    spec_lex_act = list(lex_sample.keys()) + list(act_sample.keys())

    inter = interaction_feature_names()
    extra = extra_nominal_feature_names()

    return nominal + spec_lex_act + cls + inter + extra


def canonical_feature_order(top_classes: list[int]) -> list[str]:
    """Ordem canonica COMPLETA (estaticas + cosines TF-IDF/emb + brand_emb + Sprint 2).

    Total esperado: 116 (base) + 31 (Sprint 1) + 10 (Sprint 2) = 157.
    """
    static = build_static_feature_names(top_classes)
    cosines = ["spec_cosine_tfidf_word", "spec_cosine_tfidf_char", "spec_cosine_emb"]
    brand_emb = brand_emb_feature_names()
    extra_advanced = extra_advanced_feature_names()
    return static + cosines + brand_emb + extra_advanced


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
    generic_detector: GenericTokenDetector | None = None,
    brand_emb_features: Mapping[str, float] | None = None,
    tfidf_char=None,
    class_pair_prior: ClassPairPrior | None = None,
) -> dict[str, float]:
    """Constroi vetor de features (dict) para um par.

    Parametros opcionais (necessarios para algumas features Sprint 1/2):
        `spec_cosines`: dict com spec_cosine_tfidf_word/char/emb (do preprocessor).
        `brand_emb_features`: dict com brand_emb_* (do preprocessor).
        `generic_detector`: para extra_nominal Sprint 1.
        `tfidf_char`: TfidfVectorizer ja fitado (usado em item-level matching).
        `class_pair_prior`: ClassPairPrior ja fitado (Sprint 2).
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

    extra: dict[str, float] = {}
    if generic_detector is not None:
        extra = build_extra_nominal_features(
            marca_a, marca_b, generic_detector,
            classe_a=int(cls_a), classe_b=int(cls_b),
            spec_emb_cosine=cosine_block["spec_cosine_emb"],
        )
    else:
        for n in extra_nominal_feature_names():
            extra[n] = 0.0

    brand_emb_block = {n: 0.0 for n in brand_emb_feature_names()}
    if brand_emb_features:
        for k in brand_emb_block:
            if k in brand_emb_features:
                brand_emb_block[k] = float(brand_emb_features[k])

    item_match: dict[str, float] | None = None
    if tfidf_char is not None:
        a_items = preprocess_spec_items(spec_a_raw)
        b_items = preprocess_spec_items(spec_b_raw)
        item_match = item_match_features(a_items, b_items, tfidf_char)

    cpp_block: dict[str, float] | None = None
    if class_pair_prior is not None and class_pair_prior.fitted:
        prior, chi2 = class_pair_prior.transform_pair(int(cls_a), int(cls_b))
        cpp_block = {
            "cls_pair_prior_pos": prior,
            "cls_pair_chi2_strength": chi2,
        }

    extra_advanced = build_extra_advanced_features(
        marca_a, marca_b, a_stem, b_stem,
        item_match=item_match,
        class_pair_prior=cpp_block,
        det=generic_detector,
    )

    feats: dict[str, float] = {}
    feats.update(nominal)
    feats.update(spec)
    feats.update(cls)
    feats.update(inter)
    feats.update(extra)
    feats.update(cosine_block)
    feats.update(brand_emb_block)
    feats.update(extra_advanced)
    return feats


__all__ = [
    "build_static_feature_names",
    "canonical_feature_order",
    "build_features_for_row",
]
