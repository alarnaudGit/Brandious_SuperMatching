"""Features avancadas Sprint 2 (10 novas features para minority class).

Cobre:

2.1 Item-level spec matching (3) -> via spec_decomp.item_match_features
2.2 Brand-as-anchor in spec (2) -> brand_anchor_features
2.3 Class-pair empirical prior (2) -> via class_priors.ClassPairPrior
2.4 Substring contiguo entre marcas (1) -> brand_longest_common_substring_norm
2.5 Phonetic da concatenacao (1) -> fon_concat_metaphone_eq
2.6 SimHash do brand (1) -> brand_simhash_hamming_norm

Este modulo expoe:
- helpers individuais (testaveis)
- `build_extra_advanced_features(...)` orquestrador POR LINHA
- `extra_advanced_feature_names()` ordem canonica das 10 chaves
"""
from __future__ import annotations

import re

from ..normalize import (
    apply_base_normalization,
    get_phonetic_key,
)
from .generics import GenericTokenDetector, tokenize_brand


# =============================================================================
# 2.2 Brand-as-anchor in spec (2 features)
# =============================================================================

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _stemize_spec_tokens(spec_stem: str) -> set[str]:
    """A spec ja vem stemizada do `preprocess_spec`. Retornamos o set de tokens."""
    if not spec_stem:
        return set()
    return {t for t in spec_stem.split() if len(t) >= 3}


def _matches_spec_tokens(brand_tok: str, spec_tokens: set[str]) -> bool:
    """Considera match se o token da marca aparece como prefixo de algum token
    stemizado da spec (cobre o caso de a stem da spec ser radical do termo).
    Tambem considera match exato.
    """
    if brand_tok in spec_tokens:
        return True
    for st in spec_tokens:
        if len(brand_tok) >= 4 and (st.startswith(brand_tok[:4]) or brand_tok.startswith(st[:4])):
            if brand_tok in st or st in brand_tok:
                return True
    return False


def brand_anchor_features(
    a_brand: str,
    b_brand: str,
    a_spec_stem: str,
    b_spec_stem: str,
    det: GenericTokenDetector | None = None,
) -> dict[str, float]:
    """Conta tokens informativos de uma marca que aparecem na spec da OUTRA.

    Retorna inteiros (em float) - serao escalados pelo StandardScaler.
    """
    if det is None:
        inf_a = [t for t in tokenize_brand(a_brand) if len(t) >= 4]
        inf_b = [t for t in tokenize_brand(b_brand) if len(t) >= 4]
    else:
        inf_a = [t for t in det.informative_tokens(a_brand) if len(t) >= 4]
        inf_b = [t for t in det.informative_tokens(b_brand) if len(t) >= 4]

    spec_a_tokens = _stemize_spec_tokens(a_spec_stem)
    spec_b_tokens = _stemize_spec_tokens(b_spec_stem)

    a_in_b = sum(1 for t in inf_a if _matches_spec_tokens(t, spec_b_tokens))
    b_in_a = sum(1 for t in inf_b if _matches_spec_tokens(t, spec_a_tokens))

    return {
        "brand_a_token_in_spec_b": float(a_in_b),
        "brand_b_token_in_spec_a": float(b_in_a),
    }


def brand_anchor_feature_names() -> list[str]:
    return ["brand_a_token_in_spec_b", "brand_b_token_in_spec_a"]


# =============================================================================
# 2.4 Marca-marca em substring contiguo (1 feature)
# =============================================================================

def _longest_common_substring(a: str, b: str) -> int:
    """LCS contigua via DP O(len_a * len_b). Para strings curtas (<=64) e barato."""
    if not a or not b:
        return 0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0
    prev = [0] * (lb + 1)
    best = 0
    for i in range(1, la + 1):
        cur = [0] * (lb + 1)
        ai = a[i - 1]
        for j in range(1, lb + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def brand_longest_common_substring_features(a: str, b: str) -> dict[str, float]:
    """Mede a maior substring CONTIGUA entre duas marcas normalizadas.

    Normalizadas SEM espacos, para capturar embed (ex.: 'oftal' em 'oftalmotop').
    """
    a_n = _NON_ALNUM.sub("", apply_base_normalization(a))
    b_n = _NON_ALNUM.sub("", apply_base_normalization(b))
    if not a_n or not b_n:
        return {"brand_longest_common_substring_norm": 0.0}
    lcs = _longest_common_substring(a_n, b_n)
    norm = lcs / max(1, min(len(a_n), len(b_n)))
    return {"brand_longest_common_substring_norm": float(norm)}


def brand_longest_common_substring_feature_names() -> list[str]:
    return ["brand_longest_common_substring_norm"]


# =============================================================================
# 2.5 Phonetic da concatenacao (1 feature)
# =============================================================================

def fon_concat_features(a: str, b: str) -> dict[str, float]:
    """Compara chaves foneticas da marca COMPLETA sem espacos.

    `OFTALMO TOP` deveria soar como `oftalmotop`. A versao por token nao captura
    isso (porque "TOP" sai da OFTA como token isolado).
    """
    a_concat = _NON_ALNUM.sub("", apply_base_normalization(a))
    b_concat = _NON_ALNUM.sub("", apply_base_normalization(b))
    if not a_concat or not b_concat:
        return {"fon_concat_metaphone_eq": 0.0}
    eq = float(get_phonetic_key(a_concat) == get_phonetic_key(b_concat))
    return {"fon_concat_metaphone_eq": eq}


def fon_concat_feature_names() -> list[str]:
    return ["fon_concat_metaphone_eq"]


# =============================================================================
# 2.6 SimHash da marca (1 feature)
# =============================================================================

def _char_trigrams(s: str) -> list[str]:
    s = "  " + s + "  "
    return [s[i:i + 3] for i in range(len(s) - 2)]


def _hash_token(tok: str) -> int:
    """Hash de 64 bits estavel. Usa Python hash com seed mascarada para 64 bits."""
    h = hash(tok) & 0xFFFFFFFFFFFFFFFF
    return h


def _simhash64(text: str) -> int:
    """SimHash de 64 bits sobre char-trigramas."""
    if not text:
        return 0
    accum = [0] * 64
    for trig in _char_trigrams(text):
        h = _hash_token(trig)
        for b in range(64):
            if (h >> b) & 1:
                accum[b] += 1
            else:
                accum[b] -= 1
    out = 0
    for b in range(64):
        if accum[b] >= 0:
            out |= (1 << b)
    return out


def _hamming64(x: int, y: int) -> int:
    return bin((x ^ y) & 0xFFFFFFFFFFFFFFFF).count("1")


def brand_simhash_features(a: str, b: str) -> dict[str, float]:
    a_n = _NON_ALNUM.sub("", apply_base_normalization(a))
    b_n = _NON_ALNUM.sub("", apply_base_normalization(b))
    if not a_n or not b_n:
        return {"brand_simhash_hamming_norm": 0.0}
    h_a = _simhash64(a_n)
    h_b = _simhash64(b_n)
    dist = _hamming64(h_a, h_b)
    sim = 1.0 - (dist / 64.0)
    return {"brand_simhash_hamming_norm": float(sim)}


def brand_simhash_feature_names() -> list[str]:
    return ["brand_simhash_hamming_norm"]


# =============================================================================
# Orquestrador (10 features)
# =============================================================================

def build_extra_advanced_features(
    a_brand: str,
    b_brand: str,
    a_spec_stem: str,
    b_spec_stem: str,
    *,
    item_match: dict[str, float] | None = None,
    class_pair_prior: dict[str, float] | None = None,
    det: GenericTokenDetector | None = None,
) -> dict[str, float]:
    """Combina as 10 features Sprint 2 em um dict.

    `item_match` deve conter as 3 chaves de spec_decomp.item_match_features
    (precisam do TfidfVectorizer ja fitado no preprocessor).
    `class_pair_prior` deve conter as 2 chaves de class_priors.ClassPairPrior
    (idem - precisa estar fitado no preprocessor).
    Se vierem None, preenchem com 0.0.
    """
    out: dict[str, float] = {}

    item_defaults = {
        "spec_item_max_cosine_tfidf": 0.0,
        "spec_item_top3_mean_cosine": 0.0,
        "spec_item_align_score": 0.0,
    }
    if item_match:
        for k in item_defaults:
            out[k] = float(item_match.get(k, 0.0))
    else:
        out.update(item_defaults)

    out.update(brand_anchor_features(a_brand, b_brand, a_spec_stem, b_spec_stem, det=det))

    cls_defaults = {"cls_pair_prior_pos": 0.0, "cls_pair_chi2_strength": 0.0}
    if class_pair_prior:
        for k in cls_defaults:
            out[k] = float(class_pair_prior.get(k, 0.0))
    else:
        out.update(cls_defaults)

    out.update(brand_longest_common_substring_features(a_brand, b_brand))
    out.update(fon_concat_features(a_brand, b_brand))
    out.update(brand_simhash_features(a_brand, b_brand))
    return out


def extra_advanced_feature_names() -> list[str]:
    return (
        ["spec_item_max_cosine_tfidf", "spec_item_top3_mean_cosine", "spec_item_align_score"]
        + brand_anchor_feature_names()
        + ["cls_pair_prior_pos", "cls_pair_chi2_strength"]
        + brand_longest_common_substring_feature_names()
        + fon_concat_feature_names()
        + brand_simhash_feature_names()
    )


__all__ = [
    "build_extra_advanced_features",
    "extra_advanced_feature_names",
    "brand_anchor_features",
    "brand_anchor_feature_names",
    "brand_longest_common_substring_features",
    "brand_longest_common_substring_feature_names",
    "fon_concat_features",
    "fon_concat_feature_names",
    "brand_simhash_features",
    "brand_simhash_feature_names",
]
