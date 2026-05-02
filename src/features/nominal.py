"""Features de similaridade nominativa (graficas + foneticas + heuristica OFTA).

Cada par (marca_a, marca_b) gera ~30 features numericas que serao consumidas pela MLP.
A heuristica OFTA do projeto entra como SINAL (input), nao como verdade.
"""
from __future__ import annotations

from typing import Any

import jellyfish
from Levenshtein import distance as lev_distance

from ..normalize import (
    apply_base_normalization,
    calculate_single_score,
    clean_strict,
    convert_to_cardinal_words,
    convert_to_literal_words,
    get_levenshtein_similarity,
    get_phonetic_key,
    get_phonetic_similarity,
    get_sorted_char_similarity,
    remove_repeated_chars,
)
from .brand_token_overlap import brand_cross_token_overlap_features
from .tokens import (
    avg_max_phonetic_token,
    char_ngrams,
    common_prefix_len,
    common_suffix_len,
    jaccard_sets,
    lcs_length,
    ofta_token_metrics,
    overlap_sets,
    token_set_metrics,
)

DRIVER_NAMES = [
    "Geral",
    "Ortografia (Escrita)",
    "Fonética (Som)",
    "Aproximação de Termos (Fuzzy)",
    "Inclusão Total de Termo",
    "Palavras em Comum",
]


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def _ratio(a: int, b: int) -> float:
    if a == 0 and b == 0:
        return 1.0
    if a == 0 or b == 0:
        return 0.0
    return min(a, b) / max(a, b)


def _len_diff(a: str, b: str) -> int:
    return abs(len(a) - len(b))


def _contains_other(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return 1.0 if (a in b or b in a) else 0.0


def graphical_features(a_norm: str, b_norm: str) -> dict[str, float]:
    """Features graficas (gramaticais) entre as duas marcas ja normalizadas (sem espacos)."""
    a_strict = clean_strict(a_norm)
    b_strict = clean_strict(b_norm)

    lev = get_levenshtein_similarity(a_strict, b_strict)
    if a_strict and b_strict:
        jaro = jellyfish.jaro_similarity(a_strict, b_strict)
        jaro_w = jellyfish.jaro_winkler_similarity(a_strict, b_strict)
        try:
            damerau = 1.0 - jellyfish.damerau_levenshtein_distance(a_strict, b_strict) / max(len(a_strict), len(b_strict))
        except Exception:
            damerau = lev
    else:
        jaro = 1.0 if not a_strict and not b_strict else 0.0
        jaro_w = jaro
        damerau = jaro

    bg_a, bg_b = char_ngrams(a_strict, 2), char_ngrams(b_strict, 2)
    tg_a, tg_b = char_ngrams(a_strict, 3), char_ngrams(b_strict, 3)

    lcs = lcs_length(a_strict, b_strict)
    lcs_norm = _safe_div(lcs, max(len(a_strict), len(b_strict)))
    pref = common_prefix_len(a_strict, b_strict)
    suf = common_suffix_len(a_strict, b_strict)
    pref_norm = _safe_div(pref, max(len(a_strict), len(b_strict)))
    suf_norm = _safe_div(suf, max(len(a_strict), len(b_strict)))

    anagram = get_sorted_char_similarity(a_norm, b_norm)

    return {
        "graf_levenshtein": float(lev),
        "graf_jaro": float(jaro),
        "graf_jaro_winkler": float(jaro_w),
        "graf_damerau": float(damerau),
        "graf_jaccard_bigram": float(jaccard_sets(bg_a, bg_b)),
        "graf_jaccard_trigram": float(jaccard_sets(tg_a, tg_b)),
        "graf_overlap_trigram": float(overlap_sets(tg_a, tg_b)),
        "graf_lcs_norm": float(lcs_norm),
        "graf_prefix_norm": float(pref_norm),
        "graf_suffix_norm": float(suf_norm),
        "graf_len_ratio": float(_ratio(len(a_strict), len(b_strict))),
        "graf_len_diff_abs": float(_len_diff(a_strict, b_strict)),
        "graf_contains": float(_contains_other(a_strict, b_strict)),
        "graf_anagram": float(anagram),
    }


def phonetic_features(a_norm: str, b_norm: str) -> dict[str, float]:
    """Features foneticas globais e por token."""
    fon_global = get_phonetic_similarity(a_norm, b_norm)
    key_a = get_phonetic_key(a_norm.replace(" ", ""))
    key_b = get_phonetic_key(b_norm.replace(" ", ""))
    key_eq = float(key_a == key_b and len(key_a) > 0)
    if key_a or key_b:
        key_lev = 1.0 - lev_distance(key_a, key_b) / max(len(key_a) or 1, len(key_b) or 1)
    else:
        key_lev = 1.0

    rep_eq = float(remove_repeated_chars(a_norm.replace(" ", "")) == remove_repeated_chars(b_norm.replace(" ", "")))

    per_token = avg_max_phonetic_token(a_norm, b_norm)

    return {
        "fon_global_sim": float(fon_global),
        "fon_key_eq": key_eq,
        "fon_key_lev_sim": float(key_lev),
        "fon_after_dedup_eq": rep_eq,
        "fon_token_mean": per_token["phon_token_mean"],
        "fon_token_max": per_token["phon_token_max"],
        "fon_token_eq_share": per_token["phon_token_eq_share"],
    }


def numeral_features(m1: str, m2: str) -> dict[str, float]:
    """Features das 4 vias literal/cardinal de calcular_score_complexo, sem o resultado final.

    Para cada combinacao de (lado convertido, via), recalcula score_orto e score_fon e
    expoe melhor/pior/spread, dando a rede a habilidade de aprender quando numerais
    importam.
    """
    base1 = apply_base_normalization(m1)
    base2 = apply_base_normalization(m2)

    has_digits = any(ch.isdigit() for ch in m1) or any(ch.isdigit() for ch in m2)

    if not has_digits:
        return {
            "num_has_digits": 0.0,
            "num_orto_best": 0.0,
            "num_orto_worst": 0.0,
            "num_orto_spread": 0.0,
            "num_fon_best": 0.0,
            "num_fon_worst": 0.0,
            "num_fon_spread": 0.0,
        }

    pairs: list[tuple[str, str]] = []
    pairs.append((apply_base_normalization(convert_to_literal_words(m1)), base2))
    pairs.append((apply_base_normalization(convert_to_cardinal_words(m1)), base2))
    pairs.append((base1, apply_base_normalization(convert_to_literal_words(m2))))
    pairs.append((base1, apply_base_normalization(convert_to_cardinal_words(m2))))

    ortos: list[float] = []
    fons: list[float] = []
    for x, y in pairs:
        x_strict = clean_strict(x)
        y_strict = clean_strict(y)
        ortos.append(get_levenshtein_similarity(x_strict, y_strict))
        fons.append(get_phonetic_similarity(x, y))

    return {
        "num_has_digits": 1.0,
        "num_orto_best": float(max(ortos)),
        "num_orto_worst": float(min(ortos)),
        "num_orto_spread": float(max(ortos) - min(ortos)),
        "num_fon_best": float(max(fons)),
        "num_fon_worst": float(min(fons)),
        "num_fon_spread": float(max(fons) - min(fons)),
    }


def ofta_features(m1: str, m2: str) -> dict[str, float]:
    """Score OFTA + vetores parciais como features. Rede ve a heuristica como input."""
    a = apply_base_normalization(m1)
    b = apply_base_normalization(m2)
    res: dict[str, Any] = calculate_single_score(a, b)
    vec = res.get("vetores", {})
    driver = res.get("driver", "Geral")

    out = {
        "ofta_final": float(res.get("final", 0.0)),
        "ofta_orto": float(vec.get("orto", 0.0)),
        "ofta_fon": float(vec.get("fon", 0.0)),
        "ofta_token": float(vec.get("token", 0.0)),
        "ofta_anagram": float(vec.get("anagram", 0.0)),
        "ofta_fuzzy": float(vec.get("fuzzy", 0.0)),
    }
    for name in DRIVER_NAMES:
        key = f"ofta_driver_{name.split(' ')[0].lower()}"
        out[key] = 1.0 if driver == name else 0.0
    return out


def nominal_feature_names() -> list[str]:
    """Nomes (em ordem canonica) de todas as features nominativas."""
    sample_a = "exemplo um"
    sample_b = "exemplo dois"
    feats = build_nominal_features(sample_a, sample_b)
    return list(feats.keys())


def build_nominal_features(m1: str, m2: str) -> dict[str, float]:
    """Constroi todas as features nominativas para um par (m1, m2)."""
    a_norm = apply_base_normalization(m1 or "")
    b_norm = apply_base_normalization(m2 or "")

    out: dict[str, float] = {}
    out.update(graphical_features(a_norm, b_norm))
    out.update(phonetic_features(a_norm, b_norm))
    out.update(token_set_metrics(a_norm, b_norm))
    out.update(brand_cross_token_overlap_features(m1 or "", m2 or ""))
    out.update(ofta_token_metrics(a_norm, b_norm))
    out.update(numeral_features(m1 or "", m2 or ""))
    out.update(ofta_features(m1 or "", m2 or ""))
    return out


__all__ = [
    "DRIVER_NAMES",
    "graphical_features",
    "phonetic_features",
    "numeral_features",
    "ofta_features",
    "build_nominal_features",
    "nominal_feature_names",
]
