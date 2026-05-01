"""Features novas Sprint 1: name_generic_*, contain_*, radical_*, lev_pure_*.

Cada gerador depende do GenericTokenDetector (que precisa estar fitado).
Saida combinada por `build_extra_nominal_features(...)`.
"""
from __future__ import annotations

from typing import Mapping

from Levenshtein import distance as lev_distance
from jellyfish import jaro_winkler_similarity

from ..normalize import apply_base_normalization, get_phonetic_key
from .generics import GenericTokenDetector, tokenize_brand


# =============================================================================
# Helpers
# =============================================================================

def _norm_concat(s: str) -> str:
    """Forma normalizada compactada (sem espacos) - usado em substring."""
    return apply_base_normalization(s).replace(" ", "")


def _common_prefix_len(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(a, b):
        if ca == cb:
            n += 1
        else:
            break
    return n


def _norm_lev(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return 1.0 - lev_distance(a, b) / max(len(a), len(b))


# =============================================================================
# 4.1 name_generic_* (10 features)
# =============================================================================

def name_generic_features(
    a: str, b: str, det: GenericTokenDetector,
) -> dict[str, float]:
    tok_a = tokenize_brand(a)
    tok_b = tokenize_brand(b)
    n_a, n_b = len(tok_a), len(tok_b)

    set_a, set_b = set(tok_a), set(tok_b)
    shared = set_a & set_b

    informative_a = det.filter_generics(tok_a)
    informative_b = det.filter_generics(tok_b)
    set_inf_a, set_inf_b = set(informative_a), set(informative_b)
    shared_inf = set_inf_a & set_inf_b

    union_inf = set_inf_a | set_inf_b
    jacc_inf = len(shared_inf) / max(1, len(union_inf))

    n_gen_a = sum(1 for t in tok_a if det.is_generic(t))
    n_gen_b = sum(1 for t in tok_b if det.is_generic(t))
    share_gen_a = n_gen_a / max(1, n_a)
    share_gen_b = n_gen_b / max(1, n_b)

    shared_idf_max = det.shared_token_idf_max(list(shared))
    shared_idf_mean = det.shared_token_idf_mean(list(shared))

    only_generics = float(bool(shared) and all(det.is_generic(t) for t in shared))

    return {
        "name_generic_share_a": share_gen_a,
        "name_generic_share_b": share_gen_b,
        "name_generic_share_max": max(share_gen_a, share_gen_b),
        "name_unique_token_a": float(len(set_inf_a)),
        "name_unique_token_b": float(len(set_inf_b)),
        "shared_unique_count": float(len(shared_inf)),
        "shared_unique_jaccard": jacc_inf,
        "shared_token_idf_max": shared_idf_max,
        "shared_token_idf_mean": shared_idf_mean,
        "shared_only_generics": only_generics,
    }


def name_generic_feature_names() -> list[str]:
    return [
        "name_generic_share_a", "name_generic_share_b", "name_generic_share_max",
        "name_unique_token_a", "name_unique_token_b",
        "shared_unique_count", "shared_unique_jaccard",
        "shared_token_idf_max", "shared_token_idf_mean",
        "shared_only_generics",
    ]


# =============================================================================
# 4.2 contain_* (6 features)
# =============================================================================

def containment_features(
    a: str, b: str, det: GenericTokenDetector,
) -> dict[str, float]:
    a_norm = _norm_concat(a)
    b_norm = _norm_concat(b)

    contain_a_in_b = float(bool(a_norm) and a_norm in b_norm)
    contain_b_in_a = float(bool(b_norm) and b_norm in a_norm)

    inf_a = det.informative_tokens(a)
    inf_b = det.informative_tokens(b)
    a_strip = "".join(inf_a)
    b_strip = "".join(inf_b)
    contain_after_strip_a = float(bool(a_strip) and bool(b_strip) and a_strip in b_strip)
    contain_after_strip_b = float(bool(b_strip) and bool(a_strip) and b_strip in a_strip)

    radical_share = _common_prefix_len(a_norm, b_norm)
    radical_share_norm = radical_share / max(1, min(len(a_norm), len(b_norm)))

    return {
        "contain_a_in_b": contain_a_in_b,
        "contain_b_in_a": contain_b_in_a,
        "contain_after_strip_a_in_b": contain_after_strip_a,
        "contain_after_strip_b_in_a": contain_after_strip_b,
        "contain_radical_share": float(radical_share),
        "contain_radical_share_norm": float(radical_share_norm),
    }


def containment_feature_names() -> list[str]:
    return [
        "contain_a_in_b", "contain_b_in_a",
        "contain_after_strip_a_in_b", "contain_after_strip_b_in_a",
        "contain_radical_share", "contain_radical_share_norm",
    ]


# =============================================================================
# 4.3 radical_* (4 features)
# =============================================================================

def _largest_informative_token(brand: str, det: GenericTokenDetector) -> str:
    """Token MAIOR e nao-generico, com len >= 4. Vazio se nao houver."""
    inf = det.informative_tokens(brand)
    cand = [t for t in inf if len(t) >= 4]
    if not cand:
        # fallback: maior token nao-generico mesmo que pequeno
        cand = inf or tokenize_brand(brand)
    if not cand:
        return ""
    return max(cand, key=len)


def radical_features(
    a: str, b: str, det: GenericTokenDetector,
) -> dict[str, float]:
    rad_a = _largest_informative_token(a, det)
    rad_b = _largest_informative_token(b, det)

    if not rad_a or not rad_b:
        return {
            "radical_a_len": float(len(rad_a)),
            "radical_b_len": float(len(rad_b)),
            "radical_lev_sim": 0.0,
            "radical_phonetic_eq": 0.0,
        }

    lev_sim = _norm_lev(rad_a, rad_b)
    jw = jaro_winkler_similarity(rad_a, rad_b)
    phonetic_eq = float(get_phonetic_key(rad_a) == get_phonetic_key(rad_b))

    # combinamos lev e jaro-winkler em uma so feature de similaridade radical
    radical_sim = max(lev_sim, jw)

    return {
        "radical_a_len": float(len(rad_a)),
        "radical_b_len": float(len(rad_b)),
        "radical_lev_sim": radical_sim,
        "radical_phonetic_eq": phonetic_eq,
    }


def radical_feature_names() -> list[str]:
    return [
        "radical_a_len", "radical_b_len",
        "radical_lev_sim", "radical_phonetic_eq",
    ]


# =============================================================================
# 4.4 lev_pure_* (3 features) - Levenshtein apos remover genericos
# =============================================================================

def lev_pure_features(
    a: str, b: str, det: GenericTokenDetector,
) -> dict[str, float]:
    a_pure = " ".join(det.informative_tokens(a))
    b_pure = " ".join(det.informative_tokens(b))

    if not a_pure or not b_pure:
        return {
            "lev_pure_norm": 0.0,
            "lev_pure_jaro_winkler": 0.0,
            "lev_pure_size_diff_abs": float(abs(len(a_pure) - len(b_pure))),
        }

    lev_sim = _norm_lev(a_pure, b_pure)
    jw = jaro_winkler_similarity(a_pure, b_pure)

    return {
        "lev_pure_norm": lev_sim,
        "lev_pure_jaro_winkler": jw,
        "lev_pure_size_diff_abs": float(abs(len(a_pure) - len(b_pure))),
    }


def lev_pure_feature_names() -> list[str]:
    return [
        "lev_pure_norm", "lev_pure_jaro_winkler", "lev_pure_size_diff_abs",
    ]


# =============================================================================
# 4.5 against_* (5 features) - evidencia contra colidencia
# =============================================================================

def against_features(
    a: str,
    b: str,
    det: GenericTokenDetector,
    *,
    classe_a: int,
    classe_b: int,
    spec_emb_cosine: float,
) -> dict[str, float]:
    inf_a = det.informative_tokens(a)
    inf_b = det.informative_tokens(b)
    set_inf_a, set_inf_b = set(inf_a), set(inf_b)
    shared_inf = set_inf_a & set_inf_b

    cls_diff = float(classe_a != classe_b and classe_a >= 0 and classe_b >= 0)

    against_distinct_market = float(cls_diff and spec_emb_cosine < 0.30)

    tok_a = tokenize_brand(a)
    tok_b = tokenize_brand(b)
    set_a, set_b = set(tok_a), set(tok_b)
    shared = set_a & set_b
    only_generic_overlap = float(
        bool(shared)
        and all(det.is_generic(t) for t in shared)
        and cls_diff
    )

    diff_len = abs(len(_norm_concat(a)) - len(_norm_concat(b)))
    against_size_disparity = float(diff_len >= 15 and len(shared) == 0)

    excl_a = set_inf_a - set_inf_b
    excl_b = set_inf_b - set_inf_a

    def _strong_count(toks: set[str]) -> float:
        cnt = 0
        for t in toks:
            if len(t) < 4:
                continue
            idf = det.idf_lookup.get(t, 0.0)
            if idf >= 3.0:
                cnt += 1
        return float(cnt)

    return {
        "against_distinct_market": against_distinct_market,
        "against_only_generic_overlap": only_generic_overlap,
        "against_size_disparity": against_size_disparity,
        "against_unique_strong_a": _strong_count(excl_a),
        "against_unique_strong_b": _strong_count(excl_b),
    }


def against_feature_names() -> list[str]:
    return [
        "against_distinct_market", "against_only_generic_overlap",
        "against_size_disparity",
        "against_unique_strong_a", "against_unique_strong_b",
    ]


# =============================================================================
# Orquestrador
# =============================================================================

def build_extra_nominal_features(
    a: str,
    b: str,
    det: GenericTokenDetector,
    *,
    classe_a: int,
    classe_b: int,
    spec_emb_cosine: float,
) -> dict[str, float]:
    out: dict[str, float] = {}
    out.update(name_generic_features(a, b, det))
    out.update(containment_features(a, b, det))
    out.update(radical_features(a, b, det))
    out.update(lev_pure_features(a, b, det))
    out.update(against_features(
        a, b, det,
        classe_a=classe_a, classe_b=classe_b,
        spec_emb_cosine=spec_emb_cosine,
    ))
    return out


def extra_nominal_feature_names() -> list[str]:
    return (
        name_generic_feature_names()
        + containment_feature_names()
        + radical_feature_names()
        + lev_pure_feature_names()
        + against_feature_names()
    )


__all__ = [
    "build_extra_nominal_features",
    "extra_nominal_feature_names",
    "name_generic_features",
    "name_generic_feature_names",
    "containment_features",
    "containment_feature_names",
    "radical_features",
    "radical_feature_names",
    "lev_pure_features",
    "lev_pure_feature_names",
    "against_features",
    "against_feature_names",
]
