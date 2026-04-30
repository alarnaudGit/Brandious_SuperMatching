"""Metricas de tokens entre as duas marcas (jaccard, overlap, fuzzy, lcs, etc.)."""
from __future__ import annotations

from typing import Iterable

from Levenshtein import distance as lev_distance

from ..normalize import (
    apply_base_normalization,
    get_phonetic_key,
    get_token_metrics,
)


def _tokens(s: str) -> list[str]:
    return [t for t in apply_base_normalization(s).split() if t]


def lcs_length(a: str, b: str) -> int:
    """Comprimento da maior subsequencia comum entre duas strings (DP O(n*m))."""
    if not a or not b:
        return 0
    n, m = len(a), len(b)
    if n > m:
        a, b = b, a
        n, m = m, n
    prev = [0] * (n + 1)
    cur = [0] * (n + 1)
    for j in range(1, m + 1):
        bj = b[j - 1]
        for i in range(1, n + 1):
            if a[i - 1] == bj:
                cur[i] = prev[i - 1] + 1
            else:
                cur[i] = cur[i - 1] if cur[i - 1] >= prev[i] else prev[i]
        prev, cur = cur, prev
    return prev[n]


def common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def common_suffix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(1, n + 1):
        if a[-i] != b[-i]:
            return i - 1
    return n


def char_ngrams(s: str, n: int) -> set[str]:
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def jaccard_sets(a: Iterable, b: Iterable) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


def overlap_sets(a: Iterable, b: Iterable) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


def token_set_metrics(s1: str, s2: str) -> dict[str, float]:
    """Conta tokens em comum/exclusivos e diferenca em quantidade."""
    t1 = _tokens(s1)
    t2 = _tokens(s2)
    set1, set2 = set(t1), set(t2)
    inter = set1 & set2

    return {
        "n_tokens_a": float(len(t1)),
        "n_tokens_b": float(len(t2)),
        "n_tokens_diff": float(abs(len(t1) - len(t2))),
        "n_tokens_common": float(len(inter)),
        "n_tokens_excl_a": float(len(set1 - set2)),
        "n_tokens_excl_b": float(len(set2 - set1)),
    }


def avg_max_phonetic_token(s1: str, s2: str) -> dict[str, float]:
    """Media e maximo da chave fonetica por token (cobertura por melhor matching)."""
    t1 = _tokens(s1)
    t2 = _tokens(s2)
    if not t1 or not t2:
        return {"phon_token_mean": 0.0, "phon_token_max": 0.0, "phon_token_eq_share": 0.0}

    keys1 = [get_phonetic_key(t) for t in t1]
    keys2 = [get_phonetic_key(t) for t in t2]

    best_per_t1: list[float] = []
    eq_count = 0
    for k1 in keys1:
        best = 0.0
        for k2 in keys2:
            if k1 == k2 and len(k1) > 0:
                best = 1.0
                eq_count += 1
                break
            if not k1 or not k2:
                continue
            d = lev_distance(k1, k2)
            sim = 1.0 - d / max(len(k1), len(k2))
            if sim > best:
                best = sim
        best_per_t1.append(best)

    return {
        "phon_token_mean": float(sum(best_per_t1) / len(best_per_t1)),
        "phon_token_max": float(max(best_per_t1)),
        "phon_token_eq_share": float(eq_count / len(t1)),
    }


def ofta_token_metrics(s1: str, s2: str) -> dict[str, float]:
    """Wrap de get_token_metrics do OFTA (jaccard/overlap/fuzzy)."""
    s1n = apply_base_normalization(s1)
    s2n = apply_base_normalization(s2)
    m = get_token_metrics(s1n, s2n)
    return {
        "tok_jaccard": float(m["jaccard"]),
        "tok_overlap": float(m["overlap"]),
        "tok_fuzzy": float(m["fuzzy"]),
    }


__all__ = [
    "lcs_length",
    "common_prefix_len",
    "common_suffix_len",
    "char_ngrams",
    "jaccard_sets",
    "overlap_sets",
    "token_set_metrics",
    "avg_max_phonetic_token",
    "ofta_token_metrics",
]
