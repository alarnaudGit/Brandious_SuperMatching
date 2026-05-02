"""Overlap explicito entre tokens das duas marcas (palavra em comum e substring cruzada).

Complementa `token_set_metrics` / `tok_*`: cobre casos em que nao ha token identico
na intersecao, mas um token de um lado e' substring de um token do outro (ex.: GMG vs GMGG).
"""
from __future__ import annotations


def _tokens_normalized(s1: str, s2: str) -> tuple[list[str], list[str]]:
    from ..normalize import apply_base_normalization

    a = apply_base_normalization(s1 or "")
    b = apply_base_normalization(s2 or "")
    t1 = [t for t in a.split() if t]
    t2 = [t for t in b.split() if t]
    return t1, t2


def _longest_common_substring_len_contiguous(a: str, b: str) -> int:
    """Comprimento da maior substring contigua comum entre a e b."""
    if not a or not b:
        return 0
    la, lb = len(a), len(b)
    best = 0
    dp = [0] * (lb + 1)
    for i in range(1, la + 1):
        prev = 0
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cur = dp[j]
            if ai == b[j - 1]:
                dp[j] = prev + 1
                if dp[j] > best:
                    best = dp[j]
            else:
                dp[j] = 0
            prev = cur
    return best


def brand_cross_token_overlap_features(m1: str, m2: str) -> dict[str, float]:
    """Features numericas para overlap marca-marca."""
    t1, t2 = _tokens_normalized(m1, m2)
    set1, set2 = set(t1), set(t2)
    inter = set1 & set2

    exact_any = 1.0 if len(inter) > 0 else 0.0
    exact_count = float(len(inter))

    best_len = 0
    best_norm = 0.0
    best_cov_min = 0.0
    for ta in t1:
        for tb in t2:
            ell = _longest_common_substring_len_contiguous(ta, tb)
            if ell <= 0:
                continue
            denom = max(len(ta), len(tb), 1)
            cov_a = ell / max(len(ta), 1)
            cov_b = ell / max(len(tb), 1)
            cov_min = min(cov_a, cov_b)
            norm = ell / float(denom)
            if ell > best_len or (
                ell == best_len and cov_min > best_cov_min
            ):
                best_len = ell
                best_norm = float(norm)
                best_cov_min = float(cov_min)

    return {
        "brand_word_exact_overlap_any": float(exact_any),
        "brand_word_exact_overlap_count": float(exact_count),
        "brand_cross_token_substring_len": float(best_len),
        "brand_cross_token_substring_norm": float(best_norm),
        "brand_cross_token_substring_cov_min": float(best_cov_min),
    }


def brand_cross_token_overlap_feature_names() -> list[str]:
    return [
        "brand_word_exact_overlap_any",
        "brand_word_exact_overlap_count",
        "brand_cross_token_substring_len",
        "brand_cross_token_substring_norm",
        "brand_cross_token_substring_cov_min",
    ]


__all__ = [
    "brand_cross_token_overlap_features",
    "brand_cross_token_overlap_feature_names",
]
