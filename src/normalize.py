"""Normalizacao PT-BR para marcas - reaproveita SimilarityOFTA quando possivel.

Centraliza acesso as funcoes de [SimilarityOFTA.py](../SimilarityOFTA.py) para que o
restante do pipeline nao precise importar do top-level do projeto e para que possamos
adicionar variantes auxiliares (ex.: vias literal/cardinal pre-aplicadas).
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Reexports do nucleo OFTA
from SimilarityOFTA import (  # noqa: E402
    apply_base_normalization,
    calcular_score_complexo,
    calcular_similaridade_ofta,
    calculate_single_score,
    clean_strict,
    convert_to_cardinal_words,
    convert_to_literal_words,
    get_levenshtein_similarity,
    get_phonetic_key,
    get_phonetic_similarity,
    get_sorted_char_similarity,
    get_token_metrics,
    number_to_words_br,
    remove_repeated_chars,
)


@lru_cache(maxsize=200_000)
def normalize_brand(s: str) -> str:
    """Normalizacao base para uma marca (lower, sem acento, sem especiais)."""
    if s is None:
        return ""
    return apply_base_normalization(str(s))


@lru_cache(maxsize=200_000)
def normalize_brand_strict(s: str) -> str:
    """Versao estrita (sem espacos, sem repeticao de char) - util p/ Levenshtein/anagrama."""
    if s is None:
        return ""
    return clean_strict(str(s))


@lru_cache(maxsize=100_000)
def variants_literal_cardinal(s: str) -> tuple[str, str]:
    """Devolve (literal_norm, cardinal_norm) para que features de numerais sejam simetricas."""
    if s is None:
        return ("", "")
    s = str(s)
    lit = apply_base_normalization(convert_to_literal_words(s))
    car = apply_base_normalization(convert_to_cardinal_words(s))
    return lit, car


__all__ = [
    "apply_base_normalization",
    "calcular_score_complexo",
    "calcular_similaridade_ofta",
    "calculate_single_score",
    "clean_strict",
    "convert_to_cardinal_words",
    "convert_to_literal_words",
    "get_levenshtein_similarity",
    "get_phonetic_key",
    "get_phonetic_similarity",
    "get_sorted_char_similarity",
    "get_token_metrics",
    "number_to_words_br",
    "remove_repeated_chars",
    "normalize_brand",
    "normalize_brand_strict",
    "variants_literal_cardinal",
]
