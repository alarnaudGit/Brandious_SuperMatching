"""Detector de tokens genericos em marcas + utilitarios.

Tokens "genericos" sao palavras que aparecem com tanta frequencia no corpus
de marcas (ex: BURGER, PIZZA, INSTITUTO, AUTO, MODA, FASHION, BRASIL) que
NAO ajudam a discriminar uma marca da outra. Quando duas marcas
compartilham apenas tokens genericos, a evidencia de colidencia e fraca.

Implementacao:
- Auto-deteccao por document-frequency: qualquer token presente em
  >= `df_threshold` (% das marcas) e classificado como generico.
- O conjunto e *fitado* sobre os tokens de todas as marcas (A e B juntas),
  uma vez por dataset.
- Usa o lookup IDF que ja calculamos em features/specs.py para os
  cosenos TF-IDF, mas agora aplicado a marcas (peso por raridade).

A lista de genericos detectada e exposta atraves de
`GenericTokenDetector.fit(...).generic_set` e e persistida via
preprocessor.save().
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from ..normalize import apply_base_normalization

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

# Stop-words estruturais (preposicoes, artigos, conjuncoes) sao sempre
# genericas, mesmo que apareçam pouco. Sao filtradas antes mesmo da
# auto-deteccao para nao poluir o IDF.
_STRUCT_STOPWORDS = frozenset({
    "a", "o", "as", "os", "e", "da", "de", "do", "das", "dos",
    "no", "na", "nos", "nas", "em", "com", "por", "para", "ou",
    "que", "se", "ao", "aos", "the", "of", "for", "and", "to",
})


def tokenize_brand(s: str) -> list[str]:
    """Tokeniza uma marca (lowercase, sem acentos, alfa-numerico, len>=2)."""
    if not s:
        return []
    norm = apply_base_normalization(s)
    return [t for t in norm.split() if len(t) >= 2 and not t.isdigit()]


def filter_structural(tokens: list[str]) -> list[str]:
    """Remove stop-words estruturais."""
    return [t for t in tokens if t not in _STRUCT_STOPWORDS]


# -----------------------------------------------------------------------------
# Detector
# -----------------------------------------------------------------------------

@dataclass
class GenericTokenDetector:
    """Auto-deteccao de tokens genericos por document-frequency.

    Apos o fit, `generic_set` contem todos os tokens cuja DF >= `df_threshold`
    fracao do total de marcas, e `idf_lookup` da o IDF de cada token (para
    pesos finos).
    """

    df_threshold: float = 0.01            # 1% das marcas
    min_token_len: int = 3                # tokens com >=3 chars
    include_structural: bool = True       # adiciona stop-words estruturais
    extra_generics: set[str] = field(default_factory=set)
    generic_set: set[str] = field(default_factory=set, init=False)
    idf_lookup: dict[str, float] = field(default_factory=dict, init=False)
    df_lookup: dict[str, float] = field(default_factory=dict, init=False)
    n_brands_fitted: int = field(default=0, init=False)

    def fit(self, brands: list[str]) -> "GenericTokenDetector":
        """Aprende os tokens genericos a partir de uma lista de marcas (A + B)."""
        n = len(brands)
        if n == 0:
            return self
        self.n_brands_fitted = n
        df_counter: Counter[str] = Counter()
        for s in brands:
            seen = set(tokenize_brand(str(s)))
            df_counter.update(seen)

        self.df_lookup = {tok: cnt / n for tok, cnt in df_counter.items()}
        self.idf_lookup = {
            tok: math.log((1.0 + n) / (1.0 + cnt)) + 1.0
            for tok, cnt in df_counter.items()
        }

        threshold = max(2, int(self.df_threshold * n))
        auto_generic = {
            tok for tok, cnt in df_counter.items()
            if cnt >= threshold and len(tok) >= self.min_token_len
        }

        self.generic_set = set(auto_generic)
        if self.include_structural:
            self.generic_set.update(_STRUCT_STOPWORDS)
        self.generic_set.update(self.extra_generics)
        return self

    # ---- API publica ----

    def is_generic(self, tok: str) -> bool:
        return tok in self.generic_set

    def filter_generics(self, tokens: list[str]) -> list[str]:
        return [t for t in tokens if t not in self.generic_set]

    def informative_tokens(self, brand: str) -> list[str]:
        """Retorna apenas os tokens NAO genericos da marca (forma normalizada)."""
        return self.filter_generics(tokenize_brand(brand))

    def shared_token_idf_max(self, shared: list[str]) -> float:
        if not shared:
            return 0.0
        return max(self.idf_lookup.get(t, 0.0) for t in shared)

    def shared_token_idf_mean(self, shared: list[str]) -> float:
        if not shared:
            return 0.0
        idfs = [self.idf_lookup.get(t, 0.0) for t in shared]
        return sum(idfs) / len(idfs)

    # ---- Persistencia (manual; FeaturePreprocessor cuida do joblib.dump) ----

    def to_state(self) -> dict:
        return {
            "df_threshold": self.df_threshold,
            "min_token_len": self.min_token_len,
            "include_structural": self.include_structural,
            "extra_generics": list(self.extra_generics),
            "generic_set": list(self.generic_set),
            "idf_lookup": self.idf_lookup,
            "df_lookup": self.df_lookup,
            "n_brands_fitted": self.n_brands_fitted,
        }

    @classmethod
    def from_state(cls, state: dict) -> "GenericTokenDetector":
        det = cls(
            df_threshold=float(state.get("df_threshold", 0.01)),
            min_token_len=int(state.get("min_token_len", 3)),
            include_structural=bool(state.get("include_structural", True)),
            extra_generics=set(state.get("extra_generics", [])),
        )
        det.generic_set = set(state.get("generic_set", []))
        det.idf_lookup = dict(state.get("idf_lookup", {}))
        det.df_lookup = dict(state.get("df_lookup", {}))
        det.n_brands_fitted = int(state.get("n_brands_fitted", 0))
        return det


__all__ = [
    "GenericTokenDetector",
    "tokenize_brand",
    "filter_structural",
]
