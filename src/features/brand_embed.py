"""Embedding semantico das *marcas* (nao das specs).

Para marcas que tem palavras de dicionario (`SOLAR ENERGIA`, `AGRO FRONTEIRA`,
`BELLA DAMA`), o embedding semantico das marcas captura proximidade
conceitual mesmo sem tokens em comum, e permite descontar similaridades
graficas espurias.

Padrao: usa o mesmo modelo de specs (multilingue) para reaproveitar o
download / GPU. Quando `model_name` esta configurado para um modelo PT-BR
especializado (BERTimbau-large STS), oferece sinal mais forte.

Cache em parquet (separado do cache de specs).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BrandEmbedder:
    """Embedder dedicado para marcas (textos curtos).

    Reutiliza um SentenceTransformer ja carregado (passe via `set_model`)
    OU carrega um proprio (mais memoria mas independente de specs).
    """

    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    cache_path: str = "artifacts/embeddings_brand_cache.parquet"
    batch_size: int = 64
    max_chars: int = 200
    _model: object | None = field(default=None, init=False, repr=False)
    _cache: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _dim: int | None = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------ public
    def set_model(self, model: object) -> None:
        """Permite injetar um SentenceTransformer ja carregado (compartilha com specs)."""
        self._model = model

    def _ensure_model(self) -> object:
        if self._model is not None:
            return self._model
        from sentence_transformers import SentenceTransformer
        logger.info("Carregando SentenceTransformer para marcas: %s", self.model_name)
        self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            m = self._ensure_model()
            self._dim = int(m.get_sentence_embedding_dimension())
        return self._dim

    def load_cache(self) -> None:
        path = Path(self.cache_path)
        if not path.exists():
            return
        try:
            df = pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Falha lendo cache brand %s: %s", path, exc)
            return
        if "text" in df.columns and "embedding" in df.columns:
            for _, row in df.iterrows():
                self._cache[row["text"]] = np.asarray(row["embedding"], dtype=np.float32)
            logger.info(
                "Cache brand carregado: %d entradas (%s)", len(self._cache), path
            )

    def save_cache(self) -> None:
        if not self._cache:
            return
        rows = [
            {"text": t, "embedding": np.asarray(v, dtype=np.float32).tolist()}
            for t, v in self._cache.items()
        ]
        df = pd.DataFrame(rows)
        Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.cache_path, index=False)
        logger.info("Cache brand salvo (%d entradas)", len(self._cache))

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        texts = [str(t)[: self.max_chars] for t in texts]
        out = np.zeros((len(texts), self.dim), dtype=np.float32)

        missing_idx: list[int] = []
        missing_text: list[str] = []
        for i, t in enumerate(texts):
            cached = self._cache.get(t)
            if cached is not None:
                out[i] = cached
            else:
                missing_idx.append(i)
                missing_text.append(t)

        if missing_text:
            model = self._ensure_model()
            new_emb = model.encode(
                missing_text,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=False,
                convert_to_numpy=True,
            ).astype(np.float32)
            for i, t, v in zip(missing_idx, missing_text, new_emb):
                self._cache[t] = v
                out[i] = v

        return out

    def to_state(self) -> dict:
        return {
            "model_name": self.model_name,
            "cache_path": self.cache_path,
            "batch_size": self.batch_size,
            "max_chars": self.max_chars,
        }

    @classmethod
    def from_state(cls, state: dict) -> "BrandEmbedder":
        return cls(**state)


# -----------------------------------------------------------------------------
# Features brand_emb_*
# -----------------------------------------------------------------------------

def _safe_cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def brand_emb_features_batch(
    embedder: BrandEmbedder,
    brands_a: list[str],
    brands_b: list[str],
    brands_a_norm: list[str],
    brands_b_norm: list[str],
) -> dict[str, np.ndarray]:
    """Calcula em batch: brand_emb_cosine, brand_emb_cosine_normalized, brand_emb_norm_max.

    Retorna dict com 3 vetores de comprimento len(brands_a).
    """
    n = len(brands_a)
    emb_a = embedder.encode(brands_a)
    emb_b = embedder.encode(brands_b)
    emb_an = embedder.encode(brands_a_norm)
    emb_bn = embedder.encode(brands_b_norm)

    cos_raw = np.zeros(n, dtype=np.float32)
    cos_norm = np.zeros(n, dtype=np.float32)
    norms_max = np.zeros(n, dtype=np.float32)

    for i in range(n):
        cos_raw[i] = _safe_cosine(emb_a[i], emb_b[i])
        cos_norm[i] = _safe_cosine(emb_an[i], emb_bn[i])
        na = float(np.linalg.norm(emb_a[i]))
        nb = float(np.linalg.norm(emb_b[i]))
        norms_max[i] = max(na, nb)

    return {
        "brand_emb_cosine": cos_raw,
        "brand_emb_cosine_normalized": cos_norm,
        "brand_emb_norm_max": norms_max,
    }


def brand_emb_feature_names() -> list[str]:
    return [
        "brand_emb_cosine",
        "brand_emb_cosine_normalized",
        "brand_emb_norm_max",
    ]


__all__ = [
    "BrandEmbedder",
    "brand_emb_features_batch",
    "brand_emb_feature_names",
]
