"""Preprocessor sklearn-like: ajusta TF-IDF + embedding cache + scaler e gera matriz de features."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ..features.builder import (
    build_features_for_row,
    build_static_feature_names,
    canonical_feature_order,
)
from ..features.classes import (
    build_class_features_matrix,
    fit_top_k_classes,
)
from ..features.nominal import build_nominal_features, nominal_feature_names
from ..features.specs import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingProvider,
    activity_match_score,
    lexical_features,
    preprocess_spec,
    preprocess_specs_batch,
    row_cosine,
)
from ..features.interactions import build_interaction_features

logger = logging.getLogger(__name__)


@dataclass
class PreprocessorConfig:
    tfidf_word_min_df: int = 3
    tfidf_word_max_features: int = 20_000
    tfidf_word_ngram_range: tuple[int, int] = (1, 2)

    tfidf_char_min_df: int = 3
    tfidf_char_max_features: int = 10_000
    tfidf_char_ngram_range: tuple[int, int] = (3, 5)

    top_k_classes: int = 15
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    use_embeddings: bool = True
    embedding_cache_path: str | None = "artifacts/embeddings_cache.parquet"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tfidf_word_min_df": self.tfidf_word_min_df,
            "tfidf_word_max_features": self.tfidf_word_max_features,
            "tfidf_word_ngram_range": list(self.tfidf_word_ngram_range),
            "tfidf_char_min_df": self.tfidf_char_min_df,
            "tfidf_char_max_features": self.tfidf_char_max_features,
            "tfidf_char_ngram_range": list(self.tfidf_char_ngram_range),
            "top_k_classes": self.top_k_classes,
            "embedding_model": self.embedding_model,
            "use_embeddings": self.use_embeddings,
            "embedding_cache_path": self.embedding_cache_path,
        }


class FeaturePreprocessor:
    """Mantem estado (vectorizers, scaler, classes top-K, idf) entre fit/transform.

    Apos `fit`, `transform` produz uma matriz `(N, F)` na ordem canonica de
    `feature_names_ordered`. O scaler e aplicado em uma copia para que features
    binarias permanecam binarias (`StandardScaler` no conjunto todo aceita; se
    desejar, pode-se desligar com `scale=False`).
    """

    def __init__(self, config: PreprocessorConfig | None = None) -> None:
        self.config = config or PreprocessorConfig()

        self.tfidf_word: TfidfVectorizer | None = None
        self.tfidf_char: TfidfVectorizer | None = None
        self.scaler: StandardScaler | None = None
        self.embedding_provider: EmbeddingProvider | None = None

        self.top_classes: list[int] = []
        self.idf_word_lookup: dict[str, float] = {}
        self.feature_names_ordered: list[str] = []

        self._fitted = False

    # ------------------------------------------------------------------ FIT
    def fit(self, df: pd.DataFrame) -> "FeaturePreprocessor":
        """Ajusta vectorizers e scaler no dataset."""
        cfg = self.config
        logger.info("Fit do preprocessor em %d linhas...", len(df))

        logger.info("Pre-processando especificacoes (stemming PT-BR)...")
        specs_a = preprocess_specs_batch(df["especificacao_monitorado"].tolist())
        specs_b = preprocess_specs_batch(df["especificacao_colidente"].tolist())
        all_specs = specs_a + specs_b

        logger.info("Ajustando TF-IDF word (1,2)...")
        self.tfidf_word = TfidfVectorizer(
            ngram_range=cfg.tfidf_word_ngram_range,
            min_df=cfg.tfidf_word_min_df,
            max_features=cfg.tfidf_word_max_features,
            sublinear_tf=True,
        )
        self.tfidf_word.fit(all_specs)

        self.idf_word_lookup = {
            term: float(idf)
            for term, idf in zip(self.tfidf_word.get_feature_names_out(), self.tfidf_word.idf_)
            if " " not in term
        }

        logger.info("Ajustando TF-IDF char (3,5)...")
        self.tfidf_char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=cfg.tfidf_char_ngram_range,
            min_df=cfg.tfidf_char_min_df,
            max_features=cfg.tfidf_char_max_features,
            sublinear_tf=True,
        )
        self.tfidf_char.fit(all_specs)

        if cfg.use_embeddings:
            cache_path = Path(cfg.embedding_cache_path) if cfg.embedding_cache_path else None
            self.embedding_provider = EmbeddingProvider(
                model_name=cfg.embedding_model,
                cache_path=cache_path,
            )

        self.top_classes = fit_top_k_classes(
            df["classe_marca_monitorada"].tolist(),
            df["classe_marca_colidente"].tolist(),
            k=cfg.top_k_classes,
        )

        self.feature_names_ordered = canonical_feature_order(self.top_classes)

        X = self._compute_matrix(df, specs_a, specs_b, fitting=True)
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.scaler.fit(X)

        self._fitted = True
        logger.info("Fit concluido. %d features na ordem canonica.", X.shape[1])
        return self

    # ------------------------------------------------------------- TRANSFORM
    def transform(
        self, df: pd.DataFrame, scale: bool = True, show_progress: bool = False
    ) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("FeaturePreprocessor.transform chamado antes de fit().")

        if show_progress:
            logger.info("Pre-processando especificacoes (transform)...")
        specs_a = preprocess_specs_batch(df["especificacao_monitorado"].tolist())
        specs_b = preprocess_specs_batch(df["especificacao_colidente"].tolist())

        X = self._compute_matrix(df, specs_a, specs_b, fitting=False, show_progress=show_progress)
        if scale and self.scaler is not None:
            X = self.scaler.transform(X)
        return X.astype(np.float32, copy=False)

    def fit_transform(self, df: pd.DataFrame, scale: bool = True) -> np.ndarray:
        self.fit(df)
        return self.transform(df, scale=scale)

    # -------------------------------------------------------------- INTERNALS
    def _compute_matrix(
        self,
        df: pd.DataFrame,
        specs_a: list[str],
        specs_b: list[str],
        fitting: bool,
        show_progress: bool = False,
    ) -> np.ndarray:
        n = len(df)
        feat_names = self.feature_names_ordered
        if not feat_names:
            feat_names = canonical_feature_order(self.top_classes)
            self.feature_names_ordered = feat_names

        X = np.zeros((n, len(feat_names)), dtype=np.float32)
        idx = {name: i for i, name in enumerate(feat_names)}

        ma = df["marca_monitorada"].astype(str).tolist()
        mb = df["marca_colidente"].astype(str).tolist()
        ca = df["classe_marca_monitorada"].astype(int).to_numpy()
        cb = df["classe_marca_colidente"].astype(int).to_numpy()

        nominal_keys = nominal_feature_names()

        iterator = range(n)
        if show_progress:
            iterator = tqdm(iterator, desc="Features nominal+lex", total=n)

        for i in iterator:
            nominal = build_nominal_features(ma[i], mb[i])
            for k in nominal_keys:
                col = idx.get(k)
                if col is not None:
                    X[i, col] = nominal[k]

            spec_lex = lexical_features(specs_a[i], specs_b[i], self.idf_word_lookup)
            for k, v in spec_lex.items():
                col = idx.get(k)
                if col is not None:
                    X[i, col] = v

            spec_act = activity_match_score(specs_a[i], specs_b[i])
            for k, v in spec_act.items():
                col = idx.get(k)
                if col is not None:
                    X[i, col] = v

        if self.tfidf_word is not None:
            wa = self.tfidf_word.transform(specs_a)
            wb = self.tfidf_word.transform(specs_b)
            cos_word = row_cosine(wa, wb)
            X[:, idx["spec_cosine_tfidf_word"]] = cos_word

        if self.tfidf_char is not None:
            ca_mat = self.tfidf_char.transform(specs_a)
            cb_mat = self.tfidf_char.transform(specs_b)
            cos_char = row_cosine(ca_mat, cb_mat)
            X[:, idx["spec_cosine_tfidf_char"]] = cos_char

        if self.embedding_provider is not None:
            unique_specs = list(dict.fromkeys(specs_a + specs_b))
            if show_progress:
                logger.info("Computando embeddings para %d textos unicos...", len(unique_specs))
            self.embedding_provider.encode(unique_specs)
            emb_a = self.embedding_provider.encode(specs_a)
            emb_b = self.embedding_provider.encode(specs_b)
            cos_emb = row_cosine(emb_a, emb_b)
            X[:, idx["spec_cosine_emb"]] = cos_emb
            self.embedding_provider.save_cache()

        cls_mat, cls_names = build_class_features_matrix(ca, cb, self.top_classes)
        for j, name in enumerate(cls_names):
            col = idx.get(name)
            if col is not None:
                X[:, col] = cls_mat[:, j]

        for i in range(n):
            nominal_block = {k: float(X[i, idx[k]]) for k in nominal_keys if k in idx}
            spec_block = {
                "spec_cosine_emb": float(X[i, idx["spec_cosine_emb"]]),
                "spec_cosine_tfidf_word": float(X[i, idx["spec_cosine_tfidf_word"]]),
                "spec_cosine_tfidf_char": float(X[i, idx["spec_cosine_tfidf_char"]]),
            }
            cls_block = {"cls_same": float(X[i, idx["cls_same"]])}

            inter = build_interaction_features(nominal_block, spec_block, cls_block)
            for k, v in inter.items():
                col = idx.get(k)
                if col is not None:
                    X[i, col] = v

        return X

    # ----------------------------------------------------------- PERSISTENCE
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        emb_state: dict[str, Any] | None = None
        if self.embedding_provider is not None:
            emb_state = {
                "model_name": self.embedding_provider.model_name,
                "cache_path": str(self.embedding_provider.cache_path) if self.embedding_provider.cache_path else None,
            }

        joblib.dump(
            {
                "config": self.config.to_dict(),
                "tfidf_word": self.tfidf_word,
                "tfidf_char": self.tfidf_char,
                "scaler": self.scaler,
                "top_classes": self.top_classes,
                "idf_word_lookup": self.idf_word_lookup,
                "feature_names_ordered": self.feature_names_ordered,
                "embedding_provider_state": emb_state,
                "fitted": self._fitted,
            },
            path,
            compress=3,
        )
        logger.info("Preprocessor salvo em %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "FeaturePreprocessor":
        path = Path(path)
        state = joblib.load(path)
        cfg = PreprocessorConfig(**{
            **PreprocessorConfig().to_dict(),
            **{
                k: tuple(v) if k.endswith("ngram_range") else v
                for k, v in state["config"].items()
                if k in PreprocessorConfig().to_dict()
            },
        })
        obj = cls(cfg)
        obj.tfidf_word = state.get("tfidf_word")
        obj.tfidf_char = state.get("tfidf_char")
        obj.scaler = state.get("scaler")
        obj.top_classes = list(state.get("top_classes", []))
        obj.idf_word_lookup = dict(state.get("idf_word_lookup", {}))
        obj.feature_names_ordered = list(state.get("feature_names_ordered", []))
        emb_state = state.get("embedding_provider_state")
        if emb_state:
            cache = emb_state.get("cache_path")
            obj.embedding_provider = EmbeddingProvider(
                model_name=emb_state.get("model_name", DEFAULT_EMBEDDING_MODEL),
                cache_path=Path(cache) if cache else None,
            )
        obj._fitted = bool(state.get("fitted", True))
        return obj


__all__ = [
    "PreprocessorConfig",
    "FeaturePreprocessor",
]
