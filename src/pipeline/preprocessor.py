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

from ..features.brand_embed import (
    BrandEmbedder,
    brand_emb_feature_names,
    brand_emb_features_batch,
)
from ..features.builder import (
    build_features_for_row,
    build_static_feature_names,
    canonical_feature_order,
)
from ..features.class_priors import (
    ClassPairPrior,
    class_pair_prior_feature_names,
)
from ..features.classes import (
    build_class_features_matrix,
    fit_top_k_classes,
)
from ..features.extra_advanced import (
    brand_anchor_features,
    brand_longest_common_substring_features,
    brand_simhash_features,
    extra_advanced_feature_names,
    fon_concat_features,
)
from ..features.extra_nominal import (
    build_extra_nominal_features,
    extra_nominal_feature_names,
)
from ..features.generics import GenericTokenDetector
from ..features.nominal import build_nominal_features, nominal_feature_names
from ..features.spec_decomp import (
    item_match_features,
    item_match_feature_names,
    preprocess_spec_items,
)
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
from ..normalize import apply_base_normalization

# Atalhos para modelos PT-BR especializados (BERTimbau STS-fine-tuned).
BERTIMBAU_BASE_STS = "rufimelo/Legal-BERTimbau-sts-base-ma-v3"
BERTIMBAU_LARGE_STS = "rufimelo/Legal-BERTimbau-sts-large-ma-v3"

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

    # ---- Sprint 1 (novas opcoes) ----
    use_brand_embeddings: bool = True
    brand_embedding_cache_path: str | None = "artifacts/embeddings_brand_cache.parquet"
    brand_embedding_model: str | None = None  # se None, usa o mesmo embedding_model
    generic_df_threshold: float = 0.01        # 1% das marcas
    generic_min_token_len: int = 3

    # ---- Sprint 2 (novas opcoes) ----
    use_extra_advanced: bool = True              # 10 features Sprint 2
    use_item_level_matching: bool = True         # spec_item_* (3 features)
    use_class_pair_prior: bool = True            # cls_pair_* (2 features)
    class_pair_prior_alpha: float = 2.0
    class_pair_prior_beta: float = 8.0
    label_column: str = "label"                  # usado no fit do prior; cai para nome bruto se nao achar

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
            "use_brand_embeddings": self.use_brand_embeddings,
            "brand_embedding_cache_path": self.brand_embedding_cache_path,
            "brand_embedding_model": self.brand_embedding_model,
            "generic_df_threshold": self.generic_df_threshold,
            "generic_min_token_len": self.generic_min_token_len,
            "use_extra_advanced": self.use_extra_advanced,
            "use_item_level_matching": self.use_item_level_matching,
            "use_class_pair_prior": self.use_class_pair_prior,
            "class_pair_prior_alpha": self.class_pair_prior_alpha,
            "class_pair_prior_beta": self.class_pair_prior_beta,
            "label_column": self.label_column,
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
        self.brand_embedder: BrandEmbedder | None = None
        self.generic_detector: GenericTokenDetector | None = None
        self.class_pair_prior: ClassPairPrior | None = None

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

        # Detector de tokens genericos (auto-deteccao via DF)
        all_brands = (
            df["marca_monitorada"].astype(str).tolist()
            + df["marca_colidente"].astype(str).tolist()
        )
        self.generic_detector = GenericTokenDetector(
            df_threshold=cfg.generic_df_threshold,
            min_token_len=cfg.generic_min_token_len,
        ).fit(all_brands)
        logger.info(
            "GenericTokenDetector ajustado: %d tokens genericos auto-detectados de %d marcas",
            len(self.generic_detector.generic_set), len(all_brands),
        )

        # Embedder de marca (separado do de specs para permitir modelo PT-BR especializado)
        if cfg.use_brand_embeddings:
            brand_model = cfg.brand_embedding_model or cfg.embedding_model
            brand_cache = (
                Path(cfg.brand_embedding_cache_path)
                if cfg.brand_embedding_cache_path else None
            )
            self.brand_embedder = BrandEmbedder(
                model_name=brand_model,
                cache_path=str(brand_cache) if brand_cache else "artifacts/embeddings_brand_cache.parquet",
            )
            self.brand_embedder.load_cache()
            # Se for o mesmo modelo de specs, reaproveita a instancia carregada
            if (
                cfg.use_embeddings
                and self.embedding_provider is not None
                and brand_model == cfg.embedding_model
            ):
                try:
                    shared = self.embedding_provider._ensure_model()  # noqa: SLF001
                    self.brand_embedder.set_model(shared)
                    logger.info(
                        "Reaproveitando SentenceTransformer entre specs e marcas (%s).",
                        brand_model,
                    )
                except Exception:  # noqa: BLE001
                    pass

        self.top_classes = fit_top_k_classes(
            df["classe_marca_monitorada"].tolist(),
            df["classe_marca_colidente"].tolist(),
            k=cfg.top_k_classes,
        )

        if cfg.use_class_pair_prior:
            label_col_to_use = None
            for candidate in (cfg.label_column, "label", "Rótulo (1=manter, 0=outro)"):
                if candidate in df.columns:
                    label_col_to_use = candidate
                    break
            if label_col_to_use is not None:
                self.class_pair_prior = ClassPairPrior(
                    alpha=cfg.class_pair_prior_alpha,
                    beta=cfg.class_pair_prior_beta,
                ).fit(
                    df["classe_marca_monitorada"].tolist(),
                    df["classe_marca_colidente"].tolist(),
                    df[label_col_to_use].tolist(),
                )
            else:
                logger.warning(
                    "use_class_pair_prior=True mas coluna '%s' (e fallbacks) ausentes do df de fit; "
                    "ClassPairPrior nao sera ajustado e cls_pair_* ficara em 0.",
                    cfg.label_column,
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

        # Brand embedding cosines (em batch para reaproveitar o modelo)
        if self.brand_embedder is not None:
            ma_norm = [apply_base_normalization(s) for s in ma]
            mb_norm = [apply_base_normalization(s) for s in mb]
            brand_feats = brand_emb_features_batch(
                self.brand_embedder, ma, mb, ma_norm, mb_norm,
            )
            for k, vec in brand_feats.items():
                col = idx.get(k)
                if col is not None:
                    X[:, col] = vec.astype(np.float32, copy=False)
            self.brand_embedder.save_cache()

        # Interactions + Extra-nominal (precisam de cosines ja calculados acima)
        extra_keys = extra_nominal_feature_names()
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

            if self.generic_detector is not None:
                extras = build_extra_nominal_features(
                    ma[i], mb[i], self.generic_detector,
                    classe_a=int(ca[i]), classe_b=int(cb[i]),
                    spec_emb_cosine=spec_block["spec_cosine_emb"],
                )
                for k in extra_keys:
                    col = idx.get(k)
                    if col is not None and k in extras:
                        X[i, col] = extras[k]

        # ----------------------- Sprint 2 features -----------------------
        if self.config.use_extra_advanced:
            self._fill_extra_advanced(X, idx, ma, mb, ca, cb, specs_a, specs_b, df)

        return X

    def _fill_extra_advanced(
        self,
        X: np.ndarray,
        idx: dict[str, int],
        ma: list[str],
        mb: list[str],
        ca: np.ndarray,
        cb: np.ndarray,
        specs_a: list[str],
        specs_b: list[str],
        df: pd.DataFrame,
    ) -> None:
        """Preenche as 10 features Sprint 2 em batch.

        - cls_pair_prior_pos / cls_pair_chi2_strength: vetorizado pelo prior
        - spec_item_*: precisa do tfidf_char + items por linha
        - brand_a_token_in_spec_b / brand_b_token_in_spec_a: row-wise
        - brand_longest_common_substring_norm / fon_concat_metaphone_eq /
          brand_simhash_hamming_norm: row-wise
        """
        n = len(ma)
        cpp = self.class_pair_prior
        if cpp is not None and cpp.fitted:
            cpp_mat = cpp.transform(ca, cb)
            col_p = idx.get("cls_pair_prior_pos")
            col_c = idx.get("cls_pair_chi2_strength")
            if col_p is not None:
                X[:, col_p] = cpp_mat[:, 0]
            if col_c is not None:
                X[:, col_c] = cpp_mat[:, 1]

        cfg_item = self.config.use_item_level_matching and self.tfidf_char is not None
        item_keys = item_match_feature_names()

        spec_a_raw = df["especificacao_monitorado"].astype(str).tolist()
        spec_b_raw = df["especificacao_colidente"].astype(str).tolist()

        for i in range(n):
            if cfg_item:
                a_items = preprocess_spec_items(spec_a_raw[i])
                b_items = preprocess_spec_items(spec_b_raw[i])
                im = item_match_features(a_items, b_items, self.tfidf_char)
                for k in item_keys:
                    col = idx.get(k)
                    if col is not None:
                        X[i, col] = im.get(k, 0.0)

            anchor = brand_anchor_features(
                ma[i], mb[i], specs_a[i], specs_b[i], det=self.generic_detector,
            )
            for k, v in anchor.items():
                col = idx.get(k)
                if col is not None:
                    X[i, col] = v

            lcs = brand_longest_common_substring_features(ma[i], mb[i])
            for k, v in lcs.items():
                col = idx.get(k)
                if col is not None:
                    X[i, col] = v

            fc = fon_concat_features(ma[i], mb[i])
            for k, v in fc.items():
                col = idx.get(k)
                if col is not None:
                    X[i, col] = v

            sh = brand_simhash_features(ma[i], mb[i])
            for k, v in sh.items():
                col = idx.get(k)
                if col is not None:
                    X[i, col] = v

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
        brand_emb_state: dict[str, Any] | None = None
        if self.brand_embedder is not None:
            brand_emb_state = self.brand_embedder.to_state()
        generic_state = (
            self.generic_detector.to_state() if self.generic_detector is not None else None
        )
        cpp_state = (
            self.class_pair_prior.to_state() if self.class_pair_prior is not None else None
        )

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
                "brand_embedder_state": brand_emb_state,
                "generic_detector_state": generic_state,
                "class_pair_prior_state": cpp_state,
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
        brand_emb_state = state.get("brand_embedder_state")
        if brand_emb_state:
            obj.brand_embedder = BrandEmbedder.from_state(brand_emb_state)
            obj.brand_embedder.load_cache()
        gen_state = state.get("generic_detector_state")
        if gen_state:
            obj.generic_detector = GenericTokenDetector.from_state(gen_state)
        cpp_state = state.get("class_pair_prior_state")
        if cpp_state:
            obj.class_pair_prior = ClassPairPrior.from_state(cpp_state)
        obj._fitted = bool(state.get("fitted", True))
        return obj


__all__ = [
    "PreprocessorConfig",
    "FeaturePreprocessor",
]
