"""Features das especificacoes INPI (atividades/produtos/servicos).

Combina:
- Pre-processamento PT-BR (lowercase, unaccent, split por ;, stopwords, RSLP stem).
- TF-IDF word n-grams (1,2) e char n-grams (3,5) - aprendidos no fit do preprocessor.
- Embeddings semanticos (sentence-transformers, multilingual MiniLM) com cache em parquet.
- Features lexicais derivadas (cosine, jaccard, overlap, termos raros em comum, etc.).
- Heuristica produto-vs-servico (ancora binaria, sem peso manual no score final).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from unidecode import unidecode

logger = logging.getLogger(__name__)

def _preferred_device() -> str:
    """Escolhe device para SentenceTransformer (GPU se disponível)."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# 1. Pre-processamento PT-BR
# ---------------------------------------------------------------------------

# Stopwords PT-BR base + lista customizada do INPI.
_INPI_NOISE = {
    "comercio", "comercial", "atraves", "qualquer", "meio", "para",
    "uso", "fins", "servico", "servicos", "produto", "produtos",
    "[", "]", "(", ")", ";", ",", ".", ":", "/",
    "outros", "outras", "tipo", "tipos", "destinado", "destinada",
    "destinados", "destinadas", "exceto", "inclui", "incluindo",
    "diversos", "varios", "variados", "preparacoes", "preparados",
    "fabricacao", "venda", "varejo", "atacado", "ramo",
}


def _ensure_nltk() -> tuple[set[str], "object"]:
    """Garante stopwords + RSLP, baixando se necessario, e retorna (sw_set, stemmer)."""
    import nltk
    from nltk.corpus import stopwords as nltk_stop

    for resource, pkg in (("corpora/stopwords", "stopwords"), ("stemmers/rslp", "rslp")):
        try:
            nltk.data.find(resource)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception as exc:
                logger.warning("Falha ao baixar NLTK %s: %s", pkg, exc)

    try:
        sw = set(nltk_stop.words("portuguese"))
    except Exception:
        sw = set()
    sw_norm = {unidecode(w).lower() for w in sw}
    sw_norm |= _INPI_NOISE

    from nltk.stem import RSLPStemmer

    stemmer = RSLPStemmer()
    return sw_norm, stemmer


_SW: set[str] | None = None
_STEMMER: "object" | None = None


def _get_nlp():
    global _SW, _STEMMER
    if _SW is None or _STEMMER is None:
        _SW, _STEMMER = _ensure_nltk()
    return _SW, _STEMMER


def preprocess_spec(text: str) -> str:
    """Normaliza uma string de especificacao INPI completa (multi-item separado por ;).

    Pipeline: lower -> unaccent -> trocar ; , [ ] ( ) por espaco ->
              tokenizacao por whitespace -> remover stopwords -> stemming RSLP.
    Retorna texto stemizado, util para vetorizacao.
    """
    if not text:
        return ""

    sw, stemmer = _get_nlp()
    s = unidecode(str(text)).lower()
    for ch in (";", ",", "[", "]", "(", ")", "/", ":", ".", "-"):
        s = s.replace(ch, " ")
    out: list[str] = []
    for tok in s.split():
        if not tok or tok.isdigit():
            continue
        if len(tok) <= 2:
            continue
        if tok in sw:
            continue
        try:
            st = stemmer.stem(tok)
        except Exception:
            st = tok
        if not st or st in sw:
            continue
        out.append(st)
    return " ".join(out)


def preprocess_specs_batch(texts: Iterable[str]) -> list[str]:
    return [preprocess_spec(t) for t in texts]


# ---------------------------------------------------------------------------
# 2. Embeddings semanticos (sentence-transformers) com cache em parquet
# ---------------------------------------------------------------------------

DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


@dataclass
class EmbeddingProvider:
    """Encapsula o sentence-transformer + cache em disco para evitar recomputo."""

    model_name: str = DEFAULT_EMBEDDING_MODEL
    cache_path: Path | None = None
    batch_size: int = 64
    _model: "object | None" = field(default=None, init=False, repr=False)
    _cache: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.cache_path is not None:
            self.cache_path = Path(self.cache_path)
            self._load_cache()

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Carregando modelo de embeddings %s ...", self.model_name)
            device = _preferred_device()
            logger.info("SentenceTransformer device: %s", device)
            self._model = SentenceTransformer(self.model_name, device=device)
        return self._model

    def _embedding_dim(self) -> int:
        m = self._ensure_model()
        ed = getattr(m, "get_embedding_dimension", None)
        if callable(ed):
            return int(ed())
        d = getattr(m, "get_sentence_embedding_dimension", None)
        if callable(d):
            return int(d())
        raise RuntimeError("Modelo de embeddings sem metodo de dimensao conhecido.")

    def _purge_cache_wrong_dim(self, expected_dim: int) -> int:
        """Remove vetores de outro modelo (ex.: 384 vs 1024 ao mudar BERTimbau)."""
        stale = [
            k for k, v in self._cache.items()
            if getattr(v, "shape", (0,))[0] != expected_dim
        ]
        for k in stale:
            del self._cache[k]
        if stale:
            logger.warning(
                "Cache embeddings (specs): removidas %d entradas com dim != %d.",
                len(stale),
                expected_dim,
            )
        return len(stale)

    def _load_cache(self) -> None:
        if self.cache_path and self.cache_path.exists():
            try:
                import pandas as pd

                df = pd.read_parquet(self.cache_path)
                for _, row in df.iterrows():
                    self._cache[str(row["text"])] = np.asarray(row["emb"], dtype=np.float32)
                logger.info("Cache de embeddings carregado: %d entradas.", len(self._cache))
            except Exception as exc:
                logger.warning("Falha ao carregar cache %s: %s", self.cache_path, exc)

    def save_cache(self) -> None:
        if not self.cache_path:
            return
        try:
            import pandas as pd

            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            data = pd.DataFrame(
                {"text": list(self._cache.keys()), "emb": [v.tolist() for v in self._cache.values()]}
            )
            data.to_parquet(self.cache_path, index=False)
            logger.info("Cache de embeddings salvo (%d entradas) em %s", len(self._cache), self.cache_path)
        except Exception as exc:
            logger.warning("Falha ao salvar cache: %s", exc)

    def encode(
        self,
        texts: Sequence[str],
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        """Devolve embeddings normalizados (norm L2) shape (N, D).

        `progress(done, total)` e chamado ao longo do calculo de textos em falta
        (util para Streamlit nao ficar parado na barra durante modelos grandes em CPU).
        """
        d = self._embedding_dim()
        if not texts:
            return np.zeros((0, d), dtype=np.float32)

        self._purge_cache_wrong_dim(d)

        missing_idx: list[int] = []
        missing_txt: list[str] = []
        for i, t in enumerate(texts):
            vec = self._cache.get(t)
            if vec is None or vec.shape[0] != d:
                if vec is not None:
                    del self._cache[t]
                missing_idx.append(i)
                missing_txt.append(t)

        if missing_txt:
            model = self._ensure_model()
            n_m = len(missing_txt)
            # Chunks maiores que batch_size interno: cada chunk gera um callback (UI).
            outer_step = max(128, self.batch_size * 2)
            for start in range(0, n_m, outer_step):
                chunk = missing_txt[start : start + outer_step]
                embs = model.encode(
                    chunk,
                    batch_size=self.batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                ).astype(np.float32)
                for t, e in zip(chunk, embs):
                    self._cache[t] = e
                done = min(start + len(chunk), n_m)
                if progress is not None:
                    progress(done, n_m)
                logger.info(
                    "Embeddings specs: %d/%d textos unicos codificados.",
                    done,
                    n_m,
                )

        out = np.zeros((len(texts), d), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i] = self._cache[t]
        return out


# ---------------------------------------------------------------------------
# 3. Heuristica produto-vs-servico (sem peso manual no score)
# ---------------------------------------------------------------------------

PRODUTO_ANCHORS = {
    "produt", "merc", "fabric", "bebid", "alimen", "vest", "veicul",
    "moveis", "cosmet", "perfum", "calcad", "brinq", "eletronic", "veler",
    "cerveja", "vinho", "remed", "medic", "papel", "limpez", "saboner",
}
SERVICO_ANCHORS = {
    "servic", "consult", "assess", "ensin", "educ", "formac", "transport",
    "armazen", "host", "saas", "softw", "publi", "marketing", "design",
    "agenc", "constru", "instala", "manut", "reparo", "represent",
}


def detect_activity_kind(stem_text: str) -> str:
    """Retorna 'produto', 'servico' ou 'misto'."""
    if not stem_text:
        return "misto"
    p = sum(1 for tok in stem_text.split() if any(tok.startswith(a) for a in PRODUTO_ANCHORS))
    s = sum(1 for tok in stem_text.split() if any(tok.startswith(a) for a in SERVICO_ANCHORS))
    if p == 0 and s == 0:
        return "misto"
    if p >= 2 * max(s, 1):
        return "produto"
    if s >= 2 * max(p, 1):
        return "servico"
    return "misto"


def activity_match_score(a_stem: str, b_stem: str) -> dict[str, float]:
    ka = detect_activity_kind(a_stem)
    kb = detect_activity_kind(b_stem)
    same = float(ka == kb and ka != "misto")
    misto = float(ka == "misto" or kb == "misto")
    return {
        "spec_same_activity_kind": same,
        "spec_any_misto": misto,
        "spec_kind_a_produto": float(ka == "produto"),
        "spec_kind_a_servico": float(ka == "servico"),
        "spec_kind_b_produto": float(kb == "produto"),
        "spec_kind_b_servico": float(kb == "servico"),
    }


# ---------------------------------------------------------------------------
# 4. Features lexicais derivadas
# ---------------------------------------------------------------------------


def lexical_features(a_stem: str, b_stem: str, idf_lookup: dict[str, float] | None = None) -> dict[str, float]:
    """Features de tokens stemmed (jaccard, overlap, comuns/exclusivos, raridade)."""
    ta = a_stem.split() if a_stem else []
    tb = b_stem.split() if b_stem else []
    sa, sb = set(ta), set(tb)
    inter = sa & sb
    union = sa | sb

    jacc = len(inter) / len(union) if union else (1.0 if not sa and not sb else 0.0)
    if not sa or not sb:
        ovl = 0.0
    else:
        ovl = len(inter) / min(len(sa), len(sb))

    raros_score = 0.0
    if idf_lookup and inter:
        raros_score = float(sum(idf_lookup.get(t, 0.0) for t in inter) / len(inter))

    return {
        "spec_lex_jaccard": float(jacc),
        "spec_lex_overlap": float(ovl),
        "spec_lex_n_common": float(len(inter)),
        "spec_lex_n_excl_a": float(len(sa - sb)),
        "spec_lex_n_excl_b": float(len(sb - sa)),
        "spec_lex_n_total_a": float(len(sa)),
        "spec_lex_n_total_b": float(len(sb)),
        "spec_lex_size_ratio": float(min(len(sa), len(sb)) / max(len(sa), len(sb)))
        if max(len(sa), len(sb)) > 0
        else 1.0,
        "spec_lex_size_diff_abs": float(abs(len(sa) - len(sb))),
        "spec_lex_idf_avg_common": float(raros_score),
    }


# ---------------------------------------------------------------------------
# 5. Cosine das matrizes vetoriais (TF-IDF e embedding)
# ---------------------------------------------------------------------------


def row_cosine(mat_a, mat_b) -> np.ndarray:
    """Cosine row-a-row entre duas matrizes (denso ou esparso). Retorna shape (N,)."""
    import scipy.sparse as sp
    from sklearn.preprocessing import normalize

    if sp.issparse(mat_a) or sp.issparse(mat_b):
        a = normalize(mat_a, norm="l2", axis=1, copy=False)
        b = normalize(mat_b, norm="l2", axis=1, copy=False)
        n = a.shape[0]
        out = np.zeros(n, dtype=np.float32)
        for i in range(n):
            out[i] = float(a[i].multiply(b[i]).sum())
        return out

    a = np.asarray(mat_a, dtype=np.float32)
    b = np.asarray(mat_b, dtype=np.float32)
    na = np.linalg.norm(a, axis=1) + 1e-9
    nb = np.linalg.norm(b, axis=1) + 1e-9
    return ((a * b).sum(axis=1) / (na * nb)).astype(np.float32)


def spec_feature_names(idf_available: bool = True) -> list[str]:
    sample = lexical_features("a b c", "a c d", {"a": 1.0} if idf_available else None)
    sample.update(activity_match_score("a", "a"))
    sample.update(
        {
            "spec_cosine_tfidf_word": 0.0,
            "spec_cosine_tfidf_char": 0.0,
            "spec_cosine_emb": 0.0,
        }
    )
    return list(sample.keys())


__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "EmbeddingProvider",
    "preprocess_spec",
    "preprocess_specs_batch",
    "detect_activity_kind",
    "activity_match_score",
    "lexical_features",
    "row_cosine",
    "spec_feature_names",
]
