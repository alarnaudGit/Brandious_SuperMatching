"""Decomposicao de especificacoes em itens individuais e matching item-a-item.

A especificacao INPI normalmente vem como uma lista de atividades separadas
por `;` (ex.: "Comercio de medicamentos; Importacao de cosmeticos; ...").
Quando comparamos a spec inteira como um unico documento, o sinal de UMA
atividade que coincide perfeitamente acaba diluido entre dezenas de outras.

Este modulo expoe utilitarios para:
- Quebrar uma spec em itens (item = um produto/servico/atividade);
- Pre-processar cada item (lowercase, unaccent, stopwords, RSLP stemming);
- Calcular matriz de cosine TF-IDF entre os itens de A e B;
- Resumir essa matriz em 3 features:
    spec_item_max_cosine_tfidf  -> max global da matriz
    spec_item_top3_mean_cosine  -> media dos 3 maiores valores
    spec_item_align_score        -> alinhamento bipartido otimo (Hungarian)
"""
from __future__ import annotations

import logging
import re
from typing import Sequence

import numpy as np
import scipy.sparse as sp

from .specs import preprocess_spec

logger = logging.getLogger(__name__)


_SPLITTERS = re.compile(r"[;\n]+")


def split_spec_items(text: str) -> list[str]:
    """Quebra a spec em itens nao-vazios usando `;` ou newline como separador.

    Retorna lista de strings BRUTAS (sem stemming) - util quando o caller quer
    aplicar pre-processamento proprio ou inspecionar os itens originais.
    """
    if not text or not isinstance(text, str):
        return []
    parts = _SPLITTERS.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def preprocess_spec_items(text: str) -> list[str]:
    """Quebra em itens e aplica `preprocess_spec` em cada um.

    Itens que ficam vazios apos o stemming sao descartados.
    """
    items = split_spec_items(text)
    out: list[str] = []
    for it in items:
        st = preprocess_spec(it)
        if st:
            out.append(st)
    return out


def _safe_cosine_matrix(a_mat, b_mat) -> np.ndarray:
    """Cosine entre todas as linhas de a_mat e todas as linhas de b_mat.

    Suporta matrizes esparsas (TfidfVectorizer.transform) e densas (numpy).
    Linhas com norma zero produzem 0.0.
    """
    if sp.issparse(a_mat):
        a_norms = np.sqrt(np.asarray(a_mat.multiply(a_mat).sum(axis=1)).ravel())
    else:
        a_norms = np.linalg.norm(a_mat, axis=1)
    if sp.issparse(b_mat):
        b_norms = np.sqrt(np.asarray(b_mat.multiply(b_mat).sum(axis=1)).ravel())
    else:
        b_norms = np.linalg.norm(b_mat, axis=1)

    if sp.issparse(a_mat) or sp.issparse(b_mat):
        dot = a_mat @ b_mat.T
        if sp.issparse(dot):
            dot = dot.toarray()
        else:
            dot = np.asarray(dot)
    else:
        dot = a_mat @ b_mat.T

    denom = np.outer(a_norms, b_norms)
    denom[denom == 0] = 1.0
    cos = dot / denom
    cos[(np.outer(a_norms, b_norms) == 0)] = 0.0
    return np.asarray(cos, dtype=np.float32)


def _align_score(M: np.ndarray) -> float:
    """Soma do alinhamento bipartido otimo (Hungarian) normalizada por min(R,C).

    M e uma matriz de similaridades [0,1] entre R itens de A e C itens de B.
    Retorna 0 se M for vazia.
    """
    if M.size == 0:
        return 0.0
    R, C = M.shape
    K = min(R, C)
    if K == 0:
        return 0.0
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(-M)
        return float(M[row_ind, col_ind].sum() / K)
    except Exception:
        # fallback guloso (caso scipy.optimize falhe por algum motivo raro)
        used_cols: set[int] = set()
        total = 0.0
        for _ in range(K):
            best = -1.0
            best_pair = (0, 0)
            for i in range(R):
                for j in range(C):
                    if j in used_cols:
                        continue
                    if M[i, j] > best:
                        best = M[i, j]
                        best_pair = (i, j)
            if best < 0:
                break
            used_cols.add(best_pair[1])
            total += best
        return float(total / K)


def item_match_features(
    a_items_stem: Sequence[str],
    b_items_stem: Sequence[str],
    tfidf_vectorizer,
) -> dict[str, float]:
    """Computa as 3 features de spec-item matching usando o vectorizer ja fitado.

    `tfidf_vectorizer` precisa ser um TfidfVectorizer ja ajustado (espera-se
    o mesmo `tfidf_char` aprendido pelo FeaturePreprocessor para reaproveitar
    o vocabulario).

    Retorna dict com 3 chaves: spec_item_max_cosine_tfidf,
    spec_item_top3_mean_cosine, spec_item_align_score.
    """
    out = {
        "spec_item_max_cosine_tfidf": 0.0,
        "spec_item_top3_mean_cosine": 0.0,
        "spec_item_align_score": 0.0,
    }
    if not a_items_stem or not b_items_stem or tfidf_vectorizer is None:
        return out

    try:
        A = tfidf_vectorizer.transform(a_items_stem)
        B = tfidf_vectorizer.transform(b_items_stem)
    except Exception as exc:  # noqa: BLE001
        logger.debug("item_match_features: transform falhou (%s)", exc)
        return out

    M = _safe_cosine_matrix(A, B)
    if M.size == 0:
        return out

    out["spec_item_max_cosine_tfidf"] = float(M.max())

    flat = np.sort(M.ravel())[::-1]
    k = min(3, flat.size)
    out["spec_item_top3_mean_cosine"] = float(flat[:k].mean())

    out["spec_item_align_score"] = _align_score(M)
    return out


def item_match_feature_names() -> list[str]:
    return [
        "spec_item_max_cosine_tfidf",
        "spec_item_top3_mean_cosine",
        "spec_item_align_score",
    ]


__all__ = [
    "split_spec_items",
    "preprocess_spec_items",
    "item_match_features",
    "item_match_feature_names",
]
