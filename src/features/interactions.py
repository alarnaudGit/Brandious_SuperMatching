"""Features de interacao cruzando nome x especificacao x classe.

Cobrem os 5 cenarios criticos do prompt:
  - nome semelhante + especificacao semelhante  -> ALTO RISCO
  - nome semelhante + especificacao distante    -> MEDIO RISCO
  - nome diferente + especificacao semelhante   -> POSSIVEL RISCO
  - classe igual + especificacao proxima        -> ALTO RISCO
  - classe diferente + semantica proxima        -> RISCO OCULTO
"""
from __future__ import annotations

from typing import Mapping

# Limiares de "alto"/"baixo". Nao sao pesos do score final - sao apenas
# indicadores binarios para a rede aprender combinacoes.
THR_NOME_ALTO = 0.70
THR_NOME_BAIXO = 0.40
THR_SPEC_ALTA = 0.60
THR_SPEC_BAIXA = 0.30


def _nome_score(nominal: Mapping[str, float]) -> float:
    """Heuristica de proxy do nivel de similaridade nominativa.

    Usa o final OFTA quando disponivel; caso contrario, max entre orto/fon/jaro_w.
    """
    if "ofta_final" in nominal:
        return float(nominal["ofta_final"])
    return max(
        float(nominal.get("graf_levenshtein", 0.0)),
        float(nominal.get("fon_global_sim", 0.0)),
        float(nominal.get("graf_jaro_winkler", 0.0)),
    )


def _spec_score(spec: Mapping[str, float]) -> float:
    """Proxy de proximidade semantica das especificacoes."""
    return max(
        float(spec.get("spec_cosine_emb", 0.0)),
        float(spec.get("spec_cosine_tfidf_word", 0.0)),
        float(spec.get("spec_cosine_tfidf_char", 0.0)),
    )


def build_interaction_features(
    nominal: Mapping[str, float],
    spec: Mapping[str, float],
    cls: Mapping[str, float],
) -> dict[str, float]:
    nm = _nome_score(nominal)
    sp = _spec_score(spec)
    same_cls = float(cls.get("cls_same", 0.0))

    spec_emb = float(spec.get("spec_cosine_emb", 0.0))
    spec_word = float(spec.get("spec_cosine_tfidf_word", 0.0))

    nome_alto = nm >= THR_NOME_ALTO
    nome_baixo = nm <= THR_NOME_BAIXO
    spec_alta = sp >= THR_SPEC_ALTA
    spec_baixa = sp <= THR_SPEC_BAIXA

    return {
        "inter_nome_x_spec_word": float(nm * spec_word),
        "inter_nome_x_spec_emb": float(nm * spec_emb),
        "inter_nome_x_spec_max": float(nm * sp),
        "inter_nome_x_same_cls": float(nm * same_cls),
        "inter_spec_x_same_cls": float(sp * same_cls),
        "inter_nome_alto_e_spec_alta": float(nome_alto and spec_alta),
        "inter_nome_alto_e_spec_baixa": float(nome_alto and spec_baixa),
        "inter_nome_baixo_e_spec_alta": float(nome_baixo and spec_alta),
        "inter_classe_diff_mas_emb_alto": float((1.0 - same_cls) * spec_emb),
        "inter_classe_igual_e_spec_proxima": float(same_cls * sp),
        "inter_proxy_nome": float(nm),
        "inter_proxy_spec": float(sp),
    }


def interaction_feature_names() -> list[str]:
    sample = build_interaction_features(
        {"ofta_final": 0.0, "graf_jaro_winkler": 0.0, "graf_levenshtein": 0.0, "fon_global_sim": 0.0},
        {"spec_cosine_emb": 0.0, "spec_cosine_tfidf_word": 0.0, "spec_cosine_tfidf_char": 0.0},
        {"cls_same": 0.0},
    )
    return list(sample.keys())


__all__ = [
    "build_interaction_features",
    "interaction_feature_names",
    "THR_NOME_ALTO",
    "THR_NOME_BAIXO",
    "THR_SPEC_ALTA",
    "THR_SPEC_BAIXA",
]
