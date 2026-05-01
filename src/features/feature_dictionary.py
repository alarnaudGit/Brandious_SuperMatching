"""Catalogo de descricoes curtas (PT-BR) para cada feature enriquecida.

Sincronizado com `DICIONARIO_DE_DADOS.md`. Usado pela UI Streamlit (tabelas
de importancia e Integrated Gradients) e pelos relatorios em `report.md`
para mostrar ao lado de cada feature uma descricao em linguagem de negocio.
"""
from __future__ import annotations

from typing import Iterable, Mapping


# ---------------------------------------------------------------------------
# Bloco GRAFICO (semelhanca da escrita)
# ---------------------------------------------------------------------------
_GRAFICO: dict[str, str] = {
    "graf_levenshtein": "Inverso normalizado das trocas de letra para A virar B; alto = escritas parecidas.",
    "graf_jaro": "Similaridade de Jaro entre as marcas; foca em letras na mesma posicao.",
    "graf_jaro_winkler": "Jaro com bonus para prefixo igual (importante em PT-BR).",
    "graf_damerau": "Damerau-Levenshtein normalizado (conta troca de letras adjacentes).",
    "graf_jaccard_bigram": "Sobreposicao de pares consecutivos de letras entre A e B.",
    "graf_jaccard_trigram": "Sobreposicao de trios consecutivos de letras entre A e B.",
    "graf_overlap_trigram": "Quanto dos trios da marca menor estao na maior (deteccao de inclusao).",
    "graf_lcs_norm": "Tamanho da maior subsequencia comum dividido pelo tamanho da maior marca.",
    "graf_prefix_norm": "Tamanho do comeco identico entre A e B, normalizado.",
    "graf_suffix_norm": "Tamanho do final identico entre A e B, normalizado.",
    "graf_len_ratio": "Razao tamanho_menor/tamanho_maior (proximo de 1 = comprimentos parecidos).",
    "graf_len_diff_abs": "Diferenca absoluta de tamanho em caracteres.",
    "graf_contains": "1 se uma marca esta contida inteira na outra; 0 caso contrario.",
    "graf_anagram": "Grau de anagrama: ate que ponto rearranjar letras de A produz B.",
}

# ---------------------------------------------------------------------------
# Bloco FONETICO (semelhanca do som)
# ---------------------------------------------------------------------------
_FONETICO: dict[str, str] = {
    "fon_global_sim": "Quao parecida e a versao fonetica completa de A vs B (capta 'soa igual').",
    "fon_key_eq": "1 se a chave fonetica das duas marcas for exatamente igual.",
    "fon_key_lev_sim": "Levenshtein direto entre as chaves foneticas (PT-BR).",
    "fon_after_dedup_eq": "1 se, apos remover letras repetidas, as marcas ficam iguais.",
    "fon_token_mean": "Casamento fonetico medio entre tokens de A e B.",
    "fon_token_max": "Melhor casamento fonetico entre qualquer token de A e B.",
    "fon_token_eq_share": "Proporcao de tokens de A com chave fonetica identica em B.",
}

# ---------------------------------------------------------------------------
# Bloco TOKEN-SET / OFTA (granularidade de palavras)
# ---------------------------------------------------------------------------
_TOKEN: dict[str, str] = {
    # token_set_metrics + ofta_token_metrics (nominal.build_nominal_features)
    "n_tokens_a": "Numero de tokens (palavras) na marca A apos normalizacao.",
    "n_tokens_b": "Numero de tokens na marca B apos normalizacao.",
    "n_tokens_diff": "Diferenca absoluta entre contagens de tokens de A e B.",
    "n_tokens_common": "Quantidade de tokens na interseicao entre A e B.",
    "n_tokens_excl_a": "Tokens que aparecem em A mas nao em B.",
    "n_tokens_excl_b": "Tokens que aparecem em B mas nao em A.",
    "tok_jaccard": (
        "Jaccard entre tokens segundo o modulo interno OFTA de metricas de tokens."
    ),
    "tok_overlap": "Overlap de tokens (OFTA) entre as duas marcas.",
    "tok_fuzzy": "Componente fuzzy de similaridade token-level (OFTA).",
    "num_has_digits": (
        "1 se pelo menos uma das marcas contem digitos (ativa ramo de features numericas)."
    ),
    "num_orto_best": (
        "Melhor similaridade ortografica entre variantes literal/cardinal dos numeros."
    ),
    "num_orto_worst": "Pior similaridade ortografica entre essas variantes.",
    "num_orto_spread": "Dispersao ortografica (melhor - pior) entre variantes numericas.",
    "num_fon_best": "Melhor similaridade fonetica entre variantes numericas.",
    "num_fon_worst": "Pior similaridade fonetica entre variantes numericas.",
    "num_fon_spread": "Dispersao fonetica entre variantes numericas.",
    "ofta_final": "Score final da heuristica OFTA (0-1) entre A e B.",
    "ofta_orto": "Componente de ortografia/escrita do vetor OFTA (sub-score interno).",
    "ofta_fon": "Componente fonetico do vetor OFTA (sub-score interno).",
    "ofta_token": "Componente de tokens/palavras do vetor OFTA (sub-score interno).",
    "ofta_anagram": "Componente de anagrama do vetor OFTA (sub-score interno).",
    "ofta_fuzzy": "Componente fuzzy/aproximacao de termos do vetor OFTA (sub-score interno).",
}

# Drivers dominantes do OFTA (one-hot; nomes com acentos conforme nominal.py)
_OFTA_DRIVER: dict[str, str] = {
    "ofta_driver_geral": (
        "1 se o driver dominante da heuristica OFTA foi 'Geral' "
        "(caso misto ou padrao)."
    ),
    "ofta_driver_ortografia": (
        "1 se o driver dominante OFTA foi ortografia/escrita "
        "(label interno: Ortografia (Escrita))."
    ),
    "ofta_driver_fonética": (
        "1 se o driver dominante OFTA foi fonetica/som "
        "(label interno: Fonética (Som))."
    ),
    "ofta_driver_aproximação": (
        "1 se o driver dominante OFTA foi aproximacao fuzzy de termos."
    ),
    "ofta_driver_inclusão": (
        "1 se o driver dominante OFTA foi inclusao total de termo "
        "(uma marca contem a outra)."
    ),
    "ofta_driver_palavras": (
        "1 se o driver dominante OFTA foi palavras em comum."
    ),
}

# ---------------------------------------------------------------------------
# Bloco SPEC LEXICAL (palavras das especificacoes apos stemming)
# ---------------------------------------------------------------------------
_SPEC_LEX: dict[str, str] = {
    "spec_lex_jaccard": "Jaccard entre tokens stemados das especificacoes de A e B.",
    "spec_lex_overlap": "Overlap dos tokens das specs sobre min(|specA|,|specB|).",
    "spec_lex_n_common": "Numero absoluto de tokens stemados em comum nas specs.",
    "spec_lex_n_excl_a": "Tokens nas specs exclusivos de A.",
    "spec_lex_n_excl_b": "Tokens nas specs exclusivos de B.",
    "spec_lex_n_total_a": "Total de tokens (apos limpeza/stem) na spec de A.",
    "spec_lex_n_total_b": "Total de tokens (apos limpeza/stem) na spec de B.",
    "spec_lex_size_ratio": "Razao tamanho_menor/tamanho_maior em tokens das specs.",
    "spec_lex_size_diff_abs": "Diferenca absoluta no numero de tokens entre as specs.",
    "spec_lex_idf_avg_common": "IDF medio dos tokens em comum (alto = palavras raras compartilhadas).",
}

# ---------------------------------------------------------------------------
# Bloco SPEC ATIVIDADE (produto vs servico, ancoras semanticas)
# ---------------------------------------------------------------------------
_SPEC_ACT: dict[str, str] = {
    "spec_same_activity_kind": "1 se A e B sao do mesmo tipo (produto ou servico) e nao mistos.",
    "spec_any_misto": "1 se pelo menos uma das specs e ambigua (produto+servico).",
    "spec_kind_a_produto": "1 se a spec de A foi classificada como produto.",
    "spec_kind_a_servico": "1 se a spec de A foi classificada como servico.",
    "spec_kind_b_produto": "1 se a spec de B foi classificada como produto.",
    "spec_kind_b_servico": "1 se a spec de B foi classificada como servico.",
}

# ---------------------------------------------------------------------------
# Cosines TF-IDF + embedding semantico
# ---------------------------------------------------------------------------
_SPEC_VEC: dict[str, str] = {
    "spec_cosine_tfidf_word": "Cosseno TF-IDF (palavras 1-2grams) entre as specs.",
    "spec_cosine_tfidf_char": "Cosseno TF-IDF (char 3-5grams) entre as specs.",
    "spec_cosine_emb": "Cosseno do embedding semantico (BERTimbau) entre as specs.",
}

# ---------------------------------------------------------------------------
# Embedding semantico das marcas (Sprint 1)
# ---------------------------------------------------------------------------
_BRAND_EMB: dict[str, str] = {
    "brand_emb_cosine": "Cosseno do embedding entre as marcas (forma original).",
    "brand_emb_cosine_normalized": "Cosseno do embedding entre as marcas normalizadas.",
    "brand_emb_norm_max": "Maior das normas L2 dos embeddings de A e B (sinal de informatividade).",
}

# ---------------------------------------------------------------------------
# Classes Nice (one-hot e estatisticas pareadas)
# ---------------------------------------------------------------------------
_CLASS_FIXED: dict[str, str] = {
    "cls_same": "1 se A e B estao na mesma classe Nice.",
    "cls_diff_abs": "Diferenca absoluta entre os numeros de classe Nice de A e B.",
    "cls_a_known": "1 se a classe de A e conhecida (>=0).",
    "cls_b_known": "1 se a classe de B e conhecida (>=0).",
    "cls_a_top_other": "1 se a classe de A nao esta no top-K mais frequentes.",
    "cls_b_top_other": "1 se a classe de B nao esta no top-K mais frequentes.",
}

# ---------------------------------------------------------------------------
# Interacoes derivadas (sinais combinados nome x spec x classe)
# Nomes exatos: src/features/interactions.py
# ---------------------------------------------------------------------------
_INTER: dict[str, str] = {
    "inter_nome_x_spec_word": (
        "Produto do proxy de nome pelo cosseno TF-IDF de palavras das specs; "
        "alto quando nome E vocabulario das especificacoes estao alinhados."
    ),
    "inter_nome_x_spec_emb": (
        "Produto do proxy de nome pelo cosseno semantico (embedding) das specs; "
        "capta mesmo sentido com palavras diferentes."
    ),
    "inter_nome_x_spec_max": (
        "Proxy de nome vezes o melhor sinal entre emb/word/char das specs."
    ),
    "inter_nome_x_same_cls": "Similaridade de nome vezes indicador de mesma classe Nice.",
    "inter_spec_x_same_cls": "Proximidade de specs vezes indicador de mesma classe Nice.",
    "inter_nome_alto_e_spec_alta": (
        "1 se nome muito parecido (>=0.7) E specs muito parecidas (>=0.6); "
        "cenario de alto risco."
    ),
    "inter_nome_alto_e_spec_baixa": (
        "1 se nome muito parecido mas specs fracas (<=0.3); risco medio "
        "(mesmo nome, mercados diferentes)."
    ),
    "inter_nome_baixo_e_spec_alta": (
        "1 se nome distante mas specs muito parecidas; possivel risco oculto."
    ),
    "inter_classe_diff_mas_emb_alto": (
        "Classes Nice diferentes mas embedding das specs alto; risco oculto "
        "(mercado parecido, classes distintas)."
    ),
    "inter_classe_igual_e_spec_proxima": (
        "Mesma classe Nice e specs proximas; reforco duplo de risco."
    ),
    "inter_proxy_nome": (
        "Copia do score nominativo usado nas interacoes (ofta_final ou max graf/fon/jw)."
    ),
    "inter_proxy_spec": (
        "Copia do melhor score de especificacao (max entre emb, TF-IDF palavra e char)."
    ),
}

# ---------------------------------------------------------------------------
# Extra-nominal (Sprint 1: genericos, conteinment, radicais, lev puro, against)
# ---------------------------------------------------------------------------
_EXTRA_NOMINAL: dict[str, str] = {
    "name_generic_share_a": (
        "Proporcao dos tokens da marca A classificados como genericos (DF no corpus)."
    ),
    "name_generic_share_b": (
        "Proporcao dos tokens da marca B classificados como genericos (DF no corpus)."
    ),
    "name_generic_share_max": "Maior das duas proporcoes generic_share_a / generic_share_b.",
    "name_unique_token_a": (
        "Quantidade de tokens nao-genericos distintos na marca A (apos filtro)."
    ),
    "name_unique_token_b": (
        "Quantidade de tokens nao-genericos distintos na marca B (apos filtro)."
    ),
    "shared_unique_count": (
        "Quantidade de tokens nao-genericos em comum entre A e B."
    ),
    "shared_unique_jaccard": (
        "Jaccard dos conjuntos de tokens informativos de A e B."
    ),
    "shared_token_idf_max": (
        "Maior IDF entre tokens na interseicao bruta de A e B."
    ),
    "shared_token_idf_mean": (
        "IDF medio dos tokens na interseicao bruta de A e B."
    ),
    "shared_only_generics": (
        "1 se ha tokens em comum entre os nomes E todos sao genericos; "
        "red flag de possivel falso alarme."
    ),
    "contain_a_in_b": (
        "1 se o nome normalizado compactado de A esta contido no de B."
    ),
    "contain_b_in_a": (
        "1 se o nome normalizado compactado de B esta contido no de A."
    ),
    "contain_after_strip_a_in_b": (
        "1 se A sem tokens genericos (concatenado) esta contido em B sem genericos."
    ),
    "contain_after_strip_b_in_a": (
        "1 se B sem tokens genericos esta contido em A sem genericos."
    ),
    "contain_radical_share": (
        "Comprimento do prefixo comum entre nomes compactados (radical visual)."
    ),
    "contain_radical_share_norm": (
        "Prefixo comum normalizado pelo tamanho do menor nome compactado."
    ),
    "radical_a_len": (
        "Tamanho do maior token informativo (>=4 chars) escolhido como radical de A."
    ),
    "radical_b_len": (
        "Tamanho do maior token informativo (>=4 chars) escolhido como radical de B."
    ),
    "radical_lev_sim": (
        "Similaridade entre radicais (max Levenshtein norm. e Jaro-Winkler)."
    ),
    "radical_phonetic_eq": (
        "1 se as chaves foneticas dos radicais de A e B coincidem."
    ),
    "lev_pure_norm": (
        "Similaridade Levenshtein normalizada entre nomes sem tokens genericos."
    ),
    "lev_pure_jaro_winkler": (
        "Jaro-Winkler entre nomes apos remover tokens genericos."
    ),
    "lev_pure_size_diff_abs": (
        "Diferenca absoluta de tamanho (caracteres) entre nomes sem genericos."
    ),
    "against_distinct_market": (
        "1 se classes Nice diferentes E embedding das specs baixo (<0.30); "
        "evidencia contra colidencia."
    ),
    "against_only_generic_overlap": (
        "1 se so ha overlap generico entre nomes E classes diferentes."
    ),
    "against_size_disparity": (
        "1 se diferenca grande de tamanho entre marcas E zero tokens em comum."
    ),
    "against_unique_strong_a": (
        "Contagem de tokens fortes (len>=4, IDF alto) exclusivos de A vs B."
    ),
    "against_unique_strong_b": (
        "Contagem de tokens fortes exclusivos de B vs A."
    ),
}

# ---------------------------------------------------------------------------
# Sprint 2: ancora, LCS, fon_concat, simhash, item-level, prior pareado
# ---------------------------------------------------------------------------
_SPRINT2: dict[str, str] = {
    "spec_item_max_cosine_tfidf": (
        "Maior cosseno TF-IDF entre qualquer item da spec A e qualquer da spec B."
    ),
    "spec_item_top3_mean_cosine": (
        "Media dos tres maiores cosenos TF-IDF entre itens das duas specs."
    ),
    "spec_item_align_score": (
        "Score de alinhamento item-a-item entre especificacoes (maior = melhor match)."
    ),
    "brand_a_token_in_spec_b": (
        "Quantidade de tokens informativos (len>=4) da marca A que casam com tokens "
        "stemizados da especificacao de B (ancora cruzada)."
    ),
    "brand_b_token_in_spec_a": (
        "Quantidade de tokens informativos da marca B que casam com tokens stemizados "
        "da especificacao de A."
    ),
    "cls_pair_prior_pos": (
        "P(positivo) historico estimado para o par de classes Nice (A, B)."
    ),
    "cls_pair_chi2_strength": (
        "Forca do desvio observado do par de classes (qui-quadrado empirico)."
    ),
    "brand_longest_common_substring_norm": (
        "Tamanho da maior substring comum entre nomes normalizados, "
        "dividido pelo menor tamanho."
    ),
    "fon_concat_metaphone_eq": (
        "1 se a chave Metaphone das marcas concatenadas (sem espacos) e igual."
    ),
    "brand_simhash_hamming_norm": (
        "Similaridade 1 - (Hamming/64) entre SimHash de char-trigramas dos nomes."
    ),
}


# ---------------------------------------------------------------------------
# Catalogo unificado (estatico)
# ---------------------------------------------------------------------------
_STATIC_CATALOG: dict[str, str] = {}
for _block in (
    _GRAFICO,
    _FONETICO,
    _TOKEN,
    _OFTA_DRIVER,
    _SPEC_LEX,
    _SPEC_ACT,
    _SPEC_VEC,
    _BRAND_EMB,
    _CLASS_FIXED,
    _INTER,
    _EXTRA_NOMINAL,
    _SPRINT2,
):
    _STATIC_CATALOG.update(_block)


def describe_feature(name: str) -> str:
    """Devolve descricao curta para uma feature.

    Trata padroes dinamicos (cls_a_top_<N>, cls_b_top_<N>) e devolve
    string vazia para features desconhecidas.
    """
    if not name:
        return ""
    if name in _STATIC_CATALOG:
        return _STATIC_CATALOG[name]
    if name.startswith("cls_a_top_"):
        cid = name.replace("cls_a_top_", "")
        return f"One-hot: marca A pertence a classe Nice {cid} (entre as Top-K mais frequentes)."
    if name.startswith("cls_b_top_"):
        cid = name.replace("cls_b_top_", "")
        return f"One-hot: marca B pertence a classe Nice {cid} (entre as Top-K mais frequentes)."
    if name.startswith("inter_"):
        return (
            f"Feature de interacao nome x especificacao x classe (`{name}`). "
            "Detalhes em src/features/interactions.py."
        )
    if name.startswith("ofta_driver_"):
        return (
            f"Indicador one-hot do driver dominante da heuristica OFTA (`{name}`). "
            "Ver DRIVER_NAMES em src/features/nominal.py."
        )
    if name.startswith("spec_item_"):
        return (
            "Feature de similaridade entre itens da especificacao (TF-IDF). "
            "Ver src/features/spec_decomp.py."
        )
    if name.startswith("against_") or name.startswith("contain_"):
        return (
            f"Feature nominal extra (`{name}`): genericos, contencao ou evidencia contra. "
            "Ver src/features/extra_nominal.py."
        )
    return ""


def describe_features(names: Iterable[str]) -> dict[str, str]:
    """Mapa nome -> descricao para uma lista de features."""
    return {n: describe_feature(n) for n in names}


def attach_descriptions(
    rows: Iterable[Mapping[str, object]],
    *,
    feature_key: str = "feature",
    description_key: str = "descricao",
) -> list[dict[str, object]]:
    """Adiciona campo de descricao a cada dict que tenha `feature_key`.

    Mantem a ordem original e nao remove nenhuma chave existente.
    """
    out: list[dict[str, object]] = []
    for r in rows:
        item = dict(r)
        fname = str(item.get(feature_key, ""))
        item[description_key] = describe_feature(fname)
        out.append(item)
    return out


__all__ = [
    "describe_feature",
    "describe_features",
    "attach_descriptions",
]
