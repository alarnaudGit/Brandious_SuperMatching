# Relatorio - Modelo de Similaridade Aprendida de Marcas

_Gerado em 2026-04-30T13:51:55+00:00 (UTC)._

## 1. Dataset

- Linhas validas: **30941**
- Positivos: **3373** (10.90%)
- Negativos: **27568**
- Linhas removidas (marcas nulas): 0
- Mesmo classe Nice: 22.13% dos pares
- Taxa de positivos (mesma classe / classes diferentes): 17.92% / 8.91%
- Hash SHA-256 do arquivo: `3b06ce047455dcd324536d38...`

## 2. Features criadas

Total: **116** features na ordem canonica.

### Graficas (graf_*) (14)

`graf_levenshtein`, `graf_jaro`, `graf_jaro_winkler`, `graf_damerau`, `graf_jaccard_bigram`, `graf_jaccard_trigram`, `graf_overlap_trigram`, `graf_lcs_norm`, `graf_prefix_norm`, `graf_suffix_norm`, `graf_len_ratio`, `graf_len_diff_abs`, `graf_contains`, `graf_anagram`

### Foneticas (fon_*) (7)

`fon_global_sim`, `fon_key_eq`, `fon_key_lev_sim`, `fon_after_dedup_eq`, `fon_token_mean`, `fon_token_max`, `fon_token_eq_share`

### Tokens marca (n_tokens_* / tok_*) (9)

`n_tokens_a`, `n_tokens_b`, `n_tokens_diff`, `n_tokens_common`, `n_tokens_excl_a`, `n_tokens_excl_b`, `tok_jaccard`, `tok_overlap`, `tok_fuzzy`

### Numerais (num_*) (7)

`num_has_digits`, `num_orto_best`, `num_orto_worst`, `num_orto_spread`, `num_fon_best`, `num_fon_worst`, `num_fon_spread`

### OFTA (ofta_*) (12)

`ofta_final`, `ofta_orto`, `ofta_fon`, `ofta_token`, `ofta_anagram`, `ofta_fuzzy`, `ofta_driver_geral`, `ofta_driver_ortografia`, `ofta_driver_fonética`, `ofta_driver_aproximação`, `ofta_driver_inclusão`, `ofta_driver_palavras`

### Especificacao lexical (spec_lex_*) (10)

`spec_lex_jaccard`, `spec_lex_overlap`, `spec_lex_n_common`, `spec_lex_n_excl_a`, `spec_lex_n_excl_b`, `spec_lex_n_total_a`, `spec_lex_n_total_b`, `spec_lex_size_ratio`, `spec_lex_size_diff_abs`, `spec_lex_idf_avg_common`

### Especificacao atividade (spec_kind_* / spec_*activity*) (6)

`spec_same_activity_kind`, `spec_any_misto`, `spec_kind_a_produto`, `spec_kind_a_servico`, `spec_kind_b_produto`, `spec_kind_b_servico`

### Especificacao cosine (spec_cosine_*) (3)

`spec_cosine_tfidf_word`, `spec_cosine_tfidf_char`, `spec_cosine_emb`

### Classe Nice (cls_*) (36)

`cls_same`, `cls_diff_abs`, `cls_a_known`, `cls_b_known`, `cls_a_top_35`, `cls_a_top_41`, `cls_a_top_42`, `cls_a_top_44`, `cls_a_top_43`, `cls_a_top_36`, `cls_a_top_9`, `cls_a_top_37`, `cls_a_top_30`, `cls_a_top_40`, `cls_a_top_25`, `cls_a_top_39`, `cls_a_top_45`, `cls_a_top_16`, `cls_a_top_3`, `cls_a_top_other`, `cls_b_top_35`, `cls_b_top_41`, `cls_b_top_42`, `cls_b_top_44`, `cls_b_top_43`, `cls_b_top_36`, `cls_b_top_9`, `cls_b_top_37`, `cls_b_top_30`, `cls_b_top_40`, `cls_b_top_25`, `cls_b_top_39`, `cls_b_top_45`, `cls_b_top_16`, `cls_b_top_3`, `cls_b_top_other`

### Interacoes (inter_*) (12)

`inter_nome_x_spec_word`, `inter_nome_x_spec_emb`, `inter_nome_x_spec_max`, `inter_nome_x_same_cls`, `inter_spec_x_same_cls`, `inter_nome_alto_e_spec_alta`, `inter_nome_alto_e_spec_baixa`, `inter_nome_baixo_e_spec_alta`, `inter_classe_diff_mas_emb_alto`, `inter_classe_igual_e_spec_proxima`, `inter_proxy_nome`, `inter_proxy_spec`

## 3. Estrategia de balanceamento

- Undersample negativos: ratio **3.00 : 1** (neg/pos alvo)
- Oversample positivos: fator **2.00x**
- Class weight (pos_weight) ativo: **True**, valor efetivo **1.500**
- Seed: 42
- Split estratificado: train=70% / val=15% / test=15%

## 4. Metricas

Threshold otimo: **0.260** (politica: `max_f1_with_recall>=0.85`, recall_floor=0.85)

### Treino
- ROC-AUC: **0.9295**
- PR-AUC:  **0.6387**
- F1:        0.4419 (threshold=0.260)
- Precision: 0.2868
- Recall:    0.9619
- Confusao: TN=13650, FP=5647, FN=90, TP=2271
- n_pos=2361, n_neg=19297

### Validacao
- ROC-AUC: **0.8664**
- PR-AUC:  **0.5214**
- F1:        0.3949 (threshold=0.260)
- Precision: 0.2568
- Recall:    0.8538
- Confusao: TN=2885, FP=1250, FN=74, TP=432
- n_pos=506, n_neg=4135

### Teste
- ROC-AUC: **0.8447**
- PR-AUC:  **0.4795**
- F1:        0.3752 (threshold=0.260)
- Precision: 0.2409
- Recall:    0.8478
- Confusao: TN=2784, FP=1352, FN=77, TP=429
- n_pos=506, n_neg=4136

## 5. Comparativo NN vs Heuristica OFTA (no conjunto de teste)

_Recall avaliado com piso de Precision >= 0.90._

| Metrica | NN | Heuristica OFTA | Delta |
| --- | ---:| ---:| ---:|
| ROC-AUC | 0.8447 | 0.5263 | +0.3184 |
| PR-AUC | 0.4795 | 0.1161 | +0.3633 |
| Recall@P>=0.9 | 0.0553 | 0.0000 | +0.0553 |

## 7. Limitacoes

- O modelo cobre exclusivamente marcas NOMINATIVAS; figurativas precisam de outro pipeline.
- Qualidade do rotulo INPI determina o teto pratico do modelo.
- Embeddings sao multilingues genericos; um modelo PT-BR especifico pode melhorar.
- Classes Nice raras viram bucket 'other' por construcao do top-K.

## 8. Recomendacoes

- Reavaliar `recall_floor` em conjunto com a area de negocio para calibrar custo de FN/FP.
- Retreinar a cada novo lote significativo de pareceres revisados.
- Monitorar drift de score em producao (PSI sobre `score_nn`).
- Considerar uma feature dedicada de "diferenca de mercado" caso o INPI passe a expor essa info estruturada.
