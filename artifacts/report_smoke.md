# Relatorio Completo - Modelo de Similaridade Aprendida de Marcas

_Gerado em 2026-05-01T00:17:19+00:00 (UTC)._

## Sumario executivo

**Veredito de prontidao:** `ACEITAVEL com monitoramento`

| Metrica | Treino | Validacao | Teste |
| --- | ---:| ---:| ---:|
| ROC-AUC | 0.8365 | 0.8138 | **0.8328** |
| PR-AUC  | 0.5730 | 0.5171 | **0.5326** |
| F1      | 0.3333 | 0.4859 | **0.5000** |
| Recall  | 1.0000 | 0.8958 | **0.9167** |
| Precision | 0.2000 | 0.3333 | **0.3438** |

**Threshold de operacao:** `0.1500` (politica: `max_f1_with_recall>=0.85`, recall_floor=0.85)

**Pontos fortes:**
- PR-AUC 0.533 (bom para base desbalanceada)
- Recall 0.917 atende o piso operacional de 85%
- F1 0.500 (bom equilibrio)

**Atencao:**
- ROC-AUC 0.833 (aceitavel, mas pode melhorar)

**Sinais de overfit (treino - teste):**

| Metrica | delta train-test |
| --- | ---:|
| ROC-AUC train-test | +0.0037 |
| PR-AUC train-test | +0.0404 |
| Recall train-test | +0.0833 |
| F1 train-test | -0.1667 |

## 1. Dataset

- Linhas validas: **30941**
- Positivos: **3373** (10.90%)
- Negativos: **27568**
- Linhas removidas (marcas nulas): 0
- Mesmo classe Nice: 22.13% dos pares
- Taxa positivos (mesma classe / classes diferentes): 17.92% / 8.91%
- Hash SHA-256: `3b06ce047455dcd324536d384ae884b6a267a2399136345a143c5819dbabd03a`

### 1.1 Comprimento das marcas (caracteres)

| Coluna | mean | p25 | p50 | p75 | max |
| --- | ---:| ---:| ---:| ---:| ---:|
| marca_monitorada | 17.7 | 11 | 15 | 22 | 90 |
| marca_colidente | 10.4 | 7 | 9 | 13 | 57 |

### 1.2 Comprimento das especificacoes (caracteres)

| Coluna | mean | p25 | p50 | p75 | max |
| --- | ---:| ---:| ---:| ---:| ---:|
| especificacao_monitorado | 364.1 | 72 | 159 | 444 | 16797 |
| especificacao_colidente | 752.0 | 128 | 311 | 702 | 32767 |

### 1.3 Top classes Nice mais frequentes

| Classe | Ocorrencias |
| ---:| ---:|
| 35 | 21882 |
| 41 | 6932 |
| 42 | 3851 |
| 44 | 2985 |
| 43 | 2886 |
| 36 | 2538 |
| 9 | 2151 |
| 37 | 1906 |
| 30 | 1671 |
| 40 | 1631 |
| 25 | 1436 |
| 39 | 1238 |
| 45 | 1116 |
| 16 | 1095 |
| 3 | 889 |

## 2. Pre-processamento e features

- TF-IDF word: ngram=(1, 2), min_df=3, max_features=3000
- TF-IDF char: ngram=(3, 5), min_df=3, max_features=2000
- Embeddings ativos: **False** (modelo: `paraphrase-multilingual-MiniLM-L12-v2`)
- Cache de embeddings: `artifacts/embeddings_cache_smoke.parquet`
- Top-10 classes Nice usadas em one-hot: [35, 41, 42, 43, 44, 36, 9, 25, 37, 30]
- Total de features na ordem canonica: **147**

### Graficas (graf_*) (14)

`graf_levenshtein`, `graf_jaro`, `graf_jaro_winkler`, `graf_damerau`, `graf_jaccard_bigram`, `graf_jaccard_trigram`, `graf_overlap_trigram`, `graf_lcs_norm`, `graf_prefix_norm`, `graf_suffix_norm`, `graf_len_ratio`, `graf_len_diff_abs`, `graf_contains`, `graf_anagram`

### Foneticas (fon_*) (8)

`fon_global_sim`, `fon_key_eq`, `fon_key_lev_sim`, `fon_after_dedup_eq`, `fon_token_mean`, `fon_token_max`, `fon_token_eq_share`, `fon_concat_metaphone_eq`

### Tokens marca (n_tokens_* / tok_*) (9)

`n_tokens_a`, `n_tokens_b`, `n_tokens_diff`, `n_tokens_common`, `n_tokens_excl_a`, `n_tokens_excl_b`, `tok_jaccard`, `tok_overlap`, `tok_fuzzy`

### Numerais (num_*) (7)

`num_has_digits`, `num_orto_best`, `num_orto_worst`, `num_orto_spread`, `num_fon_best`, `num_fon_worst`, `num_fon_spread`

### OFTA (ofta_*) (12)

`ofta_final`, `ofta_orto`, `ofta_fon`, `ofta_token`, `ofta_anagram`, `ofta_fuzzy`, `ofta_driver_geral`, `ofta_driver_ortografia`, `ofta_driver_fonética`, `ofta_driver_aproximação`, `ofta_driver_inclusão`, `ofta_driver_palavras`

### Especificacao lexical (spec_lex_*) (10)

`spec_lex_jaccard`, `spec_lex_overlap`, `spec_lex_n_common`, `spec_lex_n_excl_a`, `spec_lex_n_excl_b`, `spec_lex_n_total_a`, `spec_lex_n_total_b`, `spec_lex_size_ratio`, `spec_lex_size_diff_abs`, `spec_lex_idf_avg_common`

### Especificacao atividade (spec_kind_* / spec_*activity*) (9)

`spec_same_activity_kind`, `spec_any_misto`, `spec_kind_a_produto`, `spec_kind_a_servico`, `spec_kind_b_produto`, `spec_kind_b_servico`, `spec_item_max_cosine_tfidf`, `spec_item_top3_mean_cosine`, `spec_item_align_score`

### Especificacao cosine (spec_cosine_*) (3)

`spec_cosine_tfidf_word`, `spec_cosine_tfidf_char`, `spec_cosine_emb`

### Classe Nice (cls_*) (28)

`cls_same`, `cls_diff_abs`, `cls_a_known`, `cls_b_known`, `cls_a_top_35`, `cls_a_top_41`, `cls_a_top_42`, `cls_a_top_43`, `cls_a_top_44`, `cls_a_top_36`, `cls_a_top_9`, `cls_a_top_25`, `cls_a_top_37`, `cls_a_top_30`, `cls_a_top_other`, `cls_b_top_35`, `cls_b_top_41`, `cls_b_top_42`, `cls_b_top_43`, `cls_b_top_44`, `cls_b_top_36`, `cls_b_top_9`, `cls_b_top_25`, `cls_b_top_37`, `cls_b_top_30`, `cls_b_top_other`, `cls_pair_prior_pos`, `cls_pair_chi2_strength`

### Interacoes (inter_*) (12)

`inter_nome_x_spec_word`, `inter_nome_x_spec_emb`, `inter_nome_x_spec_max`, `inter_nome_x_same_cls`, `inter_spec_x_same_cls`, `inter_nome_alto_e_spec_alta`, `inter_nome_alto_e_spec_baixa`, `inter_nome_baixo_e_spec_alta`, `inter_classe_diff_mas_emb_alto`, `inter_classe_igual_e_spec_proxima`, `inter_proxy_nome`, `inter_proxy_spec`

### Outras (35)

`name_generic_share_a`, `name_generic_share_b`, `name_generic_share_max`, `name_unique_token_a`, `name_unique_token_b`, `shared_unique_count`, `shared_unique_jaccard`, `shared_token_idf_max`, `shared_token_idf_mean`, `shared_only_generics`, `contain_a_in_b`, `contain_b_in_a`, `contain_after_strip_a_in_b`, `contain_after_strip_b_in_a`, `contain_radical_share`, `contain_radical_share_norm`, `radical_a_len`, `radical_b_len`, `radical_lev_sim`, `radical_phonetic_eq`, `lev_pure_norm`, `lev_pure_jaro_winkler`, `lev_pure_size_diff_abs`, `against_distinct_market`, `against_only_generic_overlap`, `against_size_disparity`, `against_unique_strong_a`, `against_unique_strong_b`, `brand_emb_cosine`, `brand_emb_cosine_normalized`, `brand_emb_norm_max`, `brand_a_token_in_spec_b`, `brand_b_token_in_spec_a`, `brand_longest_common_substring_norm`, `brand_simhash_hamming_norm`

## 3. Estatisticas das features (apos StandardScaler)

- Linhas usadas para estatistica: 1200
- Features com variancia ZERO: **9** (possiveis candidatas a remocao)

Features sem variancia detectadas: `ofta_driver_geral`, `ofta_driver_palavras`, `cls_b_known`, `inter_nome_x_spec_emb`, `inter_classe_diff_mas_emb_alto`, `spec_cosine_emb`, `brand_emb_cosine`, `brand_emb_cosine_normalized`, `brand_emb_norm_max`

### 3.1 Top features por desvio padrao (apos scaler - util p/ debug)

| Feature | media | desvio | min | max |
| --- | --- | --- | --- | --- |
| cls_a_top_9 | -0.0000 | 1.0000 | -0.1234 | 8.1035 |
| cls_b_top_44 | -0.0000 | 1.0000 | -0.2192 | 4.5627 |
| cls_a_top_37 | -0.0000 | 1.0000 | -0.1881 | 5.3168 |
| contain_after_strip_a_in_b | 0.0000 | 1.0000 | -0.2233 | 4.4780 |
| num_has_digits | -0.0000 | 1.0000 | -0.2412 | 4.1451 |
| inter_nome_alto_e_spec_alta | -0.0000 | 1.0000 | -0.1335 | 7.4929 |
| radical_phonetic_eq | 0.0000 | 1.0000 | -0.5966 | 1.6762 |
| graf_contains | 0.0000 | 1.0000 | -0.2708 | 3.6924 |
| cls_a_top_43 | 0.0000 | 1.0000 | -0.2526 | 3.9581 |
| ofta_driver_inclusão | 0.0000 | 1.0000 | -0.2354 | 4.2482 |
| cls_a_top_30 | -0.0000 | 1.0000 | -0.1708 | 5.8561 |
| spec_kind_a_servico | 0.0000 | 1.0000 | -0.6560 | 1.5245 |
| spec_kind_b_servico | 0.0000 | 1.0000 | -0.7419 | 1.3479 |
| cls_b_top_25 | 0.0000 | 1.0000 | -0.2019 | 4.9530 |
| name_generic_share_b | -0.0000 | 1.0000 | -0.3837 | 5.6638 |
| num_orto_spread | -0.0000 | 1.0000 | -0.1776 | 13.7386 |
| cls_a_top_44 | -0.0000 | 1.0000 | -0.2192 | 4.5627 |
| cls_a_top_42 | 0.0000 | 1.0000 | -0.2334 | 4.2843 |
| cls_a_top_35 | -0.0000 | 1.0000 | -0.7325 | 1.3653 |
| cls_a_top_41 | -0.0000 | 1.0000 | -0.3765 | 2.6559 |

### 3.2 Top features por |correlacao de Pearson com a label|

| Feature | Pearson |
| --- | --- |
| cls_pair_prior_pos | 0.3996 |
| spec_item_max_cosine_tfidf | 0.3613 |
| spec_item_top3_mean_cosine | 0.3483 |
| spec_lex_idf_avg_common | 0.3272 |
| spec_item_align_score | 0.3115 |
| spec_lex_overlap | 0.3112 |
| spec_cosine_tfidf_char | 0.3102 |
| inter_proxy_spec | 0.3082 |
| spec_lex_n_common | 0.2592 |
| spec_cosine_tfidf_word | 0.2476 |
| cls_pair_chi2_strength | 0.2420 |
| inter_classe_igual_e_spec_proxima | 0.2273 |
| inter_spec_x_same_cls | 0.2273 |
| spec_lex_jaccard | 0.2217 |
| inter_nome_x_spec_max | 0.2209 |
| ofta_token | 0.1982 |
| ofta_fuzzy | 0.1982 |
| tok_fuzzy | 0.1982 |
| against_distinct_market | -0.1947 |
| cls_same | 0.1848 |

## 4. Estrategia de balanceamento

- Undersample negativos: ratio **2.50 : 1** (neg/pos alvo)
- Oversample positivos: fator **2.00x**
- Class weight (pos_weight) ativo: **True**, valor efetivo **1.250**
- Seed: 42
- Split estratificado: train=60% / val=20% / test=20%

## 5. Arquitetura e hiperparametros

### 5.1 Arquitetura

- input_dim: **147**
- hidden_dims: **None**
- ativacao: **None**
- dropout: **0.45**
- batchnorm: **True**
- output: Linear(., 1) -> Sigmoid
- **# parametros aprendiveis: 148**

### 5.2 Hiperparametros de treino

- epochs: **3**
- batch_size: **64**
- lr: **0.001**
- weight_decay: **0.0001**
- early_stopping_patience: **20**
- scheduler_patience: **4**
- scheduler_factor: **0.5**
- min_lr: **1e-06**
- grad_clip: **1.0**
- device: **auto**
- seed: **42**
- best_epoch: **3**
- best_pr_auc_val: **0.5171230765342824**
- n_train_after_balancing: **648**

- Otimizador: **AdamW**, Loss: **BCEWithLogitsLoss(pos_weight)**, Scheduler: **ReduceLROnPlateau** (max em val PR-AUC)

## 6. Historico do treino (epoca a epoca)

| epoch | train_loss | lr | train_pr_auc | val_pr_auc | val_roc_auc | val_recall@0.5 | val_f1@0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.4378 | 0.0010 | 0.6770 | 0.4207 | 0.7436 | 0.5208 | 0.4464 |
| 2 | 0.3509 | 0.0010 | 0.7651 | 0.4800 | 0.7831 | 0.2917 | 0.3636 |
| 3 | 0.3061 | 0.0010 | 0.8198 | 0.5171 | 0.8138 | 0.2708 | 0.3662 |


## 7. Metricas em detalhe

Threshold otimo: **0.1500** (politica: `max_f1_with_recall>=0.85`, recall_floor=0.85)

### Treino
- ROC-AUC: **0.8365**
- PR-AUC:  **0.5730**
- F1:        0.3333 (threshold=0.150)
- Precision: 0.2000
- Recall:    1.0000
- Confusao: TN=0, FP=576, FN=0, TP=144
- n_pos=144, n_neg=576

### Validacao
- ROC-AUC: **0.8138**
- PR-AUC:  **0.5171**
- F1:        0.4859 (threshold=0.150)
- Precision: 0.3333
- Recall:    0.8958
- Confusao: TN=106, FP=86, FN=5, TP=43
- n_pos=48, n_neg=192

### Teste
- ROC-AUC: **0.8328**
- PR-AUC:  **0.5326**
- F1:        0.5000 (threshold=0.150)
- Precision: 0.3438
- Recall:    0.9167
- Confusao: TN=108, FP=84, FN=4, TP=44
- n_pos=48, n_neg=192

## 8. Distribuicao de scores no teste

| Classe | n | mean | std | p10 | p25 | p50 | p75 | p90 |
| --- | ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:|
| pos | 48 | 0.3474 | 0.1538 | 0.1693 | 0.2468 | 0.3125 | 0.4504 | 0.5469 |
| neg | 192 | 0.1669 | 0.1224 | 0.0492 | 0.0769 | 0.1343 | 0.2215 | 0.3326 |

_Diferenca de media (pos - neg): **+0.1805**. Quanto maior, melhor a separabilidade._

### 8.1 Decis de score (10 = scores mais altos)

| Decil | score_min | score_max | n | positivos | % positivos |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.4009 | 0.6898 | 24 | 17 | 0.7083 |
| 1 | 0.3066 | 0.3978 | 24 | 8 | 0.3333 |
| 2 | 0.2530 | 0.3057 | 24 | 10 | 0.4167 |
| 3 | 0.2062 | 0.2523 | 24 | 4 | 0.1667 |
| 4 | 0.1624 | 0.2048 | 24 | 5 | 0.2083 |
| 5 | 0.1293 | 0.1622 | 24 | 0 | 0.0000 |
| 6 | 0.1022 | 0.1288 | 24 | 0 | 0.0000 |
| 7 | 0.0775 | 0.1015 | 24 | 4 | 0.1667 |
| 8 | 0.0557 | 0.0749 | 24 | 0 | 0.0000 |
| 9 | 0.0082 | 0.0555 | 24 | 0 | 0.0000 |

## 9. Analise de erros (no teste)

- Total de pares no teste: **240** (threshold=0.1500)
- **Grupo 0** (rotulo real = 0, NAO colidente): 192 pares no total. **84 foram classificados como colidentes (taxa de erro 43.75%)**.
- **Grupo 1** (rotulo real = 1, colidentes): 48 pares no total. **4 escaparam da classificacao (taxa de erro 8.33%)**.
- Lista completa Grupo 0 errados (CSV): `artifacts\falsos_positivos_grupo_0_smoke.csv`
- Lista completa Grupo 1 errados (CSV): `artifacts\falsos_negativos_grupo_1_smoke.csv`

### 9.1 Top 10 Falsos Positivos (Grupo 0 - alarmes mais graves)

| Marca A | Marca B | cls A | cls B | score |
| --- | --- | --- | --- | --- |
| Master Multi | Mastery7 | 35 | 35 | 0.6898 |
| D&F MODA FITNESS | Life Move Moda Fitness | 35 | 25 | 0.6037 |
| VIDRAÇARIA BRASÍLIA | VIDRAÇARIA MK | 37 | 37 | 0.5858 |
| VALE ZELAR | VALE JJR | 37 | 37 | 0.5643 |
| ESPETINHO DA KARLA DESDE 1996 | ESPETINHO DO BEBÊ | 43 | 43 | 0.5622 |
| RI ARQUITETURA | ACAIÁ ARQUITETURA | 42 | 42 | 0.5547 |
| MALU | MALULU KIDS | 16 | 25 | 0.4692 |
| MALU | MA5 | 16 | 37 | 0.3978 |
| Y.Consultoria | ZV CONSULTORIA | 35 | 35 | 0.3873 |
| CASA CONCRETA | CONCREJOTA | 41 | 40 | 0.3856 |

### 9.2 Top 10 Falsos Negativos (Grupo 1 - escapes mais graves)

| Marca A | Marca B | cls A | cls B | score |
| --- | --- | --- | --- | --- |
| AVESTA | A.V.I. | 35 | 36 | 0.0776 |
| SEVERINO APP | SEU SEVERINO | 42 | 35 | 0.0879 |
| CONTABILIDADE MASTER | contabilidade.com | 35 | 9 | 0.0909 |
| CAFER PANIFICADORA | Cafezo | 35 | 30 | 0.0931 |

### 9.3 LISTA COMPLETA - Grupo 0 errados (rotulo=0, classificados como colidentes) - 84 pares

_Ordenados por score decrescente (alarmes de maior "confianca errada" primeiro). Cada linha eh um par que o modelo achou que era colidencia mas NAO era segundo o rotulo INPI._

| Marca A | Marca B | cls A | cls B | score |
| --- | --- | --- | --- | --- |
| Master Multi | Mastery7 | 35 | 35 | 0.6898 |
| D&F MODA FITNESS | Life Move Moda Fitness | 35 | 25 | 0.6037 |
| VIDRAÇARIA BRASÍLIA | VIDRAÇARIA MK | 37 | 37 | 0.5858 |
| VALE ZELAR | VALE JJR | 37 | 37 | 0.5643 |
| ESPETINHO DA KARLA DESDE 1996 | ESPETINHO DO BEBÊ | 43 | 43 | 0.5622 |
| RI ARQUITETURA | ACAIÁ ARQUITETURA | 42 | 42 | 0.5547 |
| MALU | MALULU KIDS | 16 | 25 | 0.4692 |
| MALU | MA5 | 16 | 37 | 0.3978 |
| Y.Consultoria | ZV CONSULTORIA | 35 | 35 | 0.3873 |
| CASA CONCRETA | CONCREJOTA | 41 | 40 | 0.3856 |
| MALU | MALAYA | 16 | 30 | 0.3848 |
| PP TREINAMENTOS | Entretreinamento | 41 | 41 | 0.3789 |
| RESTAURANTE GUARANI SELF SERVICE UBATUBA - SP | Baloi Restaurante | 40 | 43 | 0.3614 |
| BARBEARIA SETE DE PAUS | BARBEARIA 288 | 44 | 44 | 0.3582 |
| Vamo Street Mall | VAMORA | 36 | 10 | 0.3433 |
| FAZENDA HOMEM DE PEDRA | FAZENDA CAMARGO | 35 | 40 | 0.3428 |
| DISTRIBUIDORA BAESSA ÁGUA GÁS E CARVÃO | DISTRIBUIDORA NACIONAL 31 | 35 | 21 | 0.3417 |
| NACIONAL | Nacional Cards | 0 | 35 | 0.3405 |
| E A ESTUDIO ARTE | ESTÚDIO NT | 38 | 41 | 0.3336 |
| RESTAURANTE MENINA GERAIS | Restaurante Biu Rico | 43 | 43 | 0.3335 |
| BAZAR BOTÂNICO | BOTANICCA | 35 | 44 | 0.3244 |
| MENTORIA EM GRANDES OPERAÇÕES | Mentoria KETA | 41 | 41 | 0.3176 |
| DROGARIA SANDES | DROGARIAS X PRIME | 35 | 35 | 0.3158 |
| VIA BRASIL | LOJAS VIA BRASIL | 35 | 28 | 0.3054 |
| REI DAS CLÍNICAS | CLINICA LINUS | 44 | 44 | 0.2940 |
| ES CONSTRUÇÃO | GWL CONSTRUÇÃO | 41 | 37 | 0.2910 |
| ELETRO SOLAR | rda eletro | 35 | 35 | 0.2897 |
| ALLTECH | Allevo Tech | 45 | 41 | 0.2873 |
| ESPIRITO SANTO FASHION | 89.1 Espírito Santo FM | 35 | 41 | 0.2824 |
| Rénové Biocosmetic Spray Hidratante Corporal (Body Moisturizing) | RenovaMap | 3 | 42 | 0.2806 |
| NACIONAL | BETNACIONAL | 0 | 42 | 0.2788 |
| Auto Posto G&D | AUTO POSTO KS | 35 | 37 | 0.2758 |
| UNIVERSIDADE DAS ARTES | UNIFLEX UNIVERSIDADE USAFLEX | 41 | 41 | 0.2685 |
| PERSONALE PLANTAS | PERSONE | 42 | 5 | 0.2656 |
| FARMÁCIA MAR & VIDA | MAIS VIDA | 35 | 35 | 0.2597 |
| GRAN MAREIRO HOTEL | Gran Marrion | 43 | 30 | 0.2580 |
| ELÉVI+ | ELEV-C | 35 | 35 | 0.2530 |
| ESCRITA E LEITURA PARA EVOLUÇÃO E VALORIZAÇÃO NA APRENDIZAGEM ELEVA | elevato | 35 | 30 | 0.2523 |
| TERRAL MARESIAS | TERRADUU | -1 | 25 | 0.2424 |
| DONA DESIGN | Donalu | 35 | 21 | 0.2404 |
| NORDESTE EMPREENDEDOR | EMPREENDEDOR BRABO | 41 | 41 | 0.2397 |
| NEW ELLEGANCE | Monivelle Élégance | 44 | 35 | 0.2362 |
| SOMA FLUX | SOMAR | 35 | 35 | 0.2342 |
| PINEE | Pineal | 21 | 35 | 0.2313 |
| C CONTROLLER | CNCCONTROL | 0 | 37 | 0.2312 |
| ZZ CAPITÃO MUZZARELA | CAPITÃO CHICO | 43 | 43 | 0.2302 |
| EU ENGLISH UNIVERSE ONLINE | English Deploy | 41 | 41 | 0.2242 |
| Jucelino Móveis Planejados | LOOV Móveis Planejados | 35 | 35 | 0.2238 |
| STUDIO OFICINA 3D | STUDIO A | 35 | 11 | 0.2207 |
| ALLTEC SISTEMAS DE CONTROLE DESDE 1995 | Alltech | 35 | 5 | 0.2198 |
| MOVE ENERGIA | ENERGIL | 35 | 5 | 0.2193 |
| AÇAÍ CURITIBA | AÇAÍ BM | 30 | 29 | 0.2159 |
| EMPÓRIO POUSO ALEGRE | Empório CTZ | 35 | 35 | 0.2153 |
| DOM PASTEL | PASTEL DA NEIDE | 0 | 35 | 0.2144 |
| CONEXÃO FOOD HALL | Conexão Tanino | 43 | 35 | 0.2117 |
| Z ZEST | Z ZEXTER | 35 | 35 | 0.2096 |
| CEMINHA | Cestinha | 41 | 42 | 0.2062 |
| SYM | SIML | 36 | 41 | 0.2048 |
| RESTAURANTE MENINA GERAIS | L&J RESTAURANTE | 43 | 43 | 0.2020 |
| PELLE SANA CLÍNICA ESPECIALIZADA EM TRATAMENTO DE FERIDAS | SAN' MIELLE | 44 | 35 | 0.2004 |
| GLOBUSINESS CENTER | RA Agrobusiness | 35 | 35 | 0.1945 |
| CONCEITO Modular e Offsite | CONCEITO 4.0 | 6 | 35 | 0.1936 |
| RODA | RÓD | 18 | 42 | 0.1895 |
| P&P CONTABILIDADE E CONSULTORIA EMPRESARIAL | MONETIZEI CONTABILIDADE | 35 | 35 | 0.1891 |
| SORVETES DO VALE | SORVETES GE-LITTÁ | 35 | 30 | 0.1881 |
| MAIS MÓVEIS | MAIS ERP | 20 | 42 | 0.1870 |
| CASTELLA | CASAELLA | 11 | 35 | 0.1868 |
| SERELEITO.COM | SERELEPE | 42 | 3 | 0.1839 |
| Quintal Cozinha pra Torar | Quintal Animal | 43 | 43 | 0.1831 |
| MALU | Maloca | 16 | 41 | 0.1767 |
| RESGATE | O RESGATE | 9 | 45 | 0.1755 |
| PEIDO PRONTO | Peido Seco | 35 | 35 | 0.1703 |
| PASSAQUI ANTECIPA | PASSETUR | 36 | 39 | 0.1679 |
| PSICO STORE | Psicosfera | 35 | 35 | 0.1659 |
| MALU GUARDANAPOS DE PAPEL | MALA MÁGICA | 16 | 25 | 0.1646 |
| SUPERMERCADO NOSSA FAMÍLIA | FAMILY AÇAÍ | 35 | 35 | 0.1624 |
| SMART IMOBILI | B=SMART | 36 | 37 | 0.1622 |
| AVESTA | Siesta | 36 | 21 | 0.1574 |
| DESMONTADORA INOVA ECO PEÇAS | TW montadora | 35 | 37 | 0.1569 |
| ESTÂNCIA DOS CAMPOS GALILÉIA - MG | ESTÂNCIA NOBRE | 35 | 43 | 0.1537 |
| SOLUÇÕES CONDOMÍNIOS & SERVIÇOS | FAC Soluções | 36 | 35 | 0.1524 |
| PLANETA DOS FOGOS | Planeta Feijão | 13 | 35 | 0.1511 |
| FAZENDA HOMEM DE PEDRA | FAZENDA ÁFRICA | 44 | 35 | 0.1508 |
| Lumiê Lingerie & Co. | LUMIÈRE | 35 | 14 | 0.1506 |

### 9.4 LISTA COMPLETA - Grupo 1 errados (rotulo=1, escaparam da classificacao) - 4 pares

_Ordenados por score crescente (colidentes que receberam o menor score primeiro - mais graves para a operacao). Cada linha eh uma colidencia que o modelo deixou passar._

| Marca A | Marca B | cls A | cls B | score |
| --- | --- | --- | --- | --- |
| AVESTA | A.V.I. | 35 | 36 | 0.0776 |
| SEVERINO APP | SEU SEVERINO | 42 | 35 | 0.0879 |
| CONTABILIDADE MASTER | contabilidade.com | 35 | 9 | 0.0909 |
| CAFER PANIFICADORA | Cafezo | 35 | 30 | 0.0931 |

## 10. Comparativo NN vs heuristica OFTA (no conjunto de teste)

_Recall avaliado com piso de Precision >= 0.90._

| Metrica | NN | Heuristica OFTA | Delta |
| --- | ---:| ---:| ---:|
| ROC-AUC | 0.8328 | 0.4412 | +0.3915 |
| PR-AUC | 0.5326 | 0.1902 | +0.3424 |
| Recall@P>=0.9 | 0.0000 | 0.0000 | +0.0000 |

## 11. Importancia das features (Permutation Importance)

> Quanto maior a importancia, mais a metrica de validacao PIORA quando essa feature e embaralhada. Use para auditar onde o modelo esta apoiado.

### 11.1 Top 30 globais

| Feature | Importance |
| --- | --- |
| spec_item_align_score | 0.0197 |
| inter_proxy_spec | 0.0191 |
| spec_cosine_tfidf_char | 0.0174 |
| cls_pair_prior_pos | 0.0149 |
| spec_lex_jaccard | 0.0147 |
| inter_classe_igual_e_spec_proxima | 0.0137 |
| spec_lex_n_common | 0.0126 |
| inter_spec_x_same_cls | 0.0112 |
| graf_jaccard_trigram | 0.0109 |
| ofta_driver_aproximação | 0.0107 |
| num_orto_best | 0.0098 |
| fon_token_eq_share | 0.0093 |
| spec_kind_a_produto | 0.0084 |
| brand_b_token_in_spec_a | 0.0083 |
| cls_b_top_43 | 0.0078 |
| cls_a_top_9 | 0.0077 |
| spec_lex_idf_avg_common | 0.0076 |
| fon_key_eq | 0.0075 |
| spec_kind_a_servico | 0.0072 |
| spec_item_top3_mean_cosine | 0.0072 |
| ofta_driver_fonética | 0.0071 |
| spec_item_max_cosine_tfidf | 0.0071 |
| tok_jaccard | 0.0070 |
| spec_cosine_tfidf_word | 0.0070 |
| graf_prefix_norm | 0.0069 |
| cls_pair_chi2_strength | 0.0068 |
| against_unique_strong_b | 0.0065 |
| contain_after_strip_a_in_b | 0.0064 |
| fon_key_lev_sim | 0.0063 |
| num_fon_spread | 0.0062 |

### 11.2 Importancia agregada por bloco

| Bloco | Soma | Share |
| --- | --- | --- |
| Classe Nice | 0.0734 | 0.1778 |
| Spec_atividade | 0.0595 | 0.1441 |
| Interacoes | 0.0452 | 0.1094 |
| Spec_lex | 0.0443 | 0.1073 |
| Outras | 0.0443 | 0.1072 |
| OFTA | 0.0313 | 0.0757 |
| Foneticas | 0.0263 | 0.0638 |
| Numerais | 0.0247 | 0.0599 |
| Spec_cosine | 0.0243 | 0.0589 |
| Graficas | 0.0223 | 0.0541 |
| Tokens | 0.0173 | 0.0419 |

## 12. Limitacoes

- O modelo cobre exclusivamente marcas NOMINATIVAS; figurativas precisam de outro pipeline.
- Qualidade do rotulo INPI determina o teto pratico do modelo.
- Embeddings sao multilingues genericos; um modelo PT-BR especifico pode melhorar.
- Classes Nice raras viram bucket 'other' por construcao do top-K.
- Heuristicas produto/servico baseiam-se em listas-ancora; podem nao cobrir setores especificos.

## 13. Recomendacoes

- monitorar drift de score em producao (PSI mensal)
- retreinar a cada novo lote de pareceres revisados
- afinar threshold com a area de negocio para custo FN/FP otimo
- Reavaliar `recall_floor` em conjunto com a area de negocio para calibrar custo de FN/FP.
- Retreinar a cada novo lote significativo de pareceres revisados.
- Monitorar drift de score em producao (PSI sobre `score_nn`).

## 14. Notas adicionais

- Smoke test executado com amostra reduzida e poucas epocas.
