# Relatorio Completo - Modelo de Similaridade Aprendida de Marcas

_Gerado em 2026-04-30T15:32:05+00:00 (UTC)._

## Sumario executivo

**Veredito de prontidao:** `ACEITAVEL com monitoramento`

| Metrica | Treino | Validacao | Teste |
| --- | ---:| ---:| ---:|
| ROC-AUC | 0.9295 | 0.8664 | **0.8447** |
| PR-AUC  | 0.6387 | 0.5214 | **0.4795** |
| F1      | 0.4419 | 0.3949 | **0.3752** |
| Recall  | 0.9619 | 0.8538 | **0.8478** |
| Precision | 0.2868 | 0.2568 | **0.2409** |

**Threshold de operacao:** `0.2600` (politica: `max_f1_with_recall>=0.85`, recall_floor=0.85)

**Pontos fortes:**
- PR-AUC 0.479 (bom para base desbalanceada)

**Atencao:**
- ROC-AUC 0.845 (aceitavel, mas pode melhorar)
- Recall 0.848 abaixo do piso 0.85 (perde colidentes)
- F1 0.375 (equilibrio modesto)

**Sinais de overfit (treino - teste):**

| Metrica | delta train-test |
| --- | ---:|
| ROC-AUC train-test | +0.0848 |
| PR-AUC train-test | +0.1592 |
| Recall train-test | +0.1141 |
| F1 train-test | +0.0667 |

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

- TF-IDF word: ngram=(1, 2), min_df=3, max_features=20000
- TF-IDF char: ngram=(3, 5), min_df=3, max_features=10000
- Embeddings ativos: **True** (modelo: `paraphrase-multilingual-MiniLM-L12-v2`)
- Cache de embeddings: `artifacts/embeddings_cache.parquet`
- Top-15 classes Nice usadas em one-hot: [35, 41, 42, 44, 43, 36, 9, 37, 30, 40, 25, 39, 45, 16, 3]
- Total de features na ordem canonica: **116**

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

## 3. Estatisticas das features (apos StandardScaler)

- Linhas usadas para estatistica: 30941
- Features com variancia ZERO: **1** (possiveis candidatas a remocao)

Features sem variancia detectadas: `cls_b_known`

### 3.1 Top features por desvio padrao (apos scaler - util p/ debug)

| Feature | media | desvio | min | max |
| --- | --- | --- | --- | --- |
| spec_kind_a_produto | -0.0000 | 1.0002 | -0.3917 | 2.5532 |
| cls_b_top_40 | 0.0000 | 1.0002 | -0.1433 | 6.9760 |
| num_fon_spread | 0.0000 | 1.0002 | -0.1625 | 19.1892 |
| inter_nome_alto_e_spec_baixa | -0.0000 | 1.0002 | -0.2923 | 3.4208 |
| spec_kind_b_produto | -0.0000 | 1.0002 | -0.4519 | 2.2126 |
| cls_b_top_39 | -0.0000 | 1.0002 | -0.1309 | 7.6412 |
| cls_b_top_37 | 0.0000 | 1.0002 | -0.1628 | 6.1420 |
| spec_kind_a_servico | 0.0000 | 1.0002 | -0.6842 | 1.4617 |
| cls_b_top_45 | -0.0000 | 1.0001 | -0.1311 | 7.6263 |
| cls_b_top_44 | 0.0000 | 1.0001 | -0.2154 | 4.6424 |
| inter_spec_x_same_cls | 0.0000 | 1.0001 | -0.4561 | 4.0203 |
| inter_classe_igual_e_spec_proxima | 0.0000 | 1.0001 | -0.4561 | 4.0203 |
| num_orto_best | -0.0000 | 1.0001 | -0.2051 | 9.6329 |
| tok_overlap | -0.0000 | 1.0001 | -0.7677 | 2.7102 |
| ofta_driver_ortografia | 0.0000 | 1.0001 | -0.3847 | 2.5993 |
| spec_kind_b_servico | -0.0000 | 1.0001 | -0.7468 | 1.3390 |
| cls_a_top_other | 0.0000 | 1.0001 | -0.3268 | 3.0597 |
| fon_after_dedup_eq | -0.0000 | 1.0001 | -0.0712 | 14.0478 |
| cls_same | 0.0000 | 1.0001 | -0.5331 | 1.8757 |
| n_tokens_excl_b | -0.0000 | 1.0001 | -2.1586 | 8.3973 |

### 3.2 Top features por |correlacao de Pearson com a label|

| Feature | Pearson |
| --- | --- |
| spec_lex_idf_avg_common | 0.2468 |
| spec_lex_overlap | 0.2161 |
| inter_proxy_spec | 0.2121 |
| spec_cosine_emb | 0.2097 |
| spec_cosine_tfidf_char | 0.2083 |
| spec_lex_n_common | 0.1777 |
| spec_cosine_tfidf_word | 0.1689 |
| ofta_token | 0.1629 |
| ofta_fuzzy | 0.1627 |
| tok_fuzzy | 0.1627 |
| inter_nome_x_spec_max | 0.1604 |
| inter_nome_x_spec_emb | 0.1597 |
| inter_spec_x_same_cls | 0.1526 |
| inter_classe_igual_e_spec_proxima | 0.1526 |
| spec_lex_jaccard | 0.1483 |
| graf_overlap_trigram | 0.1390 |
| graf_jaccard_bigram | 0.1360 |
| fon_token_mean | 0.1279 |
| inter_nome_x_spec_word | 0.1260 |
| graf_jaccard_trigram | 0.1248 |

## 4. Estrategia de balanceamento

- Undersample negativos: ratio **3.00 : 1** (neg/pos alvo)
- Oversample positivos: fator **2.00x**
- Class weight (pos_weight) ativo: **True**, valor efetivo **1.500**
- Seed: 42
- Split estratificado: train=70% / val=15% / test=15%

## 5. Arquitetura e hiperparametros

### 5.1 Arquitetura

- input_dim: **116**
- hidden_dims: **[128, 64, 32]**
- ativacao: **relu**
- dropout: **0.3**
- batchnorm: **True**
- output: Linear(., 1) -> Sigmoid
- **# parametros aprendiveis: 25,793**

### 5.2 Hiperparametros de treino

- epochs: **60**
- batch_size: **256**
- lr: **0.001**
- weight_decay: **0.0001**
- early_stopping_patience: **10**
- scheduler_patience: **4**
- scheduler_factor: **0.5**
- min_lr: **1e-06**
- grad_clip: **1.0**
- device: **auto**
- seed: **42**
- best_epoch: **32**
- best_pr_auc_val: **0.5214376718414918**
- n_train_after_balancing: **11805**

- Otimizador: **AdamW**, Loss: **BCEWithLogitsLoss(pos_weight)**, Scheduler: **ReduceLROnPlateau** (max em val PR-AUC)

## 6. Historico do treino (epoca a epoca)

| epoch | train_loss | lr | train_pr_auc | val_pr_auc | val_roc_auc | val_recall@0.5 | val_f1@0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.7393 | 0.0010 | 0.7515 | 0.3937 | 0.8117 | 0.7233 | 0.3841 |
| 2 | 0.6360 | 0.0010 | 0.7907 | 0.4426 | 0.8384 | 0.7905 | 0.3945 |
| 3 | 0.6042 | 0.0010 | 0.8123 | 0.4637 | 0.8470 | 0.7708 | 0.4207 |
| 4 | 0.5766 | 0.0010 | 0.8274 | 0.4762 | 0.8550 | 0.7885 | 0.4213 |
| 5 | 0.5628 | 0.0010 | 0.8418 | 0.4851 | 0.8577 | 0.7609 | 0.4353 |
| 6 | 0.5477 | 0.0010 | 0.8511 | 0.4931 | 0.8616 | 0.7905 | 0.4348 |
| 7 | 0.5316 | 0.0010 | 0.8587 | 0.4845 | 0.8611 | 0.7609 | 0.4418 |
| 8 | 0.5286 | 0.0010 | 0.8654 | 0.5062 | 0.8667 | 0.7866 | 0.4400 |
| 9 | 0.5138 | 0.0010 | 0.8733 | 0.4955 | 0.8630 | 0.7648 | 0.4400 |
| 10 | 0.5121 | 0.0010 | 0.8794 | 0.4933 | 0.8617 | 0.7589 | 0.4220 |
| 11 | 0.4913 | 0.0010 | 0.8845 | 0.4978 | 0.8615 | 0.7490 | 0.4230 |
| 12 | 0.4945 | 0.0010 | 0.8901 | 0.5006 | 0.8637 | 0.7431 | 0.4466 |
| 13 | 0.4775 | 0.0005 | 0.8983 | 0.4985 | 0.8619 | 0.7431 | 0.4416 |
| 14 | 0.4630 | 0.0005 | 0.9032 | 0.5073 | 0.8651 | 0.7668 | 0.4355 |
| 15 | 0.4571 | 0.0005 | 0.9067 | 0.5069 | 0.8659 | 0.7510 | 0.4581 |
| 16 | 0.4541 | 0.0005 | 0.9097 | 0.5061 | 0.8659 | 0.7194 | 0.4622 |
| 17 | 0.4428 | 0.0005 | 0.9135 | 0.5100 | 0.8652 | 0.7708 | 0.4412 |
| 18 | 0.4444 | 0.0005 | 0.9157 | 0.5090 | 0.8637 | 0.7688 | 0.4508 |
| 19 | 0.4394 | 0.0005 | 0.9165 | 0.5157 | 0.8642 | 0.7668 | 0.4362 |
| 20 | 0.4406 | 0.0005 | 0.9204 | 0.5128 | 0.8646 | 0.7708 | 0.4455 |
| 21 | 0.4283 | 0.0005 | 0.9216 | 0.5145 | 0.8634 | 0.7688 | 0.4368 |
| 22 | 0.4292 | 0.0005 | 0.9248 | 0.5172 | 0.8651 | 0.7530 | 0.4504 |
| 23 | 0.4236 | 0.0005 | 0.9274 | 0.5084 | 0.8621 | 0.7391 | 0.4536 |
| 24 | 0.4286 | 0.0005 | 0.9281 | 0.5082 | 0.8627 | 0.7648 | 0.4383 |
| 25 | 0.4180 | 0.0005 | 0.9292 | 0.5135 | 0.8637 | 0.7490 | 0.4420 |
| 26 | 0.4279 | 0.0005 | 0.9310 | 0.5152 | 0.8651 | 0.7569 | 0.4438 |
| 27 | 0.4122 | 0.0003 | 0.9337 | 0.5155 | 0.8645 | 0.7411 | 0.4459 |
| 28 | 0.4065 | 0.0003 | 0.9357 | 0.5148 | 0.8649 | 0.7490 | 0.4512 |
| 29 | 0.3986 | 0.0003 | 0.9371 | 0.5133 | 0.8647 | 0.7767 | 0.4384 |
| 30 | 0.3952 | 0.0003 | 0.9387 | 0.5079 | 0.8644 | 0.7668 | 0.4594 |
| 31 | 0.3976 | 0.0003 | 0.9403 | 0.5208 | 0.8652 | 0.7292 | 0.4674 |
| 32 | 0.3968 | 0.0003 | 0.9394 | 0.5214 | 0.8664 | 0.7727 | 0.4512 |
| 33 | 0.3940 | 0.0003 | 0.9404 | 0.5214 | 0.8670 | 0.7530 | 0.4571 |
| 34 | 0.3977 | 0.0003 | 0.9425 | 0.5156 | 0.8653 | 0.7391 | 0.4669 |
| 35 | 0.3932 | 0.0003 | 0.9429 | 0.5101 | 0.8648 | 0.7352 | 0.4694 |
| 36 | 0.3855 | 0.0003 | 0.9435 | 0.5176 | 0.8655 | 0.7411 | 0.4582 |
| 37 | 0.3861 | 0.0001 | 0.9435 | 0.5193 | 0.8655 | 0.7609 | 0.4687 |
| 38 | 0.3858 | 0.0001 | 0.9448 | 0.5166 | 0.8649 | 0.7411 | 0.4702 |
| 39 | 0.3821 | 0.0001 | 0.9457 | 0.5159 | 0.8639 | 0.7332 | 0.4658 |
| 40 | 0.3864 | 0.0001 | 0.9457 | 0.5193 | 0.8656 | 0.7490 | 0.4588 |
| 41 | 0.3833 | 0.0001 | 0.9462 | 0.5192 | 0.8660 | 0.7451 | 0.4657 |
| 42 | 0.3780 | 0.0001 | 0.9467 | 0.5200 | 0.8665 | 0.7628 | 0.4590 |


## 7. Metricas em detalhe

Threshold otimo: **0.2600** (politica: `max_f1_with_recall>=0.85`, recall_floor=0.85)

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

## 8. Distribuicao de scores no teste

| Classe | n | mean | std | p10 | p25 | p50 | p75 | p90 |
| --- | ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:|
| pos | 506 | 0.6933 | 0.3127 | 0.1133 | 0.5017 | 0.8379 | 0.9378 | 0.9817 |
| neg | 4136 | 0.2422 | 0.3003 | 0.0021 | 0.0120 | 0.0769 | 0.4346 | 0.7835 |

_Diferenca de media (pos - neg): **+0.4511**. Quanto maior, melhor a separabilidade._

### 8.1 Decis de score (10 = scores mais altos)

| Decil | score_min | score_max | n | positivos | % positivos |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.8588 | 0.9996 | 465 | 237 | 0.5097 |
| 1 | 0.6821 | 0.8585 | 464 | 94 | 0.2026 |
| 2 | 0.4553 | 0.6818 | 464 | 60 | 0.1293 |
| 3 | 0.2300 | 0.4552 | 464 | 43 | 0.0927 |
| 4 | 0.1123 | 0.2299 | 464 | 22 | 0.0474 |
| 5 | 0.0525 | 0.1119 | 464 | 19 | 0.0409 |
| 6 | 0.0232 | 0.0525 | 464 | 12 | 0.0259 |
| 7 | 0.0092 | 0.0231 | 464 | 8 | 0.0172 |
| 8 | 0.0026 | 0.0092 | 464 | 10 | 0.0216 |
| 9 | 0.0000 | 0.0025 | 465 | 1 | 0.0022 |

## 9. Analise de erros (no teste)

- Total de pares no teste: **4642** (threshold=0.2600)
- **Grupo 0** (rotulo real = 0, NAO colidente): 4136 pares no total. **1352 foram classificados como colidentes (taxa de erro 32.69%)**.
- **Grupo 1** (rotulo real = 1, colidentes): 506 pares no total. **77 escaparam da classificacao (taxa de erro 15.22%)**.
- Lista completa Grupo 0 errados (CSV): `artifacts\falsos_positivos_grupo_0.csv`
- Lista completa Grupo 1 errados (CSV): `artifacts\falsos_negativos_grupo_1.csv`

### 9.1 Top 10 Falsos Positivos (Grupo 0 - alarmes mais graves)

| Marca A | Marca B | cls A | cls B | score |
| --- | --- | --- | --- | --- |
| A2R ENERGIA SOLAR | A2R2 | 35 | 40 | 0.9934 |
| SUPERMERCASA | SUPERMERCADO PAIVA | 35 | 5 | 0.9929 |
| GESTAÇÃO | Gestar + | 9 | 42 | 0.9914 |
| BEONFASHION ON-LINE STYLE | Jan Fashion | 25 | 25 | 0.9911 |
| FA FASHION ATACADISTA O VAREJO NO PRECINHO DE ATACADO | DIVA FASHION | 25 | 25 | 0.9908 |
| SISAL JEANS | SIX | 25 | 40 | 0.9899 |
| VEST GOSPEL | VEXTER | 25 | 25 | 0.9885 |
| UNIFOR-MG CENTRO UNIVERSITÁRIO DE FORMIGA | UNISAT UNIFORMES | 25 | 25 | 0.9879 |
| DOM PASTEL | PASTÉ | 42 | 43 | 0.9875 |
| OdontoCape centro odontológico | ODONTOUR | 41 | 44 | 0.9869 |

### 9.2 Top 10 Falsos Negativos (Grupo 1 - escapes mais graves)

| Marca A | Marca B | cls A | cls B | score |
| --- | --- | --- | --- | --- |
| PRINCIPIA CONSULTORIA E TREINAMENTO | PA PrincipiAndo | 41 | 41 | 0.0018 |
| Ô CREPE! | CREPE 33 | 43 | 39 | 0.0047 |
| AMPLA | AMPLAPACK | 0 | 16 | 0.0048 |
| REVESTIMENTOS CASA 7 | +CASA | 35 | 37 | 0.0049 |
| OFTALMOAMIGO | OFTALMO TOP | 44 | 41 | 0.0049 |
| UMTELECOM | A.R TELECOM | 9 | 42 | 0.0061 |
| Brasil Farma Prime | FARMER BRAZIL | 35 | 36 | 0.0075 |
| AGROPEREIRA | AGRO FRONTEIRA | 35 | 39 | 0.0077 |
| X BRONZE BRONZEAMENTO NATURAL | Seu Bronze | 44 | 44 | 0.0078 |
| Q.SABOR PIZZARIA DELIVERY | Sabor e Sabores | 39 | 35 | 0.0089 |

### 9.3 LISTA COMPLETA - Grupo 0 errados (rotulo=0, classificados como colidentes) - 1352 pares

_Ordenados por score decrescente (alarmes de maior "confianca errada" primeiro). Cada linha eh um par que o modelo achou que era colidencia mas NAO era segundo o rotulo INPI._

| Marca A | Marca B | cls A | cls B | score |
| --- | --- | --- | --- | --- |
| A2R ENERGIA SOLAR | A2R2 | 35 | 40 | 0.9934 |
| SUPERMERCASA | SUPERMERCADO PAIVA | 35 | 5 | 0.9929 |
| GESTAÇÃO | Gestar + | 9 | 42 | 0.9914 |
| BEONFASHION ON-LINE STYLE | Jan Fashion | 25 | 25 | 0.9911 |
| FA FASHION ATACADISTA O VAREJO NO PRECINHO DE ATACADO | DIVA FASHION | 25 | 25 | 0.9908 |
| SISAL JEANS | SIX | 25 | 40 | 0.9899 |
| VEST GOSPEL | VEXTER | 25 | 25 | 0.9885 |
| UNIFOR-MG CENTRO UNIVERSITÁRIO DE FORMIGA | UNISAT UNIFORMES | 25 | 25 | 0.9879 |
| DOM PASTEL | PASTÉ | 42 | 43 | 0.9875 |
| OdontoCape centro odontológico | ODONTOUR | 41 | 44 | 0.9869 |
| LULUI BIKINIS | LUBE | 25 | 40 | 0.9868 |
| MD SISTEMAS | SISTEMAG | 42 | 42 | 0.9866 |
| FARDIN | Farin | 30 | 30 | 0.9823 |
| BELLA DAMA COSMÉTICOS | D'Bella | 35 | 3 | 0.9804 |
| UMTELECOM | HG TELECOM | 38 | 38 | 0.9789 |
| COLMEIA | KOLMEIA | 42 | 42 | 0.9771 |
| PASTEL DA TERRA | PASTELÊ | 43 | 30 | 0.9765 |
| Virtua Vet Coworking Veterinário | VETTES | 44 | 35 | 0.9765 |
| ATELIÊ BOLDO | ATELIÊ DA GI | 25 | 25 | 0.9760 |
| G GRA VIAGENS | grão. | 39 | 39 | 0.9757 |
| MARCENARIA MLM LUCAS MÓVEIS | MARCENAR | 20 | 20 | 0.9754 |
| Scanauto | Scana | 37 | 42 | 0.9743 |
| MARLEX SELAS E ACESSÓRIOS | MARLELI | 35 | 25 | 0.9742 |
| FAMÍLIA CHURRASQUINHO | HAMAGLIA | 35 | 25 | 0.9740 |
| Hconnect | ZOECONNECT | 42 | 42 | 0.9735 |
| CB CASA BETEL | CASAE | 35 | 20 | 0.9732 |
| Multihype | Multifeme | 28 | 41 | 0.9724 |
| GREEN GUPPY | GREEN MONEY | 25 | 35 | 0.9716 |
| UMTELECOM | A9 Telecom | 38 | 38 | 0.9706 |
| MISS NINA | Nini | 25 | 35 | 0.9703 |
| DURAFORT | ARAFORT | 11 | 9 | 0.9700 |
| RELEVANTE | R RELEVANTE NEWS | 41 | 41 | 0.9697 |
| LINAUS CLINIC | LINAR | 44 | 9 | 0.9696 |
| CONTÁBIL TURMALINA | CONTABILEIS | 35 | 35 | 0.9688 |
| UMTELECOM | F6 TELECOM | 38 | 38 | 0.9669 |
| EU TE PEGO | ICE EU TE PEGO! | 35 | 33 | 0.9656 |
| SANAUTO | SOSAN AUTO | 12 | 7 | 0.9648 |
| R W | RW | 35 | 35 | 0.9646 |
| MÉTODO 3T | MÉTODO DES | 41 | 41 | 0.9628 |
| LY BEACHWEAR | LIVELI | 25 | 25 | 0.9620 |
| A ALTERNATIVA | ALTIVA | 35 | 25 | 0.9609 |
| ArteSanto Feira Nacional do Artesanato do Espírito Santo | ARTESANOU | 35 | 35 | 0.9602 |
| LIFE SPACE | theSpace | 42 | 42 | 0.9596 |
| SMART SCOOTERS | B=SMART | 35 | 35 | 0.9580 |
| AIZU ENGENHARIA E ARQUITETURA | AIZI. | 42 | 35 | 0.9572 |
| FESTIVAL DE OLHO NO PORCO | DE OLHO NO ESPORTE | 41 | 41 | 0.9571 |
| Virtua Vet Coworking Veterinário | TAVET | 44 | 5 | 0.9569 |
| BOGHOS BOYADJIAN CADI CENTRO AVANÇADO DE DIAGNOSTICO POR IMAGEM | CADI CENTRO AVANÇADO DE DIAGNÓSTICO POR IMAGEM | 44 | 10 | 0.9567 |
| XES BRASIL | SE7ES | 25 | 25 | 0.9566 |
| SR. & SRA. FIT | SRX | 35 | 5 | 0.9563 |
| QUINTAL DO COCO | QUINTAL DO ZICO | 39 | 43 | 0.9561 |
| IVIDROS | NM VIDROS | 35 | 19 | 0.9553 |
| MÓVEIS PAGANINI | Harmoni MÓVEIS | 20 | 20 | 0.9552 |
| CHAMA EMPREENDEDORA | Moda Empreendedora | 41 | 41 | 0.9518 |
| Temperos Magia | TEMPERO DO EDU | 30 | 30 | 0.9513 |
| ALPHA CO. | ALPHA PRIME | 25 | 40 | 0.9490 |
| ARENA BEACH | OWL BEACH ARENA | 38 | 41 | 0.9489 |
| CASA D INTERIORES | Casa do Véu | 35 | 25 | 0.9488 |
| AUTO PEÇAS MATOS | AUTOPECASLTZ2020 | 35 | 35 | 0.9485 |
| Festival de Inverno de Garanhuns FIG | Festival de Vinhos de Inverno | 41 | 41 | 0.9463 |
| A MAR A VIDA | Vida | 41 | 41 | 0.9456 |
| PASTEPIZZA LANCHES | PASTÉ | 43 | 43 | 0.9454 |
| Elevant Consultoria | ELEVATE | 35 | 44 | 0.9453 |
| DOCES DELLÉ | DOCES + | 35 | 30 | 0.9450 |
| M MANIX | Manyoo | 25 | 18 | 0.9437 |
| data pool.io | datapoli | 42 | 35 | 0.9437 |
| ANA CLARA | Ana Clara | 40 | 20 | 0.9414 |
| SUPERMERCASA | SUPERMERCADO PAIVA | 35 | 8 | 0.9412 |
| DEEP COLLECTION | Alvile collection | 25 | 25 | 0.9410 |
| BOUTIQUE PET LUCKY JR | LT BOUTIQUE | 35 | 40 | 0.9405 |
| AXÉ DA SORTE | axexo | 36 | 36 | 0.9404 |
| Virtua Vet Coworking Veterinário | Uravet | 44 | 10 | 0.9394 |
| ELETRO SOLAR | SIM Eletro | 35 | 35 | 0.9394 |
| CLINICA D.O.C SAÚDE PERSONALIZADA | Clini | 44 | 42 | 0.9390 |
| AMAZON PRODUTOS NATURAIS | amazoca | 30 | 43 | 0.9382 |
| MOODZ COLLECTION | WE COLLECTION | 25 | 25 | 0.9381 |
| CONFEITARIA AURORA ITABIRITO | AL confeitaria | 43 | 35 | 0.9372 |
| Trivora Utilidades | Trivo | 35 | 43 | 0.9371 |
| PARADISE CHANNELS | Channels | 38 | 41 | 0.9370 |
| É DE CASA ENERGIA SOLAR | +CASA | 42 | 37 | 0.9369 |
| SANTO AFETO TEEN & KIDS | Santo Axé | 25 | 35 | 0.9363 |
| CONTGA | CONTGO | 37 | 42 | 0.9359 |
| E.CONTROL | ABACONTROL | 9 | 42 | 0.9355 |
| MINISTÉRIO SEMELHANTE | MINISTÉRIO ÉFOZ | 41 | 41 | 0.9349 |
| SUPER FRANGUINHO | SUPERMERCADO MORANGUINHO | 35 | 30 | 0.9341 |
| D PRIMES AÇAIZINHO | PRIMEBOI | 30 | 35 | 0.9340 |
| VirtuaMed O hub da saúde coworking médico | ORAMED | 44 | 44 | 0.9337 |
| COFFEE BREAK COCO BAMBU | BAMBUÍ | 43 | 43 | 0.9336 |
| P COFFEE | Coffeenet | 43 | 30 | 0.9330 |
| MÉTODO 3T | Método RGP | 41 | 41 | 0.9319 |
| Be You Beauty Clinic By Vanessa Rosalem | By Be You | 44 | 35 | 0.9318 |
| Vila Alimentos | VYLO | 30 | 9 | 0.9300 |
| CENTRO DE TREINAMENTO THE MOVEMENT | ULTRA5 CENTRO DE TREINAMENTO | 41 | 41 | 0.9299 |
| Prato Pet | PRATO ÚNICO | 31 | 43 | 0.9298 |
| Atrium Parfum | Atrium | 35 | 30 | 0.9295 |
| MÉTODO 3T | Método RNA | 41 | 41 | 0.9294 |
| AMOR E SORRISOS IMPLANTES | Sorriso real | 35 | 44 | 0.9291 |
| SUPER FRANGUINHO | SUPERMERCADO MORANGUINHO | 35 | 39 | 0.9289 |
| SUPER LAGOA | SUPER LUZ | 30 | 35 | 0.9272 |
| AUTHENTIC BIKE SHOP | AUTENTICE | 35 | 35 | 0.9241 |
| BISCOITOS SARETTA | Biscoiterices | 30 | 30 | 0.9227 |
| CV / INVEST DTVM | INVEXUS | 36 | 36 | 0.9225 |
| GABRIEL SOUZA | Gabriel | 41 | 41 | 0.9213 |
| ELETRO SOLAR | ELETROAMAPÁ | 35 | 40 | 0.9209 |
| NORMATEL INCORPORAÇÕES | NORMANI | 36 | 37 | 0.9204 |
| PROFIX | ProFio | 35 | 44 | 0.9194 |
| AVESTA | AVI | 36 | 36 | 0.9192 |
| SABORES EM FRUTA | Saboreô | 43 | 43 | 0.9192 |
| Marketing Erótico | Get Marketing | 41 | 41 | 0.9178 |
| TRU LOGÍSTICA | Trevo | 43 | 30 | 0.9175 |
| ÁGUA MINERAL QUALITY LIFE | EQUALITY | 32 | 35 | 0.9170 |
| CAFÉ BARREIRO | Café Vô Guerreiro | 40 | 40 | 0.9170 |
| Empório Casa & Corpo | EMPÓRIO OF KING | 35 | 3 | 0.9167 |
| LOVE PET A MARCA DE ACESSÓRIOS DO SEU PET... | Pet Lobs | 7 | 35 | 0.9165 |
| POP RIO | PRIO | 44 | 42 | 0.9164 |
| Prato Pet | PRATO CHEIO | 31 | 35 | 0.9159 |
| NATTÚRIAS | NATURY | 35 | 29 | 0.9158 |
| UNA GRANDE DESENVOLVIMENTO | GRANDO | 36 | 35 | 0.9153 |
| BOUTIQUE PET LUCKY JR | BOUTIQUE DAYIL | 35 | 25 | 0.9152 |
| AGÊNCIA ART | AGÊNCIACROSBY. | 35 | 35 | 0.9144 |
| START COMUNICAÇÃO INTEGRADA | STARKE | 35 | 35 | 0.9138 |
| TEXAS BURGER | TEXATERAS | 43 | 30 | 0.9136 |
| DOE DE CORAÇÃO | CORAÇÃO DE MÃE | 41 | 38 | 0.9134 |
| MÉTODO DENTISTA MILIONÁRIO | Protagonista Milionário | 41 | 41 | 0.9122 |
| EMPÓRIO ESTILO | EMPÓRIO POMARKET | 35 | 35 | 0.9119 |
| AKAI AÇAÍ DO BRASIL | AKAE | 35 | 35 | 0.9119 |
| IVIDROS | NM VIDROS | 21 | 35 | 0.9094 |
| HARDFERRAMENTAS | Pinho Ferramentas | 35 | 8 | 0.9092 |
| CAMPOS DO JORDÃO | campos do jordão design festival | 0 | 41 | 0.9089 |
| açaí Vitaly Pura energia | VITALE | 35 | 33 | 0.9078 |
| DELÍCIAS DA XÚ LANCHONETE | Delícias da Tarde | 43 | 43 | 0.9068 |
| Reduto | REDUTO | 42 | 33 | 0.9065 |
| EMPÓRIO POUSO ALEGRE | Emporio WEB | 35 | 21 | 0.9064 |
| RANCHO DO ACARAJÉ | RANCHO BENNU | 43 | 30 | 0.9062 |
| Janutri | Sua Nutri | 42 | 9 | 0.9060 |
| ELETRO SOLAR | ELETRONLIMA | 35 | 42 | 0.9054 |
| A MAR A VIDA | MARV | 35 | 25 | 0.9050 |
| Chama | CHAMA! | 35 | 34 | 0.9048 |
| LULUI BIKINIS | LULU BROWNIE | 25 | 30 | 0.9048 |
| HEAD MADE | Headii | 42 | 9 | 0.9042 |
| PLANO DE CARREIRA | CARREIRA LIVE | 41 | 41 | 0.9041 |
| DIARIO NO CAMPO | CODIGO DIARIO | 41 | 41 | 0.9033 |
| MY BIRDS | flybirds | 35 | 35 | 0.9030 |
| energia natural | A NATURAL | 30 | 35 | 0.9027 |
| INFOPRODUTO 360º | Infoproduto Fura Fila | 41 | 41 | 0.9020 |
| VITOMAR SUPERMERCADO | VYBE SUPERMERCADO | 35 | 35 | 0.9020 |
| DOMUS | Domus Prime | 35 | 35 | 0.9020 |
| JAPA | JAPA FAST | 40 | 43 | 0.9018 |
| ÓTICA MIDO | ÓTICAS CLIN | 35 | 35 | 0.9012 |
| DELÍCIAS DA MIRA DO DOCE AO SALGADO | DELÍCIAS DA LENNY | 35 | 35 | 0.9007 |
| Virtua Office Coworking Ideas | VIRAW | 35 | 35 | 0.9004 |
| BA COFFEE | COFFEX | 43 | 35 | 0.9000 |
| OKA PLANEJADOS | OKARI | 35 | 30 | 0.8994 |
| predicta | predictus | 36 | 42 | 0.8991 |
| PP TREINAMENTOS | Aí TREINAMENTOS | 41 | 41 | 0.8984 |
| SUPERMERCASA | SUPER C | 35 | 35 | 0.8971 |
| BARRA CONNECT PROVEDOR DE INTERNET | CONNECTWAY | 38 | 35 | 0.8970 |
| GRUPO S | GRUPO LEH | 35 | 35 | 0.8962 |
| MASTER ALUMÍNIO | MASTERIZA | 35 | 35 | 0.8955 |
| DOE DE CORAÇÃO | CORAÇÃO DE PEDRA | 41 | 41 | 0.8951 |
| EMPREENDER NÃO É BAGUNÇA | Empreendo-Me | 41 | 41 | 0.8940 |
| SMART ELETRÔNICOS | SMARTCOMP | 35 | 35 | 0.8938 |
| RI ARQUITETURA | Arquitetura Cocriadora | 42 | 42 | 0.8936 |
| CI CALHAS IPATINGA | CALHAS DIAS | 6 | 35 | 0.8929 |
| PORTAL DOS REIS RECEPÇÕES EVENTOS | Portal dos Sonhos | 41 | 43 | 0.8928 |
| SEU DIREITO | Direito mais café | 38 | 41 | 0.8924 |
| ELETRO SOLAR | NIELETRO | 35 | 11 | 0.8921 |
| AGÊNCIA ART | AGÊNCIA K1 | 35 | 35 | 0.8895 |
| PÃO DE QUEIJO TINÔ | Pão de Queijo Dona Nice | 30 | 35 | 0.8892 |
| NORMATEL INCORPORAÇÕES | NORMANI | 37 | 37 | 0.8883 |
| BOIADEIRINHA | BOIADEIRA | 41 | 38 | 0.8878 |
| MEDICALVIS | MEDICAL DENTE | 35 | 44 | 0.8877 |
| OHIMÓVEIS | A3 Imóveis | 36 | 36 | 0.8872 |
| Método Post Irresistível | MÉTODO HOKMÃ | 41 | 41 | 0.8857 |
| ELEVA POR RENATA PAULA SANTIAGO | Elevus | 35 | 35 | 0.8855 |
| EU ENGLISH UNIVERSE | English Growth | 41 | 41 | 0.8845 |
| MUNDO EM MENU | MUNDO MAR | 38 | 41 | 0.8844 |
| Reduto | reduto | 42 | 35 | 0.8842 |
| SUPERMERCASA | SUMMER SUPERMERCADOS | 35 | 35 | 0.8833 |
| Virtua Vet Coworking Veterinário | VIRAW | 35 | 35 | 0.8828 |
| Six Pilates Studio | ZENA PILATES | 41 | 41 | 0.8819 |
| SABOR DO VALE ALIMENTOS | SABOR DO VERÃO | 29 | 43 | 0.8818 |
| SER SAÚDE | Saudee | 41 | 35 | 0.8803 |
| FORTALEZA MAR HOTEL | PORTALE | 43 | 43 | 0.8797 |
| REIS CELL | RLY REIS | 35 | 35 | 0.8795 |
| VIA VERDE | VY.A | 44 | 35 | 0.8791 |
| HARDFERRAMENTAS | JOTAFER Ferramentas | 35 | 7 | 0.8786 |
| MY | MY | 35 | 35 | 0.8782 |
| Hama Decor | FAMA Casa & Decor | 24 | 35 | 0.8781 |
| LÁ NO PASTEL | IPASTEL | 43 | 39 | 0.8773 |
| Agência - Assistência de Elite | AGÊNCIA PAPO | 35 | 35 | 0.8768 |
| Academia Brasileira de Psicologia Baseada em Evidências | ACADEMIA BRASILEIRA DE CINEMA | 35 | 41 | 0.8763 |
| MERCADO DA MAKE | MERCADO DA CHAPA | 35 | 35 | 0.8754 |
| REDE BRASIL DE NEGÓCIOS | BRASILIA | 35 | 12 | 0.8740 |
| PRO PRODUÇÕES E PROPAGANDA | ZMX PRODUÇÕES | 41 | 41 | 0.8738 |
| EMPÓRIO COCADAS OLIMPIO | EMPÓRIO GAMBITO | 35 | 35 | 0.8729 |
| ENGENHAR CONSULTORIA | Engenorte Engenharia | 42 | 37 | 0.8728 |
| ACRIMAX CAST ACRYLIC | LUCRIMAX | 17 | 35 | 0.8722 |
| SYM | SISMIX | 36 | 36 | 0.8719 |
| INSTITUTO REVER SETE QUEDAS | INSTITUTO OPERA | 41 | 41 | 0.8719 |
| REDE SUPER RURAL | SUPER C | 35 | 35 | 0.8712 |
| Bella NOBRE DOCES | Bellanda | 29 | 29 | 0.8712 |
| EMPÓRIO COCADAS OLIMPIO | EMPÓRIO ZOE | 35 | 35 | 0.8706 |
| Bella NOBRE DOCES | BEL | 35 | 35 | 0.8700 |
| GRUPO MS MAIS QUE SAÚDE | GRUPO JL | 35 | 35 | 0.8693 |
| ELECTROSOL ENERGIA SOLAR | Electrly | 35 | 37 | 0.8692 |
| VIXSYSTEM SOLUÇÕES EM TECNOLOGIA DA INFORMAÇÃO | VIXTE | 42 | 35 | 0.8687 |
| VIRTUA COFFEE WORK | VIRAW | 35 | 35 | 0.8678 |
| TOTUS DESIGN | TOKYO DESIGN | 20 | 20 | 0.8674 |
| SORÓ | ZOROASTRO | 33 | 35 | 0.8670 |
| Clínica NV | CLINIC | 41 | 44 | 0.8666 |
| COXIBURGUER | GIBI BURGUER | 30 | 29 | 0.8665 |
| Marketing Erótico | RAP MARKETING | 41 | 38 | 0.8664 |
| ALLWELL | ALLCELL | 35 | 35 | 0.8664 |
| PROFIX | PETROFIX | 35 | 21 | 0.8655 |
| BRASILGÁS | BRASILIA | 35 | 12 | 0.8650 |
| HARDMÁQUINAS | NE MÁQUINAS | 35 | 35 | 0.8646 |
| Cantinho do Bolinha Pet Boutique | CANTINHO DO BONI | 35 | 43 | 0.8644 |
| TERRAMAXX SOLUÇÕES EM IMPLEMENTOS FLORESTAIS | Terramor | 35 | 35 | 0.8639 |
| Pousada Vila dos Santos | SANTOSHA | 43 | 43 | 0.8636 |
| ALPHA CO. | ALPHA SPEED | 25 | 35 | 0.8633 |
| MARKINHOS FESTAS & LANCHES | MARAVINHOS | 41 | 35 | 0.8624 |
| Academia Brasileira de Psicologia Baseada em Evidências | ACADEMIA BRASILEIRA DE POKER | 35 | 41 | 0.8622 |
| KONIK | KONICA | -1 | 9 | 0.8612 |
| CV / INVEST DTVM | Investcare | 36 | 36 | 0.8604 |
| EMPÓRIO DO BOLO | EMPÓRIO DO Ó | 30 | 35 | 0.8604 |
| BABY CENTER | Body Center | 25 | 35 | 0.8599 |
| PANTA'S | Pantanal | 9 | 10 | 0.8598 |
| COMOL CONSULTORIA M.L. | Comot | 42 | 35 | 0.8585 |
| NOSSA MARCENARIA ! | ATM MARCENARIA | 20 | 20 | 0.8585 |
| Usina de Negócios | Negócios SP | 41 | 41 | 0.8580 |
| D' TUDO EMBALAGENS E DESCARTÁVEIS | TUDO DE PET | 35 | 35 | 0.8580 |
| ZENITH GOURMET | zenity | 43 | 35 | 0.8569 |
| PAPO CARREIRA | CARREIRA LIVE | 41 | 41 | 0.8566 |
| COLÉGIO CONEXÃO | CONEXÃO WK | 43 | 35 | 0.8563 |
| ENGENHAR CONSULTORIA | ENGMAD ENGENHARIA | 42 | 37 | 0.8561 |
| TERRA DO GELO AR CONDICIONADO | TERRAGOTA | 35 | 35 | 0.8553 |
| E-CLEAN | CLEANSUI | 11 | 11 | 0.8552 |
| TERRAS DO TEJO | TERRAZUL | 35 | 35 | 0.8538 |
| UMTELECOM | WT TELECOM | 42 | 38 | 0.8535 |
| HAPFARMA | MOZA PHARMA | 35 | 35 | 0.8531 |
| REI DA LINGERIE | ROVINNI LINGERIE | 25 | 35 | 0.8525 |
| BURGER'S ARTESANAIS LAFOME | KL BURGER'S | 43 | 43 | 0.8521 |
| INNEX | Innex | -1 | 42 | 0.8518 |
| SERTANEJA | SERTANEJO EYEWEAR | 7 | 9 | 0.8516 |
| PADRÃO | PÁTIO BOQUEIRÃO | 42 | 35 | 0.8515 |
| EMPÓRIO POUSO ALEGRE | EmpórioTek | 35 | 35 | 0.8510 |
| C CASA DAS MÁQUINAS | Casa das MÁQUINAS & REFRIGERAÇÃO | 37 | 37 | 0.8508 |
| LATIN MASTER | Master-G | 35 | 9 | 0.8498 |
| SUPERMERCASA | Supermercado Brilho | 35 | 35 | 0.8494 |
| É HIT | Hiit | 38 | 41 | 0.8490 |
| EMPÓRIO DO BOI CASA DE CARNES | EmpórioTek | 35 | 35 | 0.8481 |
| PÃO DE QUEIJO UNAÍ | PÃO DE QUEIJO VOVÓZANA | 43 | 40 | 0.8478 |
| EMPÓRIO VÓ OLIVIA | Empório Vilarin | 35 | 25 | 0.8471 |
| SER SAÚDE | Saúde[IA] | 35 | 35 | 0.8470 |
| POUSADA LUAR DA SERRA | LUARA | 43 | 29 | 0.8459 |
| Agência - Assistência de Elite | AGÊNCIA PITT | 35 | 35 | 0.8459 |
| RNP BRASIL ATACADO CAMA MESA E BANHO | CF BRASIL | 35 | 25 | 0.8456 |
| MERCADINHO PRIMATA ITAMAMBUCA - UBATUBA | PARÇA MERCADINHO | 35 | 35 | 0.8452 |
| ALUMÍNIO ALIANÇA | ALOX ALUMINIUM | 35 | 35 | 0.8452 |
| PADARIA DO CARLITO DESDE: 1930 | PADOCARIA | 35 | 43 | 0.8432 |
| TECH WIND SERVICES | TECH WIX | 37 | 37 | 0.8425 |
| MÉTODO 3T | MÉTODO PPG | 41 | 41 | 0.8409 |
| CEARÁ MOTOS | CAPIVARA MOTOS | 35 | 35 | 0.8409 |
| INICIATIVA EDUCA! | EducaGov | 35 | 35 | 0.8409 |
| LATIN MASTER | MASTER WIN | 12 | 12 | 0.8406 |
| A MAR A VIDA | AMARA | 25 | 35 | 0.8403 |
| TOMAZ DISTRIBUIDORA | TOMASI | 35 | 11 | 0.8401 |
| Terra dos Grandes Festivais | TERRA DO SOL | 41 | 41 | 0.8401 |
| PERSONALE PLANTAS | Personia | 42 | 42 | 0.8396 |
| COLISÃO DISTRIBUIDORA EMBALAGENS EM GERAL | FM DISTRIBUIDORA | 35 | 35 | 0.8394 |
| Fertnutri por Paula Tâmara | PERNUTRI | 35 | 44 | 0.8393 |
| INSTITUTO REVER SETE QUEDAS | INSTITUTO VIETRI | 41 | 41 | 0.8387 |
| CENTRO EDUCACIONAL PEQUENINOS JÚNIOR | PRÓ FUTURO CENTRO EDUCACIONAL | 41 | 41 | 0.8387 |
| M METALBRASIL | SAL BRASIL | 35 | 18 | 0.8384 |
| DOM CHURRASCO | CHURRASCO DO ITALO | 43 | 35 | 0.8381 |
| INICIATIVA EDUCA! | EDUCATION + | 41 | 41 | 0.8373 |
| Academia de Astrologia | ACADEMIA IATI | 41 | 41 | 0.8365 |
| Academia de Astrologia | ACADEMIA IATI | 41 | 41 | 0.8365 |
| PIZZA DOS HERÓIS | Pizzaqui | 43 | 43 | 0.8359 |
| SOLUÇÕES ADMINISTRADORA DE CONDOMÍNIOS | VIÜ SOLUÇÕES | 36 | 36 | 0.8357 |
| ENSINAÇÃO | ENSINA AGR | 41 | 41 | 0.8349 |
| LATIN MASTER | MASTER WIN | 35 | 12 | 0.8348 |
| DELÍCIAS DA MIRA DO DOCE AO SALGADO | DELÍCIAS DA TARDE | 35 | 29 | 0.8345 |
| UMTELECOM | DTELECOM | 35 | 42 | 0.8345 |
| MERCADINHO PRIMATA ITAMAMBUCA - UBATUBA | MERCADINHO ANIMAL | 35 | 31 | 0.8332 |
| Paradinha | HARAS DA RAINHA | 30 | 35 | 0.8331 |
| ALFA GNV INJECTION | AlfaZoo | 35 | 35 | 0.8331 |
| META ENGENHARIA SERVIÇOS | Meta | 35 | 35 | 0.8329 |
| GARAGEM 59 | LÉO GARAGEM | 35 | 35 | 0.8328 |
| HARDFERRAMENTAS | JOTAFER Ferramentas | 35 | 21 | 0.8327 |
| EMPÓRIO VSN+ VENTURA SPORTS NUTRITION | EMPÓRIO 3S | 35 | 35 | 0.8324 |
| Engenharia Academy | ACADEMISEC | 41 | 41 | 0.8323 |
| EMPORIUM SOFÁ SOFÁ E DECORAÇÃO FÁBRICA SANTOS | Empório CTZ | 35 | 35 | 0.8321 |
| NOCA ACESSÓRIOS | N NOCA | 35 | 30 | 0.8320 |
| LULUI BIKINIS | LUNG | 25 | 35 | 0.8311 |
| NORMATEL DESIGN | NORMANI | 35 | 20 | 0.8307 |
| DELÍCIAS DA MIRA DO DOCE AO SALGADO | DELÍCIAS DA AMAZÔNIA | 35 | 43 | 0.8307 |
| V VIRTUDE MARIANA MODA MODESTA | MAR.IANA | 35 | 35 | 0.8306 |
| SOLUÇÕES ADMINISTRADORA DE CONDOMÍNIOS | SOLUÇÕES S NALDI | 35 | 40 | 0.8298 |
| VIRTUA COFFEE WORK | CONEXÃO VIRTUAL | 35 | 35 | 0.8285 |
| PASTELÃO PAULISTA | Pastel da Mi | 43 | 29 | 0.8280 |
| MERCADO DAS COISAS | MINAS MERCADO | 35 | 43 | 0.8275 |
| CENTRAL ÁUDIO SOM E ACESSÓRIOS | CENTRA | 37 | 42 | 0.8269 |
| Virtua Coworking | .viralata | 35 | 35 | 0.8269 |
| Multihype | Multifeme | 8 | 41 | 0.8265 |
| A S SANTOS SUPERMERCADO | SANTOSHA | 35 | 43 | 0.8264 |
| MD SISTEMAS | Sistema Laser | 42 | 42 | 0.8261 |
| ESPÍRITO SANTO CONSTRUÇÃO BRASIL | ESCOLA DO ESPÍRITO SANTO | 41 | 41 | 0.8260 |
| HORTALÍCIA | Hortaliças BUGNO | 43 | 39 | 0.8255 |
| MALU | MATIC | 16 | 20 | 0.8250 |
| Polaris Energia Solar | POLARIX | 42 | 21 | 0.8242 |
| SÓ PISO CACULÉ | Pisos+ | 35 | 27 | 0.8240 |
| NEO AGROAMBIENTAL | NEOID | 42 | 35 | 0.8239 |
| MASTER SILVA SEGURANÇA ELETRÔNICA | MASTER SMART | 37 | 37 | 0.8238 |
| INSTITUTO ROSANE CARDOSO | INSTITUTO CCV | 41 | 41 | 0.8237 |
| UMTELECOM | OP TELECOM | 42 | 35 | 0.8237 |
| CONSTRUTORA COLMEIA | COLMEIARIA | 0 | 35 | 0.8230 |
| INSTITUTO ROSANE CARDOSO | INSTITUTO KRONA | 41 | 41 | 0.8223 |
| MEDITERRÂNEO REDE DE POSTOS | MEDITERRANEUM | 40 | 35 | 0.8213 |
| Virtua Coworking | VIREN | 35 | 35 | 0.8208 |
| GRÁFICA PARCERIA | GRÁFICA BELLA | 40 | 40 | 0.8208 |
| MISS NINA | NINA | 35 | 21 | 0.8205 |
| RN RECIFE NOTÍCIAS | IN IRECÊ NOTÍCIAS | 41 | 41 | 0.8201 |
| Agência - Assistência de Elite | Agência Oak | 35 | 35 | 0.8199 |
| Plano do Gabarito | PLANO X | 41 | 9 | 0.8193 |
| GFORCE | GFORCE | 35 | 9 | 0.8193 |
| FF PONTES | HF | 29 | 31 | 0.8188 |
| EMPÓRIO COCADAS OLIMPIO | EMPÓRIO DOM RUDÁ | 35 | 35 | 0.8173 |
| Private Construction Incorporation | PRIVATE CONSTRUTORA | 36 | 42 | 0.8172 |
| FIBRATTO MÓVEIS ARTESANAIS | FRATTO | 35 | 29 | 0.8169 |
| MENTORIA EM GRANDES OPERAÇÕES | Mentoria Flash | 41 | 41 | 0.8168 |
| Virtua Coworking | CONEXÃO VIRTUAL | 35 | 35 | 0.8167 |
| Virtua Office Coworking Ideas | .VIRALATA | 35 | 35 | 0.8164 |
| GabrielBot | Gabriel Byte | 42 | 35 | 0.8154 |
| EMPÓRIO ESTILO | CIA EMPÓRIO | 35 | 35 | 0.8146 |
| BM AUTOPARTS | BM | 35 | 42 | 0.8146 |
| MÉTODO 3T | Método PAP | 41 | 41 | 0.8145 |
| Virtua Vet Coworking Veterinário | VIRX | 35 | 42 | 0.8143 |
| ARCON CONSTRUTORA | ARC | 37 | 36 | 0.8140 |
| XES BRASIL | ZEZK | 25 | 35 | 0.8137 |
| ISA | Isao | 29 | 42 | 0.8132 |
| Nutris no Online | NUTRI+ | 35 | 30 | 0.8131 |
| HAPFARMA | FCE PHARMA | 35 | 35 | 0.8130 |
| BOI E BRASA | Braza Box | 43 | 39 | 0.8129 |
| SOLAR BANHO E PISCINA | SOLAR BEST | 35 | 37 | 0.8125 |
| Nutris no Online | NUTRIE | 35 | 35 | 0.8124 |
| STUDIO HYPE CONCEPT ARQUITETURA E ENGENHARIA | STUDIO PREGO | 37 | 42 | 0.8123 |
| CASA CONCRETA | CONCRETECH | 42 | 42 | 0.8120 |
| Bara | BÁRBARA | 35 | 18 | 0.8117 |
| STOCK PISOS | STOCKFISH | 42 | 35 | 0.8116 |
| G.O. CLINIC | Clini | 44 | 42 | 0.8110 |
| ANAMAC | ANAMAI | 35 | 14 | 0.8093 |
| AÇAÍ FOOD | AÇAÍ 24H | 35 | 30 | 0.8092 |
| MÃOS QUE ENCANTAM | Encanto & Cor | 35 | 35 | 0.8091 |
| GABRIELE | GABRIELA PAIVA | 0 | 41 | 0.8090 |
| CENTRO EDUCACIONAL PEQUENINOS JÚNIOR | CENTRO EDUCACIONAL ELO | 41 | 41 | 0.8089 |
| MOODZ COLLECTION | DOMI COLLECTION | 25 | 35 | 0.8088 |
| IDEIA ULTRALED | ultra | 20 | 35 | 0.8083 |
| XPORTS EVENTOS | GL SPORTS | 41 | 35 | 0.8079 |
| Virtua Office | VIRTUOSO | 35 | 43 | 0.8068 |
| PRIME AUDIOVISUAL | PRIME PUB | 41 | 43 | 0.8060 |
| ARCÁDIA | ArcadiA | 44 | 41 | 0.8059 |
| HORTALÍCIA | PORTALE | 43 | 43 | 0.8055 |
| DELLA & DELLE | Padelle | 35 | 14 | 0.8054 |
| ENGENHAR CONSULTORIA | EXER ENGENHARIA | 42 | 42 | 0.8053 |
| ARQUISTUDIO arquitetura e urbanismo | Identidade Arquitetura e Urbanismo | 42 | 42 | 0.8050 |
| VIRTUA COFFEE WORK | GUDY COFFEE | 43 | 43 | 0.8049 |
| MULTIGRAN MÁRMORES & GRANITOS | MultigaES | 35 | 37 | 0.8042 |
| QUEIROZ PARTICIPAÇÕES | Queiroz Express | 35 | 21 | 0.8036 |
| NUTRI FIBER | nutrigo | 31 | 30 | 0.8034 |
| .Simplifica Inglês | Casa Simplifica | 35 | 35 | 0.8032 |
| VERBA | VERBA | 9 | 39 | 0.8031 |
| NANA NENÉM | NENÉM NA NUVEM | 35 | 20 | 0.8028 |
| CASA CONCRETA | CONCREJOTA | 42 | 37 | 0.8024 |
| NEO SOLUÇÕES AMBIENTAIS & PROJETOS AGROPÉCUARIOS | NEOID | 42 | 42 | 0.8022 |
| VAMO | VAMORA | 36 | 10 | 0.8015 |
| ANDER SABORES | and wander | 30 | 35 | 0.8015 |
| PIZZA FONE pizza | Pizza Flow | 43 | 43 | 0.8014 |
| CV / INVEST DTVM | INVEST4U | 36 | 35 | 0.8014 |
| ALPHA MOTORS | ALPHA DC | 35 | 35 | 0.8010 |
| INICIATIVA EDUCA! | EDUCAVIVE | 41 | 41 | 0.8009 |
| B BARÃO IBITURUNA | BARA | 36 | 37 | 0.8007 |
| Insight Strategic Social Media | Insights Hub | 35 | 35 | 0.7998 |
| AQ AYR QUALITY MOURA | AYRA | 37 | 35 | 0.7997 |
| CASTELO ENTRETENIMENTO | L&P entretenimento | 41 | 41 | 0.7996 |
| SOUL ZEN | Soul Serena | 35 | 18 | 0.7994 |
| MONSTERBEE SUPLEMENTOS | MONSTERA | 35 | 35 | 0.7987 |
| E.CONTROL | CLUB CONTROL | 9 | 42 | 0.7985 |
| IMPACTO | IMPACCTO | 40 | 35 | 0.7968 |
| CASA SOLAR | SolarChef | 42 | 11 | 0.7950 |
| The People Restôbar | THE BASIC PEOPLE | 43 | 35 | 0.7941 |
| INSTITUTO DE MÚSICA CLARITASPULCHRI | INSTITUTO CCV | 16 | 41 | 0.7934 |
| ESSENCIAL MOTOS ATACADO E VAREJO | Essencial baby | 12 | 35 | 0.7933 |
| COMUNICADO | Comunicamake | 16 | 41 | 0.7923 |
| modo conectado | CONECTA D2C | 35 | 35 | 0.7916 |
| DIÁRIO NOTÍCIAS | NOTÍCIAS SJC | 41 | 35 | 0.7915 |
| SYM | SIORA | 36 | 9 | 0.7914 |
| EMPÓRIO DO CARANGUEJO | EMPÓRIO DO RIO | 43 | 35 | 0.7911 |
| TROPICAL CHIC | Tropical-X | 35 | 43 | 0.7910 |
| DESCOMPLICANDO O ALEMÃO | DESCOMPLICANDO RÓTULOS | 41 | 41 | 0.7904 |
| Multihype | Multifeme | 16 | 41 | 0.7902 |
| Prato Pet | PRATA & PRATOS | 31 | 43 | 0.7902 |
| BOM LEITE | LEITE | 35 | 24 | 0.7892 |
| D PRIMES AÇAIZINHO | PRIME PUB | 30 | 43 | 0.7887 |
| MUSICAL BMB | MUSICAL BEATS | 41 | 44 | 0.7874 |
| ARENA BEACH | Areal Beach | 38 | 41 | 0.7870 |
| UMTELECOM | A.R TELECOM | 42 | 37 | 0.7865 |
| CAPIM | CAPIM | 41 | 42 | 0.7861 |
| Marketing Erótico | Missão Marketing | 41 | 35 | 0.7860 |
| G.O. CLINIC HOSPITAL DIA EM GINECOLOGIA | EP CLINIC | 44 | 44 | 0.7854 |
| FRAGRANCE MAISON | MAISON ALISSÉ | 35 | 25 | 0.7849 |
| FARMÁCIA ULTRA BARATO | FARMÁCIAS Rodolfo | 35 | 35 | 0.7836 |
| DESCOMPLICANDO O ALEMÃO | DESCOMPLICANDO TPM | 41 | 41 | 0.7835 |
| GABRIELE | Gabriel Pardo | 0 | 41 | 0.7835 |
| VELOZ | VELORA | 37 | 35 | 0.7828 |
| CONTABEM | CONTABILÊS | 35 | 41 | 0.7810 |
| ILANA BRAGA | KILANA | 35 | 35 | 0.7805 |
| OI NEGÓCIOS IMOBILIÁRIOS | PBD NEGÓCIOS | 36 | 36 | 0.7795 |
| SMART ELETRÔNICOS | B=SMART | 35 | 7 | 0.7790 |
| PURE NUTRE | PURE | 35 | 28 | 0.7777 |
| AÇAÍ FOOD AÇAÍZINHO | AÇAÍ POWER | 30 | 43 | 0.7776 |
| IMPERIALLE | IMPERIALIS | 9 | 20 | 0.7775 |
| GRUPO S | GRUPO V8 | 35 | 35 | 0.7757 |
| SUSTENTARTE | SUSTENTA AI | 35 | 35 | 0.7750 |
| MINAS DE MINAS | MINASA | 35 | 8 | 0.7740 |
| CONSTRUCICLO | CICLO VAC | 35 | 37 | 0.7738 |
| PRIMÍCIAS CORTINAS | Primax | 24 | 21 | 0.7738 |
| "KARNE KEIJO" | KARUI | -1 | 30 | 0.7735 |
| PRATINHA | Pratinho no ponto | 0 | 43 | 0.7734 |
| NEW TECH BLINDAGENS | NEW AGRO TECH | 40 | 35 | 0.7733 |
| CONFIA + SOLUÇÕES FINANCEIRAS | Conecta Soluções Financeiras | 36 | 36 | 0.7725 |
| BELLA MARIA | BELLA MANIA | 40 | 30 | 0.7724 |
| CASA DO RESTAURANTE | CASADEMÃE | 35 | 35 | 0.7723 |
| MALUKÃO DAS FABRICAS | MALUKIES | 40 | 30 | 0.7715 |
| MOODZ COLLECTION | D DOCA'S COLLECTION | 25 | 35 | 0.7703 |
| EMPÓRIO DO PÃO | EMPÓRIO DO MAR | 35 | 35 | 0.7698 |
| IGREJA EVANGÉLICA ASSEMBLEIA DE DEUS MINISTÉRIO DO TEMPLO CENTRAL | AD ASSEMBLÉIA DE DEUS CASCAVEL | 45 | 41 | 0.7691 |
| ACAÍ DO TIO REI | AÇAIZIM | 30 | 30 | 0.7688 |
| ESCRITA E LEITURA PARA EVOLUÇÃO E VALORIZAÇÃO NA APRENDIZAGEM ELEVA | E - LEVE | 35 | 35 | 0.7686 |
| Concept Hospital Dia | CONCEPT | 36 | 35 | 0.7682 |
| NEW PLASTIC | PLASTLEV | 21 | 35 | 0.7680 |
| GRUPO S | GRUPO DUMA | 35 | 35 | 0.7675 |
| neo núcleo de emergências odontológicas | NEOON | 44 | 44 | 0.7673 |
| GABRIEL SOUZA | GABRIELA | 41 | 41 | 0.7671 |
| P COFFEE | JM COFFEE | 43 | 43 | 0.7668 |
| DOM PASTEL | Pastel da Mi | 0 | 35 | 0.7664 |
| VIVA PIZZA | vivan | 30 | 35 | 0.7656 |
| COXIBURGUER | CUBE BURGUER | 35 | 30 | 0.7650 |
| GOLDEN TEAR A DARK FANTASY TTRPG | GOLDEN TEAM | 41 | 41 | 0.7642 |
| APLICA BUS | Aplicap | 35 | 35 | 0.7642 |
| AMAZON PRODUTOS NATURAIS | AMAZONRIP | 30 | 35 | 0.7637 |
| MD SISTEMAS | MD | 42 | 19 | 0.7634 |
| CLÍNICA VETERÍNARIA E PET SHOP BELA VIDA | Vila Canina Pet Shop e Clínica Veterinária | 44 | 35 | 0.7633 |
| MAIDA HEALTH | Health Makers | 35 | 41 | 0.7627 |
| CASA DO RESTAURANTE | CASA DO PERFIL | 35 | 40 | 0.7618 |
| AUTO ESCOLA ATUAL CENTRO DE FORMAÇÃO DE CONDUTORES | autoescola NAÇÕES | 41 | 41 | 0.7616 |
| ALIGNMED | Aligned | 35 | 35 | 0.7611 |
| HORTILOG | Pórtico | 9 | 9 | 0.7605 |
| Distribuidora Classic | GENESYS DISTRIBUIDORA | 35 | 35 | 0.7602 |
| M MÓBILE SUPERMERCADO | Mobilee | 35 | 1 | 0.7599 |
| PBENERGIASOLAR | ASGE ENERGIA SOLAR | 37 | 37 | 0.7596 |
| Temperos Magia | TEMPERO REAL | 30 | 31 | 0.7581 |
| ÚNICO CAMINHO | uni | 35 | 43 | 0.7571 |
| PERSONALE PLANTAS | Personare | 42 | 45 | 0.7570 |
| Toca fina cozinha | FITOCA | 30 | 30 | 0.7569 |
| PARADISE CHANNELS | Channels | 38 | 35 | 0.7565 |
| REI DAS CLÍNICAS | CLINICAWEB | 41 | 42 | 0.7559 |
| Liberty Energia | LIBERT | 42 | 40 | 0.7558 |
| VILLAR ELÉTROS | VILLA MAG | 35 | 43 | 0.7556 |
| PARTIU CAFÉ | partiu | 43 | 42 | 0.7555 |
| ELETRO SOLAR | ELETROBIGELI | 35 | 35 | 0.7549 |
| ¿Flash Life Business | Flipê | 41 | 41 | 0.7546 |
| IN CONNECTION | COMPANY CONNECTION | 41 | 28 | 0.7538 |
| BRASIL SOLUÇÕES INTELIGENTES | RMP+ EHS SOLUÇÕES INTELIGENTES | 36 | 42 | 0.7538 |
| SALÃO e CONGRESSO BRASILEIRO da CACHAÇA | CONGRESSO BRASILEIRO DE COSMETOLOGIA | 41 | 41 | 0.7530 |
| ACADEMIA DO VAREJO | Konnen Academia | 35 | 41 | 0.7529 |
| SER SAÚDE | Saudee | 16 | 35 | 0.7528 |
| Z ZEST | ZEST SNACKS | 35 | 30 | 0.7528 |
| The People Restôbar | THE BASIC PEOPLE | 43 | 40 | 0.7525 |
| MALU GUARDANAPOS DE PAPEL | MAIG | 16 | 35 | 0.7523 |
| CENTRO DE TREINAMENTO THE MOVEMENT | Fábrica Centro de Treinamento | 41 | 41 | 0.7502 |
| açaí Vitaly Pura energia | VITAM | 30 | 35 | 0.7499 |
| IGREJA BATISTA ESSÊNCIA DO ESPÍRITO | IGREJA BATISTA REINO | 45 | 45 | 0.7496 |
| PASTEL DA TERRA | Pastel da Mi | 43 | 30 | 0.7475 |
| VITAL CONFORT | VITALOK | 24 | 21 | 0.7474 |
| Águas Curitiba | ÁGUA SURDA | 39 | 43 | 0.7464 |
| MARCENARIA J. MEDEIROS | JPR MARCENARIA | 20 | 20 | 0.7463 |
| MAIDA.HEALTH | MAYDAY | 9 | 35 | 0.7462 |
| PRONET TELECOM | FRON | 38 | 41 | 0.7451 |
| CVPAR DIGITAL | DIGITAL | 35 | 37 | 0.7447 |
| hidra hair | HYDRA | 12 | 35 | 0.7442 |
| FESAN INTELIGÊNCIA IMOBILIÁRIA | RIGHT SPOT INTELIGÊNCIA IMOBILIÁRIA - W N E S- | 36 | 42 | 0.7439 |
| SERTANEJA | MESA SERTANEJA | 35 | 38 | 0.7437 |
| MANÁ TEMAKERIA & SUSHI BAR DELIVERY E EVENTOS | Manai | 30 | 43 | 0.7424 |
| ANEL FARMA MANIPULAÇÃO | FÀNEL | 44 | 35 | 0.7424 |
| INTEGRAL CLUBE DE BENEFÍCIOS | INTEGRAMEI | 35 | 35 | 0.7412 |
| FAMÍLIA DE CACHOS | FAMILIAR | 3 | 3 | 0.7411 |
| ES CONSTRUÇÃO | HS CONSTRUÇÃO | 41 | 35 | 0.7409 |
| MG MINAS GERAIS CERTIFICADORA DIGITAL | POSTO MINAS GERAIS | 42 | 37 | 0.7392 |
| LÁ NO PASTEL | Pastel HUB | 43 | 43 | 0.7386 |
| PASTEPIZZA LANCHES | Pazzi di Pizza | 43 | 43 | 0.7378 |
| EMPÓRIO COCADAS OLIMPIO | EMPÓRIO PETALI | 35 | 29 | 0.7376 |
| CASA DO BOLO VOVÓ TIANA | Casa do 3D | 30 | 40 | 0.7371 |
| CASA D INTERIORES | Casa 5G | 35 | 38 | 0.7370 |
| FA FASHION ATACADISTA O VAREJO NO PRECINHO DE ATACADO | FAHLENA ECO FASHION | 25 | 35 | 0.7360 |
| COMPREX | Compre sua Peça | 37 | 35 | 0.7355 |
| PRATINHA | PRATINI | 0 | 35 | 0.7352 |
| BRASIL MÓBILE | ATT BRAZIL | 35 | 35 | 0.7350 |
| ALUMÍNIO ALIANÇA | MP Alumínio | 35 | 21 | 0.7350 |
| MÉTODOCOIMBRA | Método C.E.O | 41 | 41 | 0.7346 |
| MED MALL | MV MED | 35 | 35 | 0.7345 |
| ELETRO SOLAR | eletromuk | 35 | 39 | 0.7341 |
| FARMÁCIA ANDRADE | Farmácia Boituva | 35 | 35 | 0.7331 |
| MANÍ | MANICUT | 35 | 44 | 0.7330 |
| INSTITUTO REVER SETE QUEDAS | INSTITUTO JUSTIÇA | 41 | 41 | 0.7326 |
| AGUARDENTE DE CANA PRATINHA | Aguardente de Cana Melindrosa | 35 | 33 | 0.7320 |
| ALLON | ALLOY | 41 | 35 | 0.7313 |
| B BRASMETAL | BRASMEDI | -1 | 35 | 0.7312 |
| VIRTUA COFFEE WORK | só toffee | 43 | 43 | 0.7308 |
| KONIK | KONICA | -1 | 2 | 0.7306 |
| NOSSA MARCENARIA ! | MARCENARIA 4.0 | 20 | 20 | 0.7300 |
| HIGH TICKET ADVANCED | CRM HIGH TICKET | 41 | 42 | 0.7294 |
| Espirito Capixaba | CAPIXABA | 35 | 31 | 0.7288 |
| START COMUNICAÇÃO INTEGRADA | Star | 35 | 45 | 0.7285 |
| EMPÓRIO VSN+ VENTURA SPORTS NUTRITION | EMPÓRIO PRIME | 35 | 43 | 0.7283 |
| Comercial Têxtil | FOCO Comercial | 35 | 35 | 0.7282 |
| VOZ DAS MINAS | MINASVOL | 41 | 35 | 0.7254 |
| AUTO PEÇAS MATOS | Autopen | 35 | 8 | 0.7252 |
| DELÍCIAS DA MIRA DO DOCE AO SALGADO | delíciaaa | 43 | 41 | 0.7248 |
| AUTO PEÇAS MATOS | Netmix Auto Peças | 35 | 12 | 0.7243 |
| CENTRO AUDITIVO LIBERTY | LYD CENTROAUDITIVO | 35 | 35 | 0.7210 |
| SER SAÚDE | SAÚDE+PE | 38 | 45 | 0.7199 |
| Agência VET | AGÊNCIA K1 | 42 | 35 | 0.7197 |
| G.O. CLINIC HOSPITAL DIA EM GINECOLOGIA | SM CLINIC | 44 | 44 | 0.7194 |
| TERRAMAXX SOLUÇÕES EM IMPLEMENTOS FLORESTAIS | Terramor | 35 | 30 | 0.7190 |
| Hconnect | PAW CONNECT | 42 | 21 | 0.7185 |
| TOTUS DESIGN | Lótus Design | 35 | 42 | 0.7179 |
| AGÊNCIA ART | agência (re)trato | 35 | 39 | 0.7156 |
| SE LIGA VM | Se liga só | 41 | 41 | 0.7153 |
| FLOR DE CACTUS LIMPEZA E IMPERMEABILIZAÇÃO | Flor de Caiena | 37 | 35 | 0.7148 |
| PÃO DE QUEIJO UNAÍ | PÃO DE QUEIJO VÓDELAIDE | 43 | 35 | 0.7148 |
| VirtuaMed O hub da saúde coworking médico | virtubem | 44 | 35 | 0.7134 |
| EMPÓRIO VÓ OLIVIA | EMPÓRIO DO HOMEM | 35 | 35 | 0.7133 |
| PET POW | PET POVÃO | 21 | 35 | 0.7125 |
| MASTER ALUMÍNIO | Master DM | 35 | 43 | 0.7119 |
| GRAMAC | Gramado | 35 | 3 | 0.7118 |
| inova FOODS | INOVAPME | 35 | 41 | 0.7108 |
| FARMÁCIAS PRIME | PISA FARMACIAS | 35 | 35 | 0.7107 |
| RESTAURANTE MENINA GERAIS | LA'DUNA RESTAURANTE | 43 | 43 | 0.7101 |
| A ASA DIGITAL INTERNET | DIGITAL.COM | 38 | 35 | 0.7085 |
| GRUPO MS MAIS QUE SAÚDE | GRUPO MX8 | 35 | 35 | 0.7082 |
| TINTAS VERGINIA | JCS TINTAS | 35 | 2 | 0.7049 |
| CALY | calico | 35 | 35 | 0.7041 |
| SPRING | SPRINT | 15 | 12 | 0.7029 |
| Acurá | ACCURA | 29 | 9 | 0.7023 |
| BOTANIK INCORPORADORA | BOTANIFIQUE | 36 | 35 | 0.7022 |
| ACADEMY DENTISTA DE SUCESSO | AIL - Academy | 41 | 41 | 0.7009 |
| I IMPACTO | IMPACTUS | 35 | 17 | 0.7008 |
| CANAL 10 | CanalJMS | 41 | 41 | 0.7005 |
| High Protector | PROTECTOR | 3 | 4 | 0.6998 |
| SMART ELETRÔNICOS | SMART PEEL | 35 | 10 | 0.6990 |
| INTEGRAL TRADING | INTEGRALMEDICA | 35 | 31 | 0.6980 |
| AGUARDENTE DE CANA PRATINHA | PRATINI | 35 | 35 | 0.6980 |
| PÃO DE QUEIJO TINÔ | Oh Pão de Queijo | 30 | 43 | 0.6972 |
| CASTELLA | CASAELLA | 21 | 35 | 0.6968 |
| WM INVESTIMENTOS | X10 INVESTIMENTOS | 36 | 36 | 0.6967 |
| GABRIEL SOUZA | GABRIEL ZUUK | 41 | 41 | 0.6967 |
| Pai Fora de Série | Fora de Série | 41 | 9 | 0.6961 |
| GABRIELE | GABRIEL FROEDE | 0 | 35 | 0.6960 |
| COXIBURGUER | prazí burguer | 30 | 43 | 0.6957 |
| Fene Distribuidora | FM DISTRIBUIDORA | 35 | 35 | 0.6956 |
| APLICA BUS | Aplicap | 35 | 35 | 0.6955 |
| HOME TOP MÓVEIS E COLCHÕES | TECHHOME | 35 | 35 | 0.6954 |
| BOUTIQUE SAÚDE | Boutique do sono | 35 | 35 | 0.6952 |
| CENTRAL PINK | CENTRAL PISCINAS | 35 | 40 | 0.6951 |
| AUTO POSTO DOIS IRMÃOS | AUTO POSTO DIAS | 35 | 35 | 0.6937 |
| SMART ELETRÔNICOS | Smartflex | 35 | 38 | 0.6935 |
| TURMINHA DIÁRIO | TURMINHA DE VALOR | 38 | 41 | 0.6932 |
| VENÂNCIO | VENENUM | 37 | 32 | 0.6917 |
| ROUP' ARTE NOVAS E USADAS | ROUZ | 35 | 35 | 0.6906 |
| PERFAL | PERFECT CDOP - LOJA | 35 | 35 | 0.6901 |
| CASA CONCRETA | CONCRETA | 35 | 37 | 0.6896 |
| I/O TECNOLOGIA INDUSTRIAL | IB TECNOLOGIA | 35 | 37 | 0.6890 |
| CLÍNICA MÉDICA GODINHO | CliniCalm | 44 | 44 | 0.6880 |
| COMUNICADO | COMUNICA BET | 16 | 41 | 0.6872 |
| D PRIMES AÇAIZINHO | Primea | 30 | 35 | 0.6870 |
| FESTIVAL DO VINHO FESTA DO VINHO | FESTIVAL SSA | 41 | 41 | 0.6868 |
| Y.Consultoria | DX3 Consultoria | 35 | 35 | 0.6866 |
| SER SAÚDE | Sr. Saúde | 41 | 44 | 0.6854 |
| ÁGUIA BRASIL DISTRIBUIDORA | ACQUAVIABRASIL | 35 | 45 | 0.6853 |
| CARNES BOM REBANHO | Casa de Carnes Bom Corte | 35 | 35 | 0.6853 |
| VILLAR ELÉTROS | Villa Cacau | 35 | 35 | 0.6852 |
| RHEUMA-IN | RHEUM.AI | 44 | 41 | 0.6842 |
| FLOR DE ELIS MODA INFANTIL | FLOR DE LIZ | 35 | 13 | 0.6832 |
| ESPAÇO ENGENHO COLONIAL | ENGENHO DO COLONO | 41 | 35 | 0.6830 |
| Portal do Desenvolvimento Infantil | Sementare DESENVOLVIMENTO INFANTIL | 41 | 41 | 0.6824 |
| DIZ, MULHER! | MULHERIA | 9 | 41 | 0.6822 |
| INSTITUTO ROSANE CARDOSO | INSTITUTO MANDI | 41 | 41 | 0.6818 |
| AGÊNCIA ART | AGÊNCIA CONTI | 35 | 35 | 0.6811 |
| LATICÍNIOS ED LEITE | BRANDO LATICÍNIOS | 29 | 35 | 0.6804 |
| RESTAURANTE MENINA GERAIS | KENOMA RESTAURANTE | 43 | 43 | 0.6801 |
| RESTAURANTE MENINA GERAIS | MAROLLO RESTAURANTE | 43 | 35 | 0.6789 |
| TOMAZ DISTRIBUIDORA | TCOMAS | 35 | 11 | 0.6789 |
| @blindador | BLINDADO | 45 | 41 | 0.6782 |
| FORT EVENTOS | HortApp | 41 | 35 | 0.6768 |
| DROGARIA SANDES | DROGARIA ZONA SUL | 35 | 35 | 0.6766 |
| VIDRAÇARIA MUNDO DOS VIDROS | EB VIDRAÇARIA | 19 | 40 | 0.6765 |
| SEU DIREITO | Direito Certo | 38 | 45 | 0.6764 |
| COMPREX | COMP | 37 | 42 | 0.6764 |
| ELETRO SOLAR | ELETRONAGRO | 35 | 37 | 0.6762 |
| AR MARCENARIA AMBIENTES PLANEJADOS | AFS Marcenaria | 20 | 20 | 0.6759 |
| HAUS ALAMOÉ | HAUS | 35 | 18 | 0.6757 |
| BRASILGÁS | BRASILIA | 0 | 12 | 0.6752 |
| Q.SABOR PIZZARIA DELIVERY | Saboreô | 39 | 35 | 0.6751 |
| CONGELADOS SANTA RITA SAPUCAÍ | Congela Congelados | 35 | 35 | 0.6746 |
| PLANETA BEBÊ | planeta a | 42 | 42 | 0.6742 |
| CHURRASCARIA ORIENTAL | CHURRASCARIA MENINO DE OURO | 43 | 35 | 0.6742 |
| CENTRAL MECÂNICA | CentralMax | 37 | 9 | 0.6740 |
| BRASIL SEGURANÇA ELETRÔNICA | BRASYS | 45 | 42 | 0.6735 |
| A Nova Saúde por Dr William Araujo | SaúdeNow | 41 | 41 | 0.6726 |
| MADEIREIRA RIO BRANCO | RIO BRANCO IMPORT STORE | 35 | 35 | 0.6719 |
| A ASA DIGITAL INTERNET | DigitalView | 38 | 35 | 0.6711 |
| OPERA HOUSE | HOUSE OF AME.N | 25 | 35 | 0.6709 |
| FRAGRANCE MAISON | MAISON SENNA | 35 | 35 | 0.6706 |
| DEEP COLLECTION | Dê Isa Collection | 25 | 35 | 0.6692 |
| Fardin | PARDINO | 35 | 29 | 0.6687 |
| PBENERGIASOLAR | EJS ENERGIA SOLAR | 37 | 42 | 0.6673 |
| SPA COLCHÕES | SPAR | 35 | 35 | 0.6662 |
| GRUPO MS | GRUPO HOMERO | 44 | 35 | 0.6642 |
| FRAGRANCE MAISON | MAISON ASTRAL | 35 | 25 | 0.6638 |
| FONTE DE ALEGRIA | ALEGRIA & FESTA | 32 | 35 | 0.6636 |
| INDAIÁ | INDAIAPEÇAS | 0 | 12 | 0.6634 |
| MOVÊH | Mover | 42 | 35 | 0.6629 |
| SVM INSIGHTS | BetInsights | 41 | 42 | 0.6619 |
| TERRAL MARESIAS | TERRAGOTA | -1 | 35 | 0.6606 |
| A AMARE HOME | AMARAM | 35 | 25 | 0.6606 |
| REI DA LINGERIE | Saranda Lingerie | 40 | 25 | 0.6598 |
| BAR E RESTAURANTE ZÉ CARLOS | Bar e Restaurante Helô | 43 | 43 | 0.6596 |
| VELOZ | Velofox | 37 | 12 | 0.6596 |
| EMPÓRIO VÓ OLIVIA | EMPÓRIO PRIME | 35 | 43 | 0.6593 |
| Dietitan | Dietitan | 44 | 9 | 0.6591 |
| CASA DOS TEMPEROS & CIA | ETEMPEROS | 35 | 30 | 0.6579 |
| SORVETES ICE TRUCK | Sorvetes do Nick | 30 | 35 | 0.6579 |
| MANÁ TEMAKERIA & SUSHI BAR DELIVERY E EVENTOS | MANAVI | 30 | 35 | 0.6576 |
| VENTURIM | Venturinni | 35 | 7 | 0.6574 |
| SHOPPING PIZZA BURGER | KUTZ BURGER & PIZZA | 43 | 43 | 0.6570 |
| CARVOEIRO BRASA E BAR | BRASEIRO | 43 | 35 | 0.6567 |
| CHURRASQUEIRAS LOBO | CHURRASQUIM | 11 | 41 | 0.6564 |
| ES ELETRO SENA | ELETRO VJ | 37 | 37 | 0.6561 |
| ANTERACAPS | ANTERA | 35 | 35 | 0.6559 |
| SORVETES ICE TRUCK | SKIPEROLA SORVETES | 30 | 35 | 0.6540 |
| NATURAL VILLE | naturalvitae | 37 | 35 | 0.6540 |
| Jucelino Móveis Planejados | IMPACTO MOVEIS PLANEJADOS | 35 | 35 | 0.6538 |
| Lumiê Lingerie & Co. | LUMY | 35 | 40 | 0.6519 |
| CANAL 10 | Canal SAF | 41 | 41 | 0.6513 |
| AR TERRAPLANAGEM ADRIANO ROCHA | DAF TERRAPLANAGEM | 37 | 37 | 0.6509 |
| Reviva by Fibra Náutica | REVIVARIO | 37 | 37 | 0.6491 |
| EMPÓRIO DA TAPIOCA | Empório CTZ | 35 | 35 | 0.6488 |
| SUPER LAGOA | SUPERLAMP | 35 | 35 | 0.6487 |
| EDUCALAB | EducaLab | 16 | 35 | 0.6485 |
| TOP BRASIL FERTILIZANTES | SLS Shop Brasil | 35 | 9 | 0.6482 |
| TURMINHA DIÁRIO | Turminha da Liga | 41 | 41 | 0.6470 |
| NINARE | NINA O | 40 | 3 | 0.6468 |
| KONIK | KONICA | -1 | 1 | 0.6454 |
| CT CERÂMICA TIJUCA | CERAMIC LINE | 35 | 21 | 0.6452 |
| Método Ressonante | Método ETC | 41 | 41 | 0.6451 |
| PADRÃO | PADRÃO A. | 42 | 41 | 0.6450 |
| SANTA PRAIA | PRAYA | 43 | 32 | 0.6439 |
| IGREJA EVANGÉLICA ASSEMBLEIA DE DEUS MINISTÉRIO DO TEMPLO CENTRAL | Assembleia de Deus Novo Começo | 45 | 45 | 0.6430 |
| COLISÃO DISTRIBUIDORA EMBALAGENS EM GERAL | Alpi Distribuidora | 35 | 35 | 0.6424 |
| ALTO BRASIL | OBJETO BRASIL | 35 | 25 | 0.6421 |
| BUTECO DO DANI | BUTECO DO ANÃO | 43 | 43 | 0.6417 |
| Terra dos Grandes Festivais | TERRA DO SOL | 41 | 43 | 0.6416 |
| COLISÃO DISTRIBUIDORA EMBALAGENS EM GERAL | BENTO'S DISTRIBUIDORA | 35 | 35 | 0.6407 |
| CEARÁ MOTOS | CAPIVARA MOTOS | 7 | 35 | 0.6406 |
| TUDO TECH | Tudo Q | 37 | 35 | 0.6394 |
| É DE CASA ENERGIA SOLAR | Casa Enkel | 42 | 42 | 0.6394 |
| ESPETINHO DA KARLA DESDE 1996 | ESPETINHO DO PATOLA | 43 | 43 | 0.6392 |
| FLÔR DO CAFÉ CONFEITARIA | FLOR DO CACAUEIRO | 43 | 30 | 0.6384 |
| FAALA NUTRI ! | NUTRI+ | 41 | 35 | 0.6384 |
| IDEAL SUPER | Idealtec | 11 | 35 | 0.6373 |
| TV DIÁRIO | Diarix | 0 | 9 | 0.6371 |
| FARMÁCIA ANDRADE | FARMÁCIA DE LUCA | 35 | 35 | 0.6361 |
| AGÊNCIA ART | AGÊNCIA SUALOGO | 35 | 45 | 0.6334 |
| ENERGIA NATURAL | Oliá Natural | -1 | 3 | 0.6322 |
| Massena | MASSETA | 37 | 35 | 0.6320 |
| VISÓTICA | VISTA TÁTICA | 40 | 35 | 0.6319 |
| FAROL ACADÊMICO | PAROMAX | 41 | 41 | 0.6307 |
| ESPETINHO DA KARLA DESDE 1996 | ESPETINHOS IPAVA | 43 | 29 | 0.6306 |
| DLINDA | DEOLINDA | 35 | 43 | 0.6303 |
| VIRTUA COFFEE WORK | Virtude Mineira | 43 | 30 | 0.6289 |
| PAPO CARREIRA | CARREIRA FEST | 38 | 41 | 0.6288 |
| NACIONAL GÁS | NACIONAL FIRE | 4 | 37 | 0.6273 |
| VERDE MERCÊ | VERDENA | 35 | 30 | 0.6270 |
| açaí Vitaly Pura energia | VitalLink | 35 | 35 | 0.6268 |
| NACIONAL GÁS | Nacionalidade | 35 | 45 | 0.6268 |
| ALUMÍNIO ALIANÇA | FOX ALUMÍNIO | 35 | 40 | 0.6261 |
| VENÂNCIO | NANCY | 37 | 43 | 0.6254 |
| ETERNUM WATCHES | ETERNITÀ | 35 | 45 | 0.6251 |
| PRO PRODUÇÕES E PROPAGANDA | News Produções | 41 | 41 | 0.6251 |
| BEM STORE | BEM + | 35 | 44 | 0.6249 |
| A AMARE HOME | AMARA | 35 | 25 | 0.6248 |
| SUPER BLOCOS | Supercubo | 40 | 9 | 0.6248 |
| MOUTRIZ ENERGIA E CONSTRUÇÃO | Motriz | 42 | 45 | 0.6246 |
| TECNO PLANT | TECNOPIG | 35 | 31 | 0.6240 |
| Florens | FLORENÇA | 41 | 35 | 0.6232 |
| COXINHA EXPRESS | Expresso | 43 | 42 | 0.6230 |
| GRÁFICA PARCERIA | L1 Gráfica | 40 | 40 | 0.6229 |
| RESTAURANTE MENINA GERAIS | RESTAURANTE H2 | 43 | 43 | 0.6225 |
| MÓVEIS PAGANINI | FERLINI MÓVEIS | 20 | 40 | 0.6224 |
| PRIME PE | PRIMEPLAS | 35 | 20 | 0.6214 |
| WSM AMBIENTAL | vsmb | 44 | 35 | 0.6210 |
| DEEP COLLECTION | Alvile collection | 35 | 25 | 0.6203 |
| TÁ, QUERIDA! | QUERIDO | 35 | 30 | 0.6190 |
| CONTA BEM | CONTA.GE | 16 | 35 | 0.6178 |
| NACIONAL | Nacionalidade | 0 | 45 | 0.6177 |
| NACIONAL GAS BUTANO | Nacionalidade | 0 | 45 | 0.6173 |
| É HIT | Hiit | 35 | 41 | 0.6172 |
| CASA CONCRETA | CONCREJOTA | 35 | 35 | 0.6171 |
| PAZ & AMOR | Amor Fati | 38 | 41 | 0.6170 |
| Quintal Cozinha pra Torar | Quintal do Pão | 43 | 30 | 0.6170 |
| Desconto de Hoje | DescontaBR | 35 | 35 | 0.6161 |
| BAR E RESTAURANTE ZÉ CARLOS | BAR E RESTAURANTE TAFAREL | 43 | 43 | 0.6157 |
| Auto Posto G&D | AUTO POSTO KS | 35 | 35 | 0.6156 |
| UNIMAQ | UNIMARRI | 35 | 25 | 0.6122 |
| NACIONALGÁS | Nacional Cards | 4 | 35 | 0.6121 |
| EMPÓRIO MAIA | CIA EMPÓRIO | 35 | 35 | 0.6114 |
| Burguer do Nô | Norka Burguer | 43 | 35 | 0.6113 |
| FRUTY BOM | Utibom | 32 | 35 | 0.6113 |
| AÇAI FOOD | AÇAÍ JP | 43 | 40 | 0.6111 |
| Black Stone | Blackstar | 41 | 42 | 0.6105 |
| SUBLIME PARRILLA-BUFFET | SubliMaju | 43 | 40 | 0.6104 |
| LA FOCACCIA IMPORTADOS | FOCACCIO | 0 | 43 | 0.6082 |
| EMPÓRIO VÓ OLIVIA | Empório Casamac | 35 | 35 | 0.6082 |
| VIVANET TELECOM | VIVA + PET | 38 | 41 | 0.6072 |
| MAIS CHURRASCO | MAIA | 35 | 25 | 0.6064 |
| AR MARCENARIA AMBIENTES PLANEJADOS | MARCENARIA GUARUJÁ | 20 | 40 | 0.6047 |
| SABOROSO PRIME | SABOR 40º | 30 | 30 | 0.6042 |
| LATICÍNIOS ED LEITE | ITAMILK LATICÍNIOS | 29 | 35 | 0.6041 |
| Comercial Têxtil | COMERCIAL PRÓ | 35 | 41 | 0.6037 |
| Virtua Office Coworking Ideas | vireimãe | 35 | 35 | 0.6033 |
| RESTAURANTE ENSEADA MANGUINHOS | CENA RESTAURANTE | 43 | 43 | 0.6032 |
| VILLAR ELÉTROS | VILAR | 35 | 44 | 0.6030 |
| COCO BRASIL RESTAURANTE | Brasil Cert | 43 | 42 | 0.6030 |
| IDEAL TOTAL | Idealtec | 11 | 9 | 0.6029 |
| CAFÉ ESPECIAL SERENGETI | we special cafés | 35 | 30 | 0.6003 |
| FAVO | FAVO BAR | 0 | 43 | 0.5988 |
| VITAL CONFORT | VITAM | 24 | 35 | 0.5983 |
| IMPACTO | IMPACTUS | 35 | 17 | 0.5981 |
| RESERVA LW INDÚSTRIA DE BEBIDAS SAUDÁVEIS | RESERVA goyá | 35 | 45 | 0.5980 |
| AO SOM DA VIOLA | SOMVIBE | 38 | 45 | 0.5974 |
| PBENERGIASOLAR | auroravolt energia solar | 37 | 37 | 0.5974 |
| ACAÍ DO TIO REI | AÇAIZIM | 30 | 31 | 0.5959 |
| VIMAQ | VIMAX | 35 | 40 | 0.5958 |
| NUMA creative design | NUMAR | 35 | 43 | 0.5954 |
| TRESSÊ | TRESSENCE | 35 | 35 | 0.5939 |
| A ASA DIGITAL INTERNET | Digital Dicas | 38 | 41 | 0.5933 |
| HIPER BR | HIPERBET | 1 | 35 | 0.5930 |
| açaí Vitaly Pura energia | GEVITALY | 30 | 35 | 0.5926 |
| LUNNA | LUNNAÈ | 35 | 14 | 0.5926 |
| IGREJA EVANGÉLICA ASSEMBLÉIA DE DEUS MINISTÉRIO NOVA ALIANÇA | IGREJA EVANGÉLICA ASSEMBLEIA DE DEUS NAÇÃO DA CRUZ | 45 | 45 | 0.5925 |
| Becker Advogados | BRANT ADVOGADOS | 45 | 45 | 0.5923 |
| Scanauto | Scana | 35 | 42 | 0.5915 |
| easybee | EASYBRAS | 35 | 35 | 0.5914 |
| FEIRA DO EMPREENDEDOR DE SOORETAMA | Rota do Empreendedor | 41 | 35 | 0.5912 |
| QUEIROZ PARTICIPAÇÕES | QUEIROZ | 35 | 17 | 0.5907 |
| SALÃO e CONGRESSO BRASILEIRO da CACHAÇA | BRANCACHAÇA | 41 | 41 | 0.5903 |
| CENTRAL MECÂNICA | CENTRA | 37 | 42 | 0.5887 |
| PURÍSSIMA | FLORÍSSIMA | 30 | 35 | 0.5884 |
| CANAL 10 | CANAL 10 | 41 | 35 | 0.5882 |
| ALPHA MOTORS | ALPHAMOLD | 35 | 21 | 0.5877 |
| Influencer ON | INFLUENCER STORE | 41 | 41 | 0.5876 |
| POWER FIX | Power Pilot | 3 | 35 | 0.5858 |
| LAG ATACADÃO | ATACADÃO DO VALE | 30 | 35 | 0.5852 |
| api.saúde | AFIP | 42 | 35 | 0.5843 |
| INSTITUTO REVER SETE QUEDAS | INSTITUTO MS | 41 | 41 | 0.5838 |
| PIZZARIA CARPATHIA | GUIA Pizzaria | 43 | 35 | 0.5835 |
| PASTEL DA INÊS | Pastel da Mi | 43 | 39 | 0.5831 |
| Fortaleza Serralheria e Vidraçaria | PORTALE | 35 | 43 | 0.5826 |
| ZAPET | SAFETY ONE | 42 | 35 | 0.5818 |
| ALUMÍNIO ALIANÇA | ALUMÍNIO ESTRELA | 35 | 6 | 0.5817 |
| NOSSO NORDESTE | LINK NORDESTE | 38 | 42 | 0.5787 |
| Zelda | SELF DRINK | 29 | 35 | 0.5786 |
| BISCOITOS VÔ PEDRO LAMPIÃO | Biscoititos | 30 | 30 | 0.5784 |
| EASYBEE | EASYBRAS | 35 | 35 | 0.5784 |
| INSTITUTO REVER SETE QUEDAS | Instituto Go | 41 | 41 | 0.5758 |
| ÍNTEGRA ENGENHARIA E SEGURANÇA | INTTEGRA + | 42 | 9 | 0.5754 |
| GRANORTE TUBOS | GRUPO PEDRA NORTE | 35 | 8 | 0.5751 |
| CANAL DEZ | Canal Dark | 0 | 38 | 0.5744 |
| NEO AGROAMBIENTAL | NEOON | 42 | 36 | 0.5741 |
| CASA D INTERIORES | CASA.CA | 35 | 35 | 0.5733 |
| PRATINHA | PRAINHA | 0 | 41 | 0.5729 |
| NN NORDESTE NOTICIAS | CITS NORDESTE | 38 | 41 | 0.5729 |
| DIVINO AGROARTE | Divino Ponto | 41 | 35 | 0.5710 |
| START COMUNICAÇÃO INTEGRADA | STAR 3 | 35 | 28 | 0.5698 |
| PORTAL DOS REIS RECEPÇÕES EVENTOS | PORTALE | 41 | 43 | 0.5698 |
| AGÊNCIA ART | Agência Work | 35 | 41 | 0.5697 |
| IHEALTH | GoHealth | 42 | 35 | 0.5691 |
| ATHENAS 3000 | ATHENAS SPORTS | 35 | 35 | 0.5688 |
| EASYBEE | Easybeer | 35 | 7 | 0.5682 |
| Digital Jurídico | digitale | 45 | 35 | 0.5679 |
| DEEP COLLECTION | COLLECTION DES CHEFS | 35 | 33 | 0.5678 |
| Agência VET | Agência Who | 35 | 41 | 0.5677 |
| CONTABEM | CONTAON | 41 | 35 | 0.5671 |
| CONTÁBIL TURMALINA | Contio Contabilidade | 35 | 35 | 0.5650 |
| SEO BAR BOTECO E BOTEQUIM | SÊO TONICO BAR | 43 | 43 | 0.5646 |
| JH CAMINHÕES | J F | 12 | 35 | 0.5645 |
| MINISTÉRIO SEMELHANTE | MINISTÉRIO NERIAH | 41 | 45 | 0.5633 |
| PROMAC | PROMAG | 35 | 40 | 0.5633 |
| Tô Fit alimentos saudáveis | TO.FISH | 35 | 43 | 0.5622 |
| BOUTIQUE PET LUCKY JR | Boutique de Luz | 35 | 14 | 0.5621 |
| FARMÁCIAS PRIME | PRIME | 35 | 35 | 0.5609 |
| CONTABEM | Contabipro | 35 | 42 | 0.5605 |
| CB CASA BACHÁ | BAC | 35 | 6 | 0.5604 |
| CENTRAL DOS CARTUCHOS INFO | Centralcart | 35 | 42 | 0.5589 |
| PRATINHA | Pratinho no ponto | 0 | 43 | 0.5588 |
| Bella NOBRE DOCES | BellaDur | 35 | 8 | 0.5584 |
| COMOL CONSULTORIA M.L. | Comot | 42 | 43 | 0.5576 |
| BRASILGAS | BRASIO | 0 | 41 | 0.5572 |
| BRASILGAS | BrasilNFT | 0 | 41 | 0.5571 |
| BE LIGHT INFRARED SAUNAS | Light Blue | 37 | 40 | 0.5563 |
| EDUARDO CONFECÇÕES | Eduardo MC | 24 | 42 | 0.5563 |
| Manacá Tecnologias Sociais | Mana | 42 | 41 | 0.5552 |
| Instituto Apollo | Instituto Apoio | 41 | 45 | 0.5550 |
| B BARÃO IBITURUNA | B BAR | 36 | 35 | 0.5547 |
| CENTRAL MECÂNICA | CENTRAL DA COP | 37 | 42 | 0.5539 |
| PURE NUTRE | ENUTRE | 43 | 41 | 0.5537 |
| Academia de Astrologia | APEX ACADEMIA | 41 | 41 | 0.5531 |
| NN NORDESTE NOTICIAS | CITS NORDESTE | 41 | 42 | 0.5514 |
| PERSONALE PLANTAS | Personare | 42 | 38 | 0.5507 |
| S HORSE | Horse Circles | 3 | 35 | 0.5503 |
| Quintal Cozinha Pra Torar | QUINTAL DO ZICO | 39 | 43 | 0.5492 |
| Reduto | REDUTOSS | 42 | 5 | 0.5481 |
| Aluris IA | ALURY | 42 | 10 | 0.5479 |
| RADIANTE CENTRO EDUCACIONAL RCE AZ 2003 | RAD-IA | 41 | 42 | 0.5468 |
| VAMO | VamoLar | 36 | 21 | 0.5466 |
| BRASILGAS | Brasiliê | 0 | 4 | 0.5465 |
| SER SAÚDE | Sr. Saúde | 38 | 9 | 0.5458 |
| EDITORA DOXA | EDITORA D+ | 35 | 9 | 0.5453 |
| NATURAL WAX | NATURALS RUN | 1 | 35 | 0.5450 |
| STAMINA DIGITAL | digitale | 35 | 35 | 0.5437 |
| RÉNOVÉ | Renova Be | 3 | 35 | 0.5437 |
| BRASILGÁS | BRASIL GOLD | 0 | 35 | 0.5433 |
| BRASILGÁS | BRASIL GOLD | 0 | 35 | 0.5433 |
| BRASILGAS | BRASIL BET | 0 | 41 | 0.5425 |
| WM INVESTIMENTOS | CBB Investimentos | 36 | 36 | 0.5423 |
| ODONTO SCAN FRANQUIAS | Scana | 44 | 42 | 0.5419 |
| TROPIGÁS | TROPIX | 0 | 35 | 0.5415 |
| TERRA SANTA AROMAS | Terra Santa Mineração | 4 | 35 | 0.5409 |
| primart | PrimaApp | 25 | 35 | 0.5405 |
| FESAN INTELIGÊNCIA IMOBILIÁRIA | RIGHT SPOT Inteligência Imobiliária | 36 | 35 | 0.5401 |
| ENGENHAR CONSULTORIA | EXER ENGENHARIA | 42 | 37 | 0.5399 |
| carmais | LAR MAIS | 37 | 35 | 0.5395 |
| NUTRI FIBER | NUTRILIFE | 31 | 44 | 0.5392 |
| Distribuidora Classic | NP Distribuidora | 35 | 35 | 0.5390 |
| G.O. CLINIC HOSPITAL DIA EM GINECOLOGIA | F6Clinic | 44 | 41 | 0.5387 |
| MIDOL | Midora | 0 | 42 | 0.5382 |
| EMPÓRIO DO PÃO | EMPÓRIO RV | 35 | 43 | 0.5380 |
| "KARNE KEIJO" | KARUI | -1 | 29 | 0.5369 |
| CENTRO DE FORMAÇÃO DE CONDUTORES VANGUARDA | AUTO ESCOLA VENTURA - CENTRO DE FORMAÇÃO DE CONDUTORES | 41 | 41 | 0.5362 |
| RS SERRALHERIA ESQUADRIA DE ALUMÍNIO E VIDROS | SERRALHERIA DEUS É FIEL | 37 | 37 | 0.5360 |
| "KARNE KEIJO" | KARUI | -1 | 29 | 0.5353 |
| DEEP COLLECTION | LUNÉ Collection | 25 | 35 | 0.5350 |
| TROPIGÁS | TROPIX | 0 | 25 | 0.5350 |
| LUAU DO SANFONEIRO | PINGO SANFONEIRO | 41 | 41 | 0.5348 |
| EMPÓRIO MAIA | MAYAD | 25 | 42 | 0.5341 |
| energia natural | ENERGIA PURA | 32 | 35 | 0.5333 |
| SORVETES DO VALE | SR.GELADO SORVETES | 35 | 35 | 0.5331 |
| HOW BE TECH | Powertech | 37 | 37 | 0.5331 |
| R.SOLAR | SOLAR.AG | 42 | 35 | 0.5330 |
| COLONIAL | Colônia Muraro | 0 | 41 | 0.5326 |
| MANÁ TEMAKERIA & SUSHI BAR DELIVERY E EVENTOS | MANAJAM | 30 | 35 | 0.5324 |
| DANIEL TRANSPORTES | Daniel Morato | 0 | 41 | 0.5304 |
| EMPÓRIO DO BOI CASA DE CARNES | EMPÓRIO DO MAR | 35 | 35 | 0.5294 |
| INSTITUTO DE MÚSICA CLARITASPULCHRI | INSTITUTO KRONA | 41 | 41 | 0.5293 |
| GABRIELE | GABRIEL BOY | 0 | 41 | 0.5284 |
| AGUARDENTE DE CANA PRATINHA | Aguardente de Cana Melindrosa | 33 | 33 | 0.5278 |
| ECO VIDA | ECOVINKA | 35 | 35 | 0.5277 |
| Pharmare | PHARMA SHARE | 35 | 41 | 0.5276 |
| Pharmare | PHARMA SHARE | 35 | 41 | 0.5276 |
| CIRCUITO DISTRIBUIDORA | CIRCUITO Speciale | 35 | 35 | 0.5273 |
| Blindador | BLINDADOS BH | 35 | 39 | 0.5266 |
| FORTALEZA MAR HOTEL | PORTA DO MAR | 43 | 39 | 0.5257 |
| MARIANA QUEIROZ ACESSORIOS | MARIANA JUNQUEIRA | 35 | 35 | 0.5252 |
| CloserX | Closerfy | 41 | 42 | 0.5244 |
| F.A.Z. Financeiro de A a Z | FAZZA | 35 | 35 | 0.5235 |
| MODA.COM | Modanna | 25 | 35 | 0.5226 |
| AGÊNCIA ORGANIZADA | Agência 100k | 42 | 35 | 0.5226 |
| EMPÓRIO COCADAS OLIMPIO | Emporio WEB | 35 | 21 | 0.5223 |
| Trampo Feito | TRAMPOLIM | 42 | 43 | 0.5220 |
| AMIGOS EM AÇÃO | ALÔ AMIGOS! | 0 | 38 | 0.5191 |
| SYM | SIMMON.AI | 36 | 42 | 0.5180 |
| VAMO | VamoLar | 36 | 35 | 0.5179 |
| DIVINO BOLO | DIVINI | 40 | 35 | 0.5176 |
| INSTITUTO REVER SETE QUEDAS | INSTITUTO ALUME | 41 | 41 | 0.5168 |
| ASA IMÓVEIS | Asali | 36 | 35 | 0.5158 |
| ECOMMERCE JUQUINHA | Vital e commerce | 35 | 35 | 0.5145 |
| EMPÓRIO CAPAS VCA | Empório Casamac | 35 | 35 | 0.5117 |
| SUPER Lagoa | SUPERLAMP | 29 | 35 | 0.5115 |
| INSTITUTO REVER SETE QUEDAS | Instituto Aura | 41 | 41 | 0.5113 |
| CASA DO RESTAURANTE | Casa Bardô | 35 | 3 | 0.5108 |
| GW LOG TRANSPORTES | W.A.G TRANSPORTES | 39 | 39 | 0.5094 |
| Aprende Aplica | APRENDIZ | 35 | 33 | 0.5092 |
| UNIMAQ | UNIMARC | 35 | 35 | 0.5081 |
| FOR YOU EVENTOS | Pure for You | 41 | 35 | 0.5080 |
| PERFAL | PERFORM | 35 | 45 | 0.5074 |
| MOBY URBANO | MOBYAN | 9 | 42 | 0.5074 |
| VELOZ | VELORA | 35 | 5 | 0.5071 |
| HOME TOP MÓVEIS E COLCHÕES | HOMETAU | 35 | 35 | 0.5070 |
| CONTÁBIL TURMALINA | HEINIG CONTÁBIL | 35 | 35 | 0.5068 |
| IÁ! COMUNICAÇÃO | IA+ | 38 | 42 | 0.5066 |
| BG BOUTIQUE GODIVA | BOUTIQUE DAYIL | 35 | 35 | 0.5052 |
| Manacá Tecnologias Sociais | Manai | 42 | 43 | 0.5045 |
| BRASILGAS | BRASIL BET | 0 | 35 | 0.5039 |
| BRASILGÁS | BRASIL BET | 0 | 35 | 0.5039 |
| STUDIO R BY RENATA SANTIAGO | STUDIOCAR | 44 | 35 | 0.5036 |
| DM AMBIENTAL | RF AMBIENTAL | 40 | 35 | 0.5034 |
| TERRAL MARESIAS | Terra Gaia | -1 | 25 | 0.5029 |
| CRISTÃ CONNECT | CRYSTAL MOON | 41 | 41 | 0.5026 |
| BRASILGAS | BRASILUZ | 0 | 37 | 0.5016 |
| api.saúde | Ápice | 42 | 35 | 0.5009 |
| ATALAIA NORONHA | ENJOY NORONHA | 39 | 39 | 0.5003 |
| VIA BRASIL | VIAJEI BRASIL | 35 | 35 | 0.4996 |
| SUPER LION | SUPERLAMP | 32 | 35 | 0.4992 |
| RESTAURANTE MENINA GERAIS | RESTAURANTE TATA | 43 | 43 | 0.4990 |
| CASA D INTERIORES | CASA LÔ | 35 | 25 | 0.4985 |
| SER SAÚDE | Sr. Saúde | 35 | 44 | 0.4976 |
| PARADA DA SERRA | Parada do Coco | 35 | 43 | 0.4976 |
| Zelda | SANTA KILDA | 29 | 35 | 0.4969 |
| ELÉTRICA GONÇALVES | ilumelétrica | 37 | 35 | 0.4963 |
| CACHOEIRA | CABANA MIRADOR DA CACHOEIRA | 30 | 43 | 0.4959 |
| CAMPOS EXTINTORES | CAMPO | 35 | 43 | 0.4952 |
| BOND BOCA | Bondcom | 35 | 1 | 0.4945 |
| ECO SOLUÇÕES EM ENERGIA | ILB SOLUÇÕES | 42 | 37 | 0.4943 |
| IDEAL POP | IDEAL 3D | 11 | 6 | 0.4940 |
| CONTABILIDADE MASTER | SATTVA CONTABILIDADE | 35 | 35 | 0.4937 |
| Zelda | ZETU | 29 | 35 | 0.4935 |
| S HORSE | Horse Now | 3 | 35 | 0.4933 |
| COLÉGIO M MASA | COLÉGIO COP KIDS | 41 | 41 | 0.4920 |
| RAIOS | Raiô | 35 | 36 | 0.4914 |
| HEALTHMAP | Health Makers | 42 | 41 | 0.4913 |
| FUNERÁRIA GONZAGA | FUNERÁRIA OMEGA | 45 | 45 | 0.4905 |
| SAÚDE CERTA MATERIAL HOSPITALAR | Alerta Saúde | 35 | 44 | 0.4904 |
| SERTANEJA | SERTANEJA ARRETADA | 35 | 33 | 0.4902 |
| CENTRO EDUCACIONAL MÚLTIPLA ESCOLHA | UP GRUPO EDUCACIONAL | 41 | 41 | 0.4883 |
| ACADEMIA DO VAREJO | ACADEMIA GO IT | 35 | 25 | 0.4868 |
| TRU LOGÍSTICA | Trevo | 43 | 35 | 0.4866 |
| AGÊNCIA ORGANIZADA | Agência Oak | 42 | 35 | 0.4865 |
| Virtua Vet Coworking Veterinário | Vireê | 35 | 40 | 0.4858 |
| RESERVA LW INDÚSTRIA DE BEBIDAS SAUDÁVEIS | RESERVA Sprint. | 35 | 35 | 0.4858 |
| DM AMBIENTAL | a ambiental | 40 | 42 | 0.4855 |
| CARBOTAR | CARBO START | -1 | 8 | 0.4847 |
| BELA MARI | BEL | 41 | 35 | 0.4839 |
| RANCHO NOVO | RANCHO 2K | 35 | 45 | 0.4830 |
| BEACH GOLDEN | BEACHING | 35 | 35 | 0.4827 |
| ARMAZÉM DAS GERAES SECOS E MOLHADOS | ARMAZÉM DAS ESPUMAS | 43 | 40 | 0.4818 |
| SANTA PRAIA | santaprint | 43 | 40 | 0.4815 |
| BRASILGAS | BRASIL X | 0 | 36 | 0.4814 |
| BRASILGÁS | BRASIL X | 0 | 36 | 0.4814 |
| FRANQUIA FANTASIA | FANTASYDRAFT | 28 | 41 | 0.4790 |
| IDEAL FARMA | idealcert | 35 | 42 | 0.4790 |
| INSTITUTO REVER SETE QUEDAS | INSTITUTO ALLWERT | 41 | 41 | 0.4789 |
| SOLUÇÕES FUNDAÇÃO | SOLUÇÕES ADM | 37 | 35 | 0.4784 |
| D&F MODA FITNESS | GRA.BELLE MODA FITNESS | 35 | 25 | 0.4779 |
| GREEN AGRO BIO | GREEN AGE | 35 | 35 | 0.4756 |
| GRANJA VILELA GV | GRANJA AVIN | 29 | 29 | 0.4745 |
| ACAIACA MALL | Açaiê | 0 | 35 | 0.4739 |
| INSTANTE URBANO | INSTANTÂNEA | 35 | 41 | 0.4728 |
| GB GIBATTAGLIA | GB | 36 | 35 | 0.4720 |
| TELEVISÃO DIÁRIO | DIÁRIO 360 | 41 | 35 | 0.4718 |
| FRISMAX | PRISMA | 11 | 35 | 0.4715 |
| ARQUITETURA INTEGRATIVA | INTEGRARTE | 41 | 41 | 0.4710 |
| BOUTIQUE SAÚDE | Bo Bardi Boutique | 35 | 25 | 0.4708 |
| E A ESTUDIO ARTE | estúdiopena | 38 | 42 | 0.4705 |
| RAPHAELA SHAKE MIX | Rapha | 43 | 35 | 0.4701 |
| SALÃO DE BELEZA INFANTIL CANTINHO DA CRIANÇA | CANTINHO DA MODA AS | 44 | 35 | 0.4688 |
| UNIVERSO KINDER | UN Universo Tennis | 16 | 35 | 0.4682 |
| RESTAURANTE MENINA GERAIS | RESTAURANTE CHINA 25 | 43 | 43 | 0.4681 |
| IGREJA EVANGÉLICA ASSEMBLEIA DE DEUS MINISTÉRIO DO TEMPLO CENTRAL | Assembleia de Deus Ministério Vila Nova | 41 | 45 | 0.4676 |
| ESQUADRIPLUS | esquadr.IA | 35 | 42 | 0.4674 |
| NUTRI FIBER | NUTRIVE | 31 | 35 | 0.4670 |
| AQUA PISCINAS | AQUAX | 19 | 35 | 0.4660 |
| DX Vendas Oficial | DBS Vendas | 41 | 35 | 0.4658 |
| FEIRA DO AGRONEGÓCIO E EMPREENDEDORISMO DE ALFREDO CHAVES | FC AGRONEGOCIOS | 41 | 44 | 0.4656 |
| TUDO DELLA | TUDO DE PET | 3 | 35 | 0.4651 |
| E - SAFE | SAFEUP | 35 | 35 | 0.4640 |
| RANCHO DO ACARAJÉ | RANCHO BENNU | 43 | 41 | 0.4631 |
| Marketing Erótico | MeuMarketing | 41 | 9 | 0.4608 |
| COLÉGIO CONEXÃO | CONEXÃO WK | 41 | 35 | 0.4606 |
| DELÍCIAS DA XÚ LANCHONETE | DELICIAS DO MILHO | 43 | 40 | 0.4603 |
| COLÉGIO M MASA | COLÉGIO CONNEXT | 41 | 41 | 0.4595 |
| ELÉTRICA TECH MATERIAIS ELÉTRICOS E AUTOMAÇÃO INDUSTRIAL | ELÉTRICA LIMA | 35 | 37 | 0.4593 |
| MINAS IMPORTS | MINASLIGAS | 35 | 35 | 0.4588 |
| MASTER SHOES | MASTER WIN | 35 | 35 | 0.4587 |
| idealize | ideale | 35 | 20 | 0.4573 |
| GYM NUTRIX | Glo ? NuTrix | 35 | 5 | 0.4567 |
| MERCA CONTABILIDADE & CONSULTORIA | Mercasa | 35 | 35 | 0.4564 |
| S.O.L FESTIVAL | SOLD | 41 | 35 | 0.4557 |
| EMPÓRIO DO PÃO | EMPÓRIO ZOE | 35 | 35 | 0.4557 |
| SUSTENTARTE | SUSTENTAPET | 35 | 40 | 0.4553 |
| SUSTENTARTE | SUSTENTAPET | 35 | 40 | 0.4553 |
| SORVETES DO VALE | DONNA SORVETES | 35 | 29 | 0.4552 |
| SORVETES DO VALE | DONNA SORVETES | 35 | 29 | 0.4552 |
| RI ARQUITETURA | Arquitetura Jurídica | 42 | 45 | 0.4551 |
| CHECKUP | Checkmob | 38 | 9 | 0.4544 |
| Virtua Vet Coworking Veterinário | Virtual Saúde | 44 | 35 | 0.4543 |
| PROMAC | PROMM | 35 | 10 | 0.4540 |
| MARCENARIA J. MEDEIROS | MARCENARIA 4.0 | 20 | 35 | 0.4538 |
| DELÍCIAS DA XÚ LANCHONETE | Delício | 43 | 42 | 0.4525 |
| LAG ATACADÃO | ATACADÃO DO VALE | 35 | 35 | 0.4518 |
| VILLAR ELÉTROS | Villas UBA | 35 | 43 | 0.4510 |
| CASA DO RESTAURANTE | CASCADO | 35 | 43 | 0.4501 |
| BRASILGAS | BRASIX | 0 | 9 | 0.4484 |
| NACIONAL GÁS BUTANO | TIRO NACIONAL | 0 | 41 | 0.4477 |
| FUNERÁRIA SANDES | Funerária Wolff | 35 | 45 | 0.4466 |
| Master Multi | MasterHit | 35 | 41 | 0.4462 |
| E-CLEAN | CLEAN CAPAS | 11 | 35 | 0.4456 |
| A MAR A VIDA | AMARAE | 41 | 35 | 0.4444 |
| MASTER SHOES | MASTER SPICES | 35 | 30 | 0.4440 |
| CONSTRUTORA COLMEIA | Colmeia Vídeos | 0 | 41 | 0.4436 |
| GARAGEM 59 | GARAGE BEACH | 35 | 41 | 0.4419 |
| BG BOUTIQUE GODIVA | B Nice Boutique | 35 | 25 | 0.4407 |
| EMPÓRIO DO PÃO | Emporio WEB | 35 | 21 | 0.4405 |
| EMPÓRIO DO BOLO | EMPÓRIO DUGÁ | 30 | 35 | 0.4385 |
| SOUZA | SOUZ | 4 | 35 | 0.4383 |
| CHECKUP | CHECKIN | 41 | 45 | 0.4379 |
| VOLTS ENERGIA TOTAL | WOLTEK | 35 | 7 | 0.4376 |
| neo núcleo de emergências odontológicas | NEOON | 44 | 35 | 0.4354 |
| VIDRAÇARIA E ESQUADRIA IMPERIAL | VIDRAÇARIA MK | 35 | 37 | 0.4352 |
| BRASILGAS | BRASIL BET | 0 | 38 | 0.4351 |
| BRASILGÁS | BRASIL BET | 0 | 38 | 0.4351 |
| BRASILGÁS | BRASIL BET | 0 | 38 | 0.4351 |
| RAÇA FORTE | FORTEC | 35 | 12 | 0.4347 |
| BOUTIQUE PET LUCKY JR | Boutique de Luz | 35 | 40 | 0.4346 |
| PURE NUTRE | PURE | 43 | 42 | 0.4340 |
| Lucky Jr Pet | LUCKY CUP | 35 | 7 | 0.4338 |
| DEEP COLLECTION | WE COLLECTION | 35 | 25 | 0.4312 |
| SUPERMERCASA | Supermercados Kern | 35 | 39 | 0.4301 |
| SORVETES DO VALE | SKIPEROLA SORVETES | 35 | 43 | 0.4295 |
| MARÉ SOCIAL | MARÉ | 41 | 41 | 0.4295 |
| COMUNICADO | Comuny | 16 | 41 | 0.4292 |
| CLAY. | Clayver | 3 | 35 | 0.4291 |
| PIZZARIA LEOO'S | MENSATO´S PIZZARIA | 43 | 43 | 0.4289 |
| GABRIELE | Gabriel | 0 | 42 | 0.4281 |
| F.A.Z. Financeiro de A a Z | Pazen | 35 | 45 | 0.4274 |
| D&F MODA FITNESS | K.FIT - MODA FITNESS | 35 | 25 | 0.4272 |
| PINHEIRO MÓVEIS | MIL PINHEIRO | 20 | 35 | 0.4249 |
| R.IMUNO CRIO | Imunofem | 5 | 35 | 0.4245 |
| ESSENCIAL MOTOS ATACADO E VAREJO | ESSÊNCIA GA.IAH | 12 | 42 | 0.4227 |
| MUNDO DA TABACARIA | ALEMÃO TABACARIA | 35 | 35 | 0.4223 |
| SMART SCOOTERS | SMARTCOMP | 35 | 35 | 0.4218 |
| SÓ PISO CACULÉ | SOFISTIC | 35 | 20 | 0.4218 |
| PBENERGIASOLAR | BRUSOLIS ENERGIA SOLAR | 37 | 37 | 0.4205 |
| ALUMÍNIO ALIANÇA | UNIALUMÍNIO | 35 | 35 | 0.4202 |
| INVENTAR | INVENCI | 35 | 18 | 0.4194 |
| BOUTIQUE DO FERRO | BOUTIQUE DO SONO | 40 | 20 | 0.4191 |
| LATICÍNIOS ED LEITE | LATICINIOS VALE DO RIO | 29 | 29 | 0.4187 |
| CACHOEIRA | Cachoeira da Chinela | 30 | 43 | 0.4185 |
| FEIRA DO EMPREENDEDOR DE SOORETAMA | SKIN EMPREENDEDORA | 41 | 41 | 0.4185 |
| G.O. CLINIC HOSPITAL DIA EM GINECOLOGIA | CLÍNICA TAZ | 44 | 44 | 0.4176 |
| VELOZ | VELOSTER | 35 | 37 | 0.4164 |
| FUNERÁRIA SANDES | FUNERÁRIA PAZ FAMILIAR | 36 | 35 | 0.4149 |
| UNIVERSO DO PÃO PADARIA E CONFEITARIA | UNIVERSO DA BELEZA | 43 | 35 | 0.4149 |
| CASA D INTERIORES | CASA D'ELLA | 35 | 3 | 0.4140 |
| VIDRAÇARIA MUNDO DOS VIDROS | VIDRAÇARIA RISPOLI | 19 | 35 | 0.4128 |
| LAFUENTE TURISMO | LA PINTÊ | 0 | 35 | 0.4118 |
| ESCOLA MÓVEL | Mover | 41 | 35 | 0.4117 |
| Virtua Office | ESPELHO VIRTUAL | 35 | 42 | 0.4117 |
| SORVETES ICE TRUCK | Sorvetes Sergel | 30 | 35 | 0.4115 |
| MERCA CONTABILIDADE & CONSULTORIA | Mercanfily | 35 | 35 | 0.4110 |
| CASA CONCRETA | CONCRETESTE | 38 | 42 | 0.4105 |
| pH Piscina | FH | 40 | 17 | 0.4101 |
| TECNO PLANT | TECNOLITA | 35 | 19 | 0.4100 |
| LULLY'S ALIMENTOS ARTESANAIS | LULLA | 30 | 35 | 0.4092 |
| Virtua Vet Coworking Veterinário | VIRTUOSO | 44 | 43 | 0.4091 |
| AVESTA | AVEN | 35 | 35 | 0.4072 |
| GRUPO MS MAIS QUE SAÚDE | GRUPO HOMERO | 35 | 35 | 0.4065 |
| COLISÃO DISTRIBUIDORA EMBALAGENS EM GERAL | Fene Distribuidora | 35 | 3 | 0.4060 |
| MÔNICA CURY | MÔNICA CELL | 25 | 35 | 0.4054 |
| FARMÁCIA ANDRADE | FARMÁCIA ARACAJU | 35 | 35 | 0.4041 |
| INSTITUTO REVER SETE QUEDAS | INSTITUTO JCO | 41 | 42 | 0.4040 |
| BOI E BRASA | BRASAL | 39 | 43 | 0.4030 |
| ANDER SABORES | and wander | 29 | 35 | 0.4024 |
| energia natural | GAIA natural | 32 | 35 | 0.4020 |
| MesoAcademy | MEF ACADEMY | 41 | 42 | 0.4018 |
| WORLD GEOMETRIC | WORLDFY | 14 | 35 | 0.4009 |
| CASA FÁTIMA AVIAMENTOS E TECIDOS | PATIS | 35 | 35 | 0.4004 |
| Next Personal | Nest | 42 | 37 | 0.3991 |
| ELÉTRICA TECH MATERIAIS ELÉTRICOS E AUTOMAÇÃO INDUSTRIAL | eletrycos | 35 | 35 | 0.3976 |
| D&F MODA FITNESS | STELICIO MODA FITNESS | 35 | 35 | 0.3971 |
| AUTO PEÇAS MATOS | CAEL AUTOPEÇAS | 35 | 35 | 0.3969 |
| RÉNOVÉ | MeRenove | 35 | 35 | 0.3956 |
| G.O. CLINIC HOSPITAL DIA EM GINECOLOGIA | CLÍNICA RS | 44 | 44 | 0.3951 |
| INTERNI | INTERSOM | 35 | 35 | 0.3948 |
| ROSA VIEIRA | A VIEIRA | 35 | 41 | 0.3918 |
| A AMARE HOME | Ama | 35 | 44 | 0.3903 |
| INSTITUTO REVER SETE QUEDAS | INSTITUTO PRISMA | 41 | 41 | 0.3902 |
| ALTO BRASIL | OBJETO BRASIL | 35 | 35 | 0.3897 |
| GULA LANCHES | Gulp | 43 | 35 | 0.3893 |
| AUTO PEÇAS MATOS | Cubatão Autopeças | 35 | 35 | 0.3892 |
| CONTÁBIL TURMALINA | CONTAON | 35 | 35 | 0.3891 |
| IHOME | IMMI HOME | 42 | 35 | 0.3890 |
| Persona Joias PJ Joias afetivas | Persona | 14 | 28 | 0.3880 |
| PROD MONITOR | FRON | 9 | 41 | 0.3873 |
| S HORSE | Horse Now | 44 | 35 | 0.3868 |
| AVESTA | AVI | 35 | 35 | 0.3862 |
| IVA INSTITUTO DE ESTUDOS E PESQUISAS DO VALE DO ACARAÚ | Instituto Ékatus | 41 | 41 | 0.3859 |
| FAROL ACADÊMICO | Parditude | 41 | 41 | 0.3853 |
| MASSA VINIL | MASSIÊ | 2 | 35 | 0.3849 |
| OKA PLANEJADOS | OKAP | 42 | 9 | 0.3845 |
| BRASIL SEGURANÇA ELETRÔNICA | mobbrasil | 45 | 42 | 0.3839 |
| IMP INSTITUTO DE MICRO HABILIDADES EM PROJETOS Transformando Visão em Ação junto com você. | IMBIMP | 41 | 42 | 0.3829 |
| FAROL ENERGIA DO BRASIL | Farol Red | 35 | 25 | 0.3818 |
| GARANTIA DOS VALES SOCIEDADE DE GARANTIA DE CRÉDITO | JGS GARANTIA | 36 | 36 | 0.3818 |
| easybee | EASYBRAS | 35 | 9 | 0.3817 |
| EM FLOR CASA DEI FIORI | FLORA | 35 | 23 | 0.3808 |
| A MAR A VIDA | MAIS VIDA | 25 | 35 | 0.3801 |
| CASTELLA | castellane | 20 | 35 | 0.3798 |
| Neuro+ Saúde | NEURON5 | 44 | 41 | 0.3798 |
| Burguer do Nô | GREGO BURGUER | 43 | 43 | 0.3795 |
| MV TOLDOS | TOLDOS MARTINS | 22 | 35 | 0.3783 |
| INSTITUTO REVER SETE QUEDAS | instituto update | 41 | 41 | 0.3781 |
| FA FASHION ATACADISTA O VAREJO NOPRECINHO DE ATACADO | FAHLENA ECO FASHION | 35 | 35 | 0.3777 |
| DOC SERVICES | DOCES + | 35 | 30 | 0.3768 |
| POLPA DE FRUTAS CONGELADAS DONA TEREZINHA | LupaFrui frutas congeladas | 29 | 29 | 0.3765 |
| SORVETES DHIJU | SORVETES ANDELI | 40 | 30 | 0.3753 |
| YAP | ïapó | 35 | 31 | 0.3749 |
| EISOLAR SISTEMA FOTOVOLTÁICO | Brisolar | 42 | 21 | 0.3745 |
| LUMINUM | Lumi | 35 | 35 | 0.3736 |
| MERCADO DAS COISAS | mercado compara | 35 | 35 | 0.3724 |
| ATENDE BIKE | ATENDMED | 35 | 44 | 0.3722 |
| MARES AO MAR | MARÉ | 41 | 41 | 0.3718 |
| amar.elo SAÚDE MENTAL | AMARETTI | 35 | 30 | 0.3711 |
| SANTA BÁRBARA | SANTA BÁRBARA BATERIAS | 0 | 37 | 0.3694 |
| GREEN GUPPY | GREEN FORCE | 25 | 35 | 0.3690 |
| CACHOEIRA DISTRIBUIDORA | CABANA MIRADOR DA CACHOEIRA | 30 | 43 | 0.3685 |
| DE OURO E PRATA | Ouro de Minas | 35 | 33 | 0.3682 |
| JIU - JITSU BRASIL | ZION JIU-JITSU | 35 | 25 | 0.3679 |
| INSTITUTO DOXA | Instituto Butiá | 35 | 45 | 0.3673 |
| AM AGRÍCOLA MACHADO | JHZ AGRICOLA | 31 | 31 | 0.3671 |
| CARBOTAR | CARBOX | -1 | 9 | 0.3671 |
| Florens | FLORÉ | 41 | 9 | 0.3669 |
| CAFÉ ESPECIAL SERENGETI | SERENA | 35 | 28 | 0.3650 |
| TERRA DO GELO AR CONDICIONADO | TERRA DO SOL | 35 | 41 | 0.3641 |
| REI DA LINGERIE | LIZES LINGERIE | 35 | 35 | 0.3639 |
| Nutris no Online | NutriSy | 35 | 35 | 0.3630 |
| ÁGUIA BRASIL DISTRIBUIDORA | BRASILIA | 35 | 12 | 0.3629 |
| açaí Vitaly Pura energia | Vitalume | 35 | 44 | 0.3621 |
| GRUPO PRIME PLUS | Primea | 39 | 35 | 0.3616 |
| SINDELETRO | SIM Eletro | 45 | 35 | 0.3614 |
| PROJETO SOCIAL PRINCÍPIOS | PROJETO SOCIAL DORCAS EM AÇÃO NA NAÇÃO | 45 | 45 | 0.3613 |
| RI ARQUITETURA | GALE ARQUITETURA | 42 | 42 | 0.3610 |
| CANAL 10 | Canal Dark | 41 | 38 | 0.3610 |
| SPACE BURGER | SPACE | 40 | 18 | 0.3608 |
| DOE DE CORAÇÃO | COLORADO DO CORAÇÃO | 41 | 41 | 0.3607 |
| GabrielBot | GABRIELA | 42 | 41 | 0.3599 |
| GabrielBot | GABRIELA | 42 | 41 | 0.3599 |
| INFLUA | influur | 41 | 42 | 0.3598 |
| PASTEL DA INÊS | Pastel da Mi | 43 | 35 | 0.3593 |
| CASA DO RESTAURANTE | CASA DO BODE | 35 | 29 | 0.3589 |
| NEW INVEST CONSÓRCIOS | Invest JP | 36 | 36 | 0.3584 |
| Carneiros Temporada | Carla Carneiro | 43 | 35 | 0.3582 |
| PARADINHA SORVETE DE IOGURTE | PANDINHA | 35 | 28 | 0.3578 |
| SPRINGS | SPRINT55 | 0 | 25 | 0.3575 |
| AR MARCENARIA AMBIENTES PLANEJADOS | Tupã Marcenaria | 20 | 20 | 0.3571 |
| BALIZA TRATAMENTO DE MADEIRA | Balise | 35 | 35 | 0.3569 |
| TALITA SECCO | TAYTA | 35 | 35 | 0.3563 |
| NATURAL VILLE | Naturaliz | 37 | 35 | 0.3562 |
| RNH LINGERIE | LINGER | 35 | 1 | 0.3561 |
| DYE ME | MED+ | 35 | 10 | 0.3555 |
| RAIOS | Raiô | 35 | 36 | 0.3555 |
| PERFAL | PERFECCI | 35 | 25 | 0.3540 |
| CENTRAL DO PRATINHA | CENTRAL DO KIT | 39 | 35 | 0.3540 |
| LAVANDERIA SELF SERVICE LAVEMI | PULITO LAVANDERIA | 35 | 40 | 0.3537 |
| CAFÉ TEU GRANO | GRANOTECA | 30 | 30 | 0.3536 |
| AKORI | AKORYZ | 35 | 41 | 0.3528 |
| BELLA LIA | BELLAVÈRA | 25 | 35 | 0.3518 |
| FARMÁCIAS PRIME | PRIMEAC | 35 | 35 | 0.3514 |
| BAR E RESTAURANTE ZÉ CARLOS | BF MUSIC Bar e Restaurante | 43 | 43 | 0.3482 |
| CONEXÃO FOOD HALL | conexão goiás | 35 | 41 | 0.3481 |
| NACIONAL GAS BUTANO | TIRO NACIONAL | 0 | 41 | 0.3481 |
| IBIT MOTOS | IBID | 35 | 35 | 0.3474 |
| PONTO CELL | PONTO RURAL | 37 | 35 | 0.3469 |
| ALLON | ALLOY | 41 | 42 | 0.3469 |
| A ASA DIGITAL INTERNET | Ascendia Digital | 38 | 35 | 0.3463 |
| FAROL ACADÊMICO | Farol Boemia | 41 | 43 | 0.3442 |
| BETÃO REFRIGERAÇÃO | BETÃO | 35 | 28 | 0.3419 |
| DELÍCIAS DA XÚ LANCHONETE | Delícias da Rochell | 43 | 30 | 0.3412 |
| NACIONAL | NACIONAL FIRE | 0 | 45 | 0.3397 |
| ESMALTEC | ESMALTAgel | 0 | 44 | 0.3389 |
| META ENGENHARIA SERVIÇOS | Metalon | 35 | 45 | 0.3388 |
| RNP BRASIL ATACADO CAMA MESA E BANHO | RP BRASIL | 35 | 35 | 0.3376 |
| SUPER LED ILUMINAÇÃO | SUPERNA | 35 | 44 | 0.3369 |
| AUTO PEÇAS MATOS | CARIOCA AUTO PEÇAS | 35 | 35 | 0.3361 |
| Favily | HAVIX | 35 | 35 | 0.3360 |
| MALU GUARDANAPOS DE PAPEL | MALIBU | 16 | 1 | 0.3350 |
| MIL PLASTIC | PLASTIC FLOOR | 35 | 37 | 0.3341 |
| HOTEL SERRA LIMA | HOTEL SERRA DO GANDARELA | 43 | 43 | 0.3339 |
| ELEVA POR RENATA PAULA SANTIAGO | EVA | 35 | 35 | 0.3334 |
| VIRTUA COFFEE WORK | Vireê | 43 | 40 | 0.3329 |
| CONEXÃOVIP INTERNET FIBRA ÓPTICA | Conexão | 38 | 41 | 0.3324 |
| DURAFORT | ARAFORT | 11 | 35 | 0.3305 |
| IA Facilita | IA+ | 35 | 44 | 0.3302 |
| MARCENARIA MLM LUCAS MÓVEIS | Tupã Marcenaria | 20 | 20 | 0.3299 |
| MANÍ | MANIA MANIA | 35 | 35 | 0.3296 |
| Influencer ON | FAST INFLUENCER | 41 | 35 | 0.3294 |
| CS HIGHTICKET | CRM HIGH TICKET | 41 | 9 | 0.3294 |
| MOVÊH | MOVERE | 42 | 35 | 0.3288 |
| OTIMIZEI | OTIMIFY | 35 | 9 | 0.3281 |
| CENTRO AUDITIVO LIBERTY | LYD CENTROAUDITIVO | 35 | 44 | 0.3281 |
| LATICÍNIOS ED LEITE | LATICÍNIOS SANTA CRUZ | 29 | 40 | 0.3277 |
| IDEAL SUPER | ideact | 11 | 35 | 0.3271 |
| CAPITAL DE FORTALEZA | FORUM CAPITAL | 0 | 35 | 0.3268 |
| SORVETES ICE TRUCK | SKIPEROLA SORVETES | 30 | 43 | 0.3264 |
| CRISTAL ORGÂNICOS | CRISTAL SKIM COAT | 40 | 1 | 0.3260 |
| AVESTA | Averse | 35 | 25 | 0.3255 |
| DISTRIBUIDORA BAESSA ÁGUA GÁS E CARVÃO | MC3 DISTRIBUIDORA | 35 | 35 | 0.3254 |
| CENTRO EDUCACIONAL MÚLTIPLA ESCOLHA | MC EDUCACIONAL | 41 | 35 | 0.3253 |
| TROPICAL CHIC | MUSA TROPICAL | 35 | 35 | 0.3247 |
| Paço | FACE | 35 | 35 | 0.3245 |
| MD SISTEMAS | SDC SISTEMAS | 42 | 42 | 0.3236 |
| FUNERÁRIA E VELÓRIO VALE DO PARAÍBA | Funerária Camaquense | 45 | 45 | 0.3236 |
| HORTILOG | Forttilo | 9 | 35 | 0.3236 |
| BRISA ESQUADRIAS DE PVC | BRISA KIDS | 40 | 40 | 0.3233 |
| LA COSTELA RESTAURANTE PIZZARIA E PETISCARIA | COSTELA 1008 | 43 | 43 | 0.3230 |
| Rénové Biocosmetic Spray Hidratante Corporal (Body Moisturizing) | Renovare | 3 | 35 | 0.3225 |
| CONTABILIDADE MASTER | NJ CONTABILIDADE | 35 | 35 | 0.3223 |
| MARIA MARIANA | MAR.IANA | 43 | 41 | 0.3220 |
| HIDRAR PISCINAS | Hydrax | 35 | 5 | 0.3217 |
| GESTAÇÃO | Gestare | 9 | 35 | 0.3211 |
| SORVETES KARYONE | DACRIS SORVETES | 30 | 29 | 0.3210 |
| EMPÓRIO DO BOI CASA DE CARNES | EMPÓRIO DOM PÉPE | 35 | 35 | 0.3183 |
| CENTRAL ÁUDIO SOM E ACESSÓRIOS | CENTRAL AUTO PARTS | 37 | 35 | 0.3180 |
| D&F MODA FITNESS | DRY ARAUJO MODA FITNESS | 35 | 35 | 0.3177 |
| GENTE | GENENTECH | 16 | 35 | 0.3176 |
| Editora Futura | FUTURO DEV | 41 | 41 | 0.3171 |
| BOM PASTELONE | Boma | 43 | 35 | 0.3171 |
| LATICINIO STELLALPINA | LATICÍNIOS NALMILK | 29 | 35 | 0.3169 |
| PÃES CONGELADOS SANTA FÉ | Congela Congelados | 35 | 35 | 0.3155 |
| IDEAL SUPER | Idealtec | 11 | 9 | 0.3153 |
| PONTO PODER | PONTO | 38 | 42 | 0.3140 |
| DINI MÓVEIS | DIVINI | 35 | 35 | 0.3136 |
| REI DAS CLÍNICAS | CLÍNICA ARVI | 41 | 44 | 0.3135 |
| AVESTA | AVA | 35 | 35 | 0.3131 |
| D&F MODA FITNESS | DANNA BANNA MODA FITNESS | 35 | 35 | 0.3126 |
| CAIPIRA MÁQUINAS DE SILAGEM | Caipira Padoca | 7 | 35 | 0.3125 |
| Fortaleza Serralheria e Vidraçaria | FORTALEZA AÇAÍ | 37 | 35 | 0.3117 |
| FB Jewelry | PB | 35 | 35 | 0.3113 |
| Ponto Metal | PONTO CERTO | 6 | 35 | 0.3104 |
| VASTO RESTAURANTE | VASTA | 41 | 44 | 0.3093 |
| BRASILGAS | BRASIL BÍBLIAS | 0 | 35 | 0.3090 |
| E-CONTROL | BSV CONTROL | 11 | 11 | 0.3084 |
| MULTIFY | MULTIPOD | 36 | 37 | 0.3082 |
| ACT ACADEMIA CEARENSE DE TÊNIS | Acton | 41 | 35 | 0.3078 |
| ES CONSTRUÇÃO | EMAQ CONSTRUÇÃO | 41 | 7 | 0.3075 |
| Prisma perfuração de poços | PRISMUN | 37 | 35 | 0.3066 |
| EDUCALAB | educa.ai | 38 | 42 | 0.3063 |
| FESTIVAL DE FRUTOS DO MAR DE ITAPEMIRIM | MARRÍ | 41 | 44 | 0.3060 |
| MELHOR.AI | MELFORT | 42 | 32 | 0.3058 |
| ECOCAST | EcoCar | 45 | 39 | 0.3053 |
| SpeedTech Telefonia, cabos e informática | SPEEDTECH | 35 | 1 | 0.3052 |
| B BRASLIMP | BRASLOM | 37 | 35 | 0.3049 |
| BOM PASTELONE | boom | 43 | 42 | 0.3046 |
| EMPÓRIO COCADAS OLIMPIO | Empório Vilarin | 35 | 25 | 0.3038 |
| C CONTROLLER | Controlline | 0 | 4 | 0.3022 |
| MULTI MAIS ECOMMERCE | MULTIVIX | 35 | 35 | 0.3017 |
| A CASA DA CONSTRUÇÃO | CASA AQUIM | 35 | 35 | 0.3014 |
| PERFECT LINE | PERFECT LASER | 41 | 35 | 0.3000 |
| FAMÍLIA CHURRASQUINHO | BEM FAMÍLIA | 35 | 33 | 0.2996 |
| NUTRI FIBER | NutriVerde | 31 | 31 | 0.2981 |
| RI ARQUITETURA | Lassi Arquitetura | 42 | 42 | 0.2980 |
| CONTROLLER GO CONTABILIDADE DIGITAL | CONTROLIFE | 35 | 45 | 0.2976 |
| AMOMERCI | AMOMÊ | 31 | 35 | 0.2973 |
| A MAR A VIDA | AMARA | 35 | 4 | 0.2962 |
| Rénové Biocosmetic-Serum Oil Skin | Renovatur | 3 | 35 | 0.2957 |
| CONFIA + SOLUÇÕES FINANCEIRAS | INOVE SOLUÇÕES FINANCEIRAS | 36 | 36 | 0.2943 |
| NOSSA MARCENARIA ! | MARCENARIA GUARUJÁ | 20 | 40 | 0.2932 |
| BARBEARIA DOM BIASI | W7 Barbearia | 44 | 35 | 0.2926 |
| MERCADOFAZENDINHA | PANDINHA | 35 | 28 | 0.2917 |
| AUTO PEÇAS MATOS | Zofort Autopeças | 35 | 35 | 0.2913 |
| AR MARCENARIA AMBIENTES PLANEJADOS | MARCENARIA CALIFÓRNIA | 20 | 20 | 0.2913 |
| Virtua Ideas | Virtuosos | 35 | 41 | 0.2903 |
| AUTO PEÇAS MATOS | B&M Auto Peças | 35 | 35 | 0.2895 |
| BROOKS BRASIL JEANS | CKS BRASIL | 25 | 35 | 0.2892 |
| DESÍGNIO | DESIGNWALL | 35 | 19 | 0.2885 |
| DISTRIBUIDORA POPEYE JR. | DISTRIBUIDORA DE BEBIDAS SKINÃO | 39 | 35 | 0.2885 |
| BOM LEITE | LEITE | 43 | 35 | 0.2885 |
| ANNA MACEDO | Manna | 35 | 31 | 0.2872 |
| ECO SOLUÇÕES EM ENERGIA | Enersolo Soluções | 37 | 37 | 0.2866 |
| Engenharia Academy | ACADEMIA MEI | 41 | 35 | 0.2863 |
| PERFAL | persalt | 35 | 1 | 0.2862 |
| VIRTUA COFFEE WORK | COFFEE CALM | 43 | 43 | 0.2862 |
| Espírito Santo Convention & Visitors Bureau | RESPIRITO | 35 | 44 | 0.2856 |
| COLISÃO DISTRIBUIDORA EMBALAGENS EM GERAL | DISTRIBUIDORA WSP | 35 | 35 | 0.2854 |
| ACADEMIA DO VAREJO | ACADEMIA DO SOLO | 35 | 41 | 0.2853 |
| EcoSoluções | ARKO SOLUÇÕES | 40 | 35 | 0.2853 |
| SYM INCORPORAÇÕES E PARTICIPAÇÕES | SIOM | 36 | 42 | 0.2842 |
| CERVEJA ARTESANAL 806 BEER | SINCORÁ CERVEJARIA ARTESANAL | 32 | 32 | 0.2838 |
| FARMÁCIA MAR & VIDA | MATE & VIDA | 35 | 30 | 0.2838 |
| ELÉTRICA GONÇALVES | L.C.S ELÉTRICA | 37 | 37 | 0.2833 |
| CONEXÃO FOOD HALL | C One | 43 | 35 | 0.2821 |
| MARU EMPREENDIMENTOS | DAMARA EMPREENDIMENTOS | 37 | 37 | 0.2819 |
| PADRÃO | PADRÃO CARIOCA | 42 | 43 | 0.2816 |
| B. COMPANY | AD1 Company | 33 | 35 | 0.2811 |
| DOIS IRMÃOS DISTRIBUIDORA DE PETRÓLEO | LATICÍNIO DOIS IRMÃOS | 37 | 35 | 0.2809 |
| JIU - JITSU BRASIL | ZION JIU-JITSU | 35 | 41 | 0.2808 |
| VIRTUOSA FASHION | VIRTUOSO | 35 | 35 | 0.2807 |
| ÓTIMO por OTIMISTA | Oti+ | 38 | 9 | 0.2805 |
| NOBRE BLU | NOBRE | 35 | 11 | 0.2796 |
| EMPÓRIO VSN+ VENTURA SPORTS NUTRITION | EMPÓRIO DO RIO | 35 | 35 | 0.2789 |
| GRANUM CAFÉ GOURMET | Granuleco | 43 | 31 | 0.2786 |
| FAZENDA HOMEM DE PEDRA | Fazenda Acos | 35 | 44 | 0.2773 |
| EASY CONTABILIDADE | EASYARC | 35 | 7 | 0.2772 |
| DIGITAL CARS | DIGITAL NEXT | 35 | 41 | 0.2765 |
| INTENSEE CASUAL | INTENSIFÓS | 35 | 1 | 0.2763 |
| BISCOITOS SARETTA | BISCOITOS PAULISTA | 30 | 30 | 0.2760 |
| LUKRE | LUKS | 35 | 40 | 0.2753 |
| PARADA DA SERRA | PARADELLA | 35 | 14 | 0.2752 |
| NACIONAL GAS BUTANO | Nacional Cards | 0 | 35 | 0.2752 |
| CASA DO RESTAURANTE | CASA D'ELLA | 35 | 3 | 0.2752 |
| Concept Hospital Dia | SET CONCEPT | 36 | 35 | 0.2747 |
| F.A.Z. Financeiro de A a Z | FASAW | 35 | 12 | 0.2741 |
| COXINHA EXPRESS | CIA DA COXINHA | 43 | 35 | 0.2722 |
| CONEXÃO FOOD HALL | CONEXLED | 35 | 11 | 0.2721 |
| SANTA BÁRBARA | Santa Clara | 0 | 31 | 0.2718 |
| Fene Distribuidora | Frez Distribuidora | 3 | 35 | 0.2718 |
| VERDURAS NONATO | Gaúcho Verduras | 31 | 35 | 0.2716 |
| P COFFEE | COFFEX | 43 | 35 | 0.2709 |
| INICIATIVA EDUCA! | Educalei | 35 | 41 | 0.2705 |
| RÉNOVÉ | Renovare | 35 | 41 | 0.2704 |
| GRUPO S | GRUPO JCP | 35 | 35 | 0.2704 |
| WM INVESTIMENTOS | 4CF INVESTIMENTOS | 36 | 36 | 0.2702 |
| ESPAÇOLIMPO | ESPAÇO LIPS | 3 | 35 | 0.2700 |
| Guaraci | Guarajú | 35 | 30 | 0.2698 |
| DURAFORT | Du fort | 35 | 24 | 0.2696 |
| Rewa SUPPLY | Renèva | 35 | 44 | 0.2694 |
| SIMMETRIA AMBIENTI | Asymmetry | 35 | 42 | 0.2691 |
| D PRIMES AÇAIZINHO | PRIMEAC | 35 | 35 | 0.2688 |
| COLISÃO DISTRIBUIDORA EMBALAGENS EM GERAL | DISTRIBUIDORA BARBOSA | 35 | 38 | 0.2685 |
| VITA STRONG | VITA A | 30 | 35 | 0.2682 |
| NO AR | NOAD | 41 | 9 | 0.2673 |
| SOLUÇÕES ADMINISTRADORA DE CONDOMÍNIOS | SGP Soluções | 35 | 20 | 0.2652 |
| EMPÓRIO MAIA | MIXMAIA | 25 | 35 | 0.2643 |
| UNIVERSIDADE DAS ARTES | UNIVERSIDADE GAMER | 41 | 9 | 0.2639 |
| FAÍSCA MATERIAIS ELÉTRICOS | Paizuca | 35 | 35 | 0.2639 |
| Distribuidora Classic | MAKE DISTRIBUIDORA | 35 | 35 | 0.2639 |
| ODONTO SCAN FRANQUIAS | SCAP | 44 | 42 | 0.2638 |
| AUTO PEÇAS MATOS | DEKAR AUTOPEÇAS | 35 | 35 | 0.2636 |
| START FIT STORE | Star | 35 | 41 | 0.2632 |
| TOP PANIFICADORA | TOP G | 30 | 35 | 0.2623 |
| P.A Manutenções | PA | 35 | 35 | 0.2608 |

### 9.4 LISTA COMPLETA - Grupo 1 errados (rotulo=1, escaparam da classificacao) - 77 pares

_Ordenados por score crescente (colidentes que receberam o menor score primeiro - mais graves para a operacao). Cada linha eh uma colidencia que o modelo deixou passar._

| Marca A | Marca B | cls A | cls B | score |
| --- | --- | --- | --- | --- |
| PRINCIPIA CONSULTORIA E TREINAMENTO | PA PrincipiAndo | 41 | 41 | 0.0018 |
| Ô CREPE! | CREPE 33 | 43 | 39 | 0.0047 |
| AMPLA | AMPLAPACK | 0 | 16 | 0.0048 |
| REVESTIMENTOS CASA 7 | +CASA | 35 | 37 | 0.0049 |
| OFTALMOAMIGO | OFTALMO TOP | 44 | 41 | 0.0049 |
| UMTELECOM | A.R TELECOM | 9 | 42 | 0.0061 |
| Brasil Farma Prime | FARMER BRAZIL | 35 | 36 | 0.0075 |
| AGROPEREIRA | AGRO FRONTEIRA | 35 | 39 | 0.0077 |
| X BRONZE BRONZEAMENTO NATURAL | Seu Bronze | 44 | 44 | 0.0078 |
| Q.SABOR PIZZARIA DELIVERY | Sabor e Sabores | 39 | 35 | 0.0089 |
| Q.SABOR PIZZARIA DELIVERY | Sabor e Sabores | 39 | 35 | 0.0089 |
| LYS LAÇOS E ACESSÓRIOS | LIZ | 26 | 35 | 0.0095 |
| Espírito Santo Convention & Visitors Bureau | 89.1 Espírito Santo FM | 35 | 41 | 0.0121 |
| DATEC SOLUTION | DATATECK | 7 | 9 | 0.0166 |
| MALU GUARDANAPOS DE PAPEL | Tia Malu | 16 | 35 | 0.0173 |
| BELLA FORMA LASER | FORMA | 44 | 41 | 0.0188 |
| VEM EMPREENDER | EMPRENDER | 9 | 35 | 0.0223 |
| GEOMAPA | GeoMatch | 44 | 42 | 0.0227 |
| GEMINI SOLUTIONS IN REAL ESTATE | GEPEX SOLUTIONS | 35 | 35 | 0.0229 |
| MOBILIÈRE MÓVEIS PERSONALIZADOS | Mobília · AI | 40 | 35 | 0.0232 |
| Six Pilates Studio | PILATES HOME | 41 | 35 | 0.0236 |
| WESTPOINT | NextPoint | 9 | 9 | 0.0251 |
| ARENA BEACH | BEACH | 41 | 35 | 0.0252 |
| Renova | Renova Be | 44 | 35 | 0.0266 |
| INICIATIVA EDUCA! | EDUC-C | 35 | 41 | 0.0274 |
| Fast MED | MEDPASS | 35 | 36 | 0.0288 |
| CASA DA COXINHA LINHARES | Coxinha e Cia | 43 | 35 | 0.0326 |
| Bella NOBRE DOCES | BELLA ROMA | 35 | 32 | 0.0369 |
| IN CONNECTION | FF.CONNECTION | 41 | 35 | 0.0439 |
| MAIS ODONTO | ODONTOMEDTEC | 36 | 37 | 0.0464 |
| NACIONAL | LOCAÇÃO NACIONAL | 0 | 37 | 0.0480 |
| FLOFF | PLOPY | 35 | 18 | 0.0556 |
| FINOTRATO | FINO TRATO ESTAMPARIA | 42 | 40 | 0.0556 |
| SOLAR BANHO E PISCINA | Solardyne | 35 | 42 | 0.0587 |
| BOG BRASIL OUTLET GRIFFES | RCG BRASIL | 40 | 35 | 0.0602 |
| F.A.Z. Financeiro de A a Z | FAZ O K | 35 | 35 | 0.0617 |
| 2 BROTHERS | BROTHERS BEACH | 40 | 43 | 0.0618 |
| INSIDER BY POMPEU VACONCELOS | INSIDE | 41 | 35 | 0.0618 |
| NEO AGROAMBIENTAL | NEOX | 42 | 9 | 0.0644 |
| BR MOTOS BENEFÍCIOS | B & G motos | 36 | 35 | 0.0646 |
| EU ENGLISH UNIVERSE | English (sem vish!) | 41 | 35 | 0.0683 |
| Ponto Metal | PONTO CERTO | 35 | 35 | 0.0697 |
| META ENGENHARIA SERVIÇOS | METALUX | 35 | 6 | 0.0757 |
| NOBRE SABOR GOURMET NO PONTO CERTO | Nobre Safra | 39 | 35 | 0.0759 |
| MANGÔO SNACKS & CO. | Mango Mix | 43 | 35 | 0.0835 |
| SER SAÚDE | Saúde sem Idade | 35 | 41 | 0.0864 |
| UMTELECOM | Une Telecom | 9 | 38 | 0.0890 |
| SOLAR BANHO E PISCINA | SOLARNORT | 37 | 35 | 0.0934 |
| Q.SABOR PIZZARIA DELIVERY | Sabor e Sabores | 39 | 43 | 0.1001 |
| Comunidade GG | Comunidade Leal | 44 | 44 | 0.1076 |
| SABOROSO PRIME | Saboreô | 30 | 35 | 0.1123 |
| LUKRE | LUVY | 35 | 5 | 0.1144 |
| Agência VET | AGÊNCIA PITT | 42 | 35 | 0.1205 |
| META ENGENHARIA SERVIÇOS | METALUX | 35 | 6 | 0.1225 |
| ARTEMIS | ArteRaiz | 41 | 35 | 0.1338 |
| VITORIA LANCHES | VITÓRIA CERVEJARIA | 30 | 35 | 0.1369 |
| SOLAR BANHO E PISCINA | Solardyne | 37 | 9 | 0.1424 |
| ECO VIDA | VINAECO | 35 | 30 | 0.1446 |
| CASA DO RESTAURANTE | CAZADORA | 35 | 35 | 0.1523 |
| NACIONAL GÁS | NACIONAL FIRE | 39 | 45 | 0.1569 |
| ROSAMANGO | ROSA MARIA | 35 | 40 | 0.1696 |
| SOLAR BANHO E PISCINA | SOLARCYCLE | 37 | 42 | 0.1708 |
| SOLAR BANHO E PISCINA | SolarWatts | 37 | 35 | 0.1733 |
| DIÁRIODONORDESTE.COM.BR | DIÁRIO CELESTE | 16 | 35 | 0.1770 |
| CRISTAL ORGÂNICOS | CRIS´AL | 40 | 29 | 0.1812 |
| HAPVIDA CLÍNICA COMANDANTE SAMPAIO NOTREDAME | HAPVIDA CLÍNICA DAS MARIAS | 44 | 44 | 0.1853 |
| VEM EMPREENDER | EMPREENDE QUE VENDE | 35 | 41 | 0.1954 |
| MASSA FRITA | MAZZA FOODS | 43 | 29 | 0.1999 |
| AFBN Incorporadora | AF CONSTRUTORA INCORPORADORA | 37 | 36 | 0.2127 |
| NORDESTE EMPREENDEDOR | PODCAST  EMPREENDEDOR | 38 | 41 | 0.2260 |
| GRANORTE TUBOS | GRANOS | 35 | 19 | 0.2280 |
| COMBINADO FOOD?S | COMBINÔ! | 30 | 35 | 0.2286 |
| BEACH SUMMER PORTO DE GALINHAS | beachx | 35 | 25 | 0.2336 |
| BOI E BRASA | BRASA BONDE | 41 | 35 | 0.2338 |
| WALL ENGENHARIA | WALLZ | 35 | 35 | 0.2338 |
| GRAP Gestante e reabilitação do assoalho pélvico | GRAFTYS | 44 | 10 | 0.2363 |
| VITORIA LANCHES | Vitoria do Cerrado | 30 | 35 | 0.2383 |

## 10. Comparativo NN vs heuristica OFTA (no conjunto de teste)

_Recall avaliado com piso de Precision >= 0.90._

| Metrica | NN | Heuristica OFTA | Delta |
| --- | ---:| ---:| ---:|
| ROC-AUC | 0.8447 | 0.5263 | +0.3184 |
| PR-AUC | 0.4795 | 0.1161 | +0.3633 |
| Recall@P>=0.9 | 0.0553 | 0.0000 | +0.0553 |

## 11. Importancia das features (Permutation Importance)

> Quanto maior a importancia, mais a metrica de validacao PIORA quando essa feature e embaralhada. Use para auditar onde o modelo esta apoiado.

### 11.1 Top 30 globais

| Feature | Importance |
| --- | --- |
| spec_cosine_tfidf_char | 0.1205 |
| spec_lex_idf_avg_common | 0.0596 |
| graf_len_ratio | 0.0477 |
| spec_cosine_emb | 0.0436 |
| spec_lex_size_diff_abs | 0.0326 |
| fon_token_mean | 0.0307 |
| ofta_token | 0.0276 |
| cls_a_top_9 | 0.0257 |
| fon_key_lev_sim | 0.0256 |
| spec_lex_overlap | 0.0201 |
| cls_a_top_25 | 0.0196 |
| spec_lex_n_common | 0.0195 |
| graf_contains | 0.0186 |
| tok_jaccard | 0.0180 |
| spec_kind_a_produto | 0.0173 |
| spec_lex_size_ratio | 0.0173 |
| ofta_driver_fonética | 0.0172 |
| inter_proxy_nome | 0.0169 |
| fon_token_max | 0.0168 |
| spec_kind_b_produto | 0.0168 |
| cls_a_top_43 | 0.0168 |
| n_tokens_b | 0.0167 |
| spec_kind_a_servico | 0.0165 |
| tok_overlap | 0.0165 |
| cls_b_top_30 | 0.0164 |
| cls_b_top_41 | 0.0158 |
| ofta_fuzzy | 0.0154 |
| cls_b_top_other | 0.0153 |
| fon_after_dedup_eq | 0.0150 |
| spec_any_misto | 0.0147 |

### 11.2 Importancia agregada por bloco

| Bloco | Soma | Share |
| --- | --- | --- |
| Classe Nice | 0.2558 | 0.2310 |
| Spec_cosine | 0.1641 | 0.1481 |
| Spec_lex | 0.1560 | 0.1408 |
| Graficas | 0.1134 | 0.1024 |
| Foneticas | 0.1002 | 0.0905 |
| OFTA | 0.0872 | 0.0787 |
| Tokens | 0.0789 | 0.0712 |
| Spec_atividade | 0.0720 | 0.0650 |
| Interacoes | 0.0717 | 0.0647 |
| Numerais | 0.0083 | 0.0075 |

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
