# Relatorio Completo - Modelo de Similaridade Aprendida de Marcas

_Gerado em 2026-04-30T15:06:32+00:00 (UTC)._

## Sumario executivo

**Veredito de prontidao:** `NAO RECOMENDADO para producao`

| Metrica | Treino | Validacao | Teste |
| --- | ---:| ---:| ---:|
| ROC-AUC | 0.8500 | 0.7389 | **0.7279** |
| PR-AUC  | 0.6251 | 0.4849 | **0.4882** |
| F1      | 0.4848 | 0.4286 | **0.3966** |
| Recall  | 0.9278 | 0.8500 | **0.7667** |
| Precision | 0.3281 | 0.2865 | **0.2674** |

**Threshold de operacao:** `0.3700` (politica: `max_f1_with_recall>=0.85`, recall_floor=0.85)

**Pontos fortes:**
- PR-AUC 0.488 (bom para base desbalanceada)

**Atencao:**
- Recall 0.767 abaixo do piso 0.85 (perde colidentes)
- F1 0.397 (equilibrio modesto)

**Bloqueadores:**
- ROC-AUC 0.728 abaixo de 0.80 (separabilidade fraca)

**Sinais de overfit (treino - teste):**

| Metrica | delta train-test |
| --- | ---:|
| ROC-AUC train-test | +0.1221 |
| PR-AUC train-test | +0.1369 |
| Recall train-test | +0.1611 |
| F1 train-test | +0.0882 |

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
- Top-10 classes Nice usadas em one-hot: [35, 41, 42, 43, 44, 36, 25, 9, 30, 37]
- Total de features na ordem canonica: **106**

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

### Classe Nice (cls_*) (26)

`cls_same`, `cls_diff_abs`, `cls_a_known`, `cls_b_known`, `cls_a_top_35`, `cls_a_top_41`, `cls_a_top_42`, `cls_a_top_43`, `cls_a_top_44`, `cls_a_top_36`, `cls_a_top_25`, `cls_a_top_9`, `cls_a_top_30`, `cls_a_top_37`, `cls_a_top_other`, `cls_b_top_35`, `cls_b_top_41`, `cls_b_top_42`, `cls_b_top_43`, `cls_b_top_44`, `cls_b_top_36`, `cls_b_top_25`, `cls_b_top_9`, `cls_b_top_30`, `cls_b_top_37`, `cls_b_top_other`

### Interacoes (inter_*) (12)

`inter_nome_x_spec_word`, `inter_nome_x_spec_emb`, `inter_nome_x_spec_max`, `inter_nome_x_same_cls`, `inter_spec_x_same_cls`, `inter_nome_alto_e_spec_alta`, `inter_nome_alto_e_spec_baixa`, `inter_nome_baixo_e_spec_alta`, `inter_classe_diff_mas_emb_alto`, `inter_classe_igual_e_spec_proxima`, `inter_proxy_nome`, `inter_proxy_spec`

## 3. Estatisticas das features (apos StandardScaler)

- Linhas usadas para estatistica: 1500
- Features com variancia ZERO: **6** (possiveis candidatas a remocao)

Features sem variancia detectadas: `ofta_driver_geral`, `ofta_driver_palavras`, `cls_b_known`, `inter_nome_x_spec_emb`, `inter_classe_diff_mas_emb_alto`, `spec_cosine_emb`

### 3.1 Top features por desvio padrao (apos scaler - util p/ debug)

| Feature | media | desvio | min | max |
| --- | --- | --- | --- | --- |
| cls_a_known | 0.0000 | 1.0000 | -11.6346 | 0.0860 |
| num_fon_worst | 0.0000 | 1.0000 | -0.2138 | 8.4192 |
| num_has_digits | -0.0000 | 1.0000 | -0.2374 | 4.2131 |
| cls_b_top_other | 0.0000 | 1.0000 | -0.5527 | 1.8093 |
| n_tokens_common | -0.0000 | 1.0000 | -0.7586 | 5.9346 |
| cls_b_top_30 | 0.0000 | 1.0000 | -0.1818 | 5.5000 |
| cls_b_top_41 | -0.0000 | 1.0000 | -0.3762 | 2.6579 |
| inter_nome_alto_e_spec_alta | 0.0000 | 1.0000 | -0.1379 | 7.2506 |
| n_tokens_excl_a | 0.0000 | 1.0000 | -1.5943 | 5.4498 |
| num_fon_spread | -0.0000 | 1.0000 | -0.1725 | 15.4387 |
| graf_suffix_norm | 0.0000 | 1.0000 | -0.3563 | 4.7171 |
| graf_contains | -0.0000 | 1.0000 | -0.2895 | 3.4541 |
| fon_key_eq | -0.0000 | 1.0000 | -0.1453 | 6.8838 |
| num_fon_best | 0.0000 | 1.0000 | -0.2167 | 8.0286 |
| spec_same_activity_kind | -0.0000 | 1.0000 | -0.4461 | 2.2414 |
| cls_a_top_30 | -0.0000 | 1.0000 | -0.1634 | 6.1206 |
| tok_overlap | 0.0000 | 1.0000 | -0.7747 | 2.7257 |
| spec_kind_b_produto | 0.0000 | 1.0000 | -0.4643 | 2.1539 |
| spec_cosine_tfidf_word | 0.0000 | 1.0000 | -0.4550 | 5.8513 |
| cls_same | -0.0000 | 1.0000 | -0.5589 | 1.7893 |

### 3.2 Top features por |correlacao de Pearson com a label|

| Feature | Pearson |
| --- | --- |
| spec_lex_idf_avg_common | 0.3051 |
| spec_cosine_tfidf_char | 0.3013 |
| spec_lex_overlap | 0.3006 |
| inter_proxy_spec | 0.2969 |
| spec_cosine_tfidf_word | 0.2399 |
| spec_lex_n_common | 0.2183 |
| inter_classe_igual_e_spec_proxima | 0.2170 |
| inter_spec_x_same_cls | 0.2170 |
| spec_lex_jaccard | 0.2168 |
| inter_nome_x_spec_max | 0.2151 |
| tok_fuzzy | 0.1920 |
| ofta_fuzzy | 0.1920 |
| ofta_token | 0.1920 |
| graf_jaro | 0.1768 |
| graf_overlap_trigram | 0.1767 |
| inter_nome_x_spec_word | 0.1738 |
| graf_jaro_winkler | 0.1644 |
| cls_same | 0.1628 |
| fon_key_lev_sim | 0.1534 |
| graf_jaccard_bigram | 0.1480 |

## 4. Estrategia de balanceamento

- Undersample negativos: ratio **2.50 : 1** (neg/pos alvo)
- Oversample positivos: fator **2.00x**
- Class weight (pos_weight) ativo: **True**, valor efetivo **1.250**
- Seed: 42
- Split estratificado: train=60% / val=20% / test=20%

## 5. Arquitetura e hiperparametros

### 5.1 Arquitetura

- input_dim: **106**
- hidden_dims: **[64, 32]**
- ativacao: **relu**
- dropout: **0.2**
- batchnorm: **True**
- output: Linear(., 1) -> Sigmoid
- **# parametros aprendiveis: 9,153**

### 5.2 Hiperparametros de treino

- epochs: **4**
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
- best_epoch: **4**
- best_pr_auc_val: **0.48494418055812394**
- n_train_after_balancing: **810**

- Otimizador: **AdamW**, Loss: **BCEWithLogitsLoss(pos_weight)**, Scheduler: **ReduceLROnPlateau** (max em val PR-AUC)

## 6. Historico do treino (epoca a epoca)

| epoch | train_loss | lr | train_pr_auc | val_pr_auc | val_roc_auc | val_recall@0.5 | val_f1@0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.7496 | 0.0010 | 0.7413 | 0.3784 | 0.6744 | 0.5500 | 0.3708 |
| 2 | 0.6532 | 0.0010 | 0.7893 | 0.4107 | 0.7090 | 0.6333 | 0.4176 |
| 3 | 0.6235 | 0.0010 | 0.8220 | 0.4452 | 0.7268 | 0.6333 | 0.4551 |
| 4 | 0.5826 | 0.0010 | 0.8524 | 0.4849 | 0.7389 | 0.6167 | 0.4568 |


## 7. Metricas em detalhe

Threshold otimo: **0.3700** (politica: `max_f1_with_recall>=0.85`, recall_floor=0.85)

### Treino
- ROC-AUC: **0.8500**
- PR-AUC:  **0.6251**
- F1:        0.4848 (threshold=0.370)
- Precision: 0.3281
- Recall:    0.9278
- Confusao: TN=378, FP=342, FN=13, TP=167
- n_pos=180, n_neg=720

### Validacao
- ROC-AUC: **0.7389**
- PR-AUC:  **0.4849**
- F1:        0.4286 (threshold=0.370)
- Precision: 0.2865
- Recall:    0.8500
- Confusao: TN=113, FP=127, FN=9, TP=51
- n_pos=60, n_neg=240

### Teste
- ROC-AUC: **0.7279**
- PR-AUC:  **0.4882**
- F1:        0.3966 (threshold=0.370)
- Precision: 0.2674
- Recall:    0.7667
- Confusao: TN=114, FP=126, FN=14, TP=46
- n_pos=60, n_neg=240

## 8. Distribuicao de scores no teste

| Classe | n | mean | std | p10 | p25 | p50 | p75 | p90 |
| --- | ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:|
| pos | 60 | 0.5530 | 0.1958 | 0.2935 | 0.3909 | 0.5527 | 0.7235 | 0.7958 |
| neg | 240 | 0.3904 | 0.1582 | 0.1968 | 0.2773 | 0.3820 | 0.4775 | 0.6127 |

_Diferenca de media (pos - neg): **+0.1625**. Quanto maior, melhor a separabilidade._

### 8.1 Decis de score (10 = scores mais altos)

| Decil | score_min | score_max | n | positivos | % positivos |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.6930 | 0.9041 | 30 | 19 | 0.6333 |
| 1 | 0.5787 | 0.6864 | 30 | 9 | 0.3000 |
| 2 | 0.4914 | 0.5771 | 30 | 7 | 0.2333 |
| 3 | 0.4455 | 0.4907 | 30 | 4 | 0.1333 |
| 4 | 0.4025 | 0.4431 | 30 | 5 | 0.1667 |
| 5 | 0.3554 | 0.4012 | 30 | 3 | 0.1000 |
| 6 | 0.3190 | 0.3550 | 30 | 4 | 0.1333 |
| 7 | 0.2723 | 0.3175 | 30 | 4 | 0.1333 |
| 8 | 0.2124 | 0.2717 | 30 | 5 | 0.1667 |
| 9 | 0.0150 | 0.2109 | 30 | 0 | 0.0000 |

## 9. Analise de erros (no teste)

- Total de pares no teste: **300** (threshold=0.3700)
- **Grupo 0** (rotulo real = 0, NAO colidente): 240 pares no total. **126 foram classificados como colidentes (taxa de erro 52.50%)**.
- **Grupo 1** (rotulo real = 1, colidentes): 60 pares no total. **14 escaparam da classificacao (taxa de erro 23.33%)**.
- Lista completa Grupo 0 errados (CSV): `artifacts\falsos_positivos_grupo_0_smoke.csv`
- Lista completa Grupo 1 errados (CSV): `artifacts\falsos_negativos_grupo_1_smoke.csv`

### 9.1 Top 10 Falsos Positivos (Grupo 0 - alarmes mais graves)

| Marca A | Marca B | cls A | cls B | score |
| --- | --- | --- | --- | --- |
| Master Multi | Mastery7 | 35 | 35 | 0.9041 |
| "KARNE KEIJO" | KARUI | -1 | 29 | 0.8232 |
| FRUIT AÇAÍ | Fruit Mania | 35 | 35 | 0.7584 |
| AUTO PEÇAS MATOS | DLC AUTO PEÇAS | 35 | 35 | 0.7516 |
| Scarlatto | SCARLAT | 35 | 40 | 0.7406 |
| MARZINHO | MARZINI | 35 | 35 | 0.7201 |
| Y.Consultoria | ZIP CONSULTORIA | 35 | 35 | 0.7125 |
| Auto Posto G&D | AUTO POSTO KS | 35 | 37 | 0.7102 |
| I/O TECNOLOGIA INDUSTRIAL | LIFTUS TECNOLOGIA | 35 | 9 | 0.7012 |
| P&P CONTABILIDADE E CONSULTORIA EMPRESARIAL | CAMPOS BRITO CONTABILIDADE | 35 | 35 | 0.6966 |

### 9.2 Top 10 Falsos Negativos (Grupo 1 - escapes mais graves)

| Marca A | Marca B | cls A | cls B | score |
| --- | --- | --- | --- | --- |
| SABOR CASEIRO SELF SERVICE | CASEIROKA O VERDADEIRO SABOR CASEIRO | 43 | 30 | 0.2126 |
| SHOPPING PIZZA BURGER | CAVEZZO BURGER & PIZZA | 43 | 35 | 0.2265 |
| RANCHO DO ACARAJÉ | RANCHO 185 | 43 | 30 | 0.2352 |
| BONITO'S BAR | Bonantos | 43 | 30 | 0.2658 |
| PSICO STORE | Psicoplace | 35 | 44 | 0.2685 |
| NEO AGROAMBIENTAL | NEOON | 42 | 44 | 0.2816 |
| É HIT | HITZ | 38 | 41 | 0.2948 |
| DIAS SOLAR ENERGIA RENOVÁVEL | Solardyne | 42 | 37 | 0.2964 |
| AGROPEREIRA | AGRO FRONTEIRA | 35 | 37 | 0.3123 |
| BONNA PIZZA | Bon Nut | 43 | 30 | 0.3190 |

### 9.3 LISTA COMPLETA - Grupo 0 errados (rotulo=0, classificados como colidentes) - 126 pares

_Ordenados por score decrescente (alarmes de maior "confianca errada" primeiro). Cada linha eh um par que o modelo achou que era colidencia mas NAO era segundo o rotulo INPI._

| Marca A | Marca B | cls A | cls B | score |
| --- | --- | --- | --- | --- |
| Master Multi | Mastery7 | 35 | 35 | 0.9041 |
| "KARNE KEIJO" | KARUI | -1 | 29 | 0.8232 |
| FRUIT AÇAÍ | Fruit Mania | 35 | 35 | 0.7584 |
| AUTO PEÇAS MATOS | DLC AUTO PEÇAS | 35 | 35 | 0.7516 |
| Scarlatto | SCARLAT | 35 | 40 | 0.7406 |
| MARZINHO | MARZINI | 35 | 35 | 0.7201 |
| Y.Consultoria | ZIP CONSULTORIA | 35 | 35 | 0.7125 |
| Auto Posto G&D | AUTO POSTO KS | 35 | 37 | 0.7102 |
| I/O TECNOLOGIA INDUSTRIAL | LIFTUS TECNOLOGIA | 35 | 9 | 0.7012 |
| P&P CONTABILIDADE E CONSULTORIA EMPRESARIAL | CAMPOS BRITO CONTABILIDADE | 35 | 35 | 0.6966 |
| ELÉVI+ | ELEVEN | 37 | 35 | 0.6930 |
| D&F MODA FITNESS | RMF MODA FITNESS | 35 | 35 | 0.6864 |
| Jucelino Móveis Planejados | LOOV Móveis Planejados | 35 | 35 | 0.6809 |
| ELEVA POR RENATA PAULA SANTIAGO | Elevy | 35 | 30 | 0.6772 |
| DISQUE & TOQUE | TOQUE | 41 | 41 | 0.6756 |
| CLÍNICA ANDRÉIA NOGUEIRA | Maria Nogueira | 44 | 44 | 0.6718 |
| PP TREINAMENTOS | Entretreinamento | 41 | 41 | 0.6683 |
| Distribuidora Classic | ABC DISTRIBUIDORA | 35 | 35 | 0.6507 |
| VALE ZELAR | VALE JJR | 37 | 37 | 0.6319 |
| SOLAR BANHO E PISCINA | Solaris | 37 | 35 | 0.6306 |
| E A ESTUDIO ARTE | ESTÚDIO NT | 38 | 38 | 0.6305 |
| METÁLIX ESTRUTURAS METÁLICAS | Meta | 37 | 42 | 0.6286 |
| SORVETES DO VALE | RITZ SORVETES | 35 | 30 | 0.6211 |
| FUCAPE FUNDAÇÃO CAPIXABA DE PESQUISAS | FUCAPI | 35 | 41 | 0.6208 |
| Imaginário Brasileiro | IMAGINY | 35 | 41 | 0.6118 |
| SIRIOS ENERGIA SOLAR | ZIRIX | 37 | 9 | 0.6081 |
| MÉTODO 3T | MÉTODO ADR | 41 | 41 | 0.6060 |
| SPACE CONDO | Space Cow | 41 | 28 | 0.5993 |
| CHECKUP | CHECKUP DE MARCAS | 41 | 38 | 0.5926 |
| DEEP COLLECTION | M.V COLLECTION | 25 | 35 | 0.5878 |
| ACAÍ DO TIO REI | AÇAÍ BM | 30 | 29 | 0.5792 |
| VELOZ | VELOX | 35 | 35 | 0.5787 |
| DEU POSITIVO? A PEQUENINA MODA INFANTIL | POSITIV.A | 35 | 3 | 0.5771 |
| COLISÃO DISTRIBUIDORA EMBALAGENS EM GERAL | ARILU DISTRIBUIDORA | 35 | 39 | 0.5655 |
| IVA INSTITUTO DE ESTUDOS E PESQUISAS DO VALE DO ACARAÚ | INSTITUTO HONORATO | 41 | 41 | 0.5611 |
| F.A.Z. Financeiro de A a Z | FASAW | 35 | 9 | 0.5565 |
| BR BRAVO ROMEU | BR BRAVOTEC | 35 | 1 | 0.5546 |
| IN CONNECTION | COMPANY CONNECTION | 41 | 16 | 0.5503 |
| CASA CONCRETA | CONCREJOTA | 35 | 40 | 0.5499 |
| açaí Vitaly Pura energia | Vitae+ | 35 | 9 | 0.5404 |
| Dom Brasil | DTIBRASIL | 35 | 35 | 0.5396 |
| RESTAURANTE MENINA GERAIS | L&J RESTAURANTE | 43 | 43 | 0.5359 |
| SAFE CARE RESIDENCIAL E HOTELARIA PARA IDOSOS | Safe care. | 44 | 42 | 0.5345 |
| CASTELO MAGAZINE | Castel Block | 35 | 28 | 0.5338 |
| LIFE UP EXPERIENCE | Banah Experience | 41 | 41 | 0.5287 |
| Master Multi | MASTER | 1 | 40 | 0.5225 |
| PRAIOW | PRAIA | 43 | 3 | 0.5199 |
| WM INVESTIMENTOS | ÁRTICO INVESTIMENTOS | 36 | 36 | 0.5182 |
| ATENDE BIKE | Atenda.bot | 35 | 9 | 0.5147 |
| S HORSE | KIDS HORSE | 35 | 35 | 0.5075 |
| THE CAROLINE CASUAL WEAR | CAROL | 35 | 31 | 0.5070 |
| VAREJÃO DAS MOTOS | Varejão da Horta | 35 | 31 | 0.5002 |
| EMPÓRIO VSN+ VENTURA SPORTS NUTRITION | Empório Vilarin | 35 | 25 | 0.4953 |
| M PRIME | primeline | 9 | 9 | 0.4914 |
| REI DAS CLÍNICAS | CLICK CLÍNICAS | 44 | 45 | 0.4914 |
| PELLE SANA CLÍNICA ESPECIALIZADA EM TRATAMENTO DE FERIDAS | SAN' MIELLE | 44 | 35 | 0.4907 |
| SER SAÚDE | SAÚDE+PE | 35 | 45 | 0.4879 |
| Polaris Energia Solar | POLAR | 42 | 7 | 0.4842 |
| CENTRAL MECÂNICA | CENTRAL DO QUEIJO | 37 | 35 | 0.4838 |
| FORRÓ DO MR. BOB | MR.BOX | 41 | 40 | 0.4782 |
| SECRET & CO | Secret Labs | 35 | 5 | 0.4773 |
| ESTÂNCIA DOS CAMPOS GALILÉIA - MG | Estância Uniformes | 35 | 25 | 0.4759 |
| LINAUS CLINIC | LINA | 44 | 35 | 0.4736 |
| Marketing Erótico | 3.3 Marketing | 41 | 35 | 0.4735 |
| LORENAS ODONTOLOGIA | LORPAX | 44 | 35 | 0.4727 |
| CASTELLA | Castelos | 6 | 35 | 0.4706 |
| RÉNOVÉ | RENOVA D3 | 3 | 5 | 0.4668 |
| ACADEMIA TOTAL FITNESS | Academia +FFRI | 41 | 41 | 0.4667 |
| AGUARDENTE DE CANA PRATINHA | PRAINHA | 35 | 41 | 0.4656 |
| SUPERMERCASA | SUPER CAST | 35 | 28 | 0.4637 |
| Instituto Apollo | INSTITUTO AFETO | 41 | 44 | 0.4594 |
| DISTRIBUIDORA POPEYE JR. | GENESYS DISTRIBUIDORA | 39 | 35 | 0.4576 |
| CHAMA EMPREENDEDORA | EMPREENDEDORAS EM POTENCIAL | 41 | 41 | 0.4570 |
| CLICK BABY | ClickOne | 41 | 42 | 0.4564 |
| RESTAURANTE GUARANI SELF SERVICE UBATUBA - SP | Restaurante MARANT | 40 | 43 | 0.4561 |
| CENTRO EDUCACIONAL MÚLTIPLA ESCOLHA | P4 EDUCACIONAL | 41 | 41 | 0.4547 |
| EMPÓRIO CAPAS VCA | Empório ZL | 35 | 9 | 0.4529 |
| DONA FLORINDA | FLORIVA | 35 | 35 | 0.4513 |
| P4 INVEST | P4 | 42 | 17 | 0.4471 |
| CASA ELIANA | CASAELLA | 44 | 35 | 0.4459 |
| AÇAÍ CURITIBA | AÇAÍ BM | 30 | 29 | 0.4455 |
| VIA BRASIL | VIVA BRASIL SP | 35 | 9 | 0.4431 |
| VERDICE | VerdiOne | 16 | 35 | 0.4391 |
| INTENSEE CASUAL | INTENSE SECRET | 35 | 9 | 0.4384 |
| REALLOC | RentalLoc | 35 | 12 | 0.4375 |
| LUMANA | LUMUU | 35 | 35 | 0.4368 |
| SOLUÇÕES FUNDAÇÃO | DAVAR SOLUÇÕES | 37 | 36 | 0.4353 |
| VirtuaMed O hub da saúde coworking médico | Virtude Mineira | 44 | 30 | 0.4343 |
| P&P CONTABILIDADE E CONSULTORIA EMPRESARIAL | FC Fonseca Contabilidade | 35 | 45 | 0.4313 |
| SERTANEJA MÁQUINAS | MESA SERTANEJA | 35 | 38 | 0.4311 |
| JC ELETRICIDADE ENERGIA SOLAR SOLAR FOTOVOLTAÍCA | MAXT ELETRICIDADE | 35 | 35 | 0.4302 |
| Virtua Office Coworking Ideas | VIRTUOSO | 35 | 43 | 0.4273 |
| MERCA CONTABILIDADE & CONSULTORIA | MERCON | 35 | 35 | 0.4269 |
| Nutris no Online | NUTRI G | 35 | 5 | 0.4268 |
| VIA ARTE CONSTRUTORA DE OBRAS | VIA FORTE | 37 | 35 | 0.4263 |
| GABRIELE | GABRIEL FROEDE | 0 | 41 | 0.4228 |
| UNIVERSO TEC | Universo Circular | 41 | 35 | 0.4218 |
| META ENGENHARIA SERVIÇOS | METV | 35 | 41 | 0.4194 |
| Clínica NV | CliniCalm | 44 | 44 | 0.4188 |
| SERTANEJA MÁQUINAS | VENDA SERTANEJA | 35 | 35 | 0.4154 |
| GRUPO S | Grupo Igna | 35 | 40 | 0.4152 |
| ESTÂNCIA DOS CAMPOS GALILÉIA - MG | ESTÂNCIA NOBRE | 35 | 43 | 0.4127 |
| DIEGO LANCHES | DIEGO BOY | 35 | 45 | 0.4123 |
| "CHEGA JÁ" | CHEFÃO | 39 | 25 | 0.4043 |
| INTELIGÊNCIA ADS | Maná Inteligência | 35 | 42 | 0.4035 |
| . A EXCLUSIVA | EXCLUSIVE STUDIO | 14 | 40 | 0.4027 |
| Virtua Vet Coworking Veterinário | Virtude Mineira | 44 | 29 | 0.4012 |
| BRASILGÁS | BRASIL BET | 0 | 9 | 0.3972 |
| MOB BRASIL MOBILIDADE URBANA | BRASIL BET | 9 | 38 | 0.3968 |
| DOUTOR GNV | Doutor Ciborg | 41 | 42 | 0.3950 |
| Vila Gourmet | VilaBox | 41 | 43 | 0.3937 |
| EMPÓRIO COCADAS OLIMPIO | EMPÓRIO DO RIO | 35 | 35 | 0.3927 |
| PERSONALE PLANTAS | PERSONE | 42 | 5 | 0.3897 |
| PARAÍSO DAS PALMEIRAS VIVEIRO ESPECIALIZADO | P PALMEIRAS | 35 | 38 | 0.3896 |
| TOURO | TOGURO | 40 | 3 | 0.3893 |
| ESPÍRITO SAMBA | ESPIGUITO | 41 | 39 | 0.3866 |
| MALU | MALLX | 16 | 9 | 0.3863 |
| FAMÍLIA MORELLI | MORE | 35 | 35 | 0.3840 |
| STUDIO R BY RENATA SANTIAGO | STUDIO ANEXO | 44 | 42 | 0.3838 |
| VALE ENTREGAS | Valeris | 39 | 9 | 0.3832 |
| açaí Lembrança | Nuvem Branca | 35 | 35 | 0.3808 |
| X-SERVICE | LAG SERVICE | 35 | 37 | 0.3806 |
| PERFAL | PERPODECK | 6 | 19 | 0.3798 |
| TRU LOGÍSTICA | TROAH | 39 | 25 | 0.3749 |
| Virtua Vet Coworking Veterinário | VIRTUA CONDOMÍNIOS | 35 | 36 | 0.3743 |
| ONE LINE PARTS | FINELINE | 35 | 9 | 0.3712 |

### 9.4 LISTA COMPLETA - Grupo 1 errados (rotulo=1, escaparam da classificacao) - 14 pares

_Ordenados por score crescente (colidentes que receberam o menor score primeiro - mais graves para a operacao). Cada linha eh uma colidencia que o modelo deixou passar._

| Marca A | Marca B | cls A | cls B | score |
| --- | --- | --- | --- | --- |
| SABOR CASEIRO SELF SERVICE | CASEIROKA O VERDADEIRO SABOR CASEIRO | 43 | 30 | 0.2126 |
| SHOPPING PIZZA BURGER | CAVEZZO BURGER & PIZZA | 43 | 35 | 0.2265 |
| RANCHO DO ACARAJÉ | RANCHO 185 | 43 | 30 | 0.2352 |
| BONITO'S BAR | Bonantos | 43 | 30 | 0.2658 |
| PSICO STORE | Psicoplace | 35 | 44 | 0.2685 |
| NEO AGROAMBIENTAL | NEOON | 42 | 44 | 0.2816 |
| É HIT | HITZ | 38 | 41 | 0.2948 |
| DIAS SOLAR ENERGIA RENOVÁVEL | Solardyne | 42 | 37 | 0.2964 |
| AGROPEREIRA | AGRO FRONTEIRA | 35 | 37 | 0.3123 |
| BONNA PIZZA | Bon Nut | 43 | 30 | 0.3190 |
| LEMON DECOR | LEMONY | 35 | 44 | 0.3335 |
| CASCAJU.COM.BR | CASCADO | 29 | 43 | 0.3353 |
| MARCAS COMUNICAÇÃO VISUAL | Marca Aí | 35 | 45 | 0.3495 |
| NEO SOLUÇÕES AMBIENTAIS & PROJETOS AGROPÉCUARIOS | Neoin | 42 | 37 | 0.3554 |

## 10. Comparativo NN vs heuristica OFTA (no conjunto de teste)

_Recall avaliado com piso de Precision >= 0.90._

| Metrica | NN | Heuristica OFTA | Delta |
| --- | ---:| ---:| ---:|
| ROC-AUC | 0.7279 | 0.4697 | +0.2582 |
| PR-AUC | 0.4882 | 0.1996 | +0.2886 |
| Recall@P>=0.9 | 0.0000 | 0.0000 | +0.0000 |

## 11. Importancia das features (Permutation Importance)

> Quanto maior a importancia, mais a metrica de validacao PIORA quando essa feature e embaralhada. Use para auditar onde o modelo esta apoiado.

### 11.1 Top 30 globais

| Feature | Importance |
| --- | --- |
| spec_lex_idf_avg_common | 0.0299 |
| spec_cosine_tfidf_char | 0.0264 |
| cls_b_top_44 | 0.0254 |
| spec_lex_overlap | 0.0198 |
| spec_any_misto | 0.0198 |
| spec_lex_n_common | 0.0184 |
| num_orto_spread | 0.0178 |
| ofta_token | 0.0177 |
| num_fon_best | 0.0158 |
| spec_kind_b_servico | 0.0151 |
| graf_overlap_trigram | 0.0124 |
| ofta_anagram | 0.0122 |
| tok_overlap | 0.0116 |
| inter_nome_x_spec_word | 0.0111 |
| num_fon_worst | 0.0105 |
| cls_a_top_25 | 0.0103 |
| n_tokens_excl_b | 0.0096 |
| ofta_final | 0.0093 |
| inter_nome_x_same_cls | 0.0090 |
| num_orto_worst | 0.0088 |
| n_tokens_common | 0.0083 |
| cls_diff_abs | 0.0081 |
| n_tokens_diff | 0.0075 |
| spec_kind_b_produto | 0.0071 |
| spec_kind_a_produto | 0.0069 |
| inter_nome_alto_e_spec_alta | 0.0066 |
| spec_lex_n_excl_a | 0.0062 |
| graf_contains | 0.0060 |
| cls_a_known | 0.0058 |
| cls_a_top_44 | 0.0057 |

### 11.2 Importancia agregada por bloco

| Bloco | Soma | Share |
| --- | --- | --- |
| Classe Nice | 0.0936 | 0.1935 |
| Spec_lex | 0.0866 | 0.1790 |
| Numerais | 0.0529 | 0.1094 |
| OFTA | 0.0498 | 0.1029 |
| Spec_atividade | 0.0490 | 0.1014 |
| Interacoes | 0.0421 | 0.0870 |
| Graficas | 0.0378 | 0.0782 |
| Tokens | 0.0371 | 0.0767 |
| Spec_cosine | 0.0264 | 0.0546 |
| Foneticas | 0.0084 | 0.0174 |

## 12. Limitacoes

- O modelo cobre exclusivamente marcas NOMINATIVAS; figurativas precisam de outro pipeline.
- Qualidade do rotulo INPI determina o teto pratico do modelo.
- Embeddings sao multilingues genericos; um modelo PT-BR especifico pode melhorar.
- Classes Nice raras viram bucket 'other' por construcao do top-K.
- Heuristicas produto/servico baseiam-se em listas-ancora; podem nao cobrir setores especificos.

## 13. Recomendacoes

- aumentar epocas e/ou paciencia do early stopping
- ajustar pos_weight (reduzir se ha overshoot, aumentar se recall esta baixo)
- revisar features de especificacao (talvez usar embedding PT-BR especifico)
- considerar mais dados rotulados ou arquitetura maior
- Reavaliar `recall_floor` em conjunto com a area de negocio para calibrar custo de FN/FP.
- Retreinar a cada novo lote significativo de pareceres revisados.
- Monitorar drift de score em producao (PSI sobre `score_nn`).

## 14. Notas adicionais

- Smoke test executado com amostra reduzida e poucas epocas.
