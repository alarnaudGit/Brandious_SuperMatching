# Plano de melhorias do modelo de similaridade de marcas

> **Análise baseada em**: `artifacts/report.md`, `artifacts/falsos_positivos_grupo_0.csv`,
> `artifacts/falsos_negativos_grupo_1.csv`, `artifacts/brand_similarity_input_view_enriched.xlsx`.
> **Data da análise**: 30 de abril de 2026.

---

## 1. Diagnóstico do modelo atual

### Métricas no teste (4.642 pares)

| Métrica | Valor | Avaliação |
|---|---:|---|
| ROC-AUC | 0,8447 | Aceitável, mas longe do ideal (≥ 0,90) |
| PR-AUC | 0,4795 | Bom para 10,9 % positivos |
| F1 | 0,3752 | Modesto |
| Recall | 0,8478 | Levemente abaixo do piso 0,85 |
| **Precision** | **0,2409** | **Crítico: 76 % dos alarmes são falsos positivos** |

### Distribuição dos erros

| Grupo | Total | Erros | Taxa de erro |
|---|---:|---:|---:|
| Grupo 0 (rótulo 0, **não** colidente) | 4.136 | **1.352 falsos positivos** | 32,7 % |
| Grupo 1 (rótulo 1, **colidente**) | 506 | **77 falsos negativos** | 15,2 % |

### Sinais de overfit

| Métrica | treino | teste | Δ train-test |
|---|---:|---:|---:|
| ROC-AUC | 0,9295 | 0,8447 | +0,085 |
| PR-AUC | 0,6387 | 0,4795 | **+0,159** |
| F1 | 0,4419 | 0,3752 | +0,067 |

A diferença de PR-AUC (+0,16) entre treino e teste é grande. O modelo está
**memorizando padrões de treino** que não generalizam.

### Importância de features (top 10 por permutation)

```
spec_cosine_tfidf_char        0.1205   <-- domina
spec_lex_idf_avg_common       0.0596
graf_len_ratio                0.0477
spec_cosine_emb               0.0436
spec_lex_size_diff_abs        0.0326
fon_token_mean                0.0307
ofta_token                    0.0276
cls_a_top_9                   0.0257   <-- viés indesejado em uma classe
fon_key_lev_sim               0.0256
spec_lex_overlap              0.0201
```

**Interpretação:** o modelo está pesadamente apoiado em similaridade de
especificação (TF-IDF + IDF dos termos comuns). É o sinal mais discriminante,
mas insuficiente sozinho — e cria viés quando uma marca tem especificação
genérica que se assemelha a muitas outras.

---

## 2. Achados quantitativos sobre os erros (mineração crítica)

### 2.1 Sobre os **1.352 falsos positivos** (Grupo 0)

| Achado | Impacto |
|---|---|
| **53,7 % não têm nenhum token compartilhado** entre as marcas | Maior categoria de FP — modelo confunde marcas só por proximidade ortográfica/fonética. Ex.: `SUPERMERCASA / SUPERMERCADO PAIVA`, `BELLA DAMA / D'Bella`, `GREEN GUPPY / GREEN MONEY` |
| **62,7 % têm `spec_cosine_emb < 0,30`** (specs semanticamente distantes) | Modelo recebe esse sinal mas o **subutiliza** — não consegue desconfirmar mesmo com evidência forte de mercado distinto |
| **8,2 % compartilham apenas tokens genéricos** | EMPÓRIO, INSTITUTO, RESTAURANTE, SORVETES, MARCENARIA, BRASIL, FITNESS, MODA, AUTO… falta feature de "qualidade dos tokens compartilhados" |
| 16 % têm diferença de comprimento ≥ 15 chars | Marcas com tamanho muito desigual têm risco baixo de colidência real |
| **Marca-monitorada repete em FPs**: `INSTITUTO REVER SETE QUEDAS` (11 FPs), `BRASILGAS` (10), `UMTELECOM` (7), `AUTO PEÇAS MATOS` (9), `SUPERMERCASA` (6) | São "frases-marca" longas com palavras genéricas, fáceis de confundir com qualquer marca curta |
| 214 FPs ocorrem dentro da classe 35–35 (dominante, 21.882 pares no dataset) | Classe 35 ("comércio") é mega-genérica; precisa de tratamento dedicado |

### 2.2 Sobre os **77 falsos negativos** (Grupo 1)

| Achado | Impacto |
|---|---|
| **40 % têm score < 0,05** | Modelo MUITO confiante de que não colide → casos perdidos sem chance de revisão humana |
| **49,4 % têm prefixo de ≥ 3 chars compartilhado**, 36 % têm prefixo ≥ 5 chars | Casos como `AGROPEREIRA / AGRO FRONTEIRA`, `NEO AGROAMBIENTAL / NEOON`, `OFTALMOAMIGO / OFTALMO TOP` — radicais semânticos fortes não estão sendo detectados |
| 57 % NÃO têm token em comum mas colidem segundo INPI | INPI considera contextos morfológicos e de mercado real que features atuais não capturam |
| 42 % têm pelo menos 1 token em comum | Para esses, falta uma "heurística de reforço" |

### 2.3 Conclusão do diagnóstico

> O modelo aprendeu **bem** a sinalizar similaridade, mas aprendeu **mal** a
> *desconfirmar* colidência quando há evidência negativa forte. Falta-lhe o
> contraponto: features que codificam **evidência contra colidência**.

---

## 3. Plano de melhorias por prioridade

Estimei o impacto esperado em PR-AUC com base no que cada bloco resolve.

### Sprint 1 — Features de alto impacto (impacto: +3 a +6 pp em PR-AUC)

| # | Feature/Mudança | Resolve | Esforço |
|---|---|---|---|
| 1 | **Tokens genéricos** (lista + IDF dinâmico) | 8,2 % FPs com tokens só genéricos + 53 % FPs sem tokens reais | Baixo |
| 2 | **Substring/contenção** | FN com contenção (40 % dos FN) | Baixo |
| 3 | **Radical/prefixo morfológico forte** | FN com prefixo ≥ 5 chars (36 % dos FN) | Baixo |
| 4 | **Embedding da marca** (não só de spec) | FPs sem token mas distantes semanticamente | Médio |
| 5 | **Evidência contra colidência** (combinada) | Direciona modelo a aprender a recusar | Baixo |

### Sprint 2 — Refinamento de features (impacto: +2 a +4 pp)

| # | Feature/Mudança | Resolve |
|---|---|---|
| 6 | Levenshtein **após retirar tokens genéricos** | Lev. mais discriminante em frases longas |
| 7 | Token-set ratio + Hungarian alignment | Alinhamento melhor entre tokens A e B |
| 8 | Top-K tokens **raros** compartilhados (IDF alto) | Diferencia "Fashion comum" de "Fashion ANCHOR" |
| 9 | Embedding PT-BR especializado (BERTimbau) | spec_cosine_emb mais forte |
| 10 | One-hot de pares Nice prováveis (35-35, 41-41) | Modelo reconhece que mesma classe não basta |

### Sprint 3 — Treino e arquitetura (impacto: +1 a +3 pp + reduz overfit)

| # | Mudança | Por quê |
|---|---|---|
| 11 | **Focal Loss** (γ=2) em vez de BCE+pos_weight | Foca em exemplos difíceis em vez de inflar todos os positivos |
| 12 | Aumentar **dropout** 0,3 → 0,4-0,5; weight_decay 1e-4 → 1e-3 | Reduz Δ train-test 0,16 → ~0,08 |
| 13 | **Reduzir oversample** 2× → 1× e ajustar pos_weight | Evita decoração de positivos repetidos |
| 14 | **Calibração** de score (Platt/Isotonic) | Score se torna probabilidade interpretável |
| 15 | **Ensemble** de 5 seeds | +0,5 a 1 pp grátis |

### Sprint 4 — Dados e avaliação (impacto: variável)

| # | Mudança |
|---|---|
| 16 | Auditar 200 FPs de score ≥ 0,90 — pode haver ruído de rótulo INPI |
| 17 | Criar **gold standard** de 200-500 pares revisados manualmente |
| 18 | Analisar por classe Nice (treinar threshold por classe) |
| 19 | Adicionar features externas: data de registro, UF, PJ vs PF |
| 20 | Pares hard-negative manuais (PIZZA HUT vs PIZZARIA HUT regional) |

---

## 4. Detalhamento das novas features (Sprint 1)

### 4.1 Tokens genéricos — `name_generic_*`

**Justificativa:** 8,2 % dos FPs compartilham apenas tokens "vazios de marca"
(EMPÓRIO, INSTITUTO, BURGER, AUTO, FASHION, MODA, BRASIL, FITNESS…). Hoje
todos esses contam igual. Vamos pesar pela raridade (IDF) e pela presença em
uma lista curada de termos de mercado.

**Lista inicial** (curada + auto-detectada):

```
suporte: ltda, me, epp, eireli, sa, comercio, industria
gastronomia: burger, pizza, pizzaria, pastel, sorvete, hamburger, lanches, bar,
             restaurante, churrascaria, cafe, padaria, esfiha
saude: clinica, odonto, dentista, medico, clinica, vet, veterinaria, saude,
       fisioterapia, oftalmologia, oftalmo
moda: fashion, moda, modas, modelo, boutique, store, shop, atacado, varejo,
      jeans, lingerie, fitness, beachwear
imobiliaria: imobiliaria, imovel, imoveis, construtora, incorporadora, obras
educacao: escola, instituto, faculdade, universidade, treinamento, cursos
tecnologia: tecnologia, tec, sistemas, soft, digital, online, web, app
auto: auto, autos, autopecas, motos, moto, veiculos
servicos: servicos, servico, solucao, solucoes, consultoria, consultorias,
          assessoria
geo: brasil, brazil, br, nordeste, sudeste, vitoria, recife, fortaleza,
     salvador, manaus
qualifiers: prime, premium, top, gold, plus, max, pro, smart, novo, super
energia: solar, solarenergia, energia, telecom
```

**Auto-detecção:** qualquer token com DF ≥ 1 % do corpus de marcas vai para a
lista (a regra atual é só DF ≥ 5, fica fraco).

**Features novas (10):**

| Nome | Cálculo |
|---|---|
| `name_generic_share_a` | % dos tokens de A na lista de genéricos |
| `name_generic_share_b` | % dos tokens de B na lista de genéricos |
| `name_generic_share_max` | max(a, b) |
| `name_unique_token_a` | # tokens NÃO genéricos em A com len ≥ 4 |
| `name_unique_token_b` | idem para B |
| `shared_unique_count` | # tokens compartilhados que NÃO são genéricos |
| `shared_unique_jaccard` | jaccard de tokens-não-genéricos |
| `shared_token_idf_max` | maior IDF entre os tokens compartilhados |
| `shared_token_idf_mean` | IDF médio dos tokens compartilhados |
| `shared_only_generics` | 1 se todos os tokens compartilhados são genéricos |

**Exemplo de uso:** `INSTITUTO REVER` vs `INSTITUTO RUTLE` — `shared_only_generics=1`
e `shared_token_idf_max` baixo (instituto é genérico). Modelo aprende a
desconfirmar.

### 4.2 Substring e contenção — `contain_*`

**Justificativa:** muitos FN têm uma marca contida na outra após
normalização (`AMPLA` em `AMPLAPACK`, `OFTALMO` em `OFTALMOAMIGO`).

**Features novas (6):**

| Nome | Cálculo |
|---|---|
| `contain_a_in_b` | 1 se forma normalizada de A está contida em B |
| `contain_b_in_a` | idem inverso |
| `contain_after_strip_a_in_b` | A contém-se em B após retirar tokens genéricos |
| `contain_after_strip_b_in_a` | idem inverso |
| `contain_radical_share` | # primeiros chars (sem espaço) iguais nas duas marcas |
| `contain_radical_share_norm` | igual ao anterior, normalizado pelo min(len) |

### 4.3 Radical morfológico forte — `radical_*`

**Justificativa:** marcas como `AGROPEREIRA` e `AGRO FRONTEIRA` têm radical
forte (AGRO) que indica mercado, mas atualmente só temos `graf_prefix_norm`
medindo prefixo de string crua. Queremos prefixo do **radical mais longo
extraído**, considerando hifenização e segmentação.

**Features novas (4):**

| Nome | Cálculo |
|---|---|
| `radical_a` | maior token de A (≥ 4 chars) que NÃO é genérico |
| `radical_b` | idem |
| `radical_lev_sim` | similaridade Levenshtein dos radicais |
| `radical_phonetic_eq` | chaves fonéticas (Metaphone PT-BR) dos radicais iguais |

### 4.4 Embedding da marca (não só da spec) — `brand_emb_*`

**Justificativa:** hoje o embedding sentence-transformer só roda em
especificações. Mas para marcas com palavras de dicionário (`SOLAR ENERGIA`,
`AGRO FRONTEIRA`, `BELLA DAMA`), o embedding da própria marca capta semântica.

**Features novas (3):**

| Nome | Cálculo |
|---|---|
| `brand_emb_cosine` | cosine entre embedding da marca A e da marca B |
| `brand_emb_cosine_normalized` | idem, mas usando normalize_brand antes |
| `brand_emb_norm_a/b` | norma do embedding (palavras de dicionário têm norma maior) |

**Modelo proposto:** o mesmo já em uso (`paraphrase-multilingual-MiniLM`),
ou se possível BERTimbau base. Cache em `embeddings_brand_cache.parquet`.

### 4.5 Evidência contra colidência — `against_*`

**Justificativa:** o modelo precisa de features explícitas que digam "este
par tem cara de NÃO colisão". Sem isso, ele aprende só a empilhar evidência
positiva.

**Features novas (5):**

| Nome | Cálculo |
|---|---|
| `against_distinct_market` | 1 se classe diferente E `spec_cosine_emb < 0,30` |
| `against_only_generic_overlap` | 1 se tokens compartilhados são todos genéricos AND `cls_diff != 0` |
| `against_size_disparity` | 1 se diff de comprimento ≥ 15 chars E sem token compartilhado |
| `against_unique_strong_a` | # tokens em A com IDF alto que NÃO existem em B |
| `against_unique_strong_b` | idem para B |

### 4.6 Levenshtein desgenericizado — `lev_pure_*`

**Justificativa:** `INSTITUTO REVER SETE QUEDAS` vs `INSTITUTO REGULA` têm
Levenshtein normalizado alto **só por causa dos genéricos**. Calcular
Levenshtein **após retirar tokens genéricos** dá visão muito mais real.

**Features novas (3):**

| Nome | Cálculo |
|---|---|
| `lev_pure_a_vs_b` | Levenshtein normalizado das marcas com tokens genéricos retirados |
| `lev_pure_jaro_winkler` | idem com Jaro-Winkler |
| `lev_pure_size_diff` | diff de tamanho após retirar genéricos |

---

## 5. Resumo visual das novas features

| Bloco novo | # features | Impacto esperado |
|---|---:|---|
| `name_generic_*` (tokens genéricos + IDF) | 10 | Mata FP "tokens genéricos" e "frase-longa-vs-marca-curta" |
| `contain_*` (substring/contenção) | 6 | Resgata FN por contenção |
| `radical_*` (radical morfológico) | 4 | Resgata FN por radical compartilhado |
| `brand_emb_*` (embedding da marca) | 3 | Resgata FN sem tokens comuns + reduz FP por proximidade só ortográfica |
| `against_*` (evidência negativa) | 5 | Reduz FP em pares com classe distinta + spec distante |
| `lev_pure_*` (Levenshtein "puro") | 3 | Reduz FP em frases longas |
| **Total** | **31** | **+5 a +8 pp em PR-AUC esperado** |

Resultado: passaríamos de **116 features para 147 features**.

---

## 6. Mudanças propostas no treino (Sprint 3)

### 6.1 Focal Loss

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(
            logits, target, reduction='none', pos_weight=self.pos_weight
        )
        p = torch.sigmoid(logits)
        p_t = p * target + (1 - p) * (1 - target)
        focal = (1 - p_t) ** self.gamma * bce
        return (self.alpha * focal).mean()
```

**Substitui** `BCEWithLogitsLoss(pos_weight)` no `train.py`.

### 6.2 Regularização mais forte

```python
TrainConfig(
    epochs=80,
    batch_size=256,
    lr=8e-4,           # ligeiramente menor (modelo maior)
    weight_decay=1e-3, # de 1e-4 para 1e-3
    early_stopping_patience=12,
)
MLPConfig(
    hidden_dims=[256, 128, 64, 32],   # mais profundo
    dropout=0.45,                      # de 0.3 para 0.45
    use_batchnorm=True,
)
BalancingConfig(
    undersample_neg_ratio=4.0,         # de 3.0 para 4.0 (mais negativos)
    oversample_pos_factor=1.0,         # SEM oversample
    use_class_weight=True,             # mantém pos_weight
)
```

### 6.3 Calibração e ensemble

- Após treino: ajustar **Platt scaling** (regressão logística sobre o
  logit do modelo no conjunto de validação) para que `score = P(y=1|x)`.
- Ensemble: treinar 5 modelos com seeds 42, 7, 13, 21, 99; score final =
  média (após calibração individual).

---

## 7. Métricas-alvo após implementação

| Cenário | ROC-AUC | PR-AUC | F1 | Recall | Precision |
|---|---:|---:|---:|---:|---:|
| **Hoje** | 0,8447 | 0,4795 | 0,3752 | 0,8478 | 0,2409 |
| Após Sprint 1 (features) | 0,86–0,88 | 0,53–0,56 | 0,42–0,46 | 0,85–0,87 | 0,28–0,32 |
| Após Sprint 2 (refinamento) | 0,88–0,90 | 0,57–0,61 | 0,46–0,50 | 0,85–0,88 | 0,32–0,38 |
| Após Sprint 3 (treino) | 0,89–0,91 | 0,60–0,65 | 0,50–0,55 | 0,86–0,89 | 0,38–0,45 |
| **Meta de produção** | **≥ 0,90** | **≥ 0,60** | **≥ 0,50** | **≥ 0,87** | **≥ 0,40** |

---

## 8. Ordem recomendada de execução

1. **Sprint 1.1** Implementar `name_generic_*` + lista de genéricos (1-2 dias)
2. **Sprint 1.2** Implementar `contain_*` + `radical_*` + `lev_pure_*` (1 dia)
3. **Sprint 1.3** Implementar `brand_emb_*` + integrar com cache (1 dia)
4. **Sprint 1.4** Implementar `against_*` (0,5 dia)
5. **Treinar** com novas features e gerar relatório comparativo
6. **Sprint 3.1** Trocar para Focal Loss + ajuste de regularização
7. **Sprint 3.2** Implementar Platt calibration
8. **Sprint 3.3** Ensemble 5 seeds
9. **Avaliar** e decidir se precisa Sprint 2 (refinamento) ou Sprint 4 (dados)

---

## 9. Mudanças no código (mapeamento)

| Arquivo | O que adicionar |
|---|---|
| `src/features/generics.py` (novo) | Lista de genéricos + utilitários |
| `src/features/nominal.py` | Adicionar `containment_features`, `radical_features`, `lev_pure_features` |
| `src/features/specs.py` | Adicionar `BrandEmbedder` reaproveitando provider |
| `src/features/interactions.py` | Adicionar `against_*` |
| `src/features/builder.py` | Atualizar `canonical_feature_order` |
| `src/pipeline/preprocessor.py` | Adicionar embed_brands_cache + cálculo no transform |
| `src/model/train.py` | Adicionar `FocalLoss` + opção via TrainConfig |
| `src/model/calibration.py` (novo) | `PlattCalibrator(fit/transform/save/load)` |
| `src/pipeline/inference.py` | Aplicar calibrador no `score_pair`/`score_batch` |
| `streamlit_app.py` | Expor opções: focal loss, ensemble seeds, calibration |

---

## 10. Próximas decisões necessárias

Antes de codificar, precisamos confirmar com você:

1. **OK adicionar BERTimbau** (modelo PT-BR especializado, ~470 MB)? Ele
   substituiria `paraphrase-multilingual-MiniLM` em specs e marcas. Ganho
   esperado: +1 a +2 pp PR-AUC. Custo: download único + cache.
2. **Auditar 200 FPs de score ≥ 0,90** para checar ruído de rótulo? Pode
   indicar que o teto prático do modelo é menor que parece.
3. **Listas de genéricos**: gerar do zero ou validar com você uma lista
   curada antes? Sugiro validar a lista comigo antes de aplicar.
4. **Ordem**: implementar **tudo de Sprint 1** e treinar uma vez, ou ir
   feature por feature treinando entre cada? Recomendo o primeiro (mais
   eficiente).

