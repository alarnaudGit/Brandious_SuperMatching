# Dicionário de Dados — Tabela Enriquecida do Modelo

> Arquivo gerado: `brand_similarity_input_view_enriched.xlsx` / `enriched.parquet`
>
> Objetivo deste documento: explicar **em linguagem de negócio**, sem jargão técnico, o que cada coluna da tabela enriquecida significa e por que ela é importante para a decisão de "essas duas marcas são potencialmente colidentes ou não?"

---

## Convenções

- **A** = lado da marca monitorada (`marca_monitorada`).
- **B** = lado da marca colidente (`marca_colidente`).
- Quase toda coluna numérica é **uma similaridade entre 0 e 1**, onde:
  - `0` = nada parecido
  - `1` = idêntico ou perfeitamente compatível
- Algumas colunas são **0/1 (sim/não)**: indicadores binários.
- Algumas são **contagens absolutas** (número inteiro de palavras, por exemplo).
- Faixa "Z-padrão" significa que a coluna foi padronizada (média 0, desvio 1) antes de entrar na rede neural — o valor original ainda está visível na planilha enriquecida; a versão padronizada fica embutida no modelo.

---

## 1. Colunas originais do INPI (entrada do sistema)

São as 7 colunas que existem na planilha de origem e ficam preservadas no arquivo enriquecido.

| Coluna | Significado de negócio | Tipo |
|---|---|---|
| `marca_monitorada` | Nome da marca que está sendo vigiada (lado A) | Texto |
| `marca_colidente` | Nome da outra marca, candidata a colidir com A (lado B) | Texto |
| `classe_marca_monitorada` | Classe Nice da marca A (1 a 45) | Inteiro |
| `classe_marca_colidente` | Classe Nice da marca B (1 a 45) | Inteiro |
| `especificacao_monitorado` | Lista (separada por ponto-e-vírgula) das atividades/produtos/serviços que A protege | Texto longo |
| `especificacao_colidente` | Lista das atividades/produtos/serviços que B protege | Texto longo |
| `label` | Resposta do INPI: 1 = par é colidente, 0 = par não é colidente. **Só existe em dados rotulados** | 0 ou 1 |

---

## 2. Bloco GRÁFICO — semelhança da escrita (14 colunas)

Mede o quanto as duas marcas se parecem **visualmente**, letra a letra. Imagine ler as duas marcas com os olhos: quão parecidas elas são na grafia?

| Coluna | O que significa em palavras simples | Faixa |
|---|---|---|
| `graf_levenshtein` | Quantas trocas de letra (inserir, apagar, substituir) seriam necessárias para transformar uma marca na outra. Quanto mais próximo de 1, menos trocas precisa. | 0–1 |
| `graf_jaro` | Outra medida clássica de "quão parecidas são duas palavras", focando em letras na mesma posição. | 0–1 |
| `graf_jaro_winkler` | Igual ao Jaro, mas dá bônus extra quando as primeiras letras batem (porque palavras em português costumam diferenciar mais no começo). | 0–1 |
| `graf_damerau` | Similar ao Levenshtein, mas conta troca de letras adjacentes (ex: "barbatto" / "barbatto" com letras invertidas) como uma única operação. | 0–1 |
| `graf_jaccard_bigram` | Pega todos os pares de letras consecutivas das duas marcas e mede quanto se sobrepõem. | 0–1 |
| `graf_jaccard_trigram` | Mesma ideia, mas com trios de letras. Mais sensível a coincidências longas. | 0–1 |
| `graf_overlap_trigram` | Quanto dos trios de letras da marca **menor** estão presentes na maior. Útil para detectar inclusão. | 0–1 |
| `graf_lcs_norm` | Tamanho da maior sequência de letras que aparece nas duas (mantendo a ordem), dividido pelo tamanho da maior marca. | 0–1 |
| `graf_prefix_norm` | Tamanho do começo idêntico ("ABC..." e "ABCD..." → 3 letras), normalizado. | 0–1 |
| `graf_suffix_norm` | Tamanho do fim idêntico ("...XYZ" e "...XYZ"), normalizado. | 0–1 |
| `graf_len_ratio` | Razão entre o tamanho da marca menor e o tamanho da maior. Marca de 4 letras vs 8 letras → 0.5. | 0–1 |
| `graf_len_diff_abs` | Diferença absoluta de tamanho em caracteres. Não é normalizado. | inteiro ≥ 0 |
| `graf_contains` | Vale 1 se uma marca está contida inteira na outra ("AGRO" dentro de "AGROLOG"); 0 caso contrário. | 0 ou 1 |
| `graf_anagram` | Se você embaralhar as letras das duas, ainda dá pra montar a mesma palavra? Mede o grau disso. Útil para casos como "BIONOVA" vs "NOVABIO". | 0–1 |

**Por que importa:** marcas que escrevem parecido têm mais chance de confundir o consumidor visualmente.

---

## 3. Bloco FONÉTICO — semelhança do som (7 colunas)

Mede o quanto as duas marcas **soam parecido**, ignorando diferenças de grafia. Crucial em português brasileiro porque "FARMÁCIA" e "PHARMACIA" têm escrita diferente mas som idêntico.

A chave fonética PT-BR converte:
- `PH` → F (PHARMACIA = FARMACIA)
- `Y` → I, `K` / `QU` → unificado
- Remove repetições e vogais internas, etc.

| Coluna | O que significa em palavras simples | Faixa |
|---|---|---|
| `fon_global_sim` | Quão parecida é a versão fonética das duas marcas inteiras. Captura "soa igual" mesmo com escrita diferente. | 0–1 |
| `fon_key_eq` | Vale 1 quando a chave fonética das duas marcas é exatamente a mesma. | 0 ou 1 |
| `fon_key_lev_sim` | Quase igual ao `fon_global_sim`, mas com cálculo direto entre as chaves fonéticas. | 0–1 |
| `fon_after_dedup_eq` | Vale 1 se, depois de remover letras repetidas (ex.: "BARBATTO" → "BARBATO"), as marcas ficam iguais. | 0 ou 1 |
| `fon_token_mean` | Quão fonéticas as palavras das duas marcas se casam, na média (palavra a palavra). | 0–1 |
| `fon_token_max` | A melhor combinação fonética entre qualquer palavra de A e qualquer palavra de B. | 0–1 |
| `fon_token_eq_share` | Proporção de palavras de A que têm chave fonética idêntica a alguma palavra de B. | 0–1 |

**Por que importa:** o INPI considera colidência fonética. "FARMÁCIA SOL" e "PHARMACIA SOLL" são quase certamente colidentes embora a escrita difira.

---

## 4. Bloco TOKENS — palavras em comum (9 colunas)

Trata cada marca como um **conjunto de palavras** e compara os conjuntos. Importante para marcas compostas como "MERCANTIL Q & Z" vs "Q Z".

| Coluna | O que significa | Faixa |
|---|---|---|
| `n_tokens_a` | Número de palavras na marca A. | inteiro |
| `n_tokens_b` | Número de palavras na marca B. | inteiro |
| `n_tokens_diff` | Diferença absoluta entre `n_tokens_a` e `n_tokens_b`. | inteiro ≥ 0 |
| `n_tokens_common` | Quantas palavras aparecem nas duas marcas (idênticas). | inteiro ≥ 0 |
| `n_tokens_excl_a` | Quantas palavras estão SÓ em A. | inteiro ≥ 0 |
| `n_tokens_excl_b` | Quantas palavras estão SÓ em B. | inteiro ≥ 0 |
| `tok_jaccard` | Proporção de palavras compartilhadas (palavras em comum / palavras totais únicas das duas). | 0–1 |
| `tok_overlap` | Proporção de palavras da marca **menor** que aparecem na maior. Mede inclusão. | 0–1 |
| `tok_fuzzy` | Igual ao overlap, mas aceita palavras "quase iguais" (com 85% de similaridade ou mais). Captura "BARBATO" e "BARBATTO" como mesma palavra. | 0–1 |

**Por que importa:** marcas que compartilham palavras-chave significativas costumam ser mais arriscadas; "ESCOLA SOL" e "INSTITUTO SOL" compartilham "SOL".

---

## 5. Bloco NUMERAIS — números nas marcas (7 colunas)

Algumas marcas têm dígitos: "100% NATURAL", "EMPRESA 24 HORAS". O sistema testa quatro formas de comparar quando há números: literal ("1" vira "um"), cardinal ("100" vira "cem"), e direções A→B, B→A. Estas colunas resumem o resultado dessas variações.

| Coluna | O que significa | Faixa |
|---|---|---|
| `num_has_digits` | Vale 1 se pelo menos uma das marcas tem números; 0 se ambas são puramente texto. | 0 ou 1 |
| `num_orto_best` | A melhor similaridade gráfica obtida entre as variantes de números (ex.: melhor entre "1" vs "um" e "100" vs "cem"). | 0–1 |
| `num_orto_worst` | A pior dessas variantes. | 0–1 |
| `num_orto_spread` | A diferença entre best e worst. Quando é grande, significa que números mudam muito o resultado. | 0–1 |
| `num_fon_best` | Idem ao `num_orto_best`, mas para a similaridade fonética. | 0–1 |
| `num_fon_worst` | Pior caso fonético entre as variantes. | 0–1 |
| `num_fon_spread` | Diferença best - worst do som. | 0–1 |

**Por que importa:** "EMPRESA 24" e "EMPRESA VINTE E QUATRO" são a mesma marca; sem essas features, o algoritmo trataria como diferentes.

---

## 6. Bloco OFTA — score do algoritmo clássico como referência (12 colunas)

A heurística `SimilarityOFTA` (algoritmo legado do projeto) é executada e seu resultado entra na rede como **mais um sinal**. A rede aprende quando confiar nesse score e quando descontar.

| Coluna | O que significa | Faixa |
|---|---|---|
| `ofta_final` | O score final que o algoritmo legado dá ao par (0 a 1). É o número que aparece na coluna "Score heurística OFTA" da tela de teste manual. | 0–1 |
| `ofta_orto` | Vetor parcial do OFTA para ortografia. | 0–1 |
| `ofta_fon` | Vetor parcial do OFTA para fonética. | 0–1 |
| `ofta_token` | Vetor parcial para palavras. | 0–1 |
| `ofta_anagram` | Vetor parcial para anagrama. | 0–1 |
| `ofta_fuzzy` | Vetor parcial para palavras "quase iguais". | 0–1 |
| `ofta_driver_geral` | 1 se o driver dominante do OFTA naquele par foi "Geral". | 0 ou 1 |
| `ofta_driver_ortografia` | 1 se o driver dominante foi a escrita. | 0 ou 1 |
| `ofta_driver_fonética` | 1 se o driver dominante foi o som. | 0 ou 1 |
| `ofta_driver_aproximação` | 1 se foi "Aproximação de Termos (Fuzzy)". | 0 ou 1 |
| `ofta_driver_inclusão` | 1 se foi "Inclusão Total de Termo" (uma marca contém a outra). | 0 ou 1 |
| `ofta_driver_palavras` | 1 se foi "Palavras em Comum". | 0 ou 1 |

**Por que importa:** dá à rede a opção de aproveitar o conhecimento histórico do algoritmo antigo, mas também **pode aprender quando a heurística erra** (caso clássico: o OFTA dá 0,81 mas a rede dá 0,12 porque viu que a especificação é distante).

---

## 7. Bloco ESPECIFICAÇÃO — lexical (10 colunas)

Compara as listas de atividades/produtos/serviços de A e B em nível de **palavras-chave estemizadas** (palavras com sufixos cortados, ex.: "comércio"/"comerciante" → "comerc").

| Coluna | O que significa | Faixa |
|---|---|---|
| `spec_lex_jaccard` | Proporção de palavras-chave compartilhadas entre as duas especificações. | 0–1 |
| `spec_lex_overlap` | Proporção de palavras-chave da especificação **menor** que estão presentes na maior. | 0–1 |
| `spec_lex_n_common` | Quantas palavras-chave aparecem nas duas especificações. | inteiro ≥ 0 |
| `spec_lex_n_excl_a` | Quantas palavras-chave estão SÓ na especificação de A. | inteiro ≥ 0 |
| `spec_lex_n_excl_b` | Quantas palavras-chave estão SÓ na especificação de B. | inteiro ≥ 0 |
| `spec_lex_n_total_a` | Tamanho da especificação de A (em palavras-chave). | inteiro ≥ 0 |
| `spec_lex_n_total_b` | Tamanho da especificação de B. | inteiro ≥ 0 |
| `spec_lex_size_ratio` | Razão entre o tamanho da menor especificação e o tamanho da maior. | 0–1 |
| `spec_lex_size_diff_abs` | Diferença absoluta de tamanho. | inteiro ≥ 0 |
| `spec_lex_idf_avg_common` | **Importância média** das palavras-chave em comum, ponderada pela raridade. Quando palavras raras (ex.: "biotecnologia") são compartilhadas, esse valor sobe muito. Compartilhar só palavras comuns ("comércio") gera valor baixo. | número ≥ 0 |

**Por que importa:** se duas marcas têm especificações com palavras-chave raras em comum (ex.: "bioinformática"), provavelmente atuam no mesmo nicho mesmo que a classe Nice formal seja diferente.

---

## 8. Bloco ESPECIFICAÇÃO — tipo de atividade (6 colunas)

Detecta se cada lado oferece **produto físico** ou **serviço**, baseado em palavras-âncora.

| Coluna | O que significa | Faixa |
|---|---|---|
| `spec_same_activity_kind` | Vale 1 se A e B são do mesmo tipo (ambos produto, ou ambos serviço). | 0 ou 1 |
| `spec_any_misto` | Vale 1 se a especificação de A ou de B é mista (mistura produto e serviço de forma indistinta). | 0 ou 1 |
| `spec_kind_a_produto` | Vale 1 se A é claramente vendedor de produto. | 0 ou 1 |
| `spec_kind_a_servico` | Vale 1 se A é claramente prestador de serviço. | 0 ou 1 |
| `spec_kind_b_produto` | Vale 1 se B é claramente produto. | 0 ou 1 |
| `spec_kind_b_servico` | Vale 1 se B é claramente serviço. | 0 ou 1 |

**Por que importa:** uma marca que vende produtos cosméticos e outra que presta consultoria têm risco de colidência muito menor, mesmo com nomes parecidos.

---

## 9. Bloco CLASSE NICE (4 colunas básicas + 32 indicadores)

A Classificação de Nice define em qual classe (1 a 45) a marca está registrada. Mesma classe = mesmo "balcão" formal.

### 9.1 Básicas (4 colunas)

| Coluna | O que significa | Faixa |
|---|---|---|
| `cls_same` | Vale 1 quando A e B estão na mesma classe Nice; 0 caso contrário. | 0 ou 1 |
| `cls_diff_abs` | Distância numérica entre as duas classes (apenas como referência; classes Nice **não** são ordenadas, mas a rede pode aprender padrões mesmo assim). | inteiro ≥ 0 |
| `cls_a_known` | Vale 1 quando a classe de A está informada na base; 0 quando ausente. | 0 ou 1 |
| `cls_b_known` | Vale 1 quando a classe de B está informada na base. | 0 ou 1 |

### 9.2 Indicadores por classe (32 colunas — 16 para A, 16 para B)

Para cada uma das **15 classes Nice mais frequentes** no dataset (e mais um bucket "outras"), há duas colunas: uma para A e outra para B. As 15 classes mais frequentes neste treino foram: **35, 41, 42, 44, 43, 36, 9, 37, 30, 40, 25, 39, 45, 16, 3**.

Cada coluna do tipo `cls_a_top_<N>` vale **1 quando a marca A está naquela classe N específica**, e 0 caso contrário. O mesmo vale para `cls_b_top_<N>` em relação a B. A coluna `..._top_other` capta o caso "tem classe registrada, mas não está entre as 15 mais frequentes".

| Coluna | Significado |
|---|---|
| `cls_a_top_35` | Vale 1 se A está na classe 35 (Publicidade, gestão comercial). |
| `cls_a_top_41` | Vale 1 se A está na classe 41 (Educação, entretenimento). |
| `cls_a_top_42` | Vale 1 se A está na classe 42 (Serviços científicos e tecnológicos). |
| `cls_a_top_44` | Vale 1 se A está na classe 44 (Serviços médicos, veterinários, agrícolas). |
| `cls_a_top_43` | Vale 1 se A está na classe 43 (Alimentação, hospedagem temporária). |
| `cls_a_top_36` | Vale 1 se A está na classe 36 (Seguros, financeiros, imobiliários). |
| `cls_a_top_9` | Vale 1 se A está na classe 9 (Aparelhos científicos, eletrônicos, software). |
| `cls_a_top_37` | Vale 1 se A está na classe 37 (Construção, instalação, reparos). |
| `cls_a_top_30` | Vale 1 se A está na classe 30 (Café, chá, açúcar, pães, condimentos). |
| `cls_a_top_40` | Vale 1 se A está na classe 40 (Tratamento de materiais). |
| `cls_a_top_25` | Vale 1 se A está na classe 25 (Vestuário, calçados, chapelaria). |
| `cls_a_top_39` | Vale 1 se A está na classe 39 (Transporte, embalagem, armazenagem). |
| `cls_a_top_45` | Vale 1 se A está na classe 45 (Serviços jurídicos e de segurança). |
| `cls_a_top_16` | Vale 1 se A está na classe 16 (Papel, papelão, material impresso). |
| `cls_a_top_3` | Vale 1 se A está na classe 3 (Cosméticos, perfumaria, limpeza). |
| `cls_a_top_other` | Vale 1 se A tem classe registrada, mas fora do top-15. |
| `cls_b_top_35` ... `cls_b_top_other` | **Idênticas às 16 acima, mas para a marca B.** |

**Por que importa:** ao invés de definir manualmente "classes 35 e 41 colidem", a rede vê os indicadores e descobre sozinha quais combinações de classes são mais perigosas a partir dos dados rotulados.

---

## 10. Bloco INTERAÇÕES — combinações cruzadas (12 colunas)

Estas são as features **mais inteligentes** do sistema: combinam o sinal de nome, especificação e classe em um único número, capturando os cenários de risco descritos pelo negócio.

| Coluna | O que significa em palavras simples | Faixa |
|---|---|---|
| `inter_nome_x_spec_word` | Multiplica "quão parecidos são os nomes" por "quão parecidas são as palavras das especificações". Alto quando ambos estão altos. | 0–1 |
| `inter_nome_x_spec_emb` | Multiplica nome parecido por **proximidade semântica** das especificações. Captura casos onde as especificações usam palavras diferentes mas significam o mesmo. | 0–1 |
| `inter_nome_x_spec_max` | Combina nome com o melhor sinal de especificação disponível. | 0–1 |
| `inter_nome_x_same_cls` | Nome parecido E classe Nice igual? Esta feature dispara. | 0–1 |
| `inter_spec_x_same_cls` | Especificação parecida E classe igual = risco máximo formal. | 0–1 |
| `inter_nome_alto_e_spec_alta` | Vale 1 quando o nome é muito parecido (>0.7) E a especificação é muito parecida (>0.6). **CENÁRIO ALTO RISCO.** | 0 ou 1 |
| `inter_nome_alto_e_spec_baixa` | Vale 1 quando o nome é muito parecido mas a especificação é distante (<0.3). **CENÁRIO MÉDIO RISCO** ("nomes parecidos mas mercados diferentes"). | 0 ou 1 |
| `inter_nome_baixo_e_spec_alta` | Vale 1 quando o nome é diferente mas a especificação é muito parecida. **CENÁRIO POSSÍVEL RISCO.** | 0 ou 1 |
| `inter_classe_diff_mas_emb_alto` | Quando classes Nice são diferentes mas o significado das atividades é parecido (cosseno semântico alto). **RISCO OCULTO**: o INPI separou em classes diferentes, mas o mercado real é o mesmo. | 0–1 |
| `inter_classe_igual_e_spec_proxima` | Classe igual + especificação próxima = duplo reforço de risco. | 0–1 |
| `inter_proxy_nome` | Cópia do score nominativo geral, exposta como feature dedicada. | 0–1 |
| `inter_proxy_spec` | Cópia do melhor score de especificação, exposta como feature dedicada. | 0–1 |

**Por que importa:** estas colunas codificam diretamente as cinco situações de risco do contrato de negócio, e são as features mais frequentemente "puxadas" pelos Integrated Gradients quando a rede explica um score alto ou baixo.

---

## 11. Bloco COSSENOS DE ESPECIFICAÇÃO (3 colunas)

Resumem em três números a similaridade entre os dois textos longos das especificações.

| Coluna | O que significa | Faixa |
|---|---|---|
| `spec_cosine_tfidf_word` | Quão parecidas são as duas especificações considerando **palavras importantes** (descontando palavras genéricas como "comércio"). É o sinal de "vocabulário em comum". | 0–1 |
| `spec_cosine_tfidf_char` | Igual, mas comparando **trechos de letras** (3 a 5 letras). Pega coincidências mesmo com pequenas variações de escrita ou conjugação. | 0–1 |
| `spec_cosine_emb` | Sinal **semântico**: usa um modelo de inteligência artificial multilíngue (MiniLM) para entender o "significado geral" de cada especificação e medir o quão próximas elas são em termos de sentido. Captura coisas como "venda de bebidas alcoólicas" vs "comércio de cervejas e vinhos" sendo semanticamente próximas mesmo com palavras diferentes. | 0–1 |

**Por que importa:** estas três colunas, junto com as 19 outras de especificação e interação, são o **diferencial principal** do modelo aprendido contra a heurística clássica — a heurística olha só as duas marcas, o modelo aprende também o contexto de mercado.

---

## 12. Bloco TOKENS GENÉRICOS — qualidade dos tokens compartilhados (10 colunas, Sprint 1)

A intuição: quando duas marcas compartilham só palavras "vazias de marca" (como BURGER, INSTITUTO, AUTO, MODA, FASHION, BRASIL), a evidência de colidência é fraca. Estas colunas medem quão informativos são os tokens compartilhados.

A lista de tokens "genéricos" é **detectada automaticamente**: qualquer palavra presente em pelo menos 1% das marcas do dataset entra na lista. Stop-words (a, o, de, da, dos, e, etc.) também são filtradas.

| Coluna | O que significa em palavras simples | Faixa |
|---|---|---|
| `name_generic_share_a` | Que fração das palavras da marca A são genéricas. Próximo de 1 = a marca A só tem palavras vazias de identidade ("INSTITUTO BRASILEIRO DE..."). | 0–1 |
| `name_generic_share_b` | Igual para a marca B. | 0–1 |
| `name_generic_share_max` | O maior valor entre as duas. Usado para detectar "frase-marca" longa em pelo menos um dos lados. | 0–1 |
| `name_unique_token_a` | Quantas palavras "informativas" (não genéricas, com 4+ letras) a marca A tem. | inteiro |
| `name_unique_token_b` | Igual para B. | inteiro |
| `shared_unique_count` | Número de palavras informativas compartilhadas pelas duas marcas (descontando os genéricos). É **muito mais discriminante** que o `n_tokens_common` original. | inteiro |
| `shared_unique_jaccard` | Jaccard restrito a palavras informativas. | 0–1 |
| `shared_token_idf_max` | Quão **rara** é a palavra mais rara compartilhada (IDF máximo). Compartilhar "ANCHOR" (rara) vale mais do que "FASHION" (comum). | número (tipo 0–8) |
| `shared_token_idf_mean` | IDF médio das palavras compartilhadas. | número |
| `shared_only_generics` | Vale 1 se as marcas só compartilham palavras genéricas (red flag de FALSO ALARME). | 0 ou 1 |

**Por que importa:** ataca diretamente os 8% de falsos positivos atuais onde o único elo entre as marcas eram tokens genéricos.

---

## 13. Bloco CONTENÇÃO — uma marca dentro da outra (6 colunas, Sprint 1)

Detecta casos como `AMPLA` dentro de `AMPLAPACK`, ou `OFTALMO` dentro de `OFTALMOAMIGO`. Resgata colidências que escapam quando uma marca é "filhote" da outra.

| Coluna | O que significa em palavras simples | Faixa |
|---|---|---|
| `contain_a_in_b` | A forma normalizada da marca A está contida na marca B? | 0 ou 1 |
| `contain_b_in_a` | E vice-versa? | 0 ou 1 |
| `contain_after_strip_a_in_b` | Igual ao primeiro, mas **depois de retirar palavras genéricas** das duas. Mais sensível para "frases-marca" longas. | 0 ou 1 |
| `contain_after_strip_b_in_a` | Vice-versa. | 0 ou 1 |
| `contain_radical_share` | Quantas letras iniciais (radical) as duas marcas compartilham. Captura `AGROPEREIRA` × `AGRO FRONTEIRA` (4 chars). | inteiro |
| `contain_radical_share_norm` | O mesmo, mas dividido pelo tamanho da marca menor. | 0–1 |

**Por que importa:** resgata 36% dos falsos negativos atuais que tinham radical comum forte mas o modelo não pegou.

---

## 14. Bloco RADICAL MORFOLÓGICO — núcleo semântico da marca (4 colunas, Sprint 1)

Identifica o "núcleo" de cada marca (a palavra-raiz mais longa e informativa) e mede similaridade entre os núcleos. É mais robusto que comparar a marca inteira porque ignora variações como "MODA", "FASHION", "STORE".

| Coluna | O que significa em palavras simples | Faixa |
|---|---|---|
| `radical_a_len` | Tamanho do núcleo identificado para a marca A. | inteiro |
| `radical_b_len` | Tamanho do núcleo identificado para a marca B. | inteiro |
| `radical_lev_sim` | Similaridade entre os núcleos (combina Levenshtein e Jaro-Winkler, pegando o maior). 1 = núcleos idênticos. | 0–1 |
| `radical_phonetic_eq` | Os núcleos têm a mesma pronúncia (chaves fonéticas iguais)? Detecta `KOLMEIA / COLMEIA`. | 0 ou 1 |

**Por que importa:** isola o sinal de "mesma raiz semântica" do ruído de tokens decorativos.

---

## 15. Bloco LEVENSHTEIN PURO — semelhança após remover genéricos (3 colunas, Sprint 1)

Hoje, `INSTITUTO REVER SETE QUEDAS` e `INSTITUTO REGULA` têm Levenshtein bom **só por causa do "INSTITUTO" em comum**. Estas colunas calculam Levenshtein **após retirar palavras genéricas** das duas marcas — visão muito mais real.

| Coluna | O que significa em palavras simples | Faixa |
|---|---|---|
| `lev_pure_norm` | Levenshtein normalizado das marcas com palavras genéricas removidas. Mais alto = mais parecidas no que importa. | 0–1 |
| `lev_pure_jaro_winkler` | Mesma ideia com Jaro-Winkler (mais permissivo no início das palavras). | 0–1 |
| `lev_pure_size_diff_abs` | Diferença de tamanho das marcas após remover genéricos. Tamanhos muito diferentes = baixo risco. | inteiro |

**Por que importa:** mata os falsos positivos do tipo "frase longa com vocabulário genérico parecida com qualquer outra frase longa do mesmo setor".

---

## 16. Bloco EVIDÊNCIA NEGATIVA — sinais contra a colidência (5 colunas, Sprint 1)

Estas colunas são a inovação mais importante do Sprint 1. O modelo anterior só recebia evidências **a favor** de colidência (similaridades altas). Estas 5 colunas codificam explicitamente **evidência contra**, ensinando o modelo a desconfirmar.

| Coluna | O que significa em palavras simples | Faixa |
|---|---|---|
| `against_distinct_market` | Vale 1 quando classes Nice são diferentes E a similaridade semântica das especificações é baixa (< 0.30). Sinal forte de "estes dois operam em mercados realmente distintos". | 0 ou 1 |
| `against_only_generic_overlap` | Vale 1 quando há tokens compartilhados, mas todos são genéricos, **e** as classes Nice diferem. Sinal de "se parecem por acaso, não por substância". | 0 ou 1 |
| `against_size_disparity` | Vale 1 quando há diferença de tamanho ≥ 15 caracteres E zero tokens em comum. Marcas muito desiguais raramente colidem. | 0 ou 1 |
| `against_unique_strong_a` | Quantas palavras informativas e raras (IDF ≥ 3.0) a marca A tem que **não existem** em B. Quanto maior, mais A se "sustenta sozinha". | inteiro |
| `against_unique_strong_b` | Igual para B. | inteiro |

**Por que importa:** atualmente o modelo tem precision 0,24 (76% dos alarmes são falso positivo) precisamente porque não tinha como **rejeitar**. Estas 5 features mudam o jogo.

---

## 17. Bloco EMBEDDING DA MARCA — significado das marcas em si (3 colunas, Sprint 1)

Hoje os embeddings semânticos só são aplicados às **especificações**. Estas 3 colunas aplicam o mesmo modelo de IA (sentence-transformers, opcionalmente BERTimbau-large para PT-BR) **às próprias marcas**. Para marcas com palavras de dicionário (como "SOLAR ENERGIA", "AGRO FRONTEIRA", "BELLA DAMA"), isto captura proximidade conceitual mesmo sem letras em comum.

| Coluna | O que significa em palavras simples | Faixa |
|---|---|---|
| `brand_emb_cosine` | Similaridade semântica entre as duas marcas, calculada pelo modelo de IA. 1 = significados idênticos, 0 = sentidos não relacionados. | 0–1 |
| `brand_emb_cosine_normalized` | Igual, mas usando a versão normalizada das marcas (sem pontuação, em minúsculas). Costuma ser mais estável. | 0–1 |
| `brand_emb_norm_max` | Quão "denso" é o significado da marca mais expressiva das duas. Valores altos indicam que pelo menos uma marca tem palavras de dicionário (não é só sigla). | 0–10 |

**Por que importa:** captura colidências que acontecem por proximidade conceitual mesmo sem token em comum (ex.: `BELLA DAMA / D'Bella` recebe sinal positivo mesmo com zero overlap textual).

**Nota técnica:** quando o usuário escolhe "BERTimbau-large STS" no Streamlit, o ganho esperado em PR-AUC é de +1 a +2 pontos percentuais comparado ao MiniLM multilíngue padrão.

---

## 18. Bloco SPEC ITEM-A-ITEM — atividade isolada que coincide (3 colunas, Sprint 2)

A especificação INPI vem como uma lista de atividades separadas por `;`. Comparar essa lista inteira como um único texto **dilui** o sinal quando UMA atividade colide perfeitamente entre dezenas de outras diferentes. Estas 3 colunas resolvem isso quebrando cada lado em itens individuais e medindo a melhor combinação.

| Coluna | O que significa em palavras simples | Faixa |
|---|---|---|
| `spec_item_max_cosine_tfidf` | A maior similaridade encontrada entre **um** item da spec A e **um** item da spec B (varrendo todas as combinações). Captura "uma atividade especifica coincide" mesmo em specs longas. | 0–1 |
| `spec_item_top3_mean_cosine` | Média das **3 maiores** similaridades item-a-item. Robusto a coincidências pontuais: 3 sinais reais valem mais que 1 acaso. | 0–1 |
| `spec_item_align_score` | Alinhamento bipartido ótimo (algoritmo Húngaro) — emparelha cada item de A com no máximo um item de B para maximizar a soma. Mede sobreposição **estrutural** entre as duas listas. | 0–1 |

**Por que importa:** é o vetor mais óbvio de melhoria do modelo Sprint 1. A feature `spec_cosine_tfidf_char` (cosine TF-IDF da spec inteira) já é a feature dominante, mas opera sobre o documento agregado. Estas 3 features ancoram no mesmo TF-IDF mas em granularidade item — capturam o caso clássico em que duas marcas colidem porque uma atividade específica é compartilhada.

---

## 19. Bloco BRAND-AS-ANCHOR — marca aparece na spec do par (2 colunas, Sprint 2)

Caso típico capturado pelo Sprint 1 em FN: `HAPVIDA CLINICA COMANDANTE SAMPAIO` × `HAPVIDA CLINICA DAS MARIAS` — ambas mencionam "HAPVIDA" como marca-mãe e "CLINICA" como atividade. Estas 2 features detectam quando um nome próprio (token informativo) de uma marca **literalmente aparece** dentro da spec da outra.

| Coluna | O que significa em palavras simples | Faixa |
|---|---|---|
| `brand_a_token_in_spec_b` | Quantos tokens informativos (não-genéricos) da marca A aparecem na spec stemizada de B. Contagem inteira. | 0–N |
| `brand_b_token_in_spec_a` | Inverso: tokens informativos de B encontrados na spec de A. | 0–N |

**Por que importa:** detecta franquias, sub-marcas e afiliações onde a marca-mãe é nominalmente próxima mas as descrições de atividade focam em diferenciar as filiais. Sem esta feature, o modelo confundia franquias da mesma rede com marcas diferentes apenas porque a spec descrevia coisas diferentes (endereços, clientes-alvo, etc.).

---

## 20. Bloco PRIOR DE PAR DE CLASSES — estatística empírica do par Nice (2 colunas, Sprint 2)

Substitui (e complementa) os 30+ one-hots por classe específica por **um único par de números densos** que sumarizam, com base em todo o histórico de treino, qual a chance de o par colidir dado as duas classes Nice envolvidas. Calculado **somente no conjunto de treino** para evitar vazamento.

| Coluna | O que significa em palavras simples | Faixa |
|---|---|---|
| `cls_pair_prior_pos` | Probabilidade empírica P(colide \| classe_A, classe_B), calculada no treino com **smoothing bayesiano Beta(2, 8)** para não cair em pares raros. Pares novos caem no prior global. | 0–1 |
| `cls_pair_chi2_strength` | Força estatística da associação entre o par de classes e o rótulo, normalizada para [0, 1]. Pares com chi² alto são "muito acima ou muito abaixo da média" — sinal forte. | 0–1 |

**Por que importa:** o INPI tem combinações de classe que sistematicamente colidem (ex.: 35×39 — comércio + transporte de alimentos) e outras que praticamente nunca colidem. Codificar esse conhecimento como prior empírico libera o modelo de redescobri-lo a partir de one-hots esparsos.

---

## 21. Bloco SUBSTRING & SIMHASH — semelhança contígua e holística (3 colunas, Sprint 2)

Refinamentos sobre a similaridade gráfica clássica.

| Coluna | O que significa em palavras simples | Faixa |
|---|---|---|
| `brand_longest_common_substring_norm` | Tamanho da **maior sequência contígua de letras** comum às duas marcas (sem espaços), dividido pelo tamanho da menor marca. Diferente de LCS por subsequência: aqui as letras precisam estar grudadas. Captura "embed" (`oftal` em `oftalmotop`). | 0–1 |
| `fon_concat_metaphone_eq` | 1 se a chave fonética da marca **inteira sem espaços** for igual nos dois lados; 0 caso contrário. `OFTALMO TOP` vs `OFTALMOTOP` deveria casar. | 0 ou 1 |
| `brand_simhash_hamming_norm` | Similaridade SimHash 64-bit dos char-trigramas das duas marcas. Mais discriminativo que Jaccard simples para textos curtos: 1 = hashes idênticos, 0 = todos os bits opostos. | 0–1 |

**Por que importa:** marcas curtas (4–8 caracteres) têm pouca margem para Levenshtein/Jaro-Winkler distinguirem variações reais de coincidência. SimHash e LCS contígua dão sinais alternativos que o modelo usa principalmente quando a marca é curta. A versão fonética da marca concatenada captura casos em que o usuário separa visualmente palavras que soam grudadas (`SUPER MERCADO` ≈ `supermercado`).

---

## 22. Colunas de SAÍDA do modelo (acrescentadas no enriquecimento)

Após o modelo treinado rodar sobre cada par, a planilha enriquecida ganha estas colunas finais:

| Coluna | O que significa | Faixa |
|---|---|---|
| `score_heuristico_ofta` | Score do algoritmo legado para esse par, em escala **0 a 100** (como aparece na heurística OFTA original). É a referência histórica para comparação. | 0–100 |
| `score_nn` | **Score do modelo de rede neural treinado**: probabilidade contínua entre 0 e 1 de o par ser colidente. Esta é a coluna que vai para produção. | 0–1 |
| `classe_prevista` | Decisão binária: 1 se `score_nn ≥ threshold_usado`, 0 caso contrário. | 0 ou 1 |
| `threshold_usado` | O ponto de corte aplicado. Foi escolhido na validação para garantir **recall ≥ 85%** (capturar ao menos 85% dos colidentes verdadeiros). Tipicamente fica entre 0.20 e 0.40. | 0–1 |
| `timestamp` | Data/hora UTC em que o enriquecimento foi gerado. | ISO-8601 |
| `versao_modelo` | Versão do código que produziu a tabela (rastreabilidade). | Texto |

---

## 23. Como ler uma linha — exemplo prático

Imagine uma linha com:

- `marca_monitorada = "AGROLOG DISTRIBUIDORA"`, `marca_colidente = "AGROLOG"`
- `classe_marca_monitorada = 35`, `classe_marca_colidente = 39`
- `especificacao_monitorado = "Comércio de medicamentos veterinários."`
- `especificacao_colidente = "Armazenagem; Frete; Transporte."`

Você lê na tabela enriquecida:

| Sinais que sobem o risco | Sinais que descem o risco |
|---|---|
| `graf_contains = 1` (AGROLOG está dentro de A) | `cls_same = 0` (classes diferentes: 35 vs 39) |
| `tok_overlap = 1.0` (todas palavras de B estão em A) | `spec_cosine_emb` baixo (medicamentos vet. ≠ logística) |
| `ofta_final ≈ 0.81` (heurística antiga acha colidente) | `inter_nome_alto_e_spec_baixa = 1` (nome bate, atuação distante) |
| `ofta_driver_inclusão = 1` | `spec_lex_jaccard ≈ 0` (zero palavras-chave em comum) |

A **rede aprendeu**, vendo milhares de exemplos rotulados, que quando o conjunto da direita está ativo, o caso de inclusão de termo da esquerda **não basta** para gerar colidência. Resultado: `score_nn ≈ 0.12`, `classe_prevista = 0`. A heurística sozinha geraria um falso positivo aqui; o modelo descontextualiza e acerta.

---

## 24. Glossário de termos técnicos usados

| Termo | Tradução em palavras simples |
|---|---|
| Token | Palavra (depois que o sistema separa a marca em palavras). |
| Stem / estemização | Cortar sufixos para que "comércio", "comerciante", "comercial" virem o mesmo radical "comerc". |
| Bigram / Trigram | Pares e trios de letras consecutivas. Útil para comparar grafias parecidas. |
| Cosine / Cosseno | Métrica matemática que vai de 0 (totalmente diferente) a 1 (totalmente igual) e ignora o tamanho dos textos. |
| TF-IDF | Técnica que dá mais peso a palavras raras e menos a palavras comuns. |
| Embedding semântico | Representação numérica que captura o "significado" de um texto, gerada por um modelo de IA. |
| Levenshtein / Jaro / Jaro-Winkler / Damerau | Famílias de algoritmos clássicos que medem o quanto duas palavras se parecem na grafia. |
| Anagrama | Palavra formada com as mesmas letras de outra, em ordem diferente (ex.: BIONOVA / NOVABIO). |
| Classe Nice | Sistema internacional de 45 classes que organiza marcas por tipo de produto/serviço. |
| Driver (na heurística OFTA) | Indica qual sub-algoritmo "venceu" no cálculo do score do algoritmo legado. |
| Threshold / ponto de corte | Valor a partir do qual o score 0–1 é convertido em decisão "sim, colide" / "não colide". |
| Recall / sensibilidade | Quantos dos colidentes verdadeiros o modelo conseguiu identificar (em %). Falso negativo é o erro mais grave neste projeto. |
| Token genérico | Palavra que aparece em muitas marcas (BURGER, AUTO, INSTITUTO, BRASIL, FASHION...). É detectado automaticamente: aparece em ≥ 1% das marcas. |
| IDF (Inverse Document Frequency) | Quão rara é uma palavra no corpus. IDF alto = palavra rara, mais discriminante (`ANCHOR`); IDF baixo = palavra comum, pouco discriminante (`COMÉRCIO`). |
| Radical morfológico | Núcleo identificável de uma marca (a palavra-raiz mais longa e informativa). Ex.: o radical de `AGRO FRONTEIRA` é `fronteira` (a palavra mais longa e não-genérica). |
| Evidência negativa | Sinal que aponta CONTRA a colidência (ex.: classes Nice diferentes + mercados distantes). Permite ao modelo **desconfirmar** colidências espúrias. |
| Brand embedding | O mesmo modelo de IA usado para entender especificações é aplicado **às próprias marcas**, capturando significado de palavras de dicionário (`SOLAR`, `AGRO`, `BELLA`). |
| Item-level matching (Sprint 2) | Quebra de uma spec longa pelos `;` em itens individuais e comparação **par a par** entre as duas listas. Resolve o problema de uma única atividade coincidente "se perder" no meio de dezenas. |
| Hungarian alignment | Algoritmo clássico de pareamento bipartido ótimo: dadas duas listas e uma matriz de "compatibilidade", encontra o emparelhamento que **maximiza a soma**. Usado em `spec_item_align_score`. |
| Brand-as-anchor (Sprint 2) | Detecção de quando um nome próprio (token informativo) de uma marca **aparece literalmente** na descrição de atividade da outra. Fundamental para detectar franquias e sub-marcas. |
| Prior bayesiano de par de classes (Sprint 2) | Probabilidade empírica de colidência dada cada combinação de classes Nice, calculada no treino com smoothing Beta(2, 8) para não viciar em pares raros. Substitui (densamente) os one-hots esparsos. |
| Smoothing Beta(α, β) | Truque bayesiano para estimar probabilidades a partir de poucos dados. Adiciona α "positivos virtuais" e β "negativos virtuais" antes de calcular a fração — assim pares com poucas observações não geram estimativas extremas. |
| Chi-quadrado (χ²) | Estatística que mede o quanto a distribuição observada se afasta do esperado se as duas variáveis fossem independentes. χ² alto = associação forte (positiva ou negativa); χ² ≈ 0 = sem relação. |
| LCS contígua | Maior **substring** comum: as letras precisam estar grudadas. Diferente de "subsequência", em que pode haver letras intermediárias. `OFTAL` é LCS contígua de `OFTALMOTOP` e `OFTALMICA` (5 letras grudadas). |
| SimHash | Função de hash que produz IDs binários (64 bits, neste projeto) onde **textos parecidos têm hashes parecidos**. A distância de Hamming entre dois SimHashes mede quão diferentes os textos são — útil para textos curtos. |
| Hamming distance | Número de bits diferentes entre dois números binários do mesmo tamanho. Ex.: 0011 e 0101 têm distância de Hamming = 2. |
| Focal Loss | Função de perda que **diminui o peso** dos exemplos fáceis (já bem classificados) e amplifica o peso dos difíceis. Útil quando há muito desbalanceamento — evita que o modelo "se contente" em prever sempre a classe majoritária. |
| Label smoothing | Substitui as etiquetas duras (0 e 1) por suaves (eps e 1-eps). Evita que o modelo se torne **excessivamente confiante** e melhora calibração. |
| Mixup (positivos) | Cria exemplos sintéticos misturando dois positivos do batch (`x = λ*x1 + (1-λ)*x2`, label = 1). Aumenta robustez sem precisar de mais dados rotulados. |
| Hard Negative Mining | A cada época (após algumas iniciais), escolhe os negativos com **maior score atual** (os que o modelo ainda confunde) para forçá-lo a aprender as fronteiras difíceis. |
| Calibração Platt | Treina uma regressão logística simples sobre os scores do modelo na validação para que `P(y=1) = score` seja uma probabilidade real. Modelos podem ranquear bem mas dar probabilidades distorcidas — calibração corrige isso. |
| Ensemble | Treinar **N versões** do modelo com seeds diferentes e usar a média como saída final. Reduz variância e quase sempre melhora levemente o PR-AUC. |
| Two-Tower (cross-attention) | Arquitetura com dois MLPs paralelos — um para features de **nome** e outro para features de **contexto** (spec/classe) — combinados por uma camada de atenção. Permite ao modelo aprender "alta similaridade nominal × distância de mercado = NÃO colide". |
| FT-Transformer | Cada feature contínua vira um pequeno embedding (token); blocos Transformer processam todos os tokens com auto-atenção. Estado-da-arte para dados tabulares em pesquisa recente. |
| Multi-Task Learning | Treinar com várias saídas simultaneamente (a principal + 3 auxiliares). As tarefas auxiliares (ex.: prever `cls_same`, `spec_cosine_emb`) servem como **regularização**, forçando representações mais ricas. |

---

> **Resumo de uma linha:** a tabela enriquecida tem **as 7 colunas originais + 157 features explicativas + 6 colunas de saída do modelo**, totalizando 170 colunas. As 157 features (116 base + 31 Sprint 1 + 10 Sprint 2) capturam, em vários ângulos diferentes (escrita, som, palavras, números, classes, semântica, combinações cruzadas, evidências negativas, prior empírico de classes e item-a-item), o quanto duas marcas se parecem **e** o quanto operam em mercados próximos — e é isso que a rede neural usa para decidir se é colidência ou não.
