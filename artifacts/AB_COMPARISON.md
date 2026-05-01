# Sprint 2 - Comparativo de arquiteturas (A/B)

_Geracao_: 2026-05-01T00:22:34.068330+00:00Z

## Configuracao do experimento

- Amostra de treino+val+test: **4,000** linhas
- Epocas maximas por variante: **5**
- Numero de features (canonical): **157**
- Loss usada em todas: `focal_smoothing`
- Calibracao Platt aplicada: `True`
- Split: 70/15/15 estratificado, seed=42
- Balanceamento: undersample 3:1 + oversample 2x + class_weight

## Tabela comparativa (ordenada por PR-AUC test)

| Pos | Variante | Params | Treino (s) | Best epoch | Val PR-AUC | Test ROC-AUC | **Test PR-AUC** | Test F1 | Test Prec | Test Recall |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **MLP_baseline** | 31,041 | 3.5 | 5 | 0.4878 | 0.8061 | **0.5230** | 0.4720 | 0.3279 | 0.8417 |
| 2 | **TwoTower_CA** | 96,385 | 2.5 | 5 | 0.4730 | 0.7771 | **0.4947** | 0.4569 | 0.3172 | 0.8167 |
| 3 | **MultiTask_MLP** | 82,756 | 1.6 | 5 | 0.4653 | 0.7760 | **0.4847** | 0.4417 | 0.2944 | 0.8833 |
| 4 | **FT_Transformer** | 11,953 | 57.1 | 5 | 0.4781 | 0.7876 | **0.4815** | 0.4730 | 0.3241 | 0.8750 |

## Veredito

**Melhor arquitetura: `MLP_baseline`** com PR-AUC = `0.5230` no test set.

Delta sobre o segundo (`TwoTower_CA`): **+0.0283** pp em PR-AUC.

> Delta razoavel - usar `MLP_baseline` em producao.


## Detalhamento por variante

### MLP_baseline (mlp)

_MLP padrao do Sprint 1 (baseline para comparacao)_

- N parametros: `31,041`
- Tempo de treino: `3.5s`
- Melhor epoca (val PR-AUC): `5` -> `0.4878`
- Threshold otimizado: `0.140`
- Matriz de confusao (test):

  | | Pred 0 | Pred 1 |
  |---|---:|---:|
  | Real 0 | 273 | 207 |
  | Real 1 | 19 | 101 |

### TwoTower_CA (two_tower)

_Two-Tower com cross-attention (nome vs contexto)_

- N parametros: `96,385`
- Tempo de treino: `2.5s`
- Melhor epoca (val PR-AUC): `5` -> `0.4730`
- Threshold otimizado: `0.140`
- Matriz de confusao (test):

  | | Pred 0 | Pred 1 |
  |---|---:|---:|
  | Real 0 | 269 | 211 |
  | Real 1 | 22 | 98 |

### MultiTask_MLP (multitask)

_Backbone unico + 3 heads auxiliares (multi-task)_

- N parametros: `82,756`
- Tempo de treino: `1.6s`
- Melhor epoca (val PR-AUC): `5` -> `0.4653`
- Threshold otimizado: `0.120`
- Matriz de confusao (test):

  | | Pred 0 | Pred 1 |
  |---|---:|---:|
  | Real 0 | 226 | 254 |
  | Real 1 | 14 | 106 |

### FT_Transformer (ft_transformer)

_FT-Transformer: 1 token por feature + 2 layers_

- N parametros: `11,953`
- Tempo de treino: `57.1s`
- Melhor epoca (val PR-AUC): `5` -> `0.4781`
- Threshold otimizado: `0.130`
- Matriz de confusao (test):

  | | Pred 0 | Pred 1 |
  |---|---:|---:|
  | Real 0 | 261 | 219 |
  | Real 1 | 15 | 105 |


## Como reproduzir

```bash
# Quick (~5 min, amostra reduzida):
python train_ab_compare.py --quick

# Full (rodar overnight):
python train_ab_compare.py --full
```
