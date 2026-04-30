# Brandious SuperMatching

Projeto para avaliar **similaridade/colisão de marcas** usando o algoritmo em `SimilarityOFTA.py`, com um frontend HTML e uma API local em Flask.

## Estrutura

- `SimilarityOFTA.py`: núcleo do cálculo (função `calcular_score_complexo`).
- `run.py`: servidor Flask que entrega `SimilarityOFTA_Frontend.html` e expõe o endpoint.
- `SimilarityOFTA_Frontend.html`: interface web para testar pares de nomes.
- `dataframe_final_colidencias.xlsx`: base/artefato de dados versionado junto do projeto.

## Requisitos

- Python 3.10+ (recomendado)

## Instalação (Windows / PowerShell)

Na pasta do projeto:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

## Como rodar

Inicie o servidor:

```bash
python run.py
```

Depois abra no navegador:

- `http://127.0.0.1:5000/`

## API

Endpoint:

- `POST /similarity-ofta`

Body (JSON):

```json
{ "nome1": "string", "nome2": "string" }
```

Resposta: um JSON com o score final e detalhes do cálculo (mesmo retorno de `calcular_score_complexo`).

