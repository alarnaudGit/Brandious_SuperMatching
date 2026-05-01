"""Exportacao de permutation importance para auditoria e poda da visao enriquecida.

Gera artefatos legiveis por humanos e por assistentes (CSV completo, JSON agregado,
guia em Markdown) com classificacao por *bloco* de origem da feature.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .features.feature_dictionary import describe_feature

logger = logging.getLogger(__name__)


def feature_block_for_pruning(name: str) -> str:
    """Agrupa cada coluna da matriz enriquecida num bloco para priorizar poda.

    Ordem: prefixos mais especificos primeiro.
    """
    n = str(name)
    if n.startswith("spec_item_"):
        return "Spec_item_TFIDF_alinhamento"
    if n.startswith("spec_cosine_"):
        return "Spec_cossenos_vetoriais"
    if n.startswith("cls_pair_"):
        return "Classe_par_empirico"
    if n.startswith("brand_emb_"):
        return "Marca_embedding"
    if n in ("brand_a_token_in_spec_b", "brand_b_token_in_spec_a"):
        return "Marca_tokens_na_spec_outra"
    if n.startswith("brand_longest_common_substring") or n.startswith("brand_simhash"):
        return "Marca_similaridade_extra"
    if n.startswith("fon_concat_"):
        return "Marca_fonetica_concatenada"
    if n.startswith("graf_"):
        return "Graficas"
    if n.startswith("fon_"):
        return "Foneticas_por_token"
    if n.startswith("ofta_"):
        return "OFTA"
    if n.startswith("num_"):
        return "Numerais"
    if n.startswith("spec_lex_"):
        return "Spec_lexico_tokens"
    if n.startswith("inter_"):
        return "Interacoes"
    if n.startswith("cls_"):
        return "Classe_Nice_onehot"
    if n.startswith("name_generic_") or n.startswith("shared_"):
        return "Extra_nominal_name_tokens"
    if n.startswith("contain_"):
        return "Extra_nominal_contencao"
    if n.startswith("radical_"):
        return "Extra_nominal_radical"
    if n.startswith("lev_pure_"):
        return "Extra_nominal_levenshtein_puro"
    if n.startswith("against_"):
        return "Extra_nominal_evidencia_contra"
    if n.startswith("tok_") or n.startswith("n_tokens"):
        return "Tokens_genericos"
    if n.startswith("spec_"):
        return "Spec_atividade_tipo"
    return "Outras"


def aggregate_by_block(rows: list[dict[str, float]]) -> list[dict[str, Any]]:
    """Soma importancias nao negativas por bloco e devolve linhas ordenadas."""
    sums: dict[str, float] = {}
    by_block: dict[str, list[dict[str, float]]] = {}
    for row in rows:
        name = str(row["feature"])
        imp = float(row.get("importance", 0.0))
        imp = max(0.0, imp)
        blk = feature_block_for_pruning(name)
        sums[blk] = sums.get(blk, 0.0) + imp
        by_block.setdefault(blk, []).append({"feature": name, "importance": float(row["importance"])})

    total = sum(sums.values()) or 1.0
    out: list[dict[str, Any]] = []
    for blk, s in sorted(sums.items(), key=lambda kv: -kv[1]):
        feats = sorted(by_block.get(blk, []), key=lambda r: -max(0.0, float(r["importance"])))[:8]
        out.append(
            {
                "bloco": blk,
                "importancia_total": round(s, 8),
                "share": round(s / total, 6),
                "top_features_no_bloco": feats,
            },
        )
    return out


def _pruning_guide_md() -> str:
    return """# Guia: usar importancia para enxugar a visao enriquecida

## O que significam os numeros

- **importance** (CSV / JSON): queda media do **PR-AUC no conjunto de validacao** (amostra)
  quando essa coluna e **embaralhada** entre linhas, mantendo o resto fixo.
  Quanto **maior**, mais o modelo depende daquela coluna para ranquear colidentes.

## Como reduzir tempo de geracao da visao

1. Abra `feature_importance_permutation_full.csv` (ordem decrescente de importancia).
2. Identifique colunas com importancia **muito baixa** frente ao topo (ex.: abaixo de 1%
  da soma total de importancias positivas, ou abaixo de um limiar absoluto que definirem).
3. Cruze com `feature_importance_by_block.json`: se um **bloco inteiro** tiver share baixa,
  e candidato a desligar no pre-processador (ex.: desativar um tipo de embedding ou um
  grupo de features Sprint 2) em vez de retirar coluna a coluna.
4. **Validar sempre** com um novo treino apos poda: permutation importance e ruidosa
  e correlaciona features; a poda agressiva pode subir ou descer PR-AUC.

## Ficheiros deste pacote

| Ficheiro | Conteudo |
|----------|----------|
| `feature_importance_permutation_full.csv` | Todas as features, rank, bloco, quotas |
| `feature_importance_by_block.json` | Soma por bloco + top features por bloco |
| `GUIA_PODA_FEATURES_ENRIQUECIDAS.md` | Este guia (copia) |

_Contrato de dados_: colunas `feature`, `importance`, `bloco`, `share_of_total_importance`,
`cumulative_share` no CSV estao em UTF-8 com cabecalho em ingles para importacao em ferramentas.
"""


def save_feature_importance_artifacts(
    report_dir: Path,
    project_root: Path,
    rows: list[dict[str, float]] | None,
    *,
    val_sample_n: int,
    n_repeats: int,
    metric: str,
    arch_bagging: bool,
) -> dict[str, str]:
    """Grava CSV, JSON e MD em `report_dir`. Devolve paths relativos a `project_root`."""
    if not rows:
        logger.info("Sem linhas de importance; artefatos de poda nao gravados.")
        return {}

    report_dir.mkdir(parents=True, exist_ok=True)
    total_pos = sum(max(0.0, float(r["importance"])) for r in rows) or 1.0
    cum = 0.0
    records: list[dict[str, Any]] = []
    for i, row in enumerate(rows, start=1):
        name = str(row["feature"])
        imp = float(row["importance"])
        imp_pos = max(0.0, imp)
        share = imp_pos / total_pos
        cum += share
        records.append(
            {
                "rank": i,
                "feature": name,
                "descricao": describe_feature(name),
                "importance_mean_metric_drop": round(imp, 8),
                "bloco": feature_block_for_pruning(name),
                "share_of_total_importance": round(share, 8),
                "cumulative_share": round(min(1.0, cum), 8),
            },
        )

    csv_path = report_dir / "feature_importance_permutation_full.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False, encoding="utf-8-sig")

    block_rows = aggregate_by_block(rows)
    json_payload: dict[str, Any] = {
        "method": "permutation_importance",
        "metric": metric,
        "interpretacao_pt": (
            "Valores maiores = o modelo depende mais da coluna (PR-AUC de validacao "
            "cai mais quando a coluna e embaralhada)."
        ),
        "val_sample_n": int(val_sample_n) if val_sample_n >= 0 else None,
        "n_repeats": int(n_repeats) if n_repeats >= 0 else None,
        "nota_amostra": (
            "Valores calculados neste export (amostra de validacao fixa)."
            if val_sample_n >= 0 and n_repeats >= 0
            else "Importancia foi fornecida antes deste export (ex.: aba Explicabilidade); "
            "amostra e n_repeats podem diferir dos usados no CSV."
        ),
        "arch_bagging": bool(arch_bagging),
        "n_features": len(rows),
        "by_block": block_rows,
        "arquivos": {
            "csv_full": "reports/feature_importance_permutation_full.csv",
            "json_blocks": "reports/feature_importance_by_block.json",
            "guia_md": "reports/GUIA_PODA_FEATURES_ENRIQUECIDAS.md",
        },
    }
    json_path = report_dir / "feature_importance_by_block.json"
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path = report_dir / "GUIA_PODA_FEATURES_ENRIQUECIDAS.md"
    md_path.write_text(_pruning_guide_md(), encoding="utf-8")

    return {
        "feature_importance_permutation_full_csv": str(csv_path.relative_to(project_root)),
        "feature_importance_by_block_json": str(json_path.relative_to(project_root)),
        "guia_poda_features_md": str(md_path.relative_to(project_root)),
    }
