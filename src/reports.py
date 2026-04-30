"""Relatorio final em markdown: features, balanceamento, metricas e comparativo NN vs OFTA."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .data import DatasetReport
from .model.dataset import BalancingConfig, SplitConfig
from .model.evaluate import EvalMetrics, recall_at_precision

logger = logging.getLogger(__name__)


def _format_metric_block(name: str, m: EvalMetrics) -> str:
    cm = m.confusion
    return (
        f"### {name}\n"
        f"- ROC-AUC: **{m.roc_auc:.4f}**\n"
        f"- PR-AUC:  **{m.pr_auc:.4f}**\n"
        f"- F1:        {m.f1:.4f} (threshold={m.threshold:.3f})\n"
        f"- Precision: {m.precision:.4f}\n"
        f"- Recall:    {m.recall:.4f}\n"
        f"- Confusao: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}\n"
        f"- n_pos={m.n_pos}, n_neg={m.n_neg}\n"
    )


def _features_by_group(feature_names: Iterable[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {
        "Graficas (graf_*)": [],
        "Foneticas (fon_*)": [],
        "Tokens marca (n_tokens_* / tok_*)": [],
        "Numerais (num_*)": [],
        "OFTA (ofta_*)": [],
        "Especificacao lexical (spec_lex_*)": [],
        "Especificacao atividade (spec_kind_* / spec_*activity*)": [],
        "Especificacao cosine (spec_cosine_*)": [],
        "Classe Nice (cls_*)": [],
        "Interacoes (inter_*)": [],
        "Outras": [],
    }
    for name in feature_names:
        if name.startswith("graf_"):
            groups["Graficas (graf_*)"].append(name)
        elif name.startswith("fon_"):
            groups["Foneticas (fon_*)"].append(name)
        elif name.startswith("ofta_"):
            groups["OFTA (ofta_*)"].append(name)
        elif name.startswith("num_"):
            groups["Numerais (num_*)"].append(name)
        elif name.startswith("spec_cosine_"):
            groups["Especificacao cosine (spec_cosine_*)"].append(name)
        elif name.startswith("spec_lex_"):
            groups["Especificacao lexical (spec_lex_*)"].append(name)
        elif name.startswith("spec_"):
            groups["Especificacao atividade (spec_kind_* / spec_*activity*)"].append(name)
        elif name.startswith("cls_"):
            groups["Classe Nice (cls_*)"].append(name)
        elif name.startswith("inter_"):
            groups["Interacoes (inter_*)"].append(name)
        elif name.startswith("tok_") or name.startswith("n_tokens"):
            groups["Tokens marca (n_tokens_* / tok_*)"].append(name)
        else:
            groups["Outras"].append(name)
    return {k: v for k, v in groups.items() if v}


def compare_against_heuristic(
    y_true: np.ndarray,
    score_nn: np.ndarray,
    score_heuristic_0_100: np.ndarray,
    target_precision: float = 0.9,
) -> dict[str, float]:
    """Compara NN contra a heuristica OFTA na MESMA particao (split de teste).

    Para a heuristica usamos o `final` (0-1) ja em score continuo - aqui recebemos a
    versao em escala 0-100 e dividimos por 100.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    score_h = (score_heuristic_0_100 / 100.0).astype(np.float64)

    out = {
        "nn_roc_auc": float(roc_auc_score(y_true, score_nn)) if len(np.unique(y_true)) > 1 else 0.5,
        "nn_pr_auc": float(average_precision_score(y_true, score_nn)),
        "nn_recall_at_p90": float(recall_at_precision(y_true, score_nn, target_precision)),
        "ofta_roc_auc": float(roc_auc_score(y_true, score_h)) if len(np.unique(y_true)) > 1 else 0.5,
        "ofta_pr_auc": float(average_precision_score(y_true, score_h)),
        "ofta_recall_at_p90": float(recall_at_precision(y_true, score_h, target_precision)),
        "target_precision": float(target_precision),
    }
    out["delta_pr_auc"] = out["nn_pr_auc"] - out["ofta_pr_auc"]
    out["delta_roc_auc"] = out["nn_roc_auc"] - out["ofta_roc_auc"]
    out["delta_recall_at_p90"] = out["nn_recall_at_p90"] - out["ofta_recall_at_p90"]
    return out


def build_report(
    *,
    feature_names: list[str],
    balancing: BalancingConfig,
    pos_weight_used: float,
    split_config: SplitConfig,
    metrics_train: EvalMetrics,
    metrics_val: EvalMetrics,
    metrics_test: EvalMetrics,
    comparison: dict[str, float],
    threshold_optimal: float,
    threshold_policy: dict[str, Any],
    dataset_report: DatasetReport,
    importance_top: list[dict[str, float]] | None = None,
) -> str:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    groups = _features_by_group(feature_names)

    lines: list[str] = []
    lines.append("# Relatorio - Modelo de Similaridade Aprendida de Marcas")
    lines.append("")
    lines.append(f"_Gerado em {now} (UTC)._")
    lines.append("")
    lines.append("## 1. Dataset")
    lines.append("")
    lines.append(f"- Linhas validas: **{dataset_report.n_rows}**")
    lines.append(f"- Positivos: **{dataset_report.n_pos}** ({dataset_report.pos_rate*100:.2f}%)")
    lines.append(f"- Negativos: **{dataset_report.n_neg}**")
    lines.append(f"- Linhas removidas (marcas nulas): {dataset_report.n_rows_dropped_null_brands}")
    lines.append(f"- Mesmo classe Nice: {dataset_report.same_class_share*100:.2f}% dos pares")
    lines.append(
        f"- Taxa de positivos (mesma classe / classes diferentes): "
        f"{dataset_report.pos_rate_same_class*100:.2f}% / {dataset_report.pos_rate_diff_class*100:.2f}%"
    )
    lines.append(f"- Hash SHA-256 do arquivo: `{dataset_report.dataset_hash[:24]}...`")
    lines.append("")

    lines.append("## 2. Features criadas")
    lines.append("")
    lines.append(f"Total: **{len(feature_names)}** features na ordem canonica.")
    lines.append("")
    for group, names in groups.items():
        lines.append(f"### {group} ({len(names)})")
        lines.append("")
        lines.append(", ".join(f"`{n}`" for n in names))
        lines.append("")

    lines.append("## 3. Estrategia de balanceamento")
    lines.append("")
    lines.append(f"- Undersample negativos: ratio **{balancing.undersample_neg_ratio:.2f} : 1** (neg/pos alvo)")
    lines.append(f"- Oversample positivos: fator **{balancing.oversample_pos_factor:.2f}x**")
    lines.append(f"- Class weight (pos_weight) ativo: **{balancing.use_class_weight}**, valor efetivo **{pos_weight_used:.3f}**")
    lines.append(f"- Seed: {balancing.seed}")
    lines.append(f"- Split estratificado: train={(1-split_config.test_size-split_config.val_size)*100:.0f}% / val={split_config.val_size*100:.0f}% / test={split_config.test_size*100:.0f}%")
    lines.append("")

    lines.append("## 4. Metricas")
    lines.append("")
    lines.append(f"Threshold otimo: **{threshold_optimal:.3f}** (politica: `{threshold_policy.get('policy')}`, recall_floor={threshold_policy.get('recall_floor')})")
    lines.append("")
    lines.append(_format_metric_block("Treino", metrics_train))
    lines.append(_format_metric_block("Validacao", metrics_val))
    lines.append(_format_metric_block("Teste", metrics_test))

    lines.append("## 5. Comparativo NN vs Heuristica OFTA (no conjunto de teste)")
    lines.append("")
    lines.append(f"_Recall avaliado com piso de Precision >= {comparison.get('target_precision', 0.9):.2f}._")
    lines.append("")
    lines.append("| Metrica | NN | Heuristica OFTA | Delta |")
    lines.append("| --- | ---:| ---:| ---:|")
    lines.append(
        f"| ROC-AUC | {comparison['nn_roc_auc']:.4f} | {comparison['ofta_roc_auc']:.4f} | "
        f"{comparison['delta_roc_auc']:+.4f} |"
    )
    lines.append(
        f"| PR-AUC | {comparison['nn_pr_auc']:.4f} | {comparison['ofta_pr_auc']:.4f} | "
        f"{comparison['delta_pr_auc']:+.4f} |"
    )
    lines.append(
        f"| Recall@P>=0.9 | {comparison['nn_recall_at_p90']:.4f} | {comparison['ofta_recall_at_p90']:.4f} | "
        f"{comparison['delta_recall_at_p90']:+.4f} |"
    )
    lines.append("")

    if importance_top:
        lines.append("## 6. Top features por permutation importance (val)")
        lines.append("")
        lines.append("| # | Feature | Importance |")
        lines.append("| ---: | --- | ---:|")
        for i, row in enumerate(importance_top[:30], start=1):
            lines.append(f"| {i} | `{row['feature']}` | {row['importance']:.5f} |")
        lines.append("")

    lines.append("## 7. Limitacoes")
    lines.append("")
    lines.append("- O modelo cobre exclusivamente marcas NOMINATIVAS; figurativas precisam de outro pipeline.")
    lines.append("- Qualidade do rotulo INPI determina o teto pratico do modelo.")
    lines.append("- Embeddings sao multilingues genericos; um modelo PT-BR especifico pode melhorar.")
    lines.append("- Classes Nice raras viram bucket 'other' por construcao do top-K.")
    lines.append("")
    lines.append("## 8. Recomendacoes")
    lines.append("")
    lines.append("- Reavaliar `recall_floor` em conjunto com a area de negocio para calibrar custo de FN/FP.")
    lines.append("- Retreinar a cada novo lote significativo de pareceres revisados.")
    lines.append("- Monitorar drift de score em producao (PSI sobre `score_nn`).")
    lines.append("- Considerar uma feature dedicada de \"diferenca de mercado\" caso o INPI passe a expor essa info estruturada.")
    lines.append("")

    return "\n".join(lines)


def save_report(text: str, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    logger.info("Relatorio salvo em %s", p)
    return p


__all__ = [
    "build_report",
    "save_report",
    "compare_against_heuristic",
]
