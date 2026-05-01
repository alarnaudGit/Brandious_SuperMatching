"""Relatorio final em markdown - versao completa para auditoria externa.

Reune TODAS as informacoes relevantes para que um analista externo possa
entender o estado do modelo e propor melhorias sem acesso ao codigo:
- Sumario executivo + veredito de prontidao para producao
- Configuracao do dataset (estatisticas, hash, distribuicoes)
- Configuracao de pre-processamento e features
- Arquitetura, hiperparametros e historico de treino
- Metricas em treino/val/teste + distribuicao de scores + analise de erros
- Comparativo com a heuristica OFTA
- Importancia de features (permutation importance)
- Limitacoes e recomendacoes
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .data import DatasetReport
from .feature_importance_export import feature_block_for_pruning
from .features.feature_dictionary import attach_descriptions
from .model.dataset import BalancingConfig, SplitConfig
from .model.evaluate import EvalMetrics, recall_at_precision

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers analiticos reaproveitados pela aba de Avaliacao do Streamlit
# =============================================================================

def production_verdict(em: EvalMetrics) -> dict[str, Any]:
    """Avalia uma EvalMetrics e retorna um dict {label, color, wins, yellows, reds, suggestions}.

    Faixas calibradas para o problema (recall positivo prioritario, base ~10% pos).
    """
    reds: list[str] = []
    yellows: list[str] = []
    wins: list[str] = []

    if em.roc_auc >= 0.90:
        wins.append(f"ROC-AUC {em.roc_auc:.3f} (forte capacidade de separar classes)")
    elif em.roc_auc >= 0.80:
        yellows.append(f"ROC-AUC {em.roc_auc:.3f} (aceitavel, mas pode melhorar)")
    else:
        reds.append(f"ROC-AUC {em.roc_auc:.3f} abaixo de 0.80 (separabilidade fraca)")

    if em.pr_auc >= 0.65:
        wins.append(f"PR-AUC {em.pr_auc:.3f} (excelente para base desbalanceada)")
    elif em.pr_auc >= 0.45:
        wins.append(f"PR-AUC {em.pr_auc:.3f} (bom para base desbalanceada)")
    elif em.pr_auc >= 0.30:
        yellows.append(f"PR-AUC {em.pr_auc:.3f} (aceitavel; monitorar precision/recall)")
    else:
        reds.append(f"PR-AUC {em.pr_auc:.3f} muito baixo (perto do baseline aleatorio)")

    if em.recall >= 0.85:
        wins.append(f"Recall {em.recall:.3f} atende o piso operacional de 85%")
    elif em.recall >= 0.70:
        yellows.append(f"Recall {em.recall:.3f} abaixo do piso 0.85 (perde colidentes)")
    else:
        reds.append(f"Recall {em.recall:.3f} muito baixo (>30% colidentes nao capturados)")

    if em.f1 >= 0.45:
        wins.append(f"F1 {em.f1:.3f} (bom equilibrio)")
    elif em.f1 >= 0.30:
        yellows.append(f"F1 {em.f1:.3f} (equilibrio modesto)")
    else:
        reds.append(f"F1 {em.f1:.3f} muito baixo")

    cm = em.confusion
    fn, tp = cm[1][0], cm[1][1]
    if (tp + fn) > 0 and fn / (tp + fn) > 0.30:
        reds.append(
            f"Falsos Negativos: {fn}/{tp + fn} colidentes ({100 * fn / (tp + fn):.1f}%) passariam"
        )

    if reds:
        label, color = "NAO RECOMENDADO para producao", "red"
        suggestions = [
            "aumentar epocas e/ou paciencia do early stopping",
            "ajustar pos_weight (reduzir se ha overshoot, aumentar se recall esta baixo)",
            "revisar features de especificacao (talvez usar embedding PT-BR especifico)",
            "considerar mais dados rotulados ou arquitetura maior",
        ]
    elif yellows:
        label, color = "ACEITAVEL com monitoramento", "yellow"
        suggestions = [
            "monitorar drift de score em producao (PSI mensal)",
            "retreinar a cada novo lote de pareceres revisados",
            "afinar threshold com a area de negocio para custo FN/FP otimo",
        ]
    else:
        label, color = "PRONTO para producao", "green"
        suggestions = [
            f"Threshold de operacao recomendado: {em.threshold:.3f} (recall_floor=0.85)",
            "monitorar drift de score em producao",
        ]

    return {
        "label": label,
        "color": color,
        "wins": wins,
        "yellows": yellows,
        "reds": reds,
        "suggestions": suggestions,
    }


def score_distribution_stats(y_true: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    """Estatisticas descritivas dos scores por classe."""
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]

    def _stats(arr: np.ndarray) -> dict[str, float]:
        if len(arr) == 0:
            return {"n": 0, "mean": 0.0, "std": 0.0, "p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0}
        return {
            "n": int(len(arr)),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "p10": float(np.percentile(arr, 10)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
        }

    return {"pos": _stats(pos), "neg": _stats(neg)}


def score_decile_table(y_true: np.ndarray, scores: np.ndarray) -> list[dict[str, Any]]:
    """Tabela de decis: divide scores em 10 faixas, conta total/positivos/% positivos.

    Util para inspecionar calibracao e onde concentrar atencao operacional.
    """
    if len(scores) == 0:
        return []
    df = pd.DataFrame({"score": scores, "y": y_true})
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["decile"] = pd.qcut(df.index, 10, labels=False, duplicates="drop")
    rows: list[dict[str, Any]] = []
    for d, g in df.groupby("decile"):
        rows.append(
            {
                "decile": int(d),
                "score_min": float(g["score"].min()),
                "score_max": float(g["score"].max()),
                "n": int(len(g)),
                "pos": int(g["y"].sum()),
                "pos_rate": float(g["y"].mean()),
            }
        )
    return rows


_ERROR_COLS = [
    "marca_monitorada", "marca_colidente",
    "classe_marca_monitorada", "classe_marca_colidente",
    "especificacao_monitorado", "especificacao_colidente",
    "score", "y", "pred",
]


def _make_errors_df(
    df: pd.DataFrame,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    if len(df) != len(y_true):
        raise ValueError(
            "df e y_true devem ter mesmo comprimento (alinhados ao split avaliado)."
        )
    work = df.copy().reset_index(drop=True)
    work["score"] = scores
    work["y"] = y_true
    work["pred"] = (scores >= threshold).astype(int)
    return work


def top_errors(
    df: pd.DataFrame,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    k: int = 10,
) -> dict[str, list[dict[str, Any]]]:
    """Identifica top-K Falsos Positivos e Falsos Negativos para inspecao manual."""
    work = _make_errors_df(df, y_true, scores, threshold)
    cols = [c for c in _ERROR_COLS if c in work.columns]

    fp = work[(work["y"] == 0) & (work["pred"] == 1)].sort_values("score", ascending=False).head(k)
    fn = work[(work["y"] == 1) & (work["pred"] == 0)].sort_values("score", ascending=True).head(k)

    return {
        "fp_top": fp[cols].to_dict(orient="records") if not fp.empty else [],
        "fn_top": fn[cols].to_dict(orient="records") if not fn.empty else [],
    }


def all_errors_separated(
    df: pd.DataFrame,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    keep_specs: bool = False,
) -> dict[str, Any]:
    """Retorna TODOS os erros separados por grupo (real=0 e real=1).

    - Grupo 0 (Falsos Positivos): pred=1, y=0; ordenados por score DESC (mais grave primeiro).
    - Grupo 1 (Falsos Negativos): pred=0, y=1; ordenados por score ASC (mais grave primeiro).
    """
    work = _make_errors_df(df, y_true, scores, threshold)
    cols = [c for c in _ERROR_COLS if c in work.columns]
    if not keep_specs:
        cols = [c for c in cols if not c.startswith("especificacao_")]

    fp = work[(work["y"] == 0) & (work["pred"] == 1)].sort_values("score", ascending=False)
    fn = work[(work["y"] == 1) & (work["pred"] == 0)].sort_values("score", ascending=True)

    n_g0_total = int((work["y"] == 0).sum())
    n_g1_total = int((work["y"] == 1).sum())

    return {
        "threshold": float(threshold),
        "n_total": int(len(work)),
        "n_grupo_0_total": n_g0_total,
        "n_grupo_1_total": n_g1_total,
        "n_grupo_0_errados": int(len(fp)),
        "n_grupo_1_errados": int(len(fn)),
        "taxa_erro_grupo_0": float(len(fp) / n_g0_total) if n_g0_total else 0.0,
        "taxa_erro_grupo_1": float(len(fn) / n_g1_total) if n_g1_total else 0.0,
        "grupo_0_errados": fp[cols].to_dict(orient="records") if not fp.empty else [],
        "grupo_1_errados": fn[cols].to_dict(orient="records") if not fn.empty else [],
        "fp_full_df": fp[cols].copy(),
        "fn_full_df": fn[cols].copy(),
    }


def feature_summary_stats(X: np.ndarray, feature_names: Sequence[str], top_k: int = 20) -> dict[str, Any]:
    """Estatisticas globais da matriz de features e top com maior variancia."""
    if X.size == 0:
        return {"n_features": 0, "zero_variance": [], "top_variance": []}

    mean = X.mean(axis=0)
    std = X.std(axis=0)

    zero_var_idx = np.where(std < 1e-9)[0]
    zero_var = [feature_names[i] for i in zero_var_idx]

    rank_idx = np.argsort(-std)
    top = []
    for j in rank_idx[:top_k]:
        top.append(
            {
                "feature": str(feature_names[j]),
                "mean": float(mean[j]),
                "std": float(std[j]),
                "min": float(X[:, j].min()),
                "max": float(X[:, j].max()),
            }
        )
    return {
        "n_features": int(X.shape[1]),
        "n_rows_used": int(X.shape[0]),
        "n_zero_variance": int(len(zero_var)),
        "zero_variance": list(zero_var)[:30],
        "top_variance": top,
    }


def features_correlation_with_label(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """Top-K features por |Pearson| com a label, no conjunto fornecido."""
    if X.size == 0 or len(np.unique(y)) < 2:
        return []
    y_centered = y - y.mean()
    X_centered = X - X.mean(axis=0)
    denom = (np.linalg.norm(X_centered, axis=0) * np.linalg.norm(y_centered) + 1e-12)
    corr = (X_centered.T @ y_centered) / denom
    rank = np.argsort(-np.abs(corr))[:top_k]
    return [
        {"feature": str(feature_names[i]), "pearson": float(corr[i])}
        for i in rank
    ]


def compare_against_heuristic(
    y_true: np.ndarray,
    score_nn: np.ndarray,
    score_heuristic_0_100: np.ndarray,
    target_precision: float = 0.9,
) -> dict[str, float]:
    """Compara NN contra a heuristica OFTA na MESMA particao."""
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


# =============================================================================
# Helpers internos
# =============================================================================

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


def _md_table(rows: list[dict[str, Any]], cols: list[str], col_aliases: dict[str, str] | None = None) -> str:
    if not rows:
        return "_(sem dados)_\n"
    aliases = col_aliases or {}
    head = "| " + " | ".join(aliases.get(c, c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body_lines: list[str] = []
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        body_lines.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep] + body_lines) + "\n"


def _format_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _arch_param_count(mlp_config: dict[str, Any] | None) -> int:
    if not mlp_config:
        return 0
    inp = int(mlp_config.get("input_dim", 0))
    hidden = list(mlp_config.get("hidden_dims", []))
    use_bn = bool(mlp_config.get("use_batchnorm", False))
    total = 0
    prev = inp
    for h in hidden:
        total += prev * h + h
        if use_bn:
            total += 2 * h
        prev = h
    total += prev * 1 + 1
    return int(total)


# =============================================================================
# build_report - relatorio completo
# =============================================================================

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
    # Novos campos opcionais que enriquecem o relatorio (todos default None)
    mlp_config: dict[str, Any] | None = None,
    train_config: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
    preprocessing_summary: dict[str, Any] | None = None,
    feature_stats: dict[str, Any] | None = None,
    feature_corr_with_label: list[dict[str, Any]] | None = None,
    score_stats_test: dict[str, Any] | None = None,
    score_deciles_test: list[dict[str, Any]] | None = None,
    errors_test: dict[str, list[dict[str, Any]]] | None = None,
    all_errors_test: dict[str, Any] | None = None,
    deltas_train_val_test: dict[str, float] | None = None,
    error_csv_paths: dict[str, str] | None = None,
    extra_notes: list[str] | None = None,
) -> str:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    groups = _features_by_group(feature_names)
    verdict = production_verdict(metrics_test)

    lines: list[str] = []
    lines.append("# Relatorio Completo - Modelo de Similaridade Aprendida de Marcas")
    lines.append("")
    lines.append(f"_Gerado em {now} (UTC)._")
    lines.append("")

    # ------------------------------------------------------------------ Sumario Executivo
    lines.append("## Sumario executivo")
    lines.append("")
    em = metrics_test
    lines.append(f"**Veredito de prontidao:** `{verdict['label']}`")
    lines.append("")
    lines.append("| Metrica | Treino | Validacao | Teste |")
    lines.append("| --- | ---:| ---:| ---:|")
    lines.append(
        f"| ROC-AUC | {metrics_train.roc_auc:.4f} | {metrics_val.roc_auc:.4f} | "
        f"**{em.roc_auc:.4f}** |"
    )
    lines.append(
        f"| PR-AUC  | {metrics_train.pr_auc:.4f} | {metrics_val.pr_auc:.4f} | "
        f"**{em.pr_auc:.4f}** |"
    )
    lines.append(
        f"| F1      | {metrics_train.f1:.4f} | {metrics_val.f1:.4f} | "
        f"**{em.f1:.4f}** |"
    )
    lines.append(
        f"| Recall  | {metrics_train.recall:.4f} | {metrics_val.recall:.4f} | "
        f"**{em.recall:.4f}** |"
    )
    lines.append(
        f"| Precision | {metrics_train.precision:.4f} | {metrics_val.precision:.4f} | "
        f"**{em.precision:.4f}** |"
    )
    lines.append("")
    lines.append(f"**Threshold de operacao:** `{threshold_optimal:.4f}` "
                 f"(politica: `{threshold_policy.get('policy')}`, recall_floor={threshold_policy.get('recall_floor')})")
    lines.append("")

    if verdict["wins"]:
        lines.append("**Pontos fortes:**")
        lines.extend(f"- {w}" for w in verdict["wins"])
        lines.append("")
    if verdict["yellows"]:
        lines.append("**Atencao:**")
        lines.extend(f"- {w}" for w in verdict["yellows"])
        lines.append("")
    if verdict["reds"]:
        lines.append("**Bloqueadores:**")
        lines.extend(f"- {w}" for w in verdict["reds"])
        lines.append("")

    if deltas_train_val_test:
        lines.append("**Sinais de overfit (treino - teste):**")
        lines.append("")
        lines.append("| Metrica | delta train-test |")
        lines.append("| --- | ---:|")
        for k, v in deltas_train_val_test.items():
            lines.append(f"| {k} | {v:+.4f} |")
        lines.append("")

    # ------------------------------------------------------------------ 1. Dataset
    lines.append("## 1. Dataset")
    lines.append("")
    lines.append(f"- Linhas validas: **{dataset_report.n_rows}**")
    lines.append(f"- Positivos: **{dataset_report.n_pos}** ({dataset_report.pos_rate*100:.2f}%)")
    lines.append(f"- Negativos: **{dataset_report.n_neg}**")
    lines.append(f"- Linhas removidas (marcas nulas): {dataset_report.n_rows_dropped_null_brands}")
    lines.append(f"- Mesmo classe Nice: {dataset_report.same_class_share*100:.2f}% dos pares")
    lines.append(
        f"- Taxa positivos (mesma classe / classes diferentes): "
        f"{dataset_report.pos_rate_same_class*100:.2f}% / {dataset_report.pos_rate_diff_class*100:.2f}%"
    )
    lines.append(f"- Hash SHA-256: `{dataset_report.dataset_hash}`")
    lines.append("")
    if dataset_report.brand_len_stats:
        lines.append("### 1.1 Comprimento das marcas (caracteres)")
        lines.append("")
        lines.append("| Coluna | mean | p25 | p50 | p75 | max |")
        lines.append("| --- | ---:| ---:| ---:| ---:| ---:|")
        for col, st in dataset_report.brand_len_stats.items():
            lines.append(
                f"| {col} | {st['mean']:.1f} | {st['p25']:.0f} | {st['p50']:.0f} | {st['p75']:.0f} | {st['max']:.0f} |"
            )
        lines.append("")
    if dataset_report.spec_len_stats:
        lines.append("### 1.2 Comprimento das especificacoes (caracteres)")
        lines.append("")
        lines.append("| Coluna | mean | p25 | p50 | p75 | max |")
        lines.append("| --- | ---:| ---:| ---:| ---:| ---:|")
        for col, st in dataset_report.spec_len_stats.items():
            lines.append(
                f"| {col} | {st['mean']:.1f} | {st['p25']:.0f} | {st['p50']:.0f} | {st['p75']:.0f} | {st['max']:.0f} |"
            )
        lines.append("")
    if dataset_report.classes_top:
        lines.append("### 1.3 Top classes Nice mais frequentes")
        lines.append("")
        lines.append("| Classe | Ocorrencias |")
        lines.append("| ---:| ---:|")
        for cl, cnt in dataset_report.classes_top:
            lines.append(f"| {cl} | {cnt} |")
        lines.append("")

    # ------------------------------------------------------------------ 2. Pre-processamento
    if preprocessing_summary:
        lines.append("## 2. Pre-processamento e features")
        lines.append("")
        cfg = preprocessing_summary.get("config", {})
        lines.append(f"- TF-IDF word: ngram={tuple(cfg.get('tfidf_word_ngram_range', []))}, "
                     f"min_df={cfg.get('tfidf_word_min_df')}, max_features={cfg.get('tfidf_word_max_features')}")
        lines.append(f"- TF-IDF char: ngram={tuple(cfg.get('tfidf_char_ngram_range', []))}, "
                     f"min_df={cfg.get('tfidf_char_min_df')}, max_features={cfg.get('tfidf_char_max_features')}")
        lines.append(f"- Embeddings ativos: **{cfg.get('use_embeddings')}** (modelo: `{cfg.get('embedding_model')}`)")
        lines.append(f"- Cache de embeddings: `{cfg.get('embedding_cache_path')}`")
        top_classes = preprocessing_summary.get("top_classes", [])
        if top_classes:
            lines.append(f"- Top-{len(top_classes)} classes Nice usadas em one-hot: {top_classes}")
        lines.append(f"- Total de features na ordem canonica: **{len(feature_names)}**")
        lines.append("")
    else:
        lines.append("## 2. Features criadas")
        lines.append("")
        lines.append(f"Total: **{len(feature_names)}** features na ordem canonica.")
        lines.append("")

    for group, names in groups.items():
        lines.append(f"### {group} ({len(names)})")
        lines.append("")
        lines.append(", ".join(f"`{n}`" for n in names))
        lines.append("")

    # ------------------------------------------------------------------ 3. Estatisticas das features
    if feature_stats:
        lines.append("## 3. Estatisticas das features (apos StandardScaler)")
        lines.append("")
        lines.append(f"- Linhas usadas para estatistica: {feature_stats.get('n_rows_used', '-')}")
        lines.append(f"- Features com variancia ZERO: **{feature_stats.get('n_zero_variance', 0)}** "
                     f"(possiveis candidatas a remocao)")
        zv = feature_stats.get("zero_variance", [])
        if zv:
            lines.append("")
            lines.append("Features sem variancia detectadas: " + ", ".join(f"`{n}`" for n in zv))
        lines.append("")
        top_var = feature_stats.get("top_variance", [])
        if top_var:
            lines.append("### 3.1 Top features por desvio padrao (apos scaler - util p/ debug)")
            lines.append("")
            lines.append(_md_table(
                top_var, ["feature", "mean", "std", "min", "max"],
                {"feature": "Feature", "mean": "media", "std": "desvio", "min": "min", "max": "max"},
            ))

    if feature_corr_with_label:
        lines.append("### 3.2 Top features por |correlacao de Pearson com a label|")
        lines.append("")
        lines.append(_md_table(
            feature_corr_with_label, ["feature", "pearson"],
            {"feature": "Feature", "pearson": "Pearson"},
        ))

    # ------------------------------------------------------------------ 4. Balanceamento + Split
    lines.append("## 4. Estrategia de balanceamento")
    lines.append("")
    _tb = getattr(balancing, "training_balance", "equal")
    if _tb == "equal":
        lines.append(
            "- **Modo treino: 50/50** — positivos (classe 1) reamostrados *com reposicao* ate o numero de "
            "negativos apos undersample (cada batch ve as duas classes com igual cardinalidade).",
        )
        lines.append(
            f"- Undersample negativos: no maximo **{balancing.undersample_neg_ratio:.2f}** negativos por "
            "positivo *original* antes de igualar contagens.",
        )
    else:
        lines.append(f"- Modo **legado**: undersample neg/pos alvo ratio **{balancing.undersample_neg_ratio:.2f} : 1**")
        lines.append(f"- Oversample positivos: fator **{balancing.oversample_pos_factor:.2f}x**")
    lines.append(f"- Class weight (pos_weight) ativo: **{balancing.use_class_weight}**, "
                 f"valor efetivo **{pos_weight_used:.3f}**")
    lines.append(f"- Seed: {balancing.seed}")
    lines.append(
        f"- Split estratificado: train={(1-split_config.test_size-split_config.val_size)*100:.0f}% / "
        f"val={split_config.val_size*100:.0f}% / test={split_config.test_size*100:.0f}%"
    )
    lines.append("")

    # ------------------------------------------------------------------ 5. Arquitetura + Hiperparametros
    if mlp_config or train_config:
        lines.append("## 5. Arquitetura e hiperparametros")
        lines.append("")
        if mlp_config:
            lines.append("### 5.1 Arquitetura")
            lines.append("")
            lines.append(f"- input_dim: **{mlp_config.get('input_dim')}**")
            lines.append(f"- hidden_dims: **{mlp_config.get('hidden_dims')}**")
            lines.append(f"- ativacao: **{mlp_config.get('activation')}**")
            lines.append(f"- dropout: **{mlp_config.get('dropout')}**")
            lines.append(f"- batchnorm: **{mlp_config.get('use_batchnorm')}**")
            lines.append(f"- output: Linear(., 1) -> Sigmoid")
            n_params = _arch_param_count(mlp_config)
            if n_params:
                lines.append(f"- **# parametros aprendiveis: {n_params:,}**")
            lines.append("")
        if train_config:
            lines.append("### 5.2 Hiperparametros de treino")
            lines.append("")
            for k in (
                "epochs", "batch_size", "lr", "weight_decay",
                "early_stopping_patience", "scheduler_patience",
                "scheduler_factor", "min_lr", "grad_clip", "device", "seed",
            ):
                if k in train_config:
                    lines.append(f"- {k}: **{train_config[k]}**")
            for k in ("best_epoch", "best_pr_auc_val", "n_train_after_balancing"):
                if k in train_config:
                    lines.append(f"- {k}: **{train_config[k]}**")
            lines.append("")
            lines.append("- Otimizador: **AdamW**, Loss: **BCEWithLogitsLoss(pos_weight)**, "
                         "Scheduler: **ReduceLROnPlateau** (max em val PR-AUC)")
            lines.append("")

    # ------------------------------------------------------------------ 6. Historico de treino
    if history:
        lines.append("## 6. Historico do treino (epoca a epoca)")
        lines.append("")
        cols_h = [
            "epoch", "train_loss", "lr",
            "train_pr_auc", "val_pr_auc", "val_roc_auc",
            "val_recall@0.5", "val_f1@0.5",
        ]
        present = [c for c in cols_h if c in history[0]]
        lines.append(_md_table(history, present))
        lines.append("")

    # ------------------------------------------------------------------ 7. Metricas detalhadas
    lines.append("## 7. Metricas em detalhe")
    lines.append("")
    lines.append(f"Threshold otimo: **{threshold_optimal:.4f}** "
                 f"(politica: `{threshold_policy.get('policy')}`, recall_floor={threshold_policy.get('recall_floor')})")
    lines.append("")
    lines.append(_format_metric_block("Treino", metrics_train))
    lines.append(_format_metric_block("Validacao", metrics_val))
    lines.append(_format_metric_block("Teste", metrics_test))

    # ------------------------------------------------------------------ 8. Distribuicao de scores no teste
    if score_stats_test:
        lines.append("## 8. Distribuicao de scores no teste")
        lines.append("")
        lines.append("| Classe | n | mean | std | p10 | p25 | p50 | p75 | p90 |")
        lines.append("| --- | ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:|")
        for cls in ("pos", "neg"):
            s = score_stats_test.get(cls, {})
            lines.append(
                f"| {cls} | {s.get('n', 0)} | {s.get('mean', 0):.4f} | {s.get('std', 0):.4f} | "
                f"{s.get('p10', 0):.4f} | {s.get('p25', 0):.4f} | {s.get('p50', 0):.4f} | "
                f"{s.get('p75', 0):.4f} | {s.get('p90', 0):.4f} |"
            )
        lines.append("")
        if score_stats_test.get("pos") and score_stats_test.get("neg"):
            sep = score_stats_test["pos"]["mean"] - score_stats_test["neg"]["mean"]
            lines.append(f"_Diferenca de media (pos - neg): **{sep:+.4f}**. "
                         "Quanto maior, melhor a separabilidade._")
            lines.append("")

    if score_deciles_test:
        lines.append("### 8.1 Decis de score (10 = scores mais altos)")
        lines.append("")
        lines.append(_md_table(
            score_deciles_test,
            ["decile", "score_min", "score_max", "n", "pos", "pos_rate"],
            {"decile": "Decil", "pos": "positivos", "pos_rate": "% positivos"},
        ))

    # ------------------------------------------------------------------ 9. Analise de erros
    if errors_test or all_errors_test:
        lines.append("## 9. Analise de erros (no teste)")
        lines.append("")

        if all_errors_test:
            lines.append(
                f"- Total de pares no teste: **{all_errors_test.get('n_total', 0)}** "
                f"(threshold={all_errors_test.get('threshold', 0):.4f})"
            )
            lines.append(
                f"- **Grupo 0** (rotulo real = 0, NAO colidente): "
                f"{all_errors_test.get('n_grupo_0_total', 0)} pares no total. "
                f"**{all_errors_test.get('n_grupo_0_errados', 0)} foram classificados como colidentes "
                f"(taxa de erro {all_errors_test.get('taxa_erro_grupo_0', 0)*100:.2f}%)**."
            )
            lines.append(
                f"- **Grupo 1** (rotulo real = 1, colidentes): "
                f"{all_errors_test.get('n_grupo_1_total', 0)} pares no total. "
                f"**{all_errors_test.get('n_grupo_1_errados', 0)} escaparam da classificacao "
                f"(taxa de erro {all_errors_test.get('taxa_erro_grupo_1', 0)*100:.2f}%)**."
            )
            if error_csv_paths:
                if "fp_csv" in error_csv_paths:
                    lines.append(
                        f"- Lista completa Grupo 0 errados (CSV): `{error_csv_paths['fp_csv']}`"
                    )
                if "fn_csv" in error_csv_paths:
                    lines.append(
                        f"- Lista completa Grupo 1 errados (CSV): `{error_csv_paths['fn_csv']}`"
                    )
            lines.append("")

        fp_top = errors_test.get("fp_top", []) if errors_test else []
        fn_top = errors_test.get("fn_top", []) if errors_test else []
        cols_short = [
            "marca_monitorada", "marca_colidente",
            "classe_marca_monitorada", "classe_marca_colidente", "score",
        ]
        aliases = {
            "marca_monitorada": "Marca A",
            "marca_colidente": "Marca B",
            "classe_marca_monitorada": "cls A",
            "classe_marca_colidente": "cls B",
        }

        if fp_top:
            lines.append("### 9.1 Top 10 Falsos Positivos (Grupo 0 - alarmes mais graves)")
            lines.append("")
            lines.append(_md_table(fp_top, cols_short, aliases))
        if fn_top:
            lines.append("### 9.2 Top 10 Falsos Negativos (Grupo 1 - escapes mais graves)")
            lines.append("")
            lines.append(_md_table(fn_top, cols_short, aliases))

        if all_errors_test:
            g0 = all_errors_test.get("grupo_0_errados", [])
            g1 = all_errors_test.get("grupo_1_errados", [])

            lines.append(
                "### 9.3 LISTA COMPLETA - Grupo 0 errados "
                f"(rotulo=0, classificados como colidentes) - {len(g0)} pares"
            )
            lines.append("")
            lines.append(
                "_Ordenados por score decrescente (alarmes de maior \"confianca errada\" "
                "primeiro). Cada linha eh um par que o modelo achou que era colidencia mas "
                "NAO era segundo o rotulo INPI._"
            )
            lines.append("")
            lines.append(_md_table(g0, cols_short, aliases))

            lines.append(
                "### 9.4 LISTA COMPLETA - Grupo 1 errados "
                f"(rotulo=1, escaparam da classificacao) - {len(g1)} pares"
            )
            lines.append("")
            lines.append(
                "_Ordenados por score crescente (colidentes que receberam o menor score "
                "primeiro - mais graves para a operacao). Cada linha eh uma colidencia que "
                "o modelo deixou passar._"
            )
            lines.append("")
            lines.append(_md_table(g1, cols_short, aliases))

    # ------------------------------------------------------------------ 10. Comparativo OFTA
    lines.append("## 10. Comparativo NN vs heuristica OFTA (no conjunto de teste)")
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

    # ------------------------------------------------------------------ 11. Importance
    if importance_top:
        lines.append("## 11. Importancia das features (Permutation Importance)")
        lines.append("")
        lines.append("> Quanto maior a importancia, mais a metrica de validacao PIORA quando "
                     "essa feature e embaralhada. Use para auditar onde o modelo esta apoiado.")
        lines.append("")
        lines.append("### 11.1 Top 30 globais")
        lines.append("")
        lines.append(
            "_Lista completa e agregados por bloco para poda da visao enriquecida: "
            "`reports/feature_importance_permutation_full.csv`, "
            "`reports/feature_importance_by_block.json`, "
            "`reports/GUIA_PODA_FEATURES_ENRIQUECIDAS.md` (espelhados em `artifacts/`)._"
        )
        lines.append("")
        lines.append(_md_table(
            attach_descriptions(list(importance_top[:30])),
            ["feature", "importance", "descricao"],
            {
                "feature": "Feature",
                "importance": "Importance",
                "descricao": "Descricao",
            },
        ))

        # Resumo por bloco (alinhado a `feature_importance_by_block.json` / CSV de poda)
        block_sums: dict[str, float] = {}
        for row in importance_top:
            name = str(row["feature"])
            imp = float(row["importance"])
            k = feature_block_for_pruning(name)
            block_sums[k] = block_sums.get(k, 0.0) + max(0.0, imp)
        if block_sums:
            total = sum(block_sums.values()) or 1.0
            rows_blk = sorted(
                [{"bloco": k, "importance_total": v, "share": v / total}
                 for k, v in block_sums.items()],
                key=lambda r: -r["importance_total"],
            )
            lines.append("### 11.2 Importancia agregada por bloco")
            lines.append("")
            lines.append(_md_table(
                rows_blk, ["bloco", "importance_total", "share"],
                {"bloco": "Bloco", "importance_total": "Soma", "share": "Share"},
            ))
    else:
        lines.append("## 11. Importancia das features")
        lines.append("")
        lines.append("> _Permutation importance nao foi calculada antes de gerar este relatorio. "
                     "Volte a aba 5 (Explicabilidade), rode a permutation importance, e gere os "
                     "artefatos novamente para ter esta secao._")
        lines.append("")

    # ------------------------------------------------------------------ 12. Limitacoes / Recomendacoes
    lines.append("## 12. Limitacoes")
    lines.append("")
    lines.append("- O modelo cobre exclusivamente marcas NOMINATIVAS; figurativas precisam de outro pipeline.")
    lines.append("- Qualidade do rotulo INPI determina o teto pratico do modelo.")
    lines.append("- Embeddings sao multilingues genericos; um modelo PT-BR especifico pode melhorar.")
    lines.append("- Classes Nice raras viram bucket 'other' por construcao do top-K.")
    lines.append("- Heuristicas produto/servico baseiam-se em listas-ancora; podem nao cobrir setores especificos.")
    lines.append("")

    lines.append("## 13. Recomendacoes")
    lines.append("")
    if verdict["suggestions"]:
        lines.extend(f"- {s}" for s in verdict["suggestions"])
    lines.append("- Reavaliar `recall_floor` em conjunto com a area de negocio para calibrar custo de FN/FP.")
    lines.append("- Retreinar a cada novo lote significativo de pareceres revisados.")
    lines.append("- Monitorar drift de score em producao (PSI sobre `score_nn`).")
    lines.append("")

    if extra_notes:
        lines.append("## 14. Notas adicionais")
        lines.append("")
        lines.extend(f"- {n}" for n in extra_notes)
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
    "production_verdict",
    "score_distribution_stats",
    "score_decile_table",
    "top_errors",
    "all_errors_separated",
    "feature_summary_stats",
    "features_correlation_with_label",
]
