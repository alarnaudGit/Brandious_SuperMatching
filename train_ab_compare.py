"""A/B test entre as 4 arquiteturas do Sprint 2.

Treina, no MESMO split estratificado (seed=42), as variantes:
    1. MLP (baseline) - mantem [128, 64, 32]
    2. Two-Tower + Cross-Attention
    3. FT-Transformer
    4. Multi-Task MLP (4 heads)

Todas usam o conjunto canonico de 157 features (116 base + 31 Sprint 1 +
10 Sprint 2). Cada variante e' treinada com a mesma BalancingConfig e
mesma loss configuravel. O resultado e' uma tabela comparativa em
`artifacts/AB_COMPARISON.md`.

Uso:
    # rapido (~5min): amostra 4000, 5 epocas
    python train_ab_compare.py --quick

    # full (use depois para o relatorio definitivo)
    python train_ab_compare.py --full
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import LABEL_COLUMN_CANON, load_dataset
from src.model.calibration import PlattCalibrator
from src.model.dataset import BalancingConfig, SplitConfig
from src.model.evaluate import (
    compute_metrics_at_threshold,
    find_optimal_threshold,
    predict_scores,
)
from src.model.mlp import MLPConfig
from src.model.train import TrainConfig, train_model
from src.pipeline.preprocessor import FeaturePreprocessor, PreprocessorConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("train_ab_compare")


@dataclass
class VariantSpec:
    name: str
    architecture: str
    hidden_dims: list[int]
    dropout: float
    description: str


VARIANTS = [
    VariantSpec(
        name="MLP_baseline",
        architecture="mlp",
        hidden_dims=[128, 64, 32],
        dropout=0.30,
        description="MLP padrao do Sprint 1 (baseline para comparacao)",
    ),
    VariantSpec(
        name="TwoTower_CA",
        architecture="two_tower",
        hidden_dims=[128, 64],
        dropout=0.30,
        description="Two-Tower com cross-attention (nome vs contexto)",
    ),
    VariantSpec(
        name="FT_Transformer",
        architecture="ft_transformer",
        hidden_dims=[],  # nao usa
        dropout=0.10,
        description="FT-Transformer: 1 token por feature + 2 layers",
    ),
    VariantSpec(
        name="MultiTask_MLP",
        architecture="multitask",
        hidden_dims=[256, 128, 64],
        dropout=0.45,
        description="Backbone unico + 3 heads auxiliares (multi-task)",
    ),
]


def _stratified_split_idx(
    y: np.ndarray, split_cfg: SplitConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split as _tts
    all_idx = np.arange(len(y))
    idx_tv, idx_test = _tts(
        all_idx, test_size=split_cfg.test_size, random_state=split_cfg.seed, stratify=y,
    )
    val_rel = split_cfg.val_size / (1.0 - split_cfg.test_size)
    idx_train, idx_val = _tts(
        idx_tv, test_size=val_rel, random_state=split_cfg.seed, stratify=y[idx_tv],
    )
    return idx_train, idx_val, idx_test


def _train_one_variant(
    variant: VariantSpec,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    feature_names: list[str],
    epochs: int,
    loss_name: str,
    use_calibration: bool,
    recall_floor: float = 0.85,
) -> dict[str, Any]:
    logger.info("============================================================")
    logger.info("Variante: %s (%s)", variant.name, variant.description)
    logger.info("============================================================")

    train_cfg = TrainConfig(
        epochs=epochs,
        batch_size=256,
        lr=1e-3,
        weight_decay=1e-4,
        early_stopping_patience=max(10, epochs // 4),
        seed=42,
        architecture=variant.architecture,
        loss_name=loss_name,
        focal_alpha=0.25,
        focal_gamma=2.0,
        label_smoothing=0.05,
    )
    bal = BalancingConfig(
        undersample_neg_ratio=3.0,
        oversample_pos_factor=2.0,
        training_balance="equal",
        use_class_weight=True,
        seed=42,
    )
    mlp_cfg = MLPConfig(
        input_dim=X_train.shape[1],
        hidden_dims=variant.hidden_dims or [64, 32],
        dropout=variant.dropout,
        use_batchnorm=True,
        activation="relu",
    )

    t0 = time.time()
    model, result = train_model(
        X_train, y_train, X_val, y_val,
        mlp_cfg, train_cfg, bal,
        feature_names=feature_names,
    )
    train_secs = time.time() - t0

    val_scores = predict_scores(model, X_val)
    if use_calibration:
        cal = PlattCalibrator().fit(val_scores, y_val)
        val_scores = cal.transform(val_scores)
    else:
        cal = None

    thr_opt, thr_pol = find_optimal_threshold(y_val, val_scores, recall_floor=recall_floor)

    test_scores = predict_scores(model, X_test)
    if cal is not None:
        test_scores = cal.transform(test_scores)
    em_test = compute_metrics_at_threshold(y_test, test_scores, threshold=thr_opt)

    logger.info(
        "[%s] best_epoch=%d best_val_PRAUC=%.4f | thr_opt=%.3f | "
        "test ROC=%.4f PR-AUC=%.4f Recall=%.3f F1=%.3f Prec=%.3f | %.1fs",
        variant.name, result.best_epoch, result.best_pr_auc_val, thr_opt,
        em_test.roc_auc, em_test.pr_auc, em_test.recall, em_test.f1, em_test.precision,
        train_secs,
    )
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "variant": variant.name,
        "architecture": variant.architecture,
        "description": variant.description,
        "n_params": int(n_params),
        "train_seconds": float(train_secs),
        "best_epoch": int(result.best_epoch),
        "best_pr_auc_val": float(result.best_pr_auc_val),
        "threshold_opt": float(thr_opt),
        "threshold_policy": thr_pol,
        "test_roc_auc": float(em_test.roc_auc),
        "test_pr_auc": float(em_test.pr_auc),
        "test_f1": float(em_test.f1),
        "test_precision": float(em_test.precision),
        "test_recall": float(em_test.recall),
        "test_confusion": em_test.confusion,
        "history": result.history,
    }


def _render_markdown(
    results: list[dict[str, Any]],
    sample_size: int,
    epochs: int,
    n_features: int,
    loss_name: str,
    use_calibration: bool,
) -> str:
    lines: list[str] = []
    lines.append("# Sprint 2 - Comparativo de arquiteturas (A/B)\n")
    lines.append(f"_Geracao_: {pd.Timestamp.utcnow().isoformat()}Z\n")
    lines.append("## Configuracao do experimento\n")
    lines.append(f"- Amostra de treino+val+test: **{sample_size:,}** linhas")
    lines.append(f"- Epocas maximas por variante: **{epochs}**")
    lines.append(f"- Numero de features (canonical): **{n_features}**")
    lines.append(f"- Loss usada em todas: `{loss_name}`")
    lines.append(f"- Calibracao Platt aplicada: `{use_calibration}`")
    lines.append("- Split: 70/15/15 estratificado, seed=42")
    lines.append("- Balanceamento: undersample 3:1 + oversample 2x + class_weight\n")

    ranked = sorted(results, key=lambda r: r["test_pr_auc"], reverse=True)

    lines.append("## Tabela comparativa (ordenada por PR-AUC test)\n")
    lines.append("| Pos | Variante | Params | Treino (s) | Best epoch | Val PR-AUC | Test ROC-AUC | **Test PR-AUC** | Test F1 | Test Prec | Test Recall |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(ranked, start=1):
        lines.append(
            f"| {i} | **{r['variant']}** | {r['n_params']:,} | {r['train_seconds']:.1f} | "
            f"{r['best_epoch']} | {r['best_pr_auc_val']:.4f} | {r['test_roc_auc']:.4f} | "
            f"**{r['test_pr_auc']:.4f}** | {r['test_f1']:.4f} | "
            f"{r['test_precision']:.4f} | {r['test_recall']:.4f} |"
        )
    lines.append("")

    lines.append("## Veredito\n")
    winner = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    delta_pr = (
        winner['test_pr_auc'] - runner_up['test_pr_auc']
        if runner_up else 0.0
    )
    lines.append(
        f"**Melhor arquitetura: `{winner['variant']}`** com PR-AUC = "
        f"`{winner['test_pr_auc']:.4f}` no test set."
    )
    if runner_up:
        lines.append(
            f"\nDelta sobre o segundo (`{runner_up['variant']}`): "
            f"**{delta_pr:+.4f}** pp em PR-AUC.\n"
        )
        if abs(delta_pr) < 0.01:
            lines.append(
                "> A diferenca e' MENOR que 1pp em PR-AUC - considere usar **ensemble** "
                "das duas para reduzir variancia. "
                f"Tradeoff: `{winner['variant']}` tem {winner['n_params']:,} params vs "
                f"`{runner_up['variant']}` com {runner_up['n_params']:,} params.\n"
            )
        else:
            lines.append(
                f"> Delta razoavel - usar `{winner['variant']}` em producao.\n"
            )

    lines.append("\n## Detalhamento por variante\n")
    for r in ranked:
        lines.append(f"### {r['variant']} ({r['architecture']})\n")
        lines.append(f"_{r['description']}_\n")
        lines.append(f"- N parametros: `{r['n_params']:,}`")
        lines.append(f"- Tempo de treino: `{r['train_seconds']:.1f}s`")
        lines.append(f"- Melhor epoca (val PR-AUC): `{r['best_epoch']}` -> `{r['best_pr_auc_val']:.4f}`")
        lines.append(f"- Threshold otimizado: `{r['threshold_opt']:.3f}`")
        lines.append("- Matriz de confusao (test):")
        lines.append("")
        cm = r["test_confusion"]
        lines.append("  | | Pred 0 | Pred 1 |")
        lines.append("  |---|---:|---:|")
        lines.append(f"  | Real 0 | {cm[0][0]} | {cm[0][1]} |")
        lines.append(f"  | Real 1 | {cm[1][0]} | {cm[1][1]} |")
        lines.append("")

    lines.append("\n## Como reproduzir\n")
    lines.append("```bash\n# Quick (~5 min, amostra reduzida):\npython train_ab_compare.py --quick\n\n# Full (rodar overnight):\npython train_ab_compare.py --full\n```\n")
    return "\n".join(lines)


def main(
    sample_size: int | None = None,
    epochs: int = 30,
    use_embeddings: bool = False,
    loss_name: str = "focal_smoothing",
    use_calibration: bool = True,
    out_dir: str = "artifacts",
) -> None:
    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    logger.info("Carregando dataset completo...")
    df_full, _ = load_dataset("dataframe_final_colidencias.xlsx")

    if sample_size and sample_size < len(df_full):
        pos = df_full[df_full[LABEL_COLUMN_CANON] == 1]
        neg = df_full[df_full[LABEL_COLUMN_CANON] == 0]
        n_pos = min(int(sample_size * 0.20), len(pos))
        n_neg = min(sample_size - n_pos, len(neg))
        df = pd.concat([
            pos.sample(n=n_pos, random_state=42),
            neg.sample(n=n_neg, random_state=42),
        ]).sample(frac=1.0, random_state=42).reset_index(drop=True)
        logger.info(
            "Amostra reduzida: %d linhas (%d pos, %d neg)", len(df), n_pos, n_neg,
        )
    else:
        df = df_full
        logger.info("Usando dataset completo: %d linhas", len(df))

    pre_cfg = PreprocessorConfig(
        tfidf_word_max_features=20_000,
        tfidf_char_max_features=10_000,
        top_k_classes=15,
        use_embeddings=use_embeddings,
        use_brand_embeddings=use_embeddings,
        embedding_cache_path="artifacts/embeddings_cache.parquet",
        brand_embedding_cache_path="artifacts/embeddings_brand_cache.parquet",
    )
    preproc = FeaturePreprocessor(pre_cfg)
    preproc.fit(df)
    X = preproc.transform(df, scale=True)
    y = df[LABEL_COLUMN_CANON].to_numpy().astype(np.int64)
    feature_names = list(preproc.feature_names_ordered)
    logger.info("Matriz de features: %s (%d features)", X.shape, X.shape[1])

    split_cfg = SplitConfig(test_size=0.15, val_size=0.15, seed=42)
    idx_tr, idx_va, idx_te = _stratified_split_idx(y, split_cfg)
    X_tr, y_tr = X[idx_tr], y[idx_tr]
    X_va, y_va = X[idx_va], y[idx_va]
    X_te, y_te = X[idx_te], y[idx_te]

    results: list[dict[str, Any]] = []
    for variant in VARIANTS:
        try:
            r = _train_one_variant(
                variant,
                X_tr, y_tr, X_va, y_va, X_te, y_te,
                feature_names=feature_names,
                epochs=epochs, loss_name=loss_name,
                use_calibration=use_calibration,
            )
            results.append(r)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Variante %s falhou: %s", variant.name, exc)
            results.append({
                "variant": variant.name,
                "architecture": variant.architecture,
                "description": variant.description,
                "error": str(exc),
                "n_params": 0,
                "train_seconds": 0.0,
                "best_epoch": 0,
                "best_pr_auc_val": 0.0,
                "threshold_opt": 0.5,
                "threshold_policy": {},
                "test_roc_auc": 0.0,
                "test_pr_auc": 0.0,
                "test_f1": 0.0,
                "test_precision": 0.0,
                "test_recall": 0.0,
                "test_confusion": [[0, 0], [0, 0]],
                "history": [],
            })

    md = _render_markdown(
        results, sample_size=len(df), epochs=epochs, n_features=X.shape[1],
        loss_name=loss_name, use_calibration=use_calibration,
    )
    md_path = out / "AB_COMPARISON.md"
    md_path.write_text(md, encoding="utf-8")
    logger.info("Comparativo salvo em %s", md_path)

    json_path = out / "AB_COMPARISON.json"
    with json_path.open("w", encoding="utf-8") as f:
        for r in results:
            r.pop("history", None)
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("JSON detalhado salvo em %s", json_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true", help="Amostra 4000 + 5 epocas")
    g.add_argument("--full", action="store_true", help="Dataset completo + 50 epocas")
    parser.add_argument("--sample", type=int, default=None, help="Amostra customizada")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--emb", action="store_true", help="Usar embeddings semanticos")
    parser.add_argument(
        "--loss", type=str, default="focal_smoothing",
        choices=["bce", "focal", "label_smoothing", "focal_smoothing"],
    )
    parser.add_argument("--no-calibrate", action="store_true")
    args = parser.parse_args()

    if args.quick:
        sample, epochs = 4000, 5
    elif args.full:
        sample, epochs = None, 50
    else:
        sample, epochs = args.sample, args.epochs

    main(
        sample_size=sample,
        epochs=epochs,
        use_embeddings=args.emb,
        loss_name=args.loss,
        use_calibration=not args.no_calibrate,
    )
