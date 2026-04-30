"""Smoke test end-to-end do pipeline.

Carrega uma amostra reduzida (estratificada) do dataset, gera features (sem
embeddings - para velocidade), treina por poucas epocas, salva artefatos e
realiza uma inferencia, verificando consistencia entre o score do modelo
salvo e o score reconstruido a partir do JSON.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.artifacts import (
    build_model_config_dict,
    save_enriched_dataframe,
    save_model_config,
    save_state_dict,
)
from src.data import LABEL_COLUMN_CANON, load_dataset
from src.model.dataset import BalancingConfig, SplitConfig, stratified_split
from src.model.evaluate import (
    compute_metrics_at_threshold,
    find_optimal_threshold,
    predict_scores,
)
from src.model.mlp import MLPConfig
from src.model.train import TrainConfig, train_model
from src.normalize import calcular_similaridade_ofta
from src.pipeline.inference import load_artifacts, score_pair
from src.pipeline.preprocessor import FeaturePreprocessor, PreprocessorConfig
from src.reports import build_report, compare_against_heuristic, save_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("smoke_test")


def main(sample_size: int = 1500, epochs: int = 4, use_embeddings: bool = False) -> None:
    t0 = time.time()
    logger.info("=== SMOKE TEST INICIADO (sample=%d, epochs=%d, emb=%s) ===", sample_size, epochs, use_embeddings)

    df_full, report = load_dataset("dataframe_final_colidencias.xlsx")

    pos = df_full[df_full[LABEL_COLUMN_CANON] == 1]
    neg = df_full[df_full[LABEL_COLUMN_CANON] == 0]
    n_pos = min(int(sample_size * 0.2), len(pos))
    n_neg = min(sample_size - n_pos, len(neg))
    df = pd.concat([
        pos.sample(n=n_pos, random_state=42),
        neg.sample(n=n_neg, random_state=42),
    ]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    logger.info("Amostra reduzida: %d linhas (%d pos, %d neg)", len(df), n_pos, n_neg)

    pre_cfg = PreprocessorConfig(
        tfidf_word_max_features=3000,
        tfidf_char_max_features=2000,
        top_k_classes=10,
        use_embeddings=use_embeddings,
        embedding_cache_path="artifacts/embeddings_cache_smoke.parquet",
    )
    preproc = FeaturePreprocessor(pre_cfg)
    preproc.fit(df)
    X = preproc.transform(df, scale=True)
    y = df[LABEL_COLUMN_CANON].to_numpy().astype(np.int64)
    logger.info("Matriz de features: %s", X.shape)

    split_cfg = SplitConfig(test_size=0.2, val_size=0.2, seed=42)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = stratified_split(X, y, split_cfg)

    bal = BalancingConfig(undersample_neg_ratio=2.5, oversample_pos_factor=2.0, seed=42)
    train_cfg = TrainConfig(epochs=epochs, batch_size=64, lr=1e-3, early_stopping_patience=20, seed=42)
    mlp_cfg = MLPConfig(input_dim=X.shape[1], hidden_dims=[64, 32], dropout=0.2, use_batchnorm=True)

    model, result = train_model(X_tr, y_tr, X_va, y_va, mlp_cfg, train_cfg, bal)
    logger.info("Treino: best_epoch=%d, best_val_PRAUC=%.4f", result.best_epoch, result.best_pr_auc_val)

    val_scores = predict_scores(model, X_va)
    thr_opt, thr_pol = find_optimal_threshold(y_va, val_scores, recall_floor=0.85)

    test_scores = predict_scores(model, X_te)
    em_test = compute_metrics_at_threshold(y_te, test_scores, threshold=thr_opt)
    em_train = compute_metrics_at_threshold(y_tr, predict_scores(model, X_tr), threshold=thr_opt)
    em_val = compute_metrics_at_threshold(y_va, val_scores, threshold=thr_opt)
    logger.info(
        "Test (thr=%.3f): ROC=%.4f PR-AUC=%.4f Recall=%.3f F1=%.3f",
        thr_opt, em_test.roc_auc, em_test.pr_auc, em_test.recall, em_test.f1,
    )

    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    preproc.save(out / "preprocessor_smoke.pkl")
    save_state_dict(model.state_dict(), out / "model_smoke.pt")

    cfg_json = build_model_config_dict(
        mlp_config=mlp_cfg,
        preprocessor=preproc,
        train_config=train_cfg,
        balancing=bal,
        split_config=split_cfg,
        threshold_optimal=thr_opt,
        threshold_policy=thr_pol,
        metrics_train=em_train,
        metrics_val=em_val,
        metrics_test=em_test,
        train_result=result,
        dataset_report=report,
        state_dict=model.state_dict(),
        embedding_used=pre_cfg.use_embeddings,
    )
    save_model_config(cfg_json, out / "brand_similarity_model_config_smoke.json")

    score_nn_full = predict_scores(model, X)
    score_h_full = np.array(
        [calcular_similaridade_ofta(a, b) for a, b in zip(df["marca_monitorada"].astype(str), df["marca_colidente"].astype(str))],
        dtype=np.float32,
    )
    save_enriched_dataframe(
        df_original=df,
        feature_matrix=X,
        feature_names=list(preproc.feature_names_ordered),
        score_nn=score_nn_full,
        score_heuristic=score_h_full,
        threshold=thr_opt,
        out_xlsx=out / "brand_similarity_input_view_enriched_smoke.xlsx",
        out_parquet=out / "enriched_smoke.parquet",
        label_col=LABEL_COLUMN_CANON,
    )

    comp = compare_against_heuristic(y_te, test_scores, score_h_full[: len(y_te)])
    logger.info("Comparativo NN vs OFTA (test): %s", comp)

    report_md = build_report(
        feature_names=list(preproc.feature_names_ordered),
        balancing=bal,
        pos_weight_used=result.pos_weight_used,
        split_config=split_cfg,
        metrics_train=em_train,
        metrics_val=em_val,
        metrics_test=em_test,
        comparison=comp,
        threshold_optimal=thr_opt,
        threshold_policy=thr_pol,
        dataset_report=report,
        importance_top=None,
    )
    save_report(report_md, out / "report_smoke.md")

    artifacts = load_artifacts(
        out / "brand_similarity_model_config_smoke.json",
        out / "preprocessor_smoke.pkl",
        out / "model_smoke.pt",
    )
    sample_pair = df.iloc[0]
    res = score_pair(
        artifacts,
        marca_a=str(sample_pair["marca_monitorada"]),
        marca_b=str(sample_pair["marca_colidente"]),
        classe_a=int(sample_pair["classe_marca_monitorada"]),
        classe_b=int(sample_pair["classe_marca_colidente"]),
        spec_a=str(sample_pair["especificacao_monitorado"]),
        spec_b=str(sample_pair["especificacao_colidente"]),
    )
    expected = float(score_nn_full[0])
    diff = abs(res["score_nn"] - expected)
    logger.info("Inferencia: score_nn=%.6f vs esperado=%.6f (diff=%.6f)", res["score_nn"], expected, diff)
    if diff > 1e-3:
        logger.warning("[!] Diferenca grande entre inferencia salva e em-memoria.")

    logger.info("=== SMOKE TEST OK em %.1fs ===", time.time() - t0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--emb", action="store_true", help="Usar embeddings semanticos")
    args = parser.parse_args()
    main(sample_size=args.sample, epochs=args.epochs, use_embeddings=args.emb)
