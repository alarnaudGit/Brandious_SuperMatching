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
from src.model.explain import permutation_importance
from src.normalize import calcular_similaridade_ofta
from src.pipeline.inference import load_artifacts, score_pair
from src.pipeline.preprocessor import FeaturePreprocessor, PreprocessorConfig
from src.reports import (
    all_errors_separated,
    build_report,
    compare_against_heuristic,
    feature_summary_stats,
    features_correlation_with_label,
    save_report,
    score_decile_table,
    score_distribution_stats,
    top_errors,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("smoke_test")


def main(
    sample_size: int = 1500,
    epochs: int = 4,
    use_embeddings: bool = False,
    architecture: str = "mlp",
    loss_name: str = "bce",
    n_seeds: int = 1,
    use_calibration: bool = False,
) -> None:
    t0 = time.time()
    logger.info(
        "=== SMOKE TEST INICIADO (sample=%d, epochs=%d, emb=%s, arch=%s, loss=%s, seeds=%d, cal=%s) ===",
        sample_size, epochs, use_embeddings, architecture, loss_name, n_seeds, use_calibration,
    )

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
        use_brand_embeddings=use_embeddings,  # se emb ligado, brand_emb tambem
        brand_embedding_cache_path="artifacts/embeddings_brand_cache_smoke.parquet",
        generic_df_threshold=0.01,
    )
    preproc = FeaturePreprocessor(pre_cfg)
    preproc.fit(df)
    X = preproc.transform(df, scale=True)
    y = df[LABEL_COLUMN_CANON].to_numpy().astype(np.int64)
    logger.info("Matriz de features: %s", X.shape)

    split_cfg = SplitConfig(test_size=0.2, val_size=0.2, seed=42)
    from sklearn.model_selection import train_test_split as _tts
    all_idx = np.arange(len(y))
    idx_tv, idx_te = _tts(all_idx, test_size=split_cfg.test_size, random_state=split_cfg.seed, stratify=y)
    val_rel = split_cfg.val_size / (1.0 - split_cfg.test_size)
    idx_tr, idx_va = _tts(idx_tv, test_size=val_rel, random_state=split_cfg.seed, stratify=y[idx_tv])
    X_tr, y_tr = X[idx_tr], y[idx_tr]
    X_va, y_va = X[idx_va], y[idx_va]
    X_te, y_te = X[idx_te], y[idx_te]

    bal = BalancingConfig(
        undersample_neg_ratio=2.5,
        oversample_pos_factor=2.0,
        training_balance="equal",
        seed=42,
    )
    train_cfg = TrainConfig(
        epochs=epochs, batch_size=64, lr=1e-3, early_stopping_patience=20, seed=42,
        architecture=architecture, loss_name=loss_name,
    )
    mlp_cfg = MLPConfig(
        input_dim=X.shape[1],
        hidden_dims=[256, 128, 64] if architecture == "multitask" else [64, 32],
        dropout=0.45 if architecture == "multitask" else 0.2,
        use_batchnorm=True,
    )

    feature_names = list(preproc.feature_names_ordered)

    if n_seeds > 1:
        from src.model.ensemble import train_ensemble, predict_ensemble
        seeds = [42 + i * 1000 for i in range(n_seeds)]
        members = train_ensemble(
            X_tr, y_tr, X_va, y_va,
            mlp_cfg, train_cfg, bal,
            seeds=seeds, feature_names=feature_names,
            calibrate=use_calibration,
        )
        model = members[0].model
        result = type("EnsR", (), {
            "best_epoch": members[0].best_epoch,
            "best_pr_auc_val": float(np.mean([m.best_pr_auc_val for m in members])),
            "history": members[0].history,
            "pos_weight_used": 1.0,
            "n_train_after_balancing": 0,
        })()
        val_scores = predict_ensemble(members, X_va, apply_calibration=use_calibration)
        test_scores = predict_ensemble(members, X_te, apply_calibration=use_calibration)
    else:
        model, result = train_model(
            X_tr, y_tr, X_va, y_va, mlp_cfg, train_cfg, bal,
            feature_names=feature_names,
        )
        val_scores = predict_scores(model, X_va)
        test_scores = predict_scores(model, X_te)
        if use_calibration:
            from src.model.calibration import PlattCalibrator
            cal = PlattCalibrator().fit(val_scores, y_va)
            val_scores = cal.transform(val_scores)
            test_scores = cal.transform(test_scores)
            cal.save(Path("artifacts") / "calibrator_smoke.json")
    logger.info("Treino: best_epoch=%d, best_val_PRAUC=%.4f", result.best_epoch, result.best_pr_auc_val)

    thr_opt, thr_pol = find_optimal_threshold(y_va, val_scores, recall_floor=0.85)

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

    arch_cfg = (
        model.config if hasattr(model, "config") else (
            model.cfg if hasattr(model, "cfg") else mlp_cfg
        )
    )
    cfg_json = build_model_config_dict(
        mlp_config=arch_cfg,
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

    logger.info("Calculando permutation importance (amostra=400, repeats=2)...")
    sample_n = min(400, len(y_va))
    rng = np.random.default_rng(42)
    pi_idx = rng.choice(len(y_va), size=sample_n, replace=False)
    importance_top = permutation_importance(
        model, X_va[pi_idx], y_va[pi_idx], list(preproc.feature_names_ordered),
        metric="pr_auc", n_repeats=2,
    )

    score_stats = score_distribution_stats(y_te, test_scores)
    score_deciles = score_decile_table(y_te, test_scores)
    df_test_aligned = df.iloc[idx_te].copy().reset_index(drop=True)
    errors_dict = top_errors(df_test_aligned, y_te, test_scores, thr_opt, k=10)
    all_errors = all_errors_separated(df_test_aligned, y_te, test_scores, thr_opt, keep_specs=False)
    fp_csv = out / "falsos_positivos_grupo_0_smoke.csv"
    fn_csv = out / "falsos_negativos_grupo_1_smoke.csv"
    all_errors["fp_full_df"].to_csv(fp_csv, index=False, encoding="utf-8-sig")
    all_errors["fn_full_df"].to_csv(fn_csv, index=False, encoding="utf-8-sig")
    error_csv_paths = {
        "fp_csv": str(fp_csv),
        "fn_csv": str(fn_csv),
    }
    logger.info(
        "Erros no teste: Grupo 0 errados=%d (taxa %.2f%%), Grupo 1 errados=%d (taxa %.2f%%)",
        all_errors["n_grupo_0_errados"], all_errors["taxa_erro_grupo_0"] * 100,
        all_errors["n_grupo_1_errados"], all_errors["taxa_erro_grupo_1"] * 100,
    )
    feat_stats = feature_summary_stats(X, list(preproc.feature_names_ordered), top_k=20)
    feat_corr = features_correlation_with_label(
        X, df[LABEL_COLUMN_CANON].to_numpy().astype(np.float32),
        list(preproc.feature_names_ordered), top_k=20,
    )
    preproc_summary = {
        "config": preproc.config.to_dict(),
        "top_classes": list(preproc.top_classes),
    }
    deltas = {
        "ROC-AUC train-test": em_train.roc_auc - em_test.roc_auc,
        "PR-AUC train-test": em_train.pr_auc - em_test.pr_auc,
        "Recall train-test": em_train.recall - em_test.recall,
        "F1 train-test": em_train.f1 - em_test.f1,
    }
    train_cfg_dict = train_cfg.to_dict()
    train_cfg_dict.update(
        best_epoch=result.best_epoch,
        best_pr_auc_val=result.best_pr_auc_val,
        n_train_after_balancing=result.n_train_after_balancing,
    )

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
        importance_top=importance_top,
        mlp_config=arch_cfg.to_dict() if hasattr(arch_cfg, "to_dict") else dict(arch_cfg.__dict__),
        train_config=train_cfg_dict,
        history=[ep for ep in result.history],
        preprocessing_summary=preproc_summary,
        feature_stats=feat_stats,
        feature_corr_with_label=feat_corr,
        score_stats_test=score_stats,
        score_deciles_test=score_deciles,
        errors_test=errors_dict,
        all_errors_test=all_errors,
        deltas_train_val_test=deltas,
        error_csv_paths=error_csv_paths,
        extra_notes=["Smoke test executado com amostra reduzida e poucas epocas."],
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
    parser.add_argument(
        "--arch", type=str, default="mlp",
        choices=["mlp", "two_tower", "ft_transformer", "multitask"],
        help="Arquitetura a treinar (Sprint 2)",
    )
    parser.add_argument(
        "--loss", type=str, default="bce",
        choices=["bce", "focal", "label_smoothing", "focal_smoothing"],
        help="Funcao de perda",
    )
    parser.add_argument("--seeds", type=int, default=1, help="Tamanho do ensemble")
    parser.add_argument("--calibrate", action="store_true", help="Aplicar Platt scaling")
    args = parser.parse_args()
    main(
        sample_size=args.sample, epochs=args.epochs, use_embeddings=args.emb,
        architecture=args.arch, loss_name=args.loss,
        n_seeds=args.seeds, use_calibration=args.calibrate,
    )
