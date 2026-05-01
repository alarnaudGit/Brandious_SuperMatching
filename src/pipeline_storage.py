"""Persistencia versionada: cada execucao de treino em `pipelines/<run_id>/`.

Estrutura:
  enriched/   — planilha e parquet da visao enriquecida
  weights/    — model.pt, calibrator.json, ensemble/, ensemble_arch_bagging/
  config/     — preprocessor.pkl, brand_similarity_model_config.json, pipeline_meta.json
  reports/    — report.md, feature_importance_*.csv/json, guia de poda
  errors/     — CSVs de falsos positivos / negativos

O ultimo bundle e tambem espelhado em `artifacts/` (raiz) para compatibilidade com scripts antigos.
"""
from __future__ import annotations

import json
import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .artifacts import (
    build_model_config_dict,
    save_enriched_dataframe,
    save_model_config,
    save_state_dict,
)
from .data import LABEL_COLUMN_CANON, DatasetReport
from .feature_importance_export import save_feature_importance_artifacts
from .model.arch_bagging import predict_architecture_bagging, save_architecture_bagging
from .model.calibration import PlattCalibrator
from .model.dataset import BalancingConfig, SplitConfig
from .model.ensemble import EnsembleMember, predict_ensemble, save_ensemble
from .model.evaluate import compute_metrics_at_threshold, predict_scores
from .model.explain import (
    permutation_importance,
    permutation_importance_arch_bagging,
    permutation_importance_hybrid,
    predict_hybrid_bagging,
)
from .model.logreg_bagging import save_logreg_bagging
from .model.mlp import MLPConfig
from .model.train import TrainConfig, TrainResult
from .normalize import calcular_similaridade_ofta
from .pipeline.preprocessor import FeaturePreprocessor
from .reports import (
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

logger = logging.getLogger(__name__)

ProgressCb = Callable[[float, str], None]


def pipelines_dir(project_root: Path) -> Path:
    return project_root / "pipelines"


def new_pipeline_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{uuid.uuid4().hex[:8]}"


def prepare_run_directories(project_root: Path, run_id: str) -> dict[str, Path]:
    root = pipelines_dir(project_root) / run_id
    return dirs_from_root(root)


def dirs_from_root(root: Path) -> dict[str, Path]:
    """Reconstroi mapa de subpastas a partir da raiz de um run ja gravado."""
    sub = {
        "root": root,
        "enriched": root / "enriched",
        "weights": root / "weights",
        "config": root / "config",
        "reports": root / "reports",
        "errors": root / "errors",
    }
    for p in sub.values():
        p.mkdir(parents=True, exist_ok=True)
    return sub


def append_registry(project_root: Path, entry: dict[str, Any]) -> None:
    reg_path = pipelines_dir(project_root) / "_registry.json"
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {"runs": []}
    if reg_path.exists():
        try:
            data = json.loads(reg_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Registry corrompido, reiniciando: %s", exc)
    runs: list[dict[str, Any]] = list(data.get("runs", []))
    eid = entry.get("id")
    if eid is not None:
        runs = [r for r in runs if r.get("id") != eid]
    runs.insert(0, entry)
    data["runs"] = runs[:120]
    reg_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_predict_full(
    model: Any,
    ensemble_members: list[EnsembleMember] | None,
    arch_bagging_members: list[Any] | None,
    logreg_bagging_members: list[Any] | None,
    calibrator: PlattCalibrator | None,
) -> Callable[[np.ndarray], np.ndarray]:
    def _predict_full(_X: np.ndarray) -> np.ndarray:
        if arch_bagging_members or logreg_bagging_members:
            return predict_hybrid_bagging(
                arch_bagging_members,
                logreg_bagging_members,
                _X,
                apply_calibration=True,
            )
        if ensemble_members:
            return predict_ensemble(
                ensemble_members,
                _X,
                apply_calibration=bool(
                    calibrator is not None and calibrator.fitted
                    or any(m.calibrator.fitted for m in ensemble_members)
                ),
            )
        s = predict_scores(model, _X)
        if calibrator is not None and calibrator.fitted:
            s = calibrator.transform(s)
        return s

    return _predict_full


def export_pipeline_bundle(
    *,
    project_root: Path,
    dirs: dict[str, Path],
    df: pd.DataFrame,
    X: np.ndarray,
    names: list[str],
    splits: dict[str, Any],
    model: Any,
    preproc: FeaturePreprocessor,
    train_cfg: TrainConfig,
    train_result: TrainResult | None,
    threshold_opt: float,
    threshold_policy: dict[str, Any],
    df_report: DatasetReport | None,
    history: list[dict[str, Any]] | None,
    ensemble_members: list[EnsembleMember] | None,
    arch_bagging_members: list[Any] | None,
    calibrator: PlattCalibrator | None,
    importance_top: list[dict[str, float]] | None,
    auto_imp: bool,
    progress_callback_enriched: ProgressCb | None = None,
    logreg_bagging_members: list[Any] | None = None,
) -> tuple[list[dict[str, float]] | None, dict[str, Any]]:
    """Grava bundle completo em `dirs` (subpastas). Devolve (importance_top, meta)."""
    d_enr = dirs["enriched"]
    d_w = dirs["weights"]
    d_cfg = dirs["config"]
    d_rep = dirs["reports"]
    d_err = dirs["errors"]
    root = dirs["root"]

    _predict_full = _make_predict_full(
        model, ensemble_members,
        arch_bagging_members, logreg_bagging_members,
        calibrator,
    )

    score_nn_full = _predict_full(X)
    score_h_full = np.array(
        [
            calcular_similaridade_ofta(a, b)
            for a, b in zip(
                df["marca_monitorada"].astype(str),
                df["marca_colidente"].astype(str),
            )
        ],
        dtype=np.float32,
    )

    preproc.save(d_cfg / "preprocessor.pkl")
    save_state_dict(model.state_dict(), d_w / "model.pt")

    if calibrator is not None and calibrator.fitted:
        calibrator.save(d_w / "calibrator.json")

    if ensemble_members:
        save_ensemble(ensemble_members, d_w / "ensemble")
    if arch_bagging_members:
        save_architecture_bagging(arch_bagging_members, d_w / "ensemble_arch_bagging")
    if logreg_bagging_members:
        save_logreg_bagging(logreg_bagging_members, d_w / "ensemble_logreg")

    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]
    scores_train = _predict_full(X_train)
    scores_val = _predict_full(X_val)
    scores_test = _predict_full(X_test)
    em_train = compute_metrics_at_threshold(y_train, scores_train, threshold_opt)
    em_val = compute_metrics_at_threshold(y_val, scores_val, threshold_opt)
    em_test_re = compute_metrics_at_threshold(y_test, scores_test, threshold_opt)

    bal: BalancingConfig = splits["bal"]
    split_cfg: SplitConfig = splits["split_cfg"]

    arch_cfg = (
        model.config if hasattr(model, "config") else (
            model.cfg if hasattr(model, "cfg") else MLPConfig(input_dim=X.shape[1])
        )
    )
    cfg_dict = build_model_config_dict(
        mlp_config=arch_cfg,
        preprocessor=preproc,
        train_config=train_cfg,
        balancing=bal,
        split_config=split_cfg,
        threshold_optimal=threshold_opt,
        threshold_policy=threshold_policy,
        metrics_train=em_train,
        metrics_val=em_val,
        metrics_test=em_test_re,
        train_result=train_result,
        dataset_report=df_report,
        state_dict=model.state_dict(),
        embedding_used=preproc.config.use_embeddings,
    )
    if arch_bagging_members:
        cfg_dict["architecture_bagging"] = {
            "enabled": True,
            "aggregation": "mean",
            "members": [m.key for m in arch_bagging_members],
            "artifact_dir": "weights/ensemble_arch_bagging",
        }
    if logreg_bagging_members:
        cfg_dict["logreg_bagging"] = {
            "enabled": True,
            "aggregation": "mean",
            "members": [m.key for m in logreg_bagging_members],
            "artifact_dir": "weights/ensemble_logreg",
        }
    if arch_bagging_members or logreg_bagging_members:
        cfg_dict["hybrid_bagging"] = {
            "enabled": True,
            "n_mlp": int(len(arch_bagging_members or [])),
            "n_logreg": int(len(logreg_bagging_members or [])),
            "aggregation": "mean_calibrated",
        }
    save_model_config(cfg_dict, d_cfg / "brand_similarity_model_config.json")

    save_enriched_dataframe(
        df_original=df,
        feature_matrix=X,
        feature_names=names,
        score_nn=score_nn_full,
        score_heuristic=score_h_full,
        threshold=threshold_opt,
        out_xlsx=d_enr / "brand_similarity_input_view_enriched.xlsx",
        out_parquet=d_enr / "enriched.parquet",
        label_col=LABEL_COLUMN_CANON,
        progress_callback=progress_callback_enriched,
    )

    imp = importance_top
    imp_computed_here = False
    imp_val_sample_n = -1
    imp_n_repeats_used = -1
    if imp is None and auto_imp:
        imp_computed_here = True
        rng = np.random.default_rng(42)
        sample_n = min(600, len(y_val))
        imp_val_sample_n = int(sample_n)
        imp_n_repeats_used = 3
        idx = rng.choice(len(y_val), size=sample_n, replace=False)
        if arch_bagging_members or logreg_bagging_members:
            imp = permutation_importance_hybrid(
                arch_bagging_members,
                logreg_bagging_members,
                X_val[idx],
                y_val[idx],
                names,
                metric="pr_auc",
                n_repeats=imp_n_repeats_used,
            )
        else:
            imp = permutation_importance(
                model, X_val[idx], y_val[idx], names, metric="pr_auc", n_repeats=imp_n_repeats_used,
            )

    if imp:
        imp_paths = save_feature_importance_artifacts(
            d_rep,
            project_root,
            imp,
            val_sample_n=imp_val_sample_n if imp_computed_here else -1,
            n_repeats=imp_n_repeats_used if imp_computed_here else -1,
            metric="pr_auc",
            arch_bagging=bool(
                arch_bagging_members or logreg_bagging_members
            ),
        )
    else:
        imp_paths = {}

    comp = compare_against_heuristic(y_test, scores_test, score_h_full[: len(y_test)])

    score_stats = score_distribution_stats(y_test, scores_test)
    score_deciles = score_decile_table(y_test, scores_test)

    idx_test_real = splits.get("idx_test")
    if idx_test_real is not None:
        df_test_aligned = df.iloc[idx_test_real].copy().reset_index(drop=True)
    else:
        df_test_aligned = df.iloc[: len(y_test)].copy().reset_index(drop=True)
    errors_dict = top_errors(df_test_aligned, y_test, scores_test, threshold_opt, k=10)
    all_errors = all_errors_separated(
        df_test_aligned, y_test, scores_test, threshold_opt, keep_specs=False,
    )

    fp_csv = d_err / "falsos_positivos_grupo_0.csv"
    fn_csv = d_err / "falsos_negativos_grupo_1.csv"
    all_errors["fp_full_df"].to_csv(fp_csv, index=False, encoding="utf-8-sig")
    all_errors["fn_full_df"].to_csv(fn_csv, index=False, encoding="utf-8-sig")
    error_csv_paths = {
        "fp_csv": str(fp_csv.relative_to(project_root)),
        "fn_csv": str(fn_csv.relative_to(project_root)),
    }

    feat_stats = feature_summary_stats(X, names, top_k=20)
    feat_corr = features_correlation_with_label(
        X, df[LABEL_COLUMN_CANON].to_numpy().astype(np.float32), names, top_k=20,
    )

    preproc_summary = {
        "config": preproc.config.to_dict(),
        "top_classes": list(preproc.top_classes),
    }

    deltas = {
        "ROC-AUC train-test": em_train.roc_auc - em_test_re.roc_auc,
        "PR-AUC train-test": em_train.pr_auc - em_test_re.pr_auc,
        "Recall train-test": em_train.recall - em_test_re.recall,
        "F1 train-test": em_train.f1 - em_test_re.f1,
    }

    train_cfg_dict = (
        train_cfg.to_dict() if hasattr(train_cfg, "to_dict") else dict(train_cfg.__dict__)
    )
    if train_result is not None:
        train_cfg_dict.update(
            best_epoch=train_result.best_epoch,
            best_pr_auc_val=train_result.best_pr_auc_val,
            n_train_after_balancing=train_result.n_train_after_balancing,
        )

    report_md = build_report(
        feature_names=names,
        balancing=bal,
        pos_weight_used=float(getattr(train_result, "pos_weight_used", 1.0)) if train_result else 1.0,
        split_config=split_cfg,
        metrics_train=em_train,
        metrics_val=em_val,
        metrics_test=em_test_re,
        comparison=comp,
        threshold_optimal=threshold_opt,
        threshold_policy=threshold_policy,
        dataset_report=df_report,
        importance_top=imp,
        mlp_config=arch_cfg.to_dict() if hasattr(arch_cfg, "to_dict") else dict(arch_cfg.__dict__),
        train_config=train_cfg_dict,
        history=history,
        preprocessing_summary=preproc_summary,
        feature_stats=feat_stats,
        feature_corr_with_label=feat_corr,
        score_stats_test=score_stats,
        score_deciles_test=score_deciles,
        errors_test=errors_dict,
        all_errors_test=all_errors,
        deltas_train_val_test=deltas,
        error_csv_paths=error_csv_paths,
    )
    save_report(report_md, d_rep / "report.md")

    meta = {
        "run_dir": str(root.relative_to(project_root)),
        "pr_auc_test": float(em_test_re.pr_auc),
        "recall_test": float(em_test_re.recall),
        "roc_auc_test": float(em_test_re.roc_auc),
        "f1_test": float(em_test_re.f1),
        "threshold": float(threshold_opt),
        "n_rows_enriched": int(len(df)),
        "n_features": int(X.shape[1]),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "permutation_importance_meta": {
            "computed_in_this_export": bool(imp_computed_here),
            "val_sample_n": int(imp_val_sample_n) if imp_computed_here else None,
            "n_repeats": int(imp_n_repeats_used) if imp_computed_here else None,
        },
        **imp_paths,
    }
    (d_cfg / "pipeline_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("Pipeline exportado em %s", root)
    return imp, meta


def mirror_latest_to_artifacts(project_root: Path, dirs: dict[str, Path]) -> None:
    """Copia ficheiros chave para `artifacts/` (layout plano legado)."""
    art = project_root / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    d_w, d_cfg, d_enr, d_rep, d_err = dirs["weights"], dirs["config"], dirs["enriched"], dirs["reports"], dirs["errors"]

    singles = [
        (d_cfg / "preprocessor.pkl", art / "preprocessor.pkl"),
        (d_w / "model.pt", art / "model.pt"),
        (d_cfg / "brand_similarity_model_config.json", art / "brand_similarity_model_config.json"),
        (d_enr / "brand_similarity_input_view_enriched.xlsx", art / "brand_similarity_input_view_enriched.xlsx"),
        (d_enr / "enriched.parquet", art / "enriched.parquet"),
        (d_rep / "report.md", art / "report.md"),
        (d_rep / "feature_importance_permutation_full.csv", art / "feature_importance_permutation_full.csv"),
        (d_rep / "feature_importance_by_block.json", art / "feature_importance_by_block.json"),
        (d_rep / "GUIA_PODA_FEATURES_ENRIQUECIDAS.md", art / "GUIA_PODA_FEATURES_ENRIQUECIDAS.md"),
        (d_err / "falsos_positivos_grupo_0.csv", art / "falsos_positivos_grupo_0.csv"),
        (d_err / "falsos_negativos_grupo_1.csv", art / "falsos_negativos_grupo_1.csv"),
    ]
    cal_src = d_w / "calibrator.json"
    if cal_src.exists():
        singles.append((cal_src, art / "calibrator.json"))

    for src, dst in singles:
        if src.exists():
            shutil.copy2(src, dst)

    for name in ("ensemble", "ensemble_arch_bagging", "ensemble_logreg"):
        src_dir = d_w / name
        dst_dir = art / name
        if src_dir.exists():
            if dst_dir.exists():
                shutil.rmtree(dst_dir, ignore_errors=True)
            shutil.copytree(src_dir, dst_dir)

    logger.info("Espelho legado atualizado em %s", art)


def collect_download_files(dirs: dict[str, Path]) -> list[Path]:
    """Lista ficheiros para download na UI."""
    candidates = [
        dirs["config"] / "preprocessor.pkl",
        dirs["config"] / "brand_similarity_model_config.json",
        dirs["config"] / "pipeline_meta.json",
        dirs["weights"] / "model.pt",
        dirs["weights"] / "calibrator.json",
        dirs["enriched"] / "brand_similarity_input_view_enriched.xlsx",
        dirs["enriched"] / "enriched.parquet",
        dirs["reports"] / "report.md",
        dirs["reports"] / "feature_importance_permutation_full.csv",
        dirs["reports"] / "feature_importance_by_block.json",
        dirs["reports"] / "GUIA_PODA_FEATURES_ENRIQUECIDAS.md",
        dirs["errors"] / "falsos_positivos_grupo_0.csv",
        dirs["errors"] / "falsos_negativos_grupo_1.csv",
        dirs["weights"] / "ensemble" / "ensemble_index.json",
        dirs["weights"] / "ensemble_arch_bagging" / "index.json",
        dirs["weights"] / "ensemble_logreg" / "index.json",
    ]
    return [p for p in candidates if p.exists()]


__all__ = [
    "append_registry",
    "collect_download_files",
    "dirs_from_root",
    "export_pipeline_bundle",
    "mirror_latest_to_artifacts",
    "new_pipeline_run_id",
    "pipelines_dir",
    "prepare_run_directories",
]
