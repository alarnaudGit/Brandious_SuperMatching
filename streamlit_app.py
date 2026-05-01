"""Brandious SuperMatching - interface profissional Streamlit.

Substitui o Flask antigo: oferece upload, EDA, enriquecimento, configuracao,
treino em tempo real, avaliacao, explicabilidade, teste manual e download
de artefatos para o modelo de similaridade aprendida de marcas.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import __version__
from src.data import (  # noqa: E402
    LABEL_COLUMN_CANON,
    DatasetReport,
    load_dataframe_from_bytes,
)
from src.model.dataset import (  # noqa: E402
    BalancingConfig,
    SplitConfig,
)
from src.features.feature_dictionary import (  # noqa: E402
    attach_descriptions,
)
from src.model.arch_bagging import (  # noqa: E402
    MLPRandomRanges,
    predict_architecture_bagging,
    predict_architecture_bagging_components,
    train_random_mlp_bagging,
)
from src.model.calibration import PlattCalibrator  # noqa: E402
from src.model.ensemble import (  # noqa: E402
    EnsembleMember,
    predict_ensemble,
)
from src.model.evaluate import (  # noqa: E402
    compute_metrics_at_threshold,
    find_optimal_threshold,
    pr_curve_data,
    predict_scores,
    roc_curve_data,
)
from src.model.explain import (  # noqa: E402
    integrated_gradients_arch_bagging_row,
    integrated_gradients_for_row,
    integrated_gradients_hybrid_row,
    permutation_importance,
    permutation_importance_arch_bagging,
    permutation_importance_hybrid,
    predict_hybrid_bagging,
    predict_hybrid_bagging_components,
)
from src.model.logreg_bagging import (  # noqa: E402
    LogRegRandomRanges,
    train_random_logreg_bagging,
)
from src.model.mlp import MLPConfig  # noqa: E402
from src.model.train import TrainConfig  # noqa: E402
from src.normalize import calcular_score_complexo, calcular_similaridade_ofta  # noqa: E402
from src.pipeline_storage import (  # noqa: E402
    append_registry,
    collect_download_files,
    dirs_from_root,
    export_pipeline_bundle,
    mirror_latest_to_artifacts,
    new_pipeline_run_id,
    prepare_run_directories,
)
from src.pipeline.preprocessor import (  # noqa: E402
    BERTIMBAU_BASE_STS,
    BERTIMBAU_LARGE_STS,
    FeaturePreprocessor,
    PreprocessorConfig,
)
from src.reports import compare_against_heuristic  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("streamlit_app")

st.set_page_config(
    page_title="Brandious SuperMatching - NN Trainer",
    page_icon="?",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Estado da sessao
# ---------------------------------------------------------------------------
def _init_session_state() -> None:
    defaults = {
        "df": None,
        "df_report": None,
        "preprocessor": None,
        "feature_matrix": None,
        "feature_names": None,
        "splits": None,
        "model": None,
        "train_result": None,
        "eval_metrics_test": None,
        "threshold_optimal": 0.5,
        "threshold_policy": {},
        "history": [],
        "stop_train": False,
        "importance_top": None,
        "score_nn_full": None,
        "score_heuristic_full": None,
        "ensemble_members": None,
        "arch_bagging_members": None,
        "logreg_bagging_members": None,
        "calibrator": None,
        "arch_config_dict": None,
        "pipeline_run_id": None,
        "pipeline_run_dir": None,
        # Status por etapa: dataset / enrichment / training / evaluation /
        # explainability / manual_test / artifacts.
        "step_status": {},
        # Tempos por etapa em segundos.
        "step_durations": {},
        # Assinatura do preprocessador usado para gerar a matriz atual.
        # Usado para detectar drift e avisar o usuario quando mudar
        # parametros sem regerar as features.
        "preproc_cfg_signature": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Helpers: status visual e cronometro por etapa
# ---------------------------------------------------------------------------
STEP_LABELS = {
    "dataset": "1. Dataset",
    "enrichment": "2. Enriquecimento",
    "training": "3. Treino",
    "evaluation": "4. Avaliacao",
    "explainability": "5. Explicabilidade",
    "manual_test": "6. Teste manual",
    "artifacts": "7. Artefatos",
}


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    s = max(0.0, float(seconds))
    minutes = int(s // 60)
    secs = s - minutes * 60
    if minutes >= 1:
        return f"{minutes}m {secs:04.1f}s"
    return f"{secs:.2f}s"


def _set_step_status(step: str, status: str) -> None:
    """status in {pending, running, ok, error}."""
    statuses = dict(st.session_state.get("step_status") or {})
    statuses[step] = status
    st.session_state.step_status = statuses


def _get_step_status(step: str) -> str:
    return (st.session_state.get("step_status") or {}).get(step, "pending")


def _record_step_duration(step: str, seconds: float) -> None:
    durations = dict(st.session_state.get("step_durations") or {})
    durations[step] = float(seconds)
    st.session_state.step_durations = durations


def _status_badge(step: str) -> str:
    status = _get_step_status(step)
    icon = {"pending": "[ ]", "running": "[~]", "ok": "[OK]", "error": "[X]"}.get(
        status, "[ ]",
    )
    label = STEP_LABELS.get(step, step)
    dur = (st.session_state.get("step_durations") or {}).get(step)
    if dur is not None and status in ("ok", "error"):
        return f"{icon} {label} ({_format_duration(dur)})"
    return f"{icon} {label}"


def _render_pipeline_progress() -> None:
    """Renderiza no topo de cada aba o status de todas as etapas + tempo total."""
    cols = st.columns(len(STEP_LABELS))
    total = 0.0
    for col, key in zip(cols, STEP_LABELS.keys()):
        col.caption(_status_badge(key))
        dur = (st.session_state.get("step_durations") or {}).get(key)
        if dur:
            total += float(dur)
    if total > 0:
        st.caption(
            f"Tempo total acumulado nas etapas concluidas: "
            f"**{_format_duration(total)}**.",
        )


class StepTimer:
    """Context manager que mede o tempo de uma etapa e atualiza session_state.

    Uso:
        with StepTimer("training"):
            ...

    Marca status `running` ao entrar, `ok` ao sair sem excecao e `error` se
    houver excecao. Tempo gasto e' gravado em `step_durations[step]`.
    """

    def __init__(self, step: str) -> None:
        self.step = step
        self.t0 = 0.0

    def __enter__(self) -> "StepTimer":
        self.t0 = time.perf_counter()
        _set_step_status(self.step, "running")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed = time.perf_counter() - self.t0
        _record_step_duration(self.step, elapsed)
        _set_step_status(self.step, "error" if exc_type is not None else "ok")
        return False


_init_session_state()


# ---------------------------------------------------------------------------
# Sidebar - configuracao global do treino
# ---------------------------------------------------------------------------
def render_sidebar() -> dict[str, Any]:
    st.sidebar.title("Configuracao")
    st.sidebar.caption(f"Brandious SuperMatching v{__version__}")

    with st.sidebar.expander("Pre-processador", expanded=True):
        use_emb = st.checkbox(
            "Usar embeddings semanticos (sentence-transformers)",
            value=True,
            help="Adiciona spec_cosine_emb. Default ON conforme escopo.",
        )

        emb_model_options = {
            "MiniLM multilingual (rapido, ~280 MB)": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "BERTimbau-base STS (PT-BR, ~440 MB)": BERTIMBAU_BASE_STS,
            "BERTimbau-large STS (PT-BR, ~1.3 GB) [Sprint 1 default]": BERTIMBAU_LARGE_STS,
        }
        emb_model_label = st.selectbox(
            "Modelo de embedding (specs e marcas)",
            options=list(emb_model_options.keys()),
            index=2,  # default no BERTimbau-large STS
            help="BERTimbau-large e mais lento mas costuma render +1 a +2 pp em PR-AUC.",
        )
        emb_model = emb_model_options[emb_model_label]

        use_brand_emb = st.checkbox(
            "Adicionar features brand_emb_* (embedding da marca)",
            value=True,
            help="3 features Sprint 1: cosine entre embeddings das marcas (raw e normalized).",
        )

        generic_df = st.slider(
            "Limiar DF (%) para token generico (auto-deteccao)",
            0.5, 5.0, 1.0, 0.1,
            help="Token presente em >= X% das marcas vira 'generico'. "
                 "Default 1.0%: ~100 tokens em 30k marcas. "
                 "Diminua para deteccao mais agressiva.",
        )

        top_k = st.slider("Top-K classes Nice (one-hot)", 5, 30, 18)
        tfidf_word_max = st.number_input("TF-IDF word max_features", 1000, 50_000, 25_000, step=1000)
        tfidf_char_max = st.number_input("TF-IDF char max_features", 1000, 50_000, 12_000, step=1000)

    with st.sidebar.expander(
        "Tamanho do bagging (MLPs + Logisticas)", expanded=True,
    ):
        n_per_kind = st.slider(
            "Modelos por tipo (N)",
            min_value=1, max_value=100, value=3, step=1,
            help=(
                "O bagging final tera **2 X N modelos**: N MLPs + N "
                "regressoes logisticas. O score por par e' a media "
                "calibrada Platt de todos os 2N modelos."
            ),
        )
        st.caption(
            f"Total de modelos no bagging: **{int(n_per_kind) * 2}** "
            f"({int(n_per_kind)} MLPs + {int(n_per_kind)} log\u00edsticas)."
        )

    with st.sidebar.expander(
        "Bagging - Parametros das MLPs (sorteio)", expanded=False,
    ):
        n_layers_str = st.text_input(
            "Numero de camadas ocultas (escolhas)",
            value="2,3,4",
            help="Lista vir-separada de quantas camadas a MLP pode ter.",
        )
        sizes_str = st.text_input(
            "Tamanhos possiveis de camada",
            value="32,64,96,128,192,256,384,512",
            help=(
                "Pool de neuronios por camada para sortear. Em arquiteturas "
                "monotonicamente decrescentes, cada camada subsequente nao "
                "tera mais neuronios que a anterior."
            ),
        )
        monotonic = st.checkbox(
            "Forcar arquitetura monotonicamente decrescente",
            value=True,
            help="Ex.: [256, 128, 64] em vez de [64, 256, 32].",
        )
        col_drp1, col_drp2 = st.columns(2)
        dropout_min = col_drp1.slider(
            "Dropout min", 0.0, 0.6, 0.10, 0.05,
        )
        dropout_max = col_drp2.slider(
            "Dropout max", 0.05, 0.7, 0.50, 0.05,
        )
        if dropout_max < dropout_min:
            dropout_max = dropout_min
        activation_choices = st.multiselect(
            "Ativacoes possiveis",
            options=["relu", "gelu", "leakyrelu", "tanh"],
            default=["relu", "gelu", "leakyrelu"],
        )
        bn_prob = st.slider(
            "Probabilidade de usar BatchNorm",
            0.0, 1.0, 0.7, 0.05,
        )

    with st.sidebar.expander(
        "Bagging - Parametros das Logisticas (sorteio)", expanded=False,
    ):
        col_c1, col_c2 = st.columns(2)
        lr_c_min = col_c1.select_slider(
            "C min (regularizacao)",
            options=[1e-3, 1e-2, 1e-1, 1.0],
            value=1e-2,
            help=(
                "C menor = mais regularizacao (mais bias, menos variancia)."
            ),
        )
        lr_c_max = col_c2.select_slider(
            "C max",
            options=[1.0, 10.0, 100.0],
            value=10.0,
        )
        lr_penalty_choices = st.multiselect(
            "Penalty possiveis",
            options=["l2", "l1"],
            default=["l2", "l1"],
            help=(
                "Solver `liblinear` suporta l1 e l2; ambos com class_weight."
            ),
        )
        lr_cw_choices = st.multiselect(
            "class_weight",
            options=["none", "balanced"],
            default=["none", "balanced"],
            help=(
                "`balanced` aumenta peso da classe minoritaria automaticamente."
            ),
        )
        lr_use_bootstrap = st.checkbox(
            "Bootstrap por modelo (amostragem com reposicao)",
            value=True,
            help=(
                "Cada logistica treina sobre uma amostra com reposicao do "
                "treino balanceado, aumentando diversidade do bagging."
            ),
        )
        lr_max_iter = st.number_input(
            "max_iter",
            min_value=200, max_value=10_000, value=2000, step=200,
        )

    with st.sidebar.expander("Treino (otimizacao)", expanded=True):
        epochs = st.slider("Epochs maximas (cada MLP)", 5, 200, 60)
        es_patience = st.slider(
            "Early stopping patience (PR-AUC val)", 3, 30, 10,
        )
        col_lr1, col_lr2 = st.columns(2)
        lr_min = col_lr1.select_slider(
            "LR min (log-uniform)",
            options=[1e-5, 3e-5, 1e-4, 3e-4],
            value=1e-4,
        )
        lr_max = col_lr2.select_slider(
            "LR max (log-uniform)",
            options=[1e-3, 3e-3, 1e-2, 3e-2],
            value=3e-3,
        )
        col_wd1, col_wd2 = st.columns(2)
        wd_zero_prob = col_wd1.slider(
            "Prob. de wd=0", 0.0, 1.0, 0.3, 0.05,
            help="Quando nao for zero, sortear log-uniforme em [wd_min, wd_max].",
        )
        wd_max = col_wd2.select_slider(
            "Weight decay max",
            options=[1e-5, 1e-4, 1e-3, 3e-3],
            value=1e-3,
        )
        batch_choices_str = st.text_input(
            "Batch sizes possiveis",
            value="128,256,512",
        )

    with st.sidebar.expander("Loss & Taticas Sprint 2", expanded=True):
        loss_options = {
            "BCE com pos_weight (default)": "bce",
            "Focal (gamma=2, alpha=0.25)": "focal",
            "Label smoothing BCE (eps=0.05)": "label_smoothing",
            "Focal + Label smoothing (recomendado)": "focal_smoothing",
        }
        loss_label = st.selectbox(
            "Funcao de perda", options=list(loss_options.keys()), index=3,
        )
        loss_value = loss_options[loss_label]
        focal_alpha = st.slider("Focal alpha", 0.05, 0.95, 0.25, 0.05)
        focal_gamma = st.slider("Focal gamma", 0.5, 5.0, 2.0, 0.5)
        label_smoothing = st.slider("Label smoothing (eps)", 0.0, 0.2, 0.05, 0.01)

        mixup_pos = st.slider(
            "Mixup entre positivos (% por batch)", 0.0, 0.5, 0.12, 0.05,
            help="Substitui parte dos pos do batch por combinacao convexa de outros pos. 0 desliga.",
        )
        hard_neg = st.checkbox(
            "Hard Negative Mining (a partir da 5a epoca)", value=True,
            help="Em cada epoca >=5, monta batch com TODOS pos + top-k neg de maior score.",
        )
        symmetry_aug = st.checkbox(
            "Symmetry augmentation (duplica dataset)", value=True,
            help="Aumenta robustez a ordenacao (A,B) vs (B,A).",
        )

    with st.sidebar.expander("Calibracao", expanded=False):
        st.caption(
            "No bagging de N MLPs, **cada modelo e' calibrado por Platt no "
            "validation** automaticamente. O score final e' a media dos K "
            "scores calibrados.",
        )

    with st.sidebar.expander("Balanceamento", expanded=True):
        _bal_labels = {
            "50/50 — replicar positivos ate igualar negativos (apos undersample) [padrao]": "equal",
            "Legado — oversample por fator (sem igualar contagens)": "legacy",
        }
        _bal_choice = st.radio(
            "Estrategia de treino (classe 1 = colidente)",
            options=list(_bal_labels.keys()),
            index=0,
            help=(
                "50/50: mesmo numero de linhas positivas e negativas no treino "
                "(positivos com reposicao). Legado: usa o fator de oversample."
            ),
        )
        training_balance = _bal_labels[_bal_choice]
        under_ratio = st.slider(
            "Undersample: max negativos por positivo *original*",
            1.0, 10.0, 2.5, 0.5,
            help="Antes do 50/50: limita quantos negativos entram; depois os positivos replicam ate esse total.",
        )
        over_factor = st.slider(
            "Oversample positivos (fator, so modo legado)", 1.0, 5.0, 2.0, 0.5,
            disabled=(training_balance == "equal"),
            help="Ignorado no modo 50/50.",
        )
        if training_balance == "equal":
            st.caption("No modo 50/50 o fator de oversample e ignorado; os positivos copiam-se ate igualar os negativos.")
        use_cw = st.checkbox("Usar class weight (pos_weight)", value=True)
        pw_override_use = st.checkbox("Override de pos_weight", value=False)
        pw_override = st.number_input("pos_weight override", 0.5, 30.0, 9.0, 0.5) if pw_override_use else None

    with st.sidebar.expander("Threshold", expanded=False):
        recall_floor = st.slider("Recall floor (politica de threshold)", 0.5, 0.99, 0.85, 0.01)

    with st.sidebar.expander("Split / Seed", expanded=False):
        seed = st.number_input("Seed", 0, 999_999, 42, step=1)
        test_size = st.slider("Test size", 0.05, 0.3, 0.15, 0.01)
        val_size = st.slider("Val size", 0.05, 0.3, 0.15, 0.01)

    st.sidebar.divider()
    st.sidebar.caption(
        "**Fluxo manual.** Execute as abas em sequencia (1 -> 2 -> 3 -> 4 -> "
        "5 -> 6 -> 7). Os artefatos sao gravados automaticamente quando o "
        "treino na aba 3 termina.",
    )

    def _parse_int_list(s: str, fallback: list[int]) -> list[int]:
        try:
            out = [int(x.strip()) for x in s.split(",") if x.strip()]
            return out or fallback
        except ValueError:
            return fallback

    n_layers_choices = _parse_int_list(n_layers_str, [2, 3, 4])
    layer_size_choices = _parse_int_list(
        sizes_str, [32, 64, 96, 128, 192, 256, 384, 512],
    )
    batch_choices = _parse_int_list(batch_choices_str, [128, 256, 512])

    mlp_ranges = MLPRandomRanges(
        n_layers_choices=n_layers_choices,
        layer_size_choices=layer_size_choices,
        monotonic_decreasing=bool(monotonic),
        dropout_min=float(dropout_min),
        dropout_max=float(dropout_max),
        activation_choices=tuple(activation_choices) or ("relu",),
        batchnorm_prob=float(bn_prob),
        lr_log_min=float(lr_min),
        lr_log_max=float(lr_max),
        wd_zero_prob=float(wd_zero_prob),
        wd_log_min=1e-5,
        wd_log_max=float(wd_max),
        batch_size_choices=tuple(batch_choices),
    )

    logreg_cw_pool: list[str | None] = []
    for cw in lr_cw_choices:
        if isinstance(cw, str) and cw.lower() == "none":
            logreg_cw_pool.append(None)
        else:
            logreg_cw_pool.append(str(cw))
    if not logreg_cw_pool:
        logreg_cw_pool = [None]

    logreg_ranges = LogRegRandomRanges(
        c_log_min=float(lr_c_min),
        c_log_max=float(lr_c_max),
        penalty_choices=tuple(lr_penalty_choices) or ("l2",),
        class_weight_choices=tuple(logreg_cw_pool),
        solver="liblinear",
        max_iter=int(lr_max_iter),
        bootstrap_train=bool(lr_use_bootstrap),
    )

    return {
        "preproc_cfg": PreprocessorConfig(
            tfidf_word_max_features=int(tfidf_word_max),
            tfidf_char_max_features=int(tfidf_char_max),
            top_k_classes=int(top_k),
            use_embeddings=bool(use_emb),
            embedding_model=str(emb_model),
            use_brand_embeddings=bool(use_brand_emb),
            generic_df_threshold=float(generic_df) / 100.0,
        ),
        "n_per_kind": int(n_per_kind),
        "mlp_ranges": mlp_ranges,
        "logreg_ranges": logreg_ranges,
        "epochs": int(epochs),
        "es_patience": int(es_patience),
        "loss_name": str(loss_value),
        "focal_alpha": float(focal_alpha),
        "focal_gamma": float(focal_gamma),
        "label_smoothing": float(label_smoothing),
        "mixup_pos": float(mixup_pos),
        "hard_neg_mining": bool(hard_neg),
        "symmetry_aug": bool(symmetry_aug),
        "training_balance": str(training_balance),
        "under_ratio": float(under_ratio),
        "over_factor": float(over_factor),
        "use_cw": bool(use_cw),
        "pos_weight_override": float(pw_override) if pw_override is not None else None,
        "recall_floor": float(recall_floor),
        "seed": int(seed),
        "test_size": float(test_size),
        "val_size": float(val_size),
    }


# ---------------------------------------------------------------------------
# Pipeline automatico (enriquecimento + treino + artefatos)
# ---------------------------------------------------------------------------
def _reset_downstream_pipeline_state() -> None:
    """Limpa estado derivado do dataset apos novo carregamento."""
    for k in (
        "preprocessor",
        "feature_matrix",
        "feature_names",
        "splits",
        "model",
        "train_result",
        "train_cfg",
        "mlp_cfg",
        "eval_metrics_test",
        "threshold_optimal",
        "threshold_policy",
        "history",
        "importance_top",
        "ensemble_members",
        "arch_bagging_members",
        "logreg_bagging_members",
        "calibrator",
        "pipeline_run_id",
        "pipeline_run_dir",
    ):
        st.session_state[k] = None if k != "history" else []
    st.session_state.threshold_optimal = 0.5
    st.session_state.stop_train = False
    st.session_state.step_status = {}
    st.session_state.step_durations = {}
    st.session_state.preproc_cfg_signature = None


_PREPROC_FIELDS_LABELS: dict[str, str] = {
    "tfidf_word_max_features": "TF-IDF word max_features",
    "tfidf_char_max_features": "TF-IDF char max_features",
    "top_k_classes": "Top-K classes Nice",
    "use_embeddings": "Embeddings semanticos (specs) ativos",
    "embedding_model": "Modelo de embedding",
    "use_brand_embeddings": "Brand embeddings ativos",
    "generic_df_threshold": "Limiar DF de tokens genericos",
}


def _preproc_cfg_signature(cfg: dict[str, Any]) -> dict[str, Any]:
    """Snapshot dos parametros do preprocessador relevantes para invalidacao
    da matriz de features."""
    pc = cfg.get("preproc_cfg")
    if pc is None:
        return {}
    out: dict[str, Any] = {}
    for k in _PREPROC_FIELDS_LABELS:
        v = getattr(pc, k, None)
        out[k] = v if not isinstance(v, float) else round(float(v), 8)
    return out


def _diff_preproc_signatures(
    old: dict[str, Any], new: dict[str, Any],
) -> list[tuple[str, Any, Any]]:
    """Devolve lista (label, valor_antigo, valor_novo) com os campos que
    mudaram entre duas assinaturas."""
    diffs: list[tuple[str, Any, Any]] = []
    for k, label in _PREPROC_FIELDS_LABELS.items():
        ov = old.get(k)
        nv = new.get(k)
        if ov != nv:
            diffs.append((label, ov, nv))
    return diffs


def _render_preproc_drift_warning(cfg: dict[str, Any]) -> bool:
    """Avisa se os parametros do preprocessador mudaram desde o ultimo
    enriquecimento. Retorna True se houve drift."""
    used = st.session_state.get("preproc_cfg_signature")
    if not used:
        return False
    current = _preproc_cfg_signature(cfg)
    diffs = _diff_preproc_signatures(used, current)
    if not diffs:
        return False
    lines = [
        f"- **{label}**: `{ov}` -> `{nv}`"
        for label, ov, nv in diffs
    ]
    st.warning(
        "Voce alterou parametros do **pre-processador** desde a ultima "
        "geracao de features. A matriz em cache **nao reflete** essas "
        "mudancas. Para usa-las, regenere as features na **aba 2 "
        "(Enriquecimento)**.\n\n"
        "Mudancas detectadas:\n" + "\n".join(lines)
    )
    return True


def _run_feature_enrichment(
    cfg: dict[str, Any],
    df: pd.DataFrame,
    progress_callback: Any,
) -> None:
    preproc = FeaturePreprocessor(cfg["preproc_cfg"])
    preproc.fit(df, progress_callback=progress_callback)
    X = preproc.transform(
        df, scale=True, show_progress=False, progress_callback=progress_callback,
    )
    st.session_state.preprocessor = preproc
    st.session_state.feature_matrix = X
    st.session_state.feature_names = list(preproc.feature_names_ordered)
    st.session_state.preproc_cfg_signature = _preproc_cfg_signature(cfg)


def _get_bagging_lists() -> tuple[list[Any], list[Any]]:
    """Devolve (mlp_members, logreg_members) do session_state (sempre listas)."""
    mlps = st.session_state.get("arch_bagging_members") or []
    lrs = st.session_state.get("logreg_bagging_members") or []
    return list(mlps), list(lrs)


def _has_hybrid_bagging() -> bool:
    mlps, lrs = _get_bagging_lists()
    return bool(mlps) or bool(lrs)


def _hybrid_predict(X: np.ndarray) -> np.ndarray:
    """Score agregado do bagging hibrido (MLPs + LogRegs)."""
    mlps, lrs = _get_bagging_lists()
    return predict_hybrid_bagging(mlps, lrs, X, apply_calibration=True)


def _execute_training_evaluation_export(
    cfg: dict[str, Any],
    *,
    on_epoch_end: Any | None = None,
    on_variant_start: Any | None = None,
    success_slot: Any | None = None,
    emit_final_success: bool = True,
) -> None:
    """Split estratificado, treino do bagging de N MLPs aleatorias,
    metricas de teste e exportacao automatica do bundle."""
    from sklearn.model_selection import train_test_split as _tts

    df = st.session_state.get("df")
    X = st.session_state.get("feature_matrix")
    if df is None or X is None:
        st.error("Dataset ou matriz de features ausente.")
        return

    y = df[LABEL_COLUMN_CANON].to_numpy().astype(np.int64)
    split_cfg = SplitConfig(
        test_size=cfg["test_size"], val_size=cfg["val_size"], seed=cfg["seed"],
    )
    bal = BalancingConfig(
        undersample_neg_ratio=cfg["under_ratio"],
        oversample_pos_factor=cfg["over_factor"],
        training_balance=str(cfg.get("training_balance", "equal")),
        use_class_weight=cfg["use_cw"],
        pos_weight_override=cfg["pos_weight_override"],
        seed=cfg["seed"],
    )
    # Hiperparametros base; cada MLP do bagging sobreescreve lr / wd /
    # batch_size com valores aleatorios (ver `train_random_mlp_bagging`).
    train_cfg = TrainConfig(
        epochs=cfg["epochs"],
        batch_size=256,
        lr=1e-3,
        weight_decay=1e-4,
        early_stopping_patience=cfg["es_patience"],
        seed=cfg["seed"],
        architecture="mlp",
        loss_name=cfg.get("loss_name", "bce"),
        focal_alpha=cfg.get("focal_alpha", 0.25),
        focal_gamma=cfg.get("focal_gamma", 2.0),
        label_smoothing=cfg.get("label_smoothing", 0.05),
        mixup_pos=cfg.get("mixup_pos", 0.0),
        hard_neg_mining=cfg.get("hard_neg_mining", False),
        symmetry_aug=cfg.get("symmetry_aug", False),
    )

    st.session_state.stop_train = False
    all_idx = np.arange(len(y))
    idx_tv, idx_test = _tts(
        all_idx, test_size=split_cfg.test_size,
        random_state=split_cfg.seed, stratify=y,
    )
    val_rel = split_cfg.val_size / (1.0 - split_cfg.test_size)
    idx_train, idx_val = _tts(
        idx_tv, test_size=val_rel,
        random_state=split_cfg.seed, stratify=y[idx_tv],
    )
    X_train, y_train = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]
    X_test, y_test = X[idx_test], y[idx_test]

    st.session_state.splits = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
        "split_cfg": split_cfg,
        "bal": bal,
        "idx_train": idx_train,
        "idx_val": idx_val,
        "idx_test": idx_test,
    }

    history: list[dict[str, Any]] = []

    def _default_epoch(info: dict[str, Any]) -> None:
        history.append(info)
        st.session_state.history = list(history)

    epoch_cb = on_epoch_end if on_epoch_end is not None else _default_epoch

    feature_names_local = st.session_state.get("feature_names") or []
    n_models = max(1, min(100, int(cfg.get("n_per_kind", 3))))
    ranges: MLPRandomRanges = cfg.get("mlp_ranges") or MLPRandomRanges()
    lr_ranges: LogRegRandomRanges = (
        cfg.get("logreg_ranges") or LogRegRandomRanges()
    )

    def _on_variant_start_mlp(key: str, i: int, total: int) -> None:
        if on_variant_start is None:
            return
        try:
            on_variant_start(key, i, total, "mlp")
        except TypeError:
            on_variant_start(key, i, total)

    def _on_variant_start_lr(key: str, i: int, total: int) -> None:
        if on_variant_start is None:
            return
        try:
            on_variant_start(key, i, total, "logreg")
        except TypeError:
            on_variant_start(key, i, total)

    with st.spinner(
        f"Treinando bagging hibrido: {n_models} MLPs + {n_models} "
        f"regressoes logisticas (score final = media calibrada)..."
    ):
        members_ab = train_random_mlp_bagging(
            X_train, y_train, X_val, y_val,
            train_cfg, bal, list(feature_names_local),
            n_models=n_models,
            ranges=ranges,
            base_seed=int(cfg["seed"]),
            calibrate=True,
            on_variant_start=_on_variant_start_mlp,
            on_epoch_end=epoch_cb,
            should_stop=lambda: bool(st.session_state.stop_train),
        )
        members_lr = train_random_logreg_bagging(
            X_train, y_train, X_val, y_val,
            bal, list(feature_names_local),
            n_models=n_models,
            ranges=lr_ranges,
            base_seed=int(cfg["seed"]),
            calibrate=True,
            on_variant_start=_on_variant_start_lr,
            should_stop=lambda: bool(st.session_state.stop_train),
        )
    if not members_ab and not members_lr:
        st.error(
            "Bagging hibrido nao completou nenhum modelo "
            "(MLPs e Logisticas falharam)."
        )
        return
    if not members_ab:
        st.warning(
            "Nenhuma MLP foi treinada com sucesso; bagging usara apenas as "
            "regressoes logisticas."
        )
    if not members_lr:
        st.warning(
            "Nenhuma regressao logistica foi treinada com sucesso; bagging "
            "usara apenas as MLPs."
        )

    if members_ab:
        model = members_ab[0].model
    else:
        # Fallback: precisamos de um nn.Module dummy em session_state.model
        # apenas para artefatos e checks de pre-condicao da UI legada.
        # Usamos a primeira logreg e marcamos no config.
        from src.model.mlp import BrandSimilarityMLP, MLPConfig as _MLPConfig
        model = BrandSimilarityMLP(
            _MLPConfig(input_dim=int(X.shape[1]))
        )
    mlp_pr_vals = [m.best_pr_auc_val for m in members_ab] if members_ab else []
    lr_pr_vals = [m.train_metric_val for m in members_lr] if members_lr else []
    all_pr = mlp_pr_vals + lr_pr_vals
    mean_pr = float(np.mean(all_pr)) if all_pr else 0.0
    last_epoch = (
        int(members_ab[-1].best_epoch) if members_ab else 0
    )
    last_history = (
        list(members_ab[-1].history) if members_ab else []
    )
    result = type(
        "RandomMLPBagResult", (),
        {
            "best_epoch": last_epoch,
            "best_pr_auc_val": mean_pr,
            "history": last_history,
            "pos_weight_used": 1.0,
            "n_train_after_balancing": 0,
        },
    )()

    st.session_state.arch_bagging_members = members_ab
    st.session_state.logreg_bagging_members = members_lr
    st.session_state.ensemble_members = None
    st.session_state.calibrator = None
    st.session_state.model = model
    st.session_state.train_result = result
    st.session_state.train_cfg = train_cfg
    if members_ab:
        st.session_state.mlp_cfg = MLPConfig(
            input_dim=X.shape[1],
            hidden_dims=list(
                members_ab[0].arch_dict.get("hidden_dims", [128, 64, 32])
            ),
            dropout=float(members_ab[0].arch_dict.get("dropout", 0.3)),
            use_batchnorm=bool(
                members_ab[0].arch_dict.get("use_batchnorm", True)
            ),
            activation=str(
                members_ab[0].arch_dict.get("activation", "relu")
            ),
        )
    else:
        st.session_state.mlp_cfg = MLPConfig(input_dim=X.shape[1])

    val_scores = predict_hybrid_bagging(
        members_ab, members_lr, X_val, apply_calibration=True,
    )
    thr_opt, thr_policy = find_optimal_threshold(
        y_val, val_scores, recall_floor=cfg["recall_floor"],
    )
    st.session_state.threshold_optimal = float(thr_opt)
    st.session_state.threshold_policy = thr_policy

    test_scores = predict_hybrid_bagging(
        members_ab, members_lr, X_test, apply_calibration=True,
    )
    em = compute_metrics_at_threshold(y_test, test_scores, threshold=thr_opt)
    st.session_state.eval_metrics_test = em

    _export_ok = False
    try:
        _rid = new_pipeline_run_id()
        _dirs = prepare_run_directories(PROJECT_ROOT, _rid)
        _imp_top, _meta = export_pipeline_bundle(
            project_root=PROJECT_ROOT,
            dirs=_dirs,
            df=st.session_state.df,
            X=st.session_state.feature_matrix,
            names=list(st.session_state.feature_names or []),
            splits=st.session_state.splits,
            model=st.session_state.model,
            preproc=st.session_state.preprocessor,
            train_cfg=st.session_state.train_cfg,
            train_result=st.session_state.train_result,
            threshold_opt=float(thr_opt),
            threshold_policy=thr_policy,
            df_report=st.session_state.get("df_report"),
            history=list(st.session_state.history) if st.session_state.history else None,
            ensemble_members=None,
            arch_bagging_members=st.session_state.get("arch_bagging_members"),
            logreg_bagging_members=st.session_state.get(
                "logreg_bagging_members"
            ),
            calibrator=None,
            importance_top=st.session_state.get("importance_top"),
            auto_imp=True,
            progress_callback_enriched=None,
        )
        if _imp_top is not None:
            st.session_state.importance_top = _imp_top
        append_registry(PROJECT_ROOT, {"id": _rid, **_meta})
        mirror_latest_to_artifacts(PROJECT_ROOT, _dirs)
        st.session_state.pipeline_run_id = _rid
        st.session_state.pipeline_run_dir = str(_dirs["root"])
        _export_ok = True
    except Exception as _pex:  # noqa: BLE001
        logger.exception("Gravacao automatica do pipeline falhou: %s", _pex)

    _pdir = st.session_state.get("pipeline_run_dir")
    if _export_ok and _pdir:
        _pextra = f" Artefatos: `{_pdir}` (+ espelho em `artifacts/`)."
    elif not _export_ok:
        _pextra = " Aviso: exportacao automatica do pipeline falhou (ver logs)."
    else:
        _pextra = ""
    msg = (
        f"Treino concluido (bagging hibrido: {len(members_ab)} MLPs + "
        f"{len(members_lr)} regressoes logisticas, calibrado Platt por "
        f"modelo). Epoca otima da ultima MLP: {result.best_epoch}, val "
        f"PR-AUC medio (todos os modelos) = {result.best_pr_auc_val:.4f}. "
        f"Threshold otimo (val) = {thr_opt:.3f}. Test PR-AUC = {em.pr_auc:.4f}, "
        f"Recall = {em.recall:.3f}.{_pextra}"
    )
    if not emit_final_success:
        return
    if success_slot is not None:
        success_slot.success(msg)
    else:
        st.success(msg)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
TABS = [
    "1. Upload & EDA",
    "2. Enriquecimento",
    "3. Treino",
    "4. Avaliacao",
    "5. Explicabilidade",
    "6. Teste manual",
    "7. Artefatos",
]


def tab_upload_eda(cfg: dict[str, Any]) -> None:
    st.header("1. Upload & EDA")
    _render_pipeline_progress()
    st.caption(
        "**Fluxo manual:** carregue o dataset aqui e depois siga para a aba "
        "**2. Enriquecimento**.",
    )

    default_path = PROJECT_ROOT / "dataframe_final_colidencias.xlsx"
    use_default = st.checkbox(
        "Usar dataset versionado do projeto (`dataframe_final_colidencias.xlsx`)",
        value=default_path.exists(),
    )

    file_bytes: bytes | None = None
    if use_default and default_path.exists():
        file_bytes = default_path.read_bytes()
    else:
        upload = st.file_uploader("Carregue o Excel do INPI", type=["xlsx"])
        if upload is not None:
            file_bytes = upload.read()

    if file_bytes and st.button("Carregar e validar dataset", type="primary"):
        try:
            with StepTimer("dataset"):
                with st.spinner("Carregando e validando..."):
                    df, report = load_dataframe_from_bytes(file_bytes)
                _reset_downstream_pipeline_state()
                st.session_state.df = df
                st.session_state.df_report = report
                _set_step_status("dataset", "ok")
        except Exception as exc:  # noqa: BLE001
            _set_step_status("dataset", "error")
            logger.exception("Falha ao carregar dataset: %s", exc)
            st.error(f"Falha ao carregar dataset: {exc}")
            return

    df: pd.DataFrame | None = st.session_state.get("df")
    report: DatasetReport | None = st.session_state.get("df_report")
    if df is None or report is None:
        st.info("Carregue um dataset para iniciar.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Linhas", f"{report.n_rows:,}")
    c2.metric("Positivos", f"{report.n_pos:,}", f"{report.pos_rate*100:.2f}%")
    c3.metric("Negativos", f"{report.n_neg:,}")
    c4.metric("Mesmo classe Nice", f"{report.same_class_share*100:.2f}%")

    st.caption(f"Hash do dataset: `{report.dataset_hash}`")
    if report.n_rows_dropped_null_brands:
        st.warning(f"{report.n_rows_dropped_null_brands} linhas removidas por marca nula/vazia.")

    st.subheader("Distribuicao por classe Nice")
    classes_df = pd.DataFrame(report.classes_top, columns=["classe", "ocorrencias"])
    st.bar_chart(classes_df, x="classe", y="ocorrencias")

    st.subheader("Estatisticas de comprimento (caracteres)")
    st.dataframe(
        pd.DataFrame({**report.brand_len_stats, **report.spec_len_stats}).T.round(2),
        use_container_width=True,
    )

    st.subheader("Preview do dataset")
    st.dataframe(df.head(20), use_container_width=True)


def tab_enrichment(cfg: dict[str, Any]) -> None:
    st.header("2. Enriquecimento de features")
    _render_pipeline_progress()

    df = st.session_state.get("df")
    if df is None:
        st.info(
            "Carregue um dataset na aba **1. Upload & EDA** antes de "
            "executar esta etapa.",
        )
        return

    st.write(
        "Fitta TF-IDF/embeddings e gera a matriz de features na ordem canonica. "
        "Os defaults do sidebar favorecem qualidade (BERTimbau-large, "
        "TF-IDF largo, Sprint 2 completo); esta etapa pode demorar varios "
        "minutos."
    )

    drift = _render_preproc_drift_warning(cfg)
    if not drift and st.session_state.get("feature_matrix") is not None:
        st.info(
            "Matriz de features ja gerada nesta sessao. So precisa rerodar "
            "esta etapa se voce alterar **parametros do pre-processador** "
            "no sidebar (TF-IDF, embeddings, top-K classes, generic_df). "
            "Mudancas em parametros de **treino** nao exigem reenriquecer.",
        )

    if st.button("Gerar features", type="primary"):
        prog = st.progress(0, text="A iniciar enriquecimento...")

        def _on_feat_progress(p: float, msg: str) -> None:
            prog.progress(
                min(1.0, max(0.0, float(p))), text=str(msg)[:95],
            )

        try:
            with StepTimer("enrichment"):
                _run_feature_enrichment(
                    cfg, df, progress_callback=_on_feat_progress,
                )
            prog.progress(1.0, text="Features geradas com sucesso.")
            X = st.session_state.get("feature_matrix")
            names = st.session_state.get("feature_names") or []
            st.success(
                f"Matriz gerada: shape "
                f"{X.shape if X is not None else '?'}, "
                f"{len(names)} features. Proxima etapa: aba **3. Treino**.",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Enriquecimento falhou: %s", exc)
            st.error(f"Enriquecimento falhou: {exc}")
            prog.progress(0.0, text="Erro.")
            return

    preproc: FeaturePreprocessor | None = st.session_state.get("preprocessor")
    X = st.session_state.get("feature_matrix")
    names = st.session_state.get("feature_names")
    if preproc is None or X is None:
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Pares", f"{X.shape[0]:,}")
    c2.metric("Features", f"{X.shape[1]:,}")
    c3.metric("Top-K classes", len(preproc.top_classes))

    with st.expander("Top classes Nice (apos fit)"):
        st.write(preproc.top_classes)

    st.subheader("Preview da matriz de features (primeiras 20 linhas)")
    preview = pd.DataFrame(X[:20], columns=names)
    st.dataframe(preview, use_container_width=True)


def _render_production_verdict(em) -> None:
    """Avalia as metricas de teste e exibe um veredito de prontidao para producao.

    Faixas calibradas para o problema (recall positivo e prioridade, base ~10% pos):
      - VERDE: pronto para producao
      - AMARELO: utilizavel, mas exige monitoramento ou ajuste
      - VERMELHO: nao recomendado para producao
    """
    issues_red: list[str] = []
    issues_yellow: list[str] = []
    wins_green: list[str] = []

    if em.roc_auc >= 0.90:
        wins_green.append(f"ROC-AUC {em.roc_auc:.3f} (forte capacidade de separar classes)")
    elif em.roc_auc >= 0.80:
        issues_yellow.append(f"ROC-AUC {em.roc_auc:.3f} (aceitavel, mas pode melhorar)")
    else:
        issues_red.append(f"ROC-AUC {em.roc_auc:.3f} abaixo de 0.80 (separabilidade fraca)")

    if em.pr_auc >= 0.65:
        wins_green.append(f"PR-AUC {em.pr_auc:.3f} (excelente para base desbalanceada)")
    elif em.pr_auc >= 0.45:
        wins_green.append(f"PR-AUC {em.pr_auc:.3f} (bom para base desbalanceada)")
    elif em.pr_auc >= 0.30:
        issues_yellow.append(f"PR-AUC {em.pr_auc:.3f} (aceitavel; monitorar precision/recall)")
    else:
        issues_red.append(f"PR-AUC {em.pr_auc:.3f} muito baixo (modelo perto do baseline aleatorio)")

    if em.recall >= 0.85:
        wins_green.append(f"Recall {em.recall:.3f} atende o piso operacional de 85%")
    elif em.recall >= 0.70:
        issues_yellow.append(
            f"Recall {em.recall:.3f} abaixo do piso 0.85 - alguns colidentes serao perdidos"
        )
    else:
        issues_red.append(
            f"Recall {em.recall:.3f} muito baixo (>= 30% dos colidentes nao seriam capturados)"
        )

    if em.f1 >= 0.45:
        wins_green.append(f"F1 {em.f1:.3f} (bom equilibrio)")
    elif em.f1 >= 0.30:
        issues_yellow.append(f"F1 {em.f1:.3f} (equilibrio modesto entre precisao e recall)")
    else:
        issues_red.append(f"F1 {em.f1:.3f} muito baixo")

    cm = em.confusion
    fn = cm[1][0]
    fp = cm[0][1]
    tp = cm[1][1]
    tn = cm[0][0]
    if (tp + fn) > 0 and fn / (tp + fn) > 0.30:
        issues_red.append(
            f"Falsos Negativos: {fn} de {tp + fn} colidentes ({100 * fn / (tp + fn):.1f}%) - "
            "passariam despercebidos em producao"
        )

    st.subheader("Veredito de prontidao para producao")
    if issues_red:
        st.error(
            "**NAO RECOMENDADO para producao** - corrigir os pontos abaixo antes de prosseguir:\n\n"
            + "\n".join(f"- {x}" for x in issues_red)
            + ("\n\n_Pontos secundarios:_\n" + "\n".join(f"- {x}" for x in issues_yellow) if issues_yellow else "")
        )
        st.caption(
            "Sugestoes: aumentar epocas, ajustar pos_weight, revisar features das especificacoes, "
            "considerar mais dados rotulados ou usar arquitetura maior."
        )
    elif issues_yellow:
        st.warning(
            "**ACEITAVEL com monitoramento** - bom para um piloto, mas observe os pontos abaixo:\n\n"
            + "\n".join(f"- {x}" for x in issues_yellow)
            + ("\n\n_Pontos fortes:_\n" + "\n".join(f"- {x}" for x in wins_green) if wins_green else "")
        )
        st.caption(
            "Sugestoes: monitorar drift de score em producao, retreinar a cada novo lote de "
            "rotulos, considerar ajuste fino do threshold com a area de negocio."
        )
    else:
        st.success(
            "**PRONTO para producao** - o modelo atende aos criterios de qualidade definidos:\n\n"
            + "\n".join(f"- {x}" for x in wins_green)
        )
        st.caption(
            f"Threshold de operacao recomendado: {em.threshold:.3f} "
            "(ajustado para recall >= 0.85, conforme politica do projeto)."
        )


def _draw_history(history: list[dict[str, float]]) -> None:
    if not history:
        return
    df = pd.DataFrame(history)

    fig = go.Figure()
    if "train_loss" in df.columns:
        fig.add_trace(go.Scatter(x=df["epoch"], y=df["train_loss"], name="train_loss"))
    fig.update_layout(title="Loss por epoca", xaxis_title="epoca", yaxis_title="loss")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    for col, name in [
        ("val_pr_auc", "val PR-AUC"),
        ("val_roc_auc", "val ROC-AUC"),
        ("val_recall@0.5", "val Recall@0.5"),
        ("val_f1@0.5", "val F1@0.5"),
    ]:
        if col in df.columns:
            fig2.add_trace(go.Scatter(x=df["epoch"], y=df[col], name=name))
    fig2.update_layout(title="Metricas de validacao", xaxis_title="epoca")
    st.plotly_chart(fig2, use_container_width=True)


def tab_train(cfg: dict[str, Any]) -> None:
    st.header("3. Treino em tempo real")
    _render_pipeline_progress()

    st.markdown(
        """
**O que esta aba faz**

Aqui o modelo aprende, a partir dos exemplos rotulados pelo INPI, qual combinacao de
sinais (escrita parecida, som parecido, especificacao parecida, classe igual etc.)
indica colidencia. O treino acontece em **epocas** - cada epoca e uma passada
completa pelos pares de marcas. A cada epoca o modelo ajusta seus pesos um pouco
para errar menos.

**Padroes atuais (qualidade > velocidade)**

- Balanceamento **50/50** no treino (positivos com reposicao ate igualar negativos apos undersample).
- **Bagging hibrido**: para o N escolhido no sidebar treina-se **N MLPs aleatorias + N regressoes logisticas aleatorias** (total = 2 X N modelos). Cada modelo e' calibrado por Platt no validation; o score final e' a **media das 2 X N probabilidades calibradas**.
- MLPs sorteiam arquitetura, dropout, ativacao, lr, weight decay e batch size; logisticas sorteiam C, penalty (l1/l2) e class_weight (none/balanced).
- Loss **Focal + label smoothing** (so MLPs), mixup de positivos, **hard negative mining** e **simetria (A,B)/(B,A)**.
- Mais epocas e paciencia de early stopping; embeddings **BERTimbau-large** e TF-IDF mais largos no enriquecimento.

**O que voce ve abaixo**

- **Loss por epoca** (apenas MLPs): as logisticas treinam por solver fechado (sem epocas),
  entao os graficos de loss/metricas por epoca cobrem somente as MLPs do bagging.
  **Loss deve cair com o tempo**. Se ficar oscilando ou crescer, o aprendizado nao esta convergindo.
- **Metricas de validacao**: medidas em pares que o modelo NAO viu durante o treino,
  para detectar se ele esta apenas decorando os exemplos (overfit).
  - **PR-AUC**: principal metrica neste projeto. Deve **subir** com as epocas.
  - **ROC-AUC**: capacidade geral de separar colidentes de nao-colidentes.
  - **Recall@0.5**: % de colidentes verdadeiros que o modelo capturaria com corte 0.5.
  - **F1@0.5**: equilibrio entre precisao e recall.

**Sinais de bom treino (verde para producao)**

- Loss caindo de forma consistente nas primeiras epocas.
- PR-AUC e ROC-AUC de validacao subindo.
- Diferenca entre treino e validacao moderada (sinal de generalizacao).
- O processo encerra por **early stopping** (para sozinho quando para de melhorar) -
  isso e desejado, indica que o modelo encontrou seu melhor ponto.

**Sinais de alerta (vermelho - nao ir para producao)**

- Loss subindo ou oscilando muito.
- PR-AUC de validacao abaixo de 0.30 ao final do treino (lembrando que a base
  positiva e ~10%, entao o baseline aleatorio fica em torno de 0.10).
- Recall@0.5 muito baixo (< 0.30) - o modelo esta deixando passar muitos colidentes.
- Treino completa todas as epocas sem early stopping E PR-AUC ainda esta subindo:
  pode ter sido cortado cedo demais; aumente "Epochs maximas" no sidebar.

> Apos o treino terminar, va para a **aba 4 (Avaliacao)** para ver as metricas no
> conjunto de teste (que nao foi usado em nenhuma decisao do treino) - esse e o
> numero que vale para decidir producao.
        """
    )

    df = st.session_state.get("df")
    X = st.session_state.get("feature_matrix")
    if df is None or X is None:
        st.info(
            "Carregue o dataset (aba 1) e gere as features (aba 2) antes "
            "de treinar.",
        )
        return

    _render_preproc_drift_warning(cfg)

    n_models = max(1, min(100, int(cfg.get("n_per_kind", 3))))
    st.caption(
        f"Sera treinado um bagging hibrido de **{n_models} MLPs + {n_models} "
        f"regressoes logisticas** (total = **{2 * n_models} modelos**) "
        f"reusando a matriz de features ja em cache "
        f"({X.shape[0]:,} pares X {X.shape[1]:,} features). "
        f"Apos o sucesso, os artefatos sao gravados automaticamente em "
        f"`pipelines/<run_id>/` e espelhados em `artifacts/`.",
    )

    c1, c2 = st.columns(2)
    train_btn = c1.button("Treinar bagging hibrido", type="primary")
    stop_btn = c2.button("Parar treino")
    if stop_btn:
        st.session_state.stop_train = True

    chart_loss = st.empty()
    chart_metrics = st.empty()
    table_slot = st.empty()
    status_slot = st.empty()

    if train_btn:
        st.session_state.stop_train = False
        st.session_state.history = []

        history: list[dict[str, float]] = []

        def on_epoch_end(info: dict[str, Any]) -> None:
            history.append(info)
            st.session_state.history = list(history)
            df_h = pd.DataFrame(history)
            with chart_loss.container():
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_h["epoch"], y=df_h["train_loss"], name="train_loss",
                ))
                fig.update_layout(
                    title="Loss por epoca", xaxis_title="epoca",
                    yaxis_title="loss", height=320,
                )
                st.plotly_chart(fig, use_container_width=True)
            with chart_metrics.container():
                fig2 = go.Figure()
                for col, name in [
                    ("val_pr_auc", "val PR-AUC"),
                    ("val_roc_auc", "val ROC-AUC"),
                    ("val_recall@0.5", "val Recall@0.5"),
                    ("val_f1@0.5", "val F1@0.5"),
                ]:
                    if col in df_h.columns:
                        fig2.add_trace(go.Scatter(
                            x=df_h["epoch"], y=df_h[col], name=name,
                        ))
                fig2.update_layout(
                    title="Metricas de validacao", xaxis_title="epoca",
                    height=320,
                )
                st.plotly_chart(fig2, use_container_width=True)
            with table_slot.container():
                st.dataframe(df_h.tail(10), use_container_width=True)
            status_slot.info(
                f"Epoca {info['epoch']} concluida "
                f"(val PR-AUC = {info['val_pr_auc']:.4f}).",
            )

        def on_variant_start(
            key: str, i: int, total: int, kind: str = "mlp",
        ) -> None:
            kind_label = (
                "MLP" if kind == "mlp" else "Regressao Logistica"
            )
            tail = (
                "nova amostra de arquitetura/hiperparametros; "
                "treino + epocas abaixo referem-se a esta rede."
                if kind == "mlp" else
                "nova amostra de C/penalty/class_weight; "
                "treino fechado (sem epocas) sobre o split balanceado."
            )
            status_slot.info(
                f"**Bagging:** {kind_label} **{i} de {total}** (`{key}`) — "
                f"{tail}"
            )

        try:
            with StepTimer("training"):
                _execute_training_evaluation_export(
                    cfg,
                    on_epoch_end=on_epoch_end,
                    on_variant_start=on_variant_start,
                    success_slot=status_slot,
                )
            # Avaliacao roda dentro do mesmo passo; marcamos OK explicitamente.
            _set_step_status("evaluation", "ok")
            _set_step_status("artifacts", "ok")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Treino falhou: %s", exc)
            st.error(f"Treino falhou: {exc}")
            return

    elif st.session_state.history:
        _draw_history(st.session_state.history)


def tab_evaluation() -> None:
    st.header("4. Avaliacao")
    _render_pipeline_progress()

    st.markdown(
        """
**O que esta aba faz**

Mede a qualidade real do modelo em pares que ele **nunca viu** durante o treino
(conjunto de teste). Esse e o numero que vale para a decisao de levar ou nao
para producao - tudo o que aparece aqui e cego ao processo de aprendizado.

**Como ler cada metrica (em palavras simples)**

- **ROC-AUC** (0 a 1): probabilidade do modelo dar score maior para um colidente
  real do que para um nao-colidente, escolhendo dois pares aleatorios. **0.5 e
  chute, 1.0 e perfeito.**
- **PR-AUC** (0 a 1): metrica que prioriza acerto na classe rara (colidentes).
  **A mais importante neste projeto** porque so 10% dos pares sao colidentes.
  Baseline aleatorio ~ 0.10.
- **Recall** (0 a 1): de todos os colidentes verdadeiros, qual % o modelo
  capturou. **Foco principal do negocio** - falso negativo e o erro mais grave.
- **F1** (0 a 1): equilibrio entre precisao (quantos dos alarmados sao reais)
  e recall (quantos dos reais foram alarmados).

**Slider de threshold**: e o ponto de corte que vira "sim, colide" (1) ou "nao
colide" (0). Mover o slider muda dinamicamente a matriz de confusao e as
metricas - serve para entender o trade-off antes de gravar a decisao final.

**Tabela de matriz de confusao**

| Celula | Significado |
|---|---|
| Real 0 / Pred 0 | Verdadeiro Negativo (acertou que nao colide) |
| Real 0 / Pred 1 | **Falso Positivo** (alarme falso - o time vai investigar a toa) |
| Real 1 / Pred 0 | **Falso Negativo** - O ERRO MAIS GRAVE (deixou passar colidente) |
| Real 1 / Pred 1 | Verdadeiro Positivo (capturou colidente) |

**Curva ROC** mostra TPR vs FPR para todos os thresholds; quanto mais "puxada para
cima e para a esquerda", melhor. **Curva PR** mostra Precision vs Recall; quanto
mais "puxada para cima e para a direita", melhor.

**Histograma de scores** mostra a distribuicao do score (NN) para positivos vs
negativos; idealmente os positivos (em uma cor) ficam concentrados a direita
(scores altos) e os negativos a esquerda (scores baixos), com pouca sobreposicao.
        """
    )

    model = st.session_state.get("model")
    splits = st.session_state.get("splits")
    ensemble_members = st.session_state.get("ensemble_members")
    mlp_members, lr_members = _get_bagging_lists()
    calibrator = st.session_state.get("calibrator")
    if model is None or splits is None:
        st.info("Treine o modelo na aba **3. Treino** primeiro.")
        return

    X_test, y_test = splits["test"]
    if mlp_members or lr_members:
        scores_test = predict_hybrid_bagging(
            mlp_members, lr_members, X_test, apply_calibration=True,
        )
        st.caption(
            f"Bagging hibrido: {len(mlp_members)} MLPs + "
            f"{len(lr_members)} regressoes logisticas. Score = media "
            f"das {len(mlp_members) + len(lr_members)} probabilidades "
            "calibradas (Platt por modelo)."
        )
    elif ensemble_members:
        scores_test = predict_ensemble(
            ensemble_members, X_test, apply_calibration=True,
        )
        st.caption(
            f"Modo ensemble (legacy) ativo ({len(ensemble_members)} membros).",
        )
    else:
        scores_test = predict_scores(model, X_test)
        if calibrator is not None and calibrator.fitted:
            scores_test = calibrator.transform(scores_test)
            st.caption("Calibracao Platt aplicada nos scores.")

    threshold = st.slider(
        "Threshold (slider dinamico)",
        0.05, 0.95, float(st.session_state.threshold_optimal), 0.01,
        help="O threshold otimo identificado no validacao esta marcado como default.",
    )

    em = compute_metrics_at_threshold(y_test, scores_test, threshold=threshold)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", f"{em.roc_auc:.4f}")
    c2.metric("PR-AUC", f"{em.pr_auc:.4f}")
    c3.metric("Recall", f"{em.recall:.4f}")
    c4.metric("F1", f"{em.f1:.4f}")

    _render_production_verdict(em)

    cm = em.confusion
    cell_labels = [["TN", "FP"], ["FN", "TP"]]
    cm_labelled = [
        [
            f"{cell_labels[r][c]} ({cm[r][c]:,})"
            for c in range(2)
        ]
        for r in range(2)
    ]
    cm_df = pd.DataFrame(
        cm_labelled,
        index=[
            "Real 0 = nao colidente",
            "Real 1 = colidente",
        ],
        columns=[
            "Pred 0 = nao colidente",
            "Pred 1 = colidente",
        ],
    )
    st.subheader("Matriz de confusao (test)")
    st.dataframe(cm_df, use_container_width=True)
    st.caption(
        "Legenda: **TN** verdadeiro negativo | **FP** falso positivo "
        "| **FN** falso negativo | **TP** verdadeiro positivo. "
        "Valores entre parenteses sao contagens de pares."
    )

    st.subheader("Curva ROC")
    roc = roc_curve_data(y_test, scores_test)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], name=f"ROC (AUC={roc['auc']:.3f})"))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="random", line=dict(dash="dash")))
    fig_roc.update_layout(xaxis_title="FPR", yaxis_title="TPR", height=400)
    st.plotly_chart(fig_roc, use_container_width=True)

    st.subheader("Curva Precision-Recall")
    pr = pr_curve_data(y_test, scores_test)
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=pr["recall"], y=pr["precision"], name=f"PR (AP={pr['auc']:.3f})"))
    fig_pr.update_layout(xaxis_title="Recall", yaxis_title="Precision", height=400)
    st.plotly_chart(fig_pr, use_container_width=True)

    st.subheader("Histograma de scores (positivos vs negativos)")
    fig_h = go.Figure()
    fig_h.add_trace(go.Histogram(x=scores_test[y_test == 0], name="neg", opacity=0.6, nbinsx=40))
    fig_h.add_trace(go.Histogram(x=scores_test[y_test == 1], name="pos", opacity=0.6, nbinsx=40))
    fig_h.update_layout(barmode="overlay", height=350)
    st.plotly_chart(fig_h, use_container_width=True)

    st.subheader("Comparativo NN vs heuristica OFTA (test)")
    if st.button("Calcular comparativo (pode levar alguns segundos)"):
        df = st.session_state.get("df")
        with st.spinner("Calculando heuristica OFTA para conjunto de teste..."):
            test_indices = splits.get("test_indices")
            if test_indices is None:
                ofta_scores = []
                ma = df["marca_monitorada"].astype(str).tolist()
                mb = df["marca_colidente"].astype(str).tolist()
                rng = np.random.default_rng(42)
                idx_full = rng.permutation(len(df))[: len(y_test)]
                for i in idx_full:
                    ofta_scores.append(calcular_similaridade_ofta(ma[i], mb[i]))
                ofta_arr = np.array(ofta_scores, dtype=np.float64)
            else:
                ma = df["marca_monitorada"].astype(str).tolist()
                mb = df["marca_colidente"].astype(str).tolist()
                ofta_arr = np.array(
                    [calcular_similaridade_ofta(ma[i], mb[i]) for i in test_indices],
                    dtype=np.float64,
                )
        comp = compare_against_heuristic(y_test, scores_test, ofta_arr)
        st.session_state.comparison = comp

    comp_saved = st.session_state.get("comparison")
    if comp_saved:
        tprec = float(comp_saved.get("target_precision", 0.9))
        cmp_tbl = pd.DataFrame(
            [
                {
                    "Metrica": "ROC-AUC",
                    "Rede neural": round(float(comp_saved["nn_roc_auc"]), 4),
                    "Heuristica OFTA": round(float(comp_saved["ofta_roc_auc"]), 4),
                    "Delta (NN - OFTA)": round(float(comp_saved["delta_roc_auc"]), 4),
                },
                {
                    "Metrica": "PR-AUC",
                    "Rede neural": round(float(comp_saved["nn_pr_auc"]), 4),
                    "Heuristica OFTA": round(float(comp_saved["ofta_pr_auc"]), 4),
                    "Delta (NN - OFTA)": round(float(comp_saved["delta_pr_auc"]), 4),
                },
                {
                    "Metrica": f"Recall @ Precision >= {tprec:.2f}",
                    "Rede neural": round(float(comp_saved["nn_recall_at_p90"]), 4),
                    "Heuristica OFTA": round(float(comp_saved["ofta_recall_at_p90"]), 4),
                    "Delta (NN - OFTA)": round(
                        float(comp_saved["delta_recall_at_p90"]), 4,
                    ),
                },
            ],
        )
        st.dataframe(cmp_tbl, hide_index=True, use_container_width=True)
        st.caption(
            f"Criterio do Recall na ultima linha: precision minima **{tprec:.2f}**. "
            "Delta positivo = NN melhor que OFTA nessa metrica.",
        )


def tab_explainability() -> None:
    st.header("5. Explicabilidade")
    _render_pipeline_progress()

    st.markdown(
        """
**O que esta aba faz**

Mostra **em quais features o modelo realmente confiou** para tomar decisoes. Isso
e essencial em projetos juridicos: precisamos poder defender por que um par foi
classificado como colidente ou nao.

**Permutation Importance (em palavras simples)**

Para cada feature, o sistema:
1. Embaralha aleatoriamente os valores daquela feature (so daquela coluna).
2. Mede quanto a metrica do modelo (PR-AUC ou ROC-AUC) PIORA depois do embaralhamento.
3. Quanto mais a metrica piora, mais o modelo dependia daquela feature.

**Como interpretar a tabela e o grafico**

- **Importance positiva alta**: feature critica - o modelo perde performance se
  ela e perturbada. Idealmente, voce ve aqui as features que fazem sentido de
  negocio: `ofta_final`, `spec_cosine_emb`, `cls_same`, `inter_*`, etc.
- **Importance proxima de zero**: feature redundante ou sem sinal - o modelo nao
  esta usando.
- **Importance negativa**: extremamente raro; pode indicar ruido na amostra ou
  feature que esta confundindo levemente. Geralmente ignoravel.

**Sinais BONS para producao (modelo confiavel)**

- As features no topo sao **interpretaveis** e fazem sentido para o negocio.
- Existe **diversidade de blocos** no top-30 (gráfica + fonetica + spec + classe +
  interacoes), nao apenas um tipo dominante.
- Features de interacao (`inter_*`) aparecem entre as importantes - significa que
  a rede aprendeu combinacoes nao-triviais, nao so similaridade nominativa.

**Sinais de ALERTA**

- O top e dominado por **uma unica feature** (ex.: so `ofta_final`) - significa
  que a rede virou um disfarce da heuristica antiga, sem agregar valor.
- Features semanticas (`spec_cosine_emb`, `inter_classe_diff_mas_emb_alto`) com
  importance proxima de zero - significa que o sinal de especificacao esta sendo
  ignorado, e o modelo pode falhar nos cenarios de "risco oculto".
- Features de interacao todas com importance baixa - a rede nao aprendeu
  combinacoes; pode estar subdimensionada (poucas camadas/neuronios) ou treinada
  por poucas epocas.

> Observacao: parametros maiores (mais amostras, mais repeticoes) geram numeros
> mais estaveis, mas demoram mais. Comece com 200-800 amostras e 2 repeticoes.
        """
    )

    model = st.session_state.get("model")
    splits = st.session_state.get("splits")
    names = st.session_state.get("feature_names")
    mlp_members, lr_members = _get_bagging_lists()
    if model is None or splits is None:
        st.info("Treine o modelo na aba **3. Treino** primeiro.")
        return

    X_val, y_val = splits["val"]
    if mlp_members or lr_members:
        st.caption(
            f"Importancia calculada sobre o score agregado do bagging "
            f"hibrido (**{len(mlp_members)} MLPs + {len(lr_members)} "
            f"regressoes logisticas**, media calibrada Platt por modelo).",
        )

    sample_max = min(2000, len(y_val))
    sample = st.slider(
        "Amostra para permutation importance",
        200, sample_max, min(800, sample_max),
    )
    n_repeats = st.slider("Repetições", 1, 5, 2)
    metric = st.selectbox("Metrica", ["pr_auc", "roc_auc"], index=0)

    if st.button("Calcular permutation importance"):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(y_val), size=sample, replace=False)
        try:
            with StepTimer("explainability"):
                with st.spinner("Calculando importance..."):
                    if mlp_members or lr_members:
                        top = permutation_importance_hybrid(
                            mlp_members,
                            lr_members,
                            X_val[idx],
                            y_val[idx],
                            names,
                            metric=metric,
                            n_repeats=n_repeats,
                        )
                    else:
                        top = permutation_importance(
                            model, X_val[idx], y_val[idx], names,
                            metric=metric, n_repeats=n_repeats,
                        )
                st.session_state.importance_top = top
        except Exception as exc:  # noqa: BLE001
            logger.exception("Permutation importance falhou: %s", exc)
            st.error(f"Permutation importance falhou: {exc}")
            return

    top = st.session_state.get("importance_top")
    if top is not None:
        df = pd.DataFrame(attach_descriptions(top))
        st.subheader(
            f"Top 30 features por permutation importance ({metric})",
        )
        st.bar_chart(df.head(30).set_index("feature")["importance"])
        cols_show = [
            c for c in ("feature", "importance", "descricao") if c in df.columns
        ]
        st.dataframe(df[cols_show], use_container_width=True)


def tab_manual_test() -> None:
    st.header("6. Teste manual")
    _render_pipeline_progress()

    model = st.session_state.get("model")
    preproc: FeaturePreprocessor | None = st.session_state.get("preprocessor")
    names = st.session_state.get("feature_names")
    thr = st.session_state.get("threshold_optimal", 0.5)
    threshold = float(0.5 if thr is None else thr)

    if model is None or preproc is None:
        st.info(
            "Treine o modelo na aba **3. Treino** antes de usar o teste "
            "manual.",
        )
        return

    # ---- Picker de pares reais do conjunto de teste ----
    st.subheader("Selecionar par real da massa de teste")
    splits = st.session_state.get("splits")
    df_full = st.session_state.get("df")
    selected_idx: int | None = None
    if splits is not None and df_full is not None:
        idx_test = splits.get("idx_test")
        if idx_test is not None and len(idx_test) > 0:
            try:
                X_test, y_test = splits["test"]
                mlp_pre, lr_pre = _get_bagging_lists()
                if mlp_pre or lr_pre:
                    test_scores_full = predict_hybrid_bagging(
                        mlp_pre, lr_pre, X_test, apply_calibration=True,
                    )
                else:
                    test_scores_full = predict_scores(model, X_test)
                pred_lbl = (test_scores_full >= threshold).astype(int)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Falha ao prever scores de teste: %s", exc)
                test_scores_full = np.full(len(idx_test), np.nan)
                pred_lbl = np.full(len(idx_test), -1)

            classes_marca_mon = (
                df_full.get("classe_marca_monitorada")
                if "classe_marca_monitorada" in df_full.columns
                else pd.Series([-1] * len(df_full))
            )
            classes_marca_col = (
                df_full.get("classe_marca_colidente")
                if "classe_marca_colidente" in df_full.columns
                else pd.Series([-1] * len(df_full))
            )

            picker_rows = []
            for k, row_idx in enumerate(idx_test):
                row = df_full.iloc[int(row_idx)]
                ma = str(row.get("marca_monitorada", ""))
                mb = str(row.get("marca_colidente", ""))
                ca = int(classes_marca_mon.iloc[int(row_idx)])
                cb = int(classes_marca_col.iloc[int(row_idx)])
                yreal = int(row.get(LABEL_COLUMN_CANON, 0))
                score = float(test_scores_full[k]) if k < len(test_scores_full) else float("nan")
                yp = int(pred_lbl[k]) if k < len(pred_lbl) else -1
                if score == score:
                    cat = "TP" if (yreal == 1 and yp == 1) else (
                        "FN" if (yreal == 1 and yp == 0) else (
                            "FP" if (yreal == 0 and yp == 1) else "TN"
                        )
                    )
                    label = (
                        f"{cat} | score={score:.3f} | y={yreal} | "
                        f"{ma} (cl {ca}) X {mb} (cl {cb})"
                    )
                else:
                    label = f"y={yreal} | {ma} (cl {ca}) X {mb} (cl {cb})"
                picker_rows.append(
                    {"k": k, "row_idx": int(row_idx), "label": label}
                )

            cat_filter = st.selectbox(
                "Filtrar por categoria",
                options=["Todos", "TP", "FN", "FP", "TN"],
                index=0,
                help=(
                    "TP/FN/FP/TN avaliados pelo bagging atual com o "
                    "threshold otimo de validacao."
                ),
            )
            if cat_filter != "Todos":
                picker_rows = [
                    r for r in picker_rows
                    if r["label"].startswith(cat_filter + " ")
                ]
            picker_rows = picker_rows[:1500]

            if not picker_rows:
                st.caption(
                    "Nenhum par disponivel para esse filtro.",
                )
            else:
                opt_labels = [r["label"] for r in picker_rows]
                pick = st.selectbox(
                    f"Pares reais ({len(opt_labels)}):",
                    options=opt_labels,
                    index=0,
                )
                if pick:
                    chosen = next(
                        r for r in picker_rows if r["label"] == pick
                    )
                    selected_idx = int(chosen["row_idx"])
                    if st.button(
                        "Carregar este par nos campos abaixo",
                    ):
                        row = df_full.iloc[selected_idx]
                        st.session_state.manual_marca_a = str(
                            row.get("marca_monitorada", ""),
                        )
                        st.session_state.manual_marca_b = str(
                            row.get("marca_colidente", ""),
                        )
                        st.session_state.manual_classe_a = int(
                            classes_marca_mon.iloc[selected_idx],
                        )
                        st.session_state.manual_classe_b = int(
                            classes_marca_col.iloc[selected_idx],
                        )
                        st.session_state.manual_spec_a = str(
                            row.get("especificacao_monitorado", ""),
                        )
                        st.session_state.manual_spec_b = str(
                            row.get("especificacao_colidente", ""),
                        )
                        st.success(
                            "Par carregado nos campos. Clique em "
                            "**Calcular score**.",
                        )
        else:
            st.caption(
                "Indices do conjunto de teste indisponiveis. Re-treine o "
                "modelo para habilitar a selecao automatica.",
            )
    else:
        st.caption(
            "Conjunto de teste indisponivel. Treine o modelo para "
            "habilitar a selecao de pares reais.",
        )

    st.divider()

    c1, c2 = st.columns(2)
    marca_a = c1.text_input(
        "Marca A",
        value=st.session_state.get("manual_marca_a", "AGROLOG DISTRIBUIDORA"),
        key="manual_marca_a",
    )
    marca_b = c2.text_input(
        "Marca B",
        value=st.session_state.get("manual_marca_b", "AGROLOG"),
        key="manual_marca_b",
    )

    classe_a = c1.number_input(
        "Classe Nice A", -1, 99,
        int(st.session_state.get("manual_classe_a", 35)),
        key="manual_classe_a",
    )
    classe_b = c2.number_input(
        "Classe Nice B", -1, 99,
        int(st.session_state.get("manual_classe_b", 39)),
        key="manual_classe_b",
    )

    spec_a = c1.text_area(
        "Especificacao A",
        value=st.session_state.get(
            "manual_spec_a", "Comercio de medicamentos.",
        ),
        key="manual_spec_a",
    )
    spec_b = c2.text_area(
        "Especificacao B",
        value=st.session_state.get(
            "manual_spec_b",
            "Afretamento;Armazenagem;Frete;Servicos de transporte.",
        ),
        key="manual_spec_b",
    )

    calc_col, phase_col = st.columns([2, 1])
    with phase_col:
        st.caption("**Fases do calculo** (atualiza ao clicar em Calcular score)")
        manual_phase_box = st.empty()

    with calc_col:
        trigger_calc = st.button("Calcular score", type="primary")

    if trigger_calc:
        manual_phase_box.info(
            "**Aguardando inicio** — botao acionado; iniciando pipeline...",
        )
        try:
            with StepTimer("manual_test"):
                with st.status(
                    "**Calculo do score em andamento** — veja as fases:",
                    expanded=True,
                ) as calc_stat:
                    calc_stat.markdown(
                        "**Fase 1/4 — Features:** montando linha do par e "
                        "`preprocessor.transform` (TF-IDF, embeddings, scaler)...",
                    )
                    manual_phase_box.info(
                        "**Fase 1/4** — Vetor de features (transformador ja "
                        "ajustado no enriquecimento).",
                    )
                    df_pair = pd.DataFrame({
                        "marca_monitorada": [marca_a],
                        "marca_colidente": [marca_b],
                        "classe_marca_monitorada": [int(classe_a)],
                        "classe_marca_colidente": [int(classe_b)],
                        "especificacao_monitorado": [spec_a],
                        "especificacao_colidente": [spec_b],
                    })
                    X_pair = preproc.transform(df_pair, scale=True)
                    ensemble_members = st.session_state.get(
                        "ensemble_members",
                    )
                    mlp_members, lr_members = _get_bagging_lists()
                    calibrator = st.session_state.get("calibrator")

                    calc_stat.markdown(
                        f"**Fase 2/4 — Bagging hibrido:** {len(mlp_members)} "
                        f"MLPs + {len(lr_members)} regressoes logisticas "
                        "(forward + Platt + media)...",
                    )
                    manual_phase_box.info(
                        "**Fase 2/4** — Forward MLPs + LogRegs + calibracao Platt.",
                    )
                    by_arch_rows: list[dict[str, Any]] | None = None
                    if mlp_members or lr_members:
                        comp_mat, comp_info = predict_hybrid_bagging_components(
                            mlp_members, lr_members, X_pair,
                            apply_calibration=True,
                        )
                        comp_row = comp_mat[0]
                        nn_score = float(comp_row.mean())
                        by_arch_rows = [
                            {
                                "tipo": info["kind"],
                                "chave": info["key"],
                                "score": float(comp_row[j]),
                            }
                            for j, info in enumerate(comp_info)
                        ]
                    elif ensemble_members:
                        nn_score = float(predict_ensemble(
                            ensemble_members, X_pair, apply_calibration=True,
                        )[0])
                    else:
                        nn_score = float(predict_scores(model, X_pair)[0])
                        if calibrator is not None and calibrator.fitted:
                            nn_score = float(
                                calibrator.transform(np.array([nn_score]))[0]
                            )

                    calc_stat.markdown(
                        "**Fase 3/4 — Heuristica OFTA:** score legado para "
                        "comparacao lado a lado...",
                    )
                    manual_phase_box.info(
                        "**Fase 3/4** — OFTA (regra de negocio historica).",
                    )
                    ofta = calcular_score_complexo(marca_a, marca_b)

                    calc_stat.markdown(
                        "**Fase 4/4 — Explicabilidade:** Integrated Gradients "
                        "(contribuicao por feature neste par)...",
                    )
                    manual_phase_box.info(
                        "**Fase 4/4** — Integrated Gradients (GPU se disponivel; "
                        "MLPs via captum, logisticas via contribuicao linear analitica).",
                    )
                    if mlp_members or lr_members:
                        attrib = integrated_gradients_hybrid_row(
                            mlp_members, lr_members, X_pair[0], names,
                        )
                    else:
                        attrib = integrated_gradients_for_row(
                            model, X_pair[0], names,
                        )
                    calc_stat.update(
                        label="**Concluido** — todas as fases terminaram.",
                        state="complete",
                    )
                    manual_phase_box.success(
                        "**Concluido** — score e IG disponiveis abaixo.",
                    )

                c1, c2, c3 = st.columns(3)
                c1.metric("Score NN (0-1)", f"{nn_score:.4f}")
                c2.metric(
                    "Score heuristica OFTA (0-1)",
                    f"{ofta['final']:.4f}",
                    delta=f"driver: {ofta.get('driver','')}",
                )
                c3.metric(
                    "Classe prevista",
                    int(nn_score >= threshold),
                    delta=f"thr={threshold:.2f}",
                )
                if by_arch_rows:
                    n_total = len(by_arch_rows)
                    n_mlp_models = sum(
                        1 for r in by_arch_rows if r["tipo"] == "mlp"
                    )
                    n_lr_models = sum(
                        1 for r in by_arch_rows if r["tipo"] == "logreg"
                    )
                    scores_only = [r["score"] for r in by_arch_rows]
                    if n_total <= 8:
                        st.caption(
                            "Scores por modelo do bagging (calibrados): "
                            + ", ".join(
                                f"{r['tipo']}/{r['chave']}={r['score']:.4f}"
                                for r in by_arch_rows
                            )
                        )
                    else:
                        st.caption(
                            f"Bagging com {n_total} modelos "
                            f"({n_mlp_models} MLPs + {n_lr_models} "
                            f"logisticas). Score agregado={nn_score:.4f}, "
                            f"min={min(scores_only):.4f}, "
                            f"max={max(scores_only):.4f}."
                        )
                        st.dataframe(
                            pd.DataFrame(by_arch_rows),
                            use_container_width=True,
                            hide_index=True,
                        )

                st.subheader(
                    "Top 10 features que mais influenciaram este score",
                )
                df_attr = pd.DataFrame(
                    attach_descriptions(attrib),
                ).head(10)
                cols_show = [
                    c for c in ("feature", "attribution", "descricao")
                    if c in df_attr.columns
                ]
                st.dataframe(df_attr[cols_show], use_container_width=True)
        except Exception as exc:  # noqa: BLE001
            manual_phase_box.error(f"**Erro:** {exc}")
            logger.exception("Teste manual falhou: %s", exc)
            st.error(f"Teste manual falhou: {exc}")


def tab_artifacts() -> None:
    st.header("7. Artefatos finais")
    _render_pipeline_progress()

    st.markdown(
        """
Cada **treino concluido** grava automaticamente um pacote em `pipelines/<data_hora_uid>/`
com subpastas **enriched**, **weights**, **config**, **reports** e **errors**, e
atualiza o espelho em `artifacts/` para compatibilidade com scripts antigos.

Use o botao abaixo para **re-exportar** o mesmo pipeline (por exemplo apos calcular
permutation importance na aba 5).
        """
    )

    df = st.session_state.get("df")
    df_report: DatasetReport | None = st.session_state.get("df_report")
    preproc: FeaturePreprocessor | None = st.session_state.get("preprocessor")
    X = st.session_state.get("feature_matrix")
    names = st.session_state.get("feature_names")
    splits = st.session_state.get("splits")
    model = st.session_state.get("model")
    em_test = st.session_state.get("eval_metrics_test")
    threshold_opt = float(st.session_state.get("threshold_optimal", 0.5))
    threshold_policy = st.session_state.get("threshold_policy", {})
    train_result = st.session_state.get("train_result")
    ensemble_members: list[EnsembleMember] | None = st.session_state.get("ensemble_members")
    arch_bagging_members = st.session_state.get("arch_bagging_members")
    logreg_bagging_members = st.session_state.get("logreg_bagging_members")
    calibrator: PlattCalibrator | None = st.session_state.get("calibrator")

    if not all([df is not None, preproc is not None, X is not None, model is not None, em_test is not None]):
        st.info("Conclua treino e avaliacao antes de exportar artefatos.")
        return

    pr = st.session_state.get("pipeline_run_dir")
    if pr:
        st.success(f"**Pipeline ativo:** `{pr}`")
        try:
            st.caption(f"Relativo ao projeto: `{Path(pr).relative_to(PROJECT_ROOT)}`")
        except ValueError:
            st.caption(str(pr))
    else:
        st.warning(
            "Ainda nao houve treino com gravacao automatica nesta sessao. "
            "Conclua o treino na aba 3 ou use o botao abaixo para criar uma nova pasta.",
        )

    auto_imp = st.checkbox(
        "Calcular permutation importance automaticamente se ausente",
        value=True,
        help="Garante que o relatorio nunca saia sem a secao 11. Usa amostra de 600 e 2 repeticoes.",
    )

    train_cfg: TrainConfig = st.session_state.get(
        "train_cfg", TrainConfig(epochs=max(1, len(st.session_state.history)))
    )

    if st.button("Re-exportar artefatos (atualiza pipeline ativo)", type="primary"):
        if pr and Path(pr).exists():
            dirs = dirs_from_root(Path(pr))
        else:
            rid = new_pipeline_run_id()
            dirs = prepare_run_directories(PROJECT_ROOT, rid)
            st.session_state.pipeline_run_id = rid
            st.session_state.pipeline_run_dir = str(dirs["root"])
        prog_enr = st.progress(0, text="A exportar pipeline...")
        try:
            imp, meta = export_pipeline_bundle(
                project_root=PROJECT_ROOT,
                dirs=dirs,
                df=df,
                X=X,
                names=list(names or []),
                splits=splits,
                model=model,
                preproc=preproc,
                train_cfg=train_cfg,
                train_result=train_result,
                threshold_opt=threshold_opt,
                threshold_policy=threshold_policy,
                df_report=df_report,
                history=list(st.session_state.history) if st.session_state.history else None,
                ensemble_members=ensemble_members,
                arch_bagging_members=arch_bagging_members,
                logreg_bagging_members=logreg_bagging_members,
                calibrator=calibrator,
                importance_top=st.session_state.get("importance_top"),
                auto_imp=auto_imp,
                progress_callback_enriched=lambda p, m: prog_enr.progress(
                    min(1.0, max(0.0, float(p))), text=str(m)[:95],
                ),
            )
            prog_enr.progress(1.0, text="Exportacao concluida.")
            if imp is not None:
                st.session_state.importance_top = imp
            rid = dirs["root"].name
            append_registry(PROJECT_ROOT, {"id": rid, **meta})
            mirror_latest_to_artifacts(PROJECT_ROOT, dirs)
            st.session_state.pipeline_run_dir = str(dirs["root"])
            st.session_state.pipeline_run_id = rid
        except Exception as exc:  # noqa: BLE001
            st.error(f"Falha na exportacao: {exc}")
            logger.exception("tab_artifacts export")
        else:
            report_path = dirs["reports"] / "report.md"
            prev = ""
            if report_path.exists():
                prev = report_path.read_text(encoding="utf-8")
            st.success(
                f"Artefatos atualizados em `{dirs['root']}` e espelho em `artifacts/`. "
                f"Registo em `pipelines/_registry.json`.",
            )
            if prev:
                with st.expander("Previa do relatorio gerado", expanded=False):
                    st.markdown(prev)

    pr2 = st.session_state.get("pipeline_run_dir")
    if pr2 and Path(pr2).exists():
        st.subheader("Downloads (pipeline ativo)")
        dirs_dl = dirs_from_root(Path(pr2))
        for fp in collect_download_files(dirs_dl):
            rel = fp.relative_to(dirs_dl["root"])
            mb = fp.stat().st_size / (1024 * 1024)
            with fp.open("rb") as f:
                st.download_button(
                    label=f"Download {rel.as_posix()} ({mb:.2f} MB)",
                    data=f.read(),
                    file_name=str(rel).replace("\\", "_").replace("/", "_"),
                    key=f"dl_{fp.as_posix()}",
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.title("Brandious SuperMatching")
    st.caption(
        "Pipeline de similaridade aprendida de marcas (PT-BR). Execute as "
        "abas em sequencia (1 -> 7); o treino na aba 3 grava artefatos "
        "automaticamente.",
    )

    cfg = render_sidebar()

    tabs = st.tabs(TABS)
    with tabs[0]:
        tab_upload_eda(cfg)
    with tabs[1]:
        tab_enrichment(cfg)
    with tabs[2]:
        tab_train(cfg)
    with tabs[3]:
        tab_evaluation()
    with tabs[4]:
        tab_explainability()
    with tabs[5]:
        tab_manual_test()
    with tabs[6]:
        tab_artifacts()


if __name__ == "__main__":
    main()
