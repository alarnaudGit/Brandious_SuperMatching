"""Brandious SuperMatching - interface profissional Streamlit.

Substitui o Flask antigo: oferece upload, EDA, enriquecimento, configuracao,
treino em tempo real, avaliacao, explicabilidade, teste manual e download
de artefatos para o modelo de similaridade aprendida de marcas.
"""
from __future__ import annotations

import logging
import sys
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
from src.artifacts import (  # noqa: E402
    build_model_config_dict,
    save_enriched_dataframe,
    save_model_config,
    save_state_dict,
)
from src.data import (  # noqa: E402
    LABEL_COLUMN_CANON,
    DatasetReport,
    load_dataframe_from_bytes,
)
from src.model.dataset import (  # noqa: E402
    BalancingConfig,
    SplitConfig,
    stratified_split,
)
from src.model.evaluate import (  # noqa: E402
    compute_metrics_at_threshold,
    find_optimal_threshold,
    pr_curve_data,
    predict_scores,
    roc_curve_data,
)
from src.model.explain import (  # noqa: E402
    integrated_gradients_for_row,
    permutation_importance,
)
from src.model.mlp import BrandSimilarityMLP, MLPConfig  # noqa: E402
from src.model.train import TrainConfig, train_model  # noqa: E402
from src.normalize import calcular_score_complexo, calcular_similaridade_ofta  # noqa: E402
from src.pipeline.preprocessor import FeaturePreprocessor, PreprocessorConfig  # noqa: E402
from src.reports import build_report, compare_against_heuristic, save_report  # noqa: E402

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
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


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
        top_k = st.slider("Top-K classes Nice (one-hot)", 5, 30, 15)
        tfidf_word_max = st.number_input("TF-IDF word max_features", 1000, 50_000, 20_000, step=1000)
        tfidf_char_max = st.number_input("TF-IDF char max_features", 1000, 50_000, 10_000, step=1000)

    with st.sidebar.expander("Arquitetura MLP", expanded=True):
        hidden_str = st.text_input("Hidden layers (vir-separadas)", value="128,64,32")
        dropout = st.slider("Dropout", 0.0, 0.7, 0.3, 0.05)
        bn = st.checkbox("BatchNorm", value=True)
        activation = st.selectbox("Ativacao", ["relu", "gelu", "leakyrelu", "tanh"], index=0)

    with st.sidebar.expander("Treino", expanded=True):
        epochs = st.slider("Epochs maximas", 5, 200, 60)
        batch = st.select_slider("Batch size", options=[64, 128, 256, 512, 1024], value=256)
        lr = st.select_slider("Learning rate", options=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2], value=1e-3)
        wd = st.select_slider("Weight decay", options=[0.0, 1e-5, 1e-4, 1e-3], value=1e-4)
        es_patience = st.slider("Early stopping patience (PR-AUC val)", 3, 30, 10)

    with st.sidebar.expander("Balanceamento", expanded=True):
        under_ratio = st.slider("Undersample neg/pos ratio", 1.0, 10.0, 3.0, 0.5)
        over_factor = st.slider("Oversample positivos (x)", 1.0, 5.0, 2.0, 0.5)
        use_cw = st.checkbox("Usar class weight (pos_weight)", value=True)
        pw_override_use = st.checkbox("Override de pos_weight", value=False)
        pw_override = st.number_input("pos_weight override", 0.5, 30.0, 9.0, 0.5) if pw_override_use else None

    with st.sidebar.expander("Threshold", expanded=False):
        recall_floor = st.slider("Recall floor (politica de threshold)", 0.5, 0.99, 0.85, 0.01)

    with st.sidebar.expander("Split / Seed", expanded=False):
        seed = st.number_input("Seed", 0, 999_999, 42, step=1)
        test_size = st.slider("Test size", 0.05, 0.3, 0.15, 0.01)
        val_size = st.slider("Val size", 0.05, 0.3, 0.15, 0.01)

    return {
        "preproc_cfg": PreprocessorConfig(
            tfidf_word_max_features=int(tfidf_word_max),
            tfidf_char_max_features=int(tfidf_char_max),
            top_k_classes=int(top_k),
            use_embeddings=bool(use_emb),
        ),
        "hidden_dims": [int(x.strip()) for x in hidden_str.split(",") if x.strip()],
        "dropout": float(dropout),
        "use_batchnorm": bool(bn),
        "activation": str(activation),
        "epochs": int(epochs),
        "batch_size": int(batch),
        "lr": float(lr),
        "weight_decay": float(wd),
        "es_patience": int(es_patience),
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


def tab_upload_eda() -> None:
    st.header("1. Upload & EDA")

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
        with st.spinner("Carregando e validando..."):
            df, report = load_dataframe_from_bytes(file_bytes)
            st.session_state.df = df
            st.session_state.df_report = report

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

    df = st.session_state.get("df")
    if df is None:
        st.info("Carregue um dataset na aba 1.")
        return

    st.write(
        "Fitta TF-IDF/embeddings e gera a matriz de features na ordem canonica. "
        "Esta etapa pode demorar (~minutos com embeddings)."
    )

    if st.button("Gerar features", type="primary"):
        with st.spinner("Pre-processando especificacoes e ajustando vectorizers..."):
            preproc = FeaturePreprocessor(cfg["preproc_cfg"])
            preproc.fit(df)
            X = preproc.transform(df, scale=True, show_progress=True)
        st.session_state.preprocessor = preproc
        st.session_state.feature_matrix = X
        st.session_state.feature_names = list(preproc.feature_names_ordered)
        st.success(f"Matriz gerada: shape {X.shape}, {len(preproc.feature_names_ordered)} features.")

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

    st.markdown(
        """
**O que esta aba faz**

Aqui o modelo aprende, a partir dos exemplos rotulados pelo INPI, qual combinacao de
sinais (escrita parecida, som parecido, especificacao parecida, classe igual etc.)
indica colidencia. O treino acontece em **epocas** - cada epoca e uma passada
completa pelos pares de marcas. A cada epoca o modelo ajusta seus pesos um pouco
para errar menos.

**O que voce ve abaixo**

- **Loss por epoca**: o erro medio que o modelo esta cometendo. **Deve cair com o
  tempo**. Se ficar oscilando ou crescer, o aprendizado nao esta convergindo.
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
        st.info("Carregue o dataset e gere as features primeiro.")
        return

    y = df[LABEL_COLUMN_CANON].to_numpy().astype(np.int64)
    split_cfg = SplitConfig(test_size=cfg["test_size"], val_size=cfg["val_size"], seed=cfg["seed"])
    bal = BalancingConfig(
        undersample_neg_ratio=cfg["under_ratio"],
        oversample_pos_factor=cfg["over_factor"],
        use_class_weight=cfg["use_cw"],
        pos_weight_override=cfg["pos_weight_override"],
        seed=cfg["seed"],
    )
    train_cfg = TrainConfig(
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        early_stopping_patience=cfg["es_patience"],
        seed=cfg["seed"],
    )
    mlp_cfg = MLPConfig(
        input_dim=X.shape[1],
        hidden_dims=list(cfg["hidden_dims"]),
        dropout=cfg["dropout"],
        use_batchnorm=cfg["use_batchnorm"],
        activation=cfg["activation"],
    )

    c1, c2 = st.columns(2)
    train_btn = c1.button("Treinar modelo", type="primary")
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

        (X_train, y_train), (X_val, y_val), (X_test, y_test) = stratified_split(X, y, split_cfg)
        st.session_state.splits = {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test),
            "split_cfg": split_cfg,
            "bal": bal,
        }

        history: list[dict[str, float]] = []

        def on_epoch_end(info: dict[str, Any]) -> None:
            history.append(info)
            st.session_state.history = list(history)
            df_h = pd.DataFrame(history)
            with chart_loss.container():
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_h["epoch"], y=df_h["train_loss"], name="train_loss"))
                fig.update_layout(title="Loss por epoca", xaxis_title="epoca", yaxis_title="loss", height=320)
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
                        fig2.add_trace(go.Scatter(x=df_h["epoch"], y=df_h[col], name=name))
                fig2.update_layout(title="Metricas de validacao", xaxis_title="epoca", height=320)
                st.plotly_chart(fig2, use_container_width=True)
            with table_slot.container():
                st.dataframe(df_h.tail(10), use_container_width=True)
            status_slot.info(f"Epoca {info['epoch']} concluida (val PR-AUC = {info['val_pr_auc']:.4f}).")

        with st.spinner("Treinando MLP..."):
            model, result = train_model(
                X_train,
                y_train,
                X_val,
                y_val,
                mlp_cfg,
                train_cfg,
                bal,
                on_epoch_end=on_epoch_end,
                should_stop=lambda: bool(st.session_state.stop_train),
            )

        st.session_state.model = model
        st.session_state.train_result = result
        st.session_state.train_cfg = train_cfg
        st.session_state.mlp_cfg = mlp_cfg

        val_scores = predict_scores(model, X_val)
        thr_opt, thr_policy = find_optimal_threshold(y_val, val_scores, recall_floor=cfg["recall_floor"])
        st.session_state.threshold_optimal = float(thr_opt)
        st.session_state.threshold_policy = thr_policy

        test_scores = predict_scores(model, X_test)
        em = compute_metrics_at_threshold(y_test, test_scores, threshold=thr_opt)
        st.session_state.eval_metrics_test = em

        status_slot.success(
            f"Treino concluido. Epoca otima: {result.best_epoch}, val PR-AUC = {result.best_pr_auc_val:.4f}. "
            f"Threshold otimo (val) = {thr_opt:.3f}. Test PR-AUC = {em.pr_auc:.4f}, Recall = {em.recall:.3f}."
        )

    elif st.session_state.history:
        _draw_history(st.session_state.history)


def tab_evaluation() -> None:
    st.header("4. Avaliacao")

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
    if model is None or splits is None:
        st.info("Treine o modelo na aba 3.")
        return

    X_test, y_test = splits["test"]
    scores_test = predict_scores(model, X_test)

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
    cm_df = pd.DataFrame(cm, index=["Real 0", "Real 1"], columns=["Pred 0", "Pred 1"])
    st.subheader("Matriz de confusao (test)")
    st.dataframe(cm_df, use_container_width=True)

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
        st.json(comp)


def tab_explainability() -> None:
    st.header("5. Explicabilidade")

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
    if model is None or splits is None:
        st.info("Treine o modelo na aba 3.")
        return

    X_val, y_val = splits["val"]

    sample_max = min(2000, len(y_val))
    sample = st.slider("Amostra para permutation importance", 200, sample_max, min(800, sample_max))
    n_repeats = st.slider("Repetições", 1, 5, 2)
    metric = st.selectbox("Metrica", ["pr_auc", "roc_auc"], index=0)

    if st.button("Calcular permutation importance"):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(y_val), size=sample, replace=False)
        with st.spinner("Calculando importance..."):
            top = permutation_importance(
                model, X_val[idx], y_val[idx], names, metric=metric, n_repeats=n_repeats,
            )
        st.session_state.importance_top = top

    top = st.session_state.get("importance_top")
    if top is not None:
        df = pd.DataFrame(top)
        st.subheader(f"Top 30 features por permutation importance ({metric})")
        st.bar_chart(df.head(30).set_index("feature")["importance"])
        st.dataframe(df, use_container_width=True)


def tab_manual_test() -> None:
    st.header("6. Teste manual")

    model = st.session_state.get("model")
    preproc: FeaturePreprocessor | None = st.session_state.get("preprocessor")
    names = st.session_state.get("feature_names")
    threshold = float(st.session_state.get("threshold_optimal", 0.5))

    if model is None or preproc is None:
        st.info("Treine o modelo na aba 3.")
        return

    c1, c2 = st.columns(2)
    marca_a = c1.text_input("Marca A", value="AGROLOG DISTRIBUIDORA")
    marca_b = c2.text_input("Marca B", value="AGROLOG")

    classe_a = c1.number_input("Classe Nice A", -1, 99, 35)
    classe_b = c2.number_input("Classe Nice B", -1, 99, 39)

    spec_a = c1.text_area("Especificacao A", value="Comércio de medicamentos.")
    spec_b = c2.text_area(
        "Especificacao B",
        value="Afretamento;Armazenagem;Frete;Servicos de transporte.",
    )

    if st.button("Calcular score", type="primary"):
        df_pair = pd.DataFrame(
            {
                "marca_monitorada": [marca_a],
                "marca_colidente": [marca_b],
                "classe_marca_monitorada": [int(classe_a)],
                "classe_marca_colidente": [int(classe_b)],
                "especificacao_monitorado": [spec_a],
                "especificacao_colidente": [spec_b],
            }
        )
        X_pair = preproc.transform(df_pair, scale=True)
        nn_score = float(predict_scores(model, X_pair)[0])
        ofta = calcular_score_complexo(marca_a, marca_b)

        c1, c2, c3 = st.columns(3)
        c1.metric("Score NN (0-1)", f"{nn_score:.4f}")
        c2.metric("Score heuristica OFTA (0-1)", f"{ofta['final']:.4f}", delta=f"driver: {ofta.get('driver','')}")
        c3.metric("Classe prevista", int(nn_score >= threshold), delta=f"thr={threshold:.2f}")

        with st.spinner("Calculando contribuicoes (Integrated Gradients)..."):
            attrib = integrated_gradients_for_row(model, X_pair[0], names)
        st.subheader("Top 10 features que mais influenciaram este score")
        df_attr = pd.DataFrame(attrib).head(10)
        st.dataframe(df_attr, use_container_width=True)


def tab_artifacts() -> None:
    st.header("7. Artefatos finais")

    df = st.session_state.get("df")
    df_report: DatasetReport | None = st.session_state.get("df_report")
    preproc: FeaturePreprocessor | None = st.session_state.get("preprocessor")
    X = st.session_state.get("feature_matrix")
    names = st.session_state.get("feature_names")
    splits = st.session_state.get("splits")
    model: BrandSimilarityMLP | None = st.session_state.get("model")
    em_test = st.session_state.get("eval_metrics_test")
    threshold_opt = float(st.session_state.get("threshold_optimal", 0.5))
    threshold_policy = st.session_state.get("threshold_policy", {})
    train_result = st.session_state.get("train_result")
    importance_top = st.session_state.get("importance_top")

    if not all([df is not None, preproc is not None, X is not None, model is not None, em_test is not None]):
        st.info("Conclua treino e avaliacao antes de exportar artefatos.")
        return

    out_dir = PROJECT_ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)

    if st.button("Gerar e salvar todos os artefatos", type="primary"):
        with st.spinner("Calculando scores no dataset completo..."):
            score_nn_full = predict_scores(model, X)
            score_h_full = np.array(
                [
                    calcular_similaridade_ofta(a, b)
                    for a, b in zip(df["marca_monitorada"].astype(str), df["marca_colidente"].astype(str))
                ],
                dtype=np.float32,
            )
        st.session_state.score_nn_full = score_nn_full
        st.session_state.score_heuristic_full = score_h_full

        preproc.save(out_dir / "preprocessor.pkl")
        save_state_dict(model.state_dict(), out_dir / "model.pt")

        X_train, y_train = splits["train"]
        X_val, y_val = splits["val"]
        X_test, y_test = splits["test"]
        scores_train = predict_scores(model, X_train)
        scores_val = predict_scores(model, X_val)
        scores_test = predict_scores(model, X_test)
        em_train = compute_metrics_at_threshold(y_train, scores_train, threshold_opt)
        em_val = compute_metrics_at_threshold(y_val, scores_val, threshold_opt)
        em_test_re = compute_metrics_at_threshold(y_test, scores_test, threshold_opt)

        bal: BalancingConfig = splits["bal"]
        split_cfg: SplitConfig = splits["split_cfg"]
        train_cfg: TrainConfig = st.session_state.get(
            "train_cfg", TrainConfig(epochs=max(1, len(st.session_state.history)))
        )

        cfg_dict = build_model_config_dict(
            mlp_config=model.config,
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
        save_model_config(cfg_dict, out_dir / "brand_similarity_model_config.json")

        save_enriched_dataframe(
            df_original=df,
            feature_matrix=X,
            feature_names=names,
            score_nn=score_nn_full,
            score_heuristic=score_h_full,
            threshold=threshold_opt,
            out_xlsx=out_dir / "brand_similarity_input_view_enriched.xlsx",
            out_parquet=out_dir / "enriched.parquet",
            label_col=LABEL_COLUMN_CANON,
        )

        comp = compare_against_heuristic(y_test, scores_test, score_h_full[: len(y_test)])
        report_md = build_report(
            feature_names=names,
            balancing=bal,
            pos_weight_used=float(train_result.pos_weight_used) if train_result else 1.0,
            split_config=split_cfg,
            metrics_train=em_train,
            metrics_val=em_val,
            metrics_test=em_test_re,
            comparison=comp,
            threshold_optimal=threshold_opt,
            threshold_policy=threshold_policy,
            dataset_report=df_report,
            importance_top=importance_top,
        )
        save_report(report_md, out_dir / "report.md")

        st.success("Artefatos salvos em " + str(out_dir))

    files = [
        out_dir / "brand_similarity_model_config.json",
        out_dir / "model.pt",
        out_dir / "preprocessor.pkl",
        out_dir / "enriched.parquet",
        out_dir / "brand_similarity_input_view_enriched.xlsx",
        out_dir / "report.md",
    ]
    for fp in files:
        if fp.exists():
            mb = fp.stat().st_size / (1024 * 1024)
            with fp.open("rb") as f:
                st.download_button(
                    label=f"Download {fp.name}  ({mb:.2f} MB)",
                    data=f.read(),
                    file_name=fp.name,
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.title("Brandious SuperMatching")
    st.caption("Pipeline de similaridade aprendida de marcas (PT-BR) - Streamlit substitui o app Flask antigo.")

    cfg = render_sidebar()

    tabs = st.tabs(TABS)
    with tabs[0]:
        tab_upload_eda()
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
