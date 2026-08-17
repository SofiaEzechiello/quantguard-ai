"""Visualização do Risk Engine."""

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from config import (
    VALOR_CARTEIRA,
    CONFIANCA,
)

from ui.charts import configurar_figura


def render_risk(resultados):

    retornos_carteira_treino = resultados[
        "retornos_carteira_treino"
    ]

    retornos_carteira_teste = resultados[
        "retornos_carteira_teste"
    ]

    var_param = resultados["var_param"]
    var_hist = resultados["var_hist"]
    es = resultados["es"]

    backtest_hist = resultados["backtest_hist"]

    # ========================================================
    # CABEÇALHO
    # ========================================================

    st.markdown("## Risk Engine")

    st.caption(
        "One-day portfolio risk estimated on the training sample and "
        "validated on unseen observations."
    )

    # ========================================================
    # MÉTRICAS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "PARAMETRIC VaR · 95%",
        f"R$ {var_param:,.0f}",
    )

    col2.metric(
        "HISTORICAL VaR · 95%",
        f"R$ {var_hist:,.0f}",
    )

    col3.metric(
        "EXPECTED SHORTFALL",
        f"R$ {es:,.0f}",
    )

    col4.metric(
        "HIST. VaR VIOLATIONS",
        f"{backtest_hist['taxa_violacoes']:.2%}",
    )

    # ========================================================
    # GRÁFICOS
    # ========================================================

    col_left, col_right = st.columns(2)

    # --------------------------------------------------------
    # DISTRIBUIÇÃO DOS RETORNOS
    # --------------------------------------------------------

    with col_left:

        st.markdown("### Return Distribution")

        distribuicao = pd.DataFrame(
            {
                "Portfolio Return":
                    retornos_carteira_treino
            }
        )

        fig_dist = px.histogram(
            distribuicao,
            x="Portfolio Return",
            nbins=55,
        )

        fig_dist.update_traces(
            marker_color="#8171C9"
        )

        limite_hist = -(
            var_hist / VALOR_CARTEIRA
        )

        fig_dist.add_vline(
            x=limite_hist,
            line_dash="dash",
            line_color="#B42318",
            annotation_text="Historical VaR",
        )

        configurar_figura(
            fig_dist,
            x_title="Daily Return",
            y_title="Frequency",
        )

        st.plotly_chart(
            fig_dist,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # BACKTEST DE VaR
    # --------------------------------------------------------

    with col_right:

        st.markdown("### VaR Backtesting")

        perdas_teste = (
            -retornos_carteira_teste
            * VALOR_CARTEIRA
        )

        backtest_df = pd.DataFrame(
            {
                "Date": perdas_teste.index,
                "Loss": perdas_teste.values,
            }
        )

        fig_backtest = go.Figure()

        fig_backtest.add_trace(
            go.Scatter(
                x=backtest_df["Date"],
                y=backtest_df["Loss"],
                mode="lines",
                name="Daily Loss",
                line=dict(
                    color="#8171C9",
                    width=1.5,
                ),
            )
        )

        violacoes_hist = (
            perdas_teste > var_hist
        )

        fig_backtest.add_trace(
            go.Scatter(
                x=perdas_teste.index[
                    violacoes_hist
                ],
                y=perdas_teste[
                    violacoes_hist
                ],
                mode="markers",
                name="VaR Violations",
                marker=dict(
                    color="#B42318",
                    size=7,
                ),
            )
        )

        fig_backtest.add_hline(
            y=var_hist,
            line_dash="dash",
            line_color="#B42318",
            annotation_text="Historical VaR",
        )

        configurar_figura(
            fig_backtest,
            x_title=None,
            y_title="Daily Loss (R$)",
        )

        st.plotly_chart(
            fig_backtest,
            use_container_width=True,
        )

    # ========================================================
    # OBSERVAÇÃO
    # ========================================================

    st.caption(
        f"Expected violation rate at "
        f"{CONFIANCA:.0%} confidence: "
        f"{1 - CONFIANCA:.0%}. "
        f"Observed historical-VaR violation rate: "
        f"{backtest_hist['taxa_violacoes']:.2%}."
    )