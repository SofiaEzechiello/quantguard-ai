"""Visualização do Risk Engine."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    VALOR_CARTEIRA,
    CONFIANCA,
)

from ui.charts import configurar_figura


def render_risk(resultados):
    """
    Renderiza métricas de risco, backtesting out-of-sample
    e validação estatística do VaR.
    """

    retornos_carteira_treino = resultados[
        "retornos_carteira_treino"
    ]

    retornos_carteira_teste = resultados[
        "retornos_carteira_teste"
    ]

    var_param = resultados["var_param"]
    var_hist = resultados["var_hist"]
    es = resultados["es"]

    backtest_param = resultados[
        "backtest_param"
    ]

    backtest_hist = resultados[
        "backtest_hist"
    ]

    kupiec_param = backtest_param[
        "kupiec"
    ]

    kupiec_hist = backtest_hist[
        "kupiec"
    ]

    # ========================================================
    # CABEÇALHO
    # ========================================================

    st.markdown("## Risk Engine")

    st.caption(
        "One-day portfolio risk estimated on the training sample "
        "and validated on unseen out-of-sample observations."
    )

    # ========================================================
    # MÉTRICAS PRINCIPAIS
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
            var_hist
            / VALOR_CARTEIRA
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
    # BACKTEST DO VaR
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
    # BACKTEST SUMMARY
    # ========================================================

    st.markdown("### Out-of-Sample Backtest")

    esperado = (
        1 - CONFIANCA
    )

    b1, b2, b3 = st.columns(3)

    b1.metric(
        "EXPECTED VIOLATION RATE",
        f"{esperado:.2%}",
    )

    b2.metric(
        "PARAMETRIC VaR",
        f"{backtest_param['taxa_violacoes']:.2%}",
    )

    b3.metric(
        "HISTORICAL VaR",
        f"{backtest_hist['taxa_violacoes']:.2%}",
    )

    st.caption(
        f"Out-of-sample observations: "
        f"{backtest_hist['observacoes']} · "
        f"Historical VaR violations: "
        f"{backtest_hist['violacoes']} · "
        f"Parametric VaR violations: "
        f"{backtest_param['violacoes']}."
    )

    # ========================================================
    # KUPIEC TEST
    # ========================================================

    st.markdown("### Kupiec Unconditional Coverage Test")

    st.caption(
        "The Kupiec test evaluates whether the observed frequency "
        "of VaR violations is statistically consistent with the "
        "expected violation probability."
    )

    col_param, col_hist = st.columns(2)

    # --------------------------------------------------------
    # PARAMETRIC VaR
    # --------------------------------------------------------

    with col_param:

        st.markdown("#### Parametric VaR")

        k1, k2 = st.columns(2)

        k1.metric(
            "Kupiec LR",
            f"{kupiec_param['estatistica_lr']:.3f}",
        )

        k2.metric(
            "p-value",
            f"{kupiec_param['p_valor']:.4f}",
        )

        st.caption(
            f"Expected violations: "
            f"{kupiec_param['taxa_esperada']:.2%} · "
            f"Observed: "
            f"{kupiec_param['taxa_observada']:.2%}"
        )

        if kupiec_param["rejeita_h0"]:

            st.error(
                "REJECT H₀ — unconditional coverage is rejected "
                "at the 5% significance level."
            )

        else:

            st.success(
                "DO NOT REJECT H₀ — unconditional coverage is "
                "not rejected at the 5% significance level."
            )

    # --------------------------------------------------------
    # HISTORICAL VaR
    # --------------------------------------------------------

    with col_hist:

        st.markdown("#### Historical VaR")

        k1, k2 = st.columns(2)

        k1.metric(
            "Kupiec LR",
            f"{kupiec_hist['estatistica_lr']:.3f}",
        )

        k2.metric(
            "p-value",
            f"{kupiec_hist['p_valor']:.4f}",
        )

        st.caption(
            f"Expected violations: "
            f"{kupiec_hist['taxa_esperada']:.2%} · "
            f"Observed: "
            f"{kupiec_hist['taxa_observada']:.2%}"
        )

        if kupiec_hist["rejeita_h0"]:

            st.error(
                "REJECT H₀ — unconditional coverage is rejected "
                "at the 5% significance level."
            )

        else:

            st.success(
                "DO NOT REJECT H₀ — unconditional coverage is "
                "not rejected at the 5% significance level."
            )

    # ========================================================
    # INTERPRETAÇÃO
    # ========================================================

    st.info(
        "A non-rejection result does not prove that a VaR model is "
        "fully valid. The Kupiec test evaluates unconditional coverage "
        "only; violation independence is outside the scope of this "
        "version of QuantGuard."
    )