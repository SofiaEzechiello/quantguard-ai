"""Visualização dos testes de stress."""

import pandas as pd
import streamlit as st
import plotly.express as px

from ui.charts import configurar_figura


def render_stress(resultados):

    resultado_covid = resultados[
        "resultado_covid"
    ]

    stress_mercado = resultados[
        "stress_mercado"
    ]

    betas_ibov = resultados[
        "betas_ibov"
    ]

    # ========================================================
    # CABEÇALHO
    # ========================================================

    st.markdown("## Stress Testing")

    st.caption(
        "Historical and factor-based scenarios applied to the "
        "maximum-Sharpe portfolio."
    )

    # ========================================================
    # RESULTADOS
    # ========================================================

    covid_impacto = resultado_covid[
        "retorno_carteira"
    ]

    ibov_impacto = stress_mercado[
        "impacto_carteira"
    ]

    # ========================================================
    # MÉTRICAS
    # ========================================================

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "COVID-19 CRASH",
        f"{covid_impacto:.2%}",
    )

    col2.metric(
        "IBOVESPA -20%",
        f"{ibov_impacto:.2%}",
    )

    col3.metric(
        "COVID LOSS",
        f"R$ {resultado_covid['perda']:,.0f}",
    )

    # ========================================================
    # IMPACTO DOS CENÁRIOS
    # ========================================================

    stress_df = pd.DataFrame(
        {
            "Scenario": [
                "COVID-19 Crash",
                "Ibovespa -20%",
            ],
            "Portfolio Impact": [
                covid_impacto,
                ibov_impacto,
            ],
        }
    )

    fig_stress = px.bar(
        stress_df,
        x="Scenario",
        y="Portfolio Impact",
        text_auto=".1%",
    )

    fig_stress.update_traces(
        marker_color="#6C4CE3"
    )

    configurar_figura(
        fig_stress,
        x_title=None,
        y_title="Portfolio Impact",
    )

    st.plotly_chart(
        fig_stress,
        use_container_width=True,
    )

    # ========================================================
    # DETALHAMENTO
    # ========================================================

    col_left, col_right = st.columns(2)

    # --------------------------------------------------------
    # COVID POR ATIVO
    # --------------------------------------------------------

    with col_left:

        st.markdown(
            "### COVID-19 Asset Impact"
        )

        covid_assets = (
            resultado_covid[
                "retornos_ativos"
            ]
            .rename("Return")
            .reset_index()
        )

        covid_assets.columns = [
            "Asset",
            "Return",
        ]

        fig_covid_assets = px.bar(
            covid_assets,
            x="Asset",
            y="Return",
            text_auto=".1%",
        )

        fig_covid_assets.update_traces(
            marker_color="#8171C9"
        )

        configurar_figura(
            fig_covid_assets,
            x_title=None,
            y_title="Return During Scenario",
        )

        st.plotly_chart(
            fig_covid_assets,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # BETAS
    # --------------------------------------------------------

    with col_right:

        st.markdown("### Market Betas")

        beta_df = (
            betas_ibov
            .rename("Beta")
            .reset_index()
        )

        beta_df.columns = [
            "Asset",
            "Beta",
        ]

        fig_beta = px.bar(
            beta_df,
            x="Asset",
            y="Beta",
            text_auto=".2f",
        )

        fig_beta.update_traces(
            marker_color="#A991FF"
        )

        configurar_figura(
            fig_beta,
            x_title=None,
            y_title="Beta vs. Ibovespa",
        )

        st.plotly_chart(
            fig_beta,
            use_container_width=True,
        )