"""Visualização do módulo de derivativos."""

import pandas as pd
import streamlit as st
import plotly.express as px

from ui.charts import configurar_figura


def render_derivatives(resultados):
    """
    Renderiza a comparação entre a solução analítica
    de Black-Scholes e a aproximação numérica por Monte Carlo.
    """

    resultado_derivativos = resultados[
        "resultado_derivativos"
    ]

    # ========================================================
    # CABEÇALHO
    # ========================================================

    st.markdown("## Derivatives Lab")

    st.caption(
        "European call pricing using the closed-form Black–Scholes "
        "solution and a Monte Carlo numerical cross-check."
    )

    # ========================================================
    # MÉTRICAS
    # ========================================================

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "BLACK–SCHOLES",
        (
            f"R$ "
            f"{resultado_derivativos['black_scholes']:.4f}"
        ),
    )

    col2.metric(
        "MONTE CARLO",
        (
            f"R$ "
            f"{resultado_derivativos['monte_carlo']:.4f}"
        ),
    )

    col3.metric(
        "NUMERICAL DIFFERENCE",
        (
            f"{resultado_derivativos['diferenca_percentual']:.2f}%"
        ),
    )

    # ========================================================
    # COMPARAÇÃO
    # ========================================================

    deriv_df = pd.DataFrame(
        {
            "Method": [
                "Black–Scholes",
                "Monte Carlo",
            ],
            "Option Price": [
                resultado_derivativos[
                    "black_scholes"
                ],
                resultado_derivativos[
                    "monte_carlo"
                ],
            ],
        }
    )

    fig_deriv = px.bar(
        deriv_df,
        x="Method",
        y="Option Price",
        text_auto=".4f",
    )

    fig_deriv.update_traces(
        marker_color="#6C4CE3"
    )

    configurar_figura(
        fig_deriv,
        x_title=None,
        y_title="European Call Price (R$)",
    )

    st.plotly_chart(
        fig_deriv,
        use_container_width=True,
    )

    # ========================================================
    # INTERPRETAÇÃO
    # ========================================================

    st.info(
        "Black–Scholes provides the analytical benchmark, while "
        "Monte Carlo estimates the same European call price numerically "
        "under consistent assumptions. Their proximity acts as a "
        "reproducible implementation cross-check."
    )

    # ========================================================
    # INPUTS
    # ========================================================

    st.caption(
        "Inputs: S = 100 | K = 100 | T = 1 year | "
        "r = 5% | σ = 20% | "
        "Monte Carlo = 100,000 simulations."
    )