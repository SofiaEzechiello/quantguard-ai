"""derivatives_view.py - visualização do laboratório de derivativos
"""

with aba_derivatives:
    st.markdown("## Derivatives Lab")
    st.caption(
        "European call pricing with closed-form Black–Scholes and "
        "Monte Carlo cross-validation."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "BLACK–SCHOLES",
        f"R$ {resultado_derivativos['black_scholes']:.4f}",
    )

    col2.metric(
        "MONTE CARLO",
        f"R$ {resultado_derivativos['monte_carlo']:.4f}",
    )

    col3.metric(
        "MODEL DEVIATION",
        f"{resultado_derivativos['diferenca_percentual']:.2f}%",
    )

    deriv_df = pd.DataFrame(
        {
            "Model": ["Black–Scholes", "Monte Carlo"],
            "Option Price": [
                resultado_derivativos["black_scholes"],
                resultado_derivativos["monte_carlo"],
            ],
        }
    )

    fig_deriv = px.bar(
        deriv_df,
        x="Model",
        y="Option Price",
        text_auto=".4f",
    )
    fig_deriv.update_traces(marker_color="#6C4CE3")

    configurar_figura(
        fig_deriv,
        x_title=None,
        y_title="Call Price (R$)",
    )

    st.plotly_chart(fig_deriv, use_container_width=True)

    st.caption(
        "Inputs: S = 100 | K = 100 | T = 1 year | "
        "r = 5% | σ = 20% | Monte Carlo = 100,000 simulations."
    )