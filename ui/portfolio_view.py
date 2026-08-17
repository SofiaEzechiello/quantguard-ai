"""portfolio_view.py - visualização do portfólio otimizado"""

with aba_portfolio:
    st.markdown("## Portfolio Intelligence")
    st.caption(
        "Mean-variance optimization trained on 2021–2024 data "
        "and evaluated out-of-sample on 2025–2026 data."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("MAX SHARPE · TRAIN", f"{sharpe_max:.2f}")
    col2.metric("EXPECTED RETURN · TRAIN", f"{retorno_max:.2%}")
    col3.metric("VOLATILITY · TRAIN", f"{risco_max:.2%}")

    st.markdown("### Efficient Frontier")
    st.caption(
        "10,000 simulated portfolios. The highlighted points represent "
        "the maximum-Sharpe and minimum-variance solutions."
    )

    fronteira = pd.DataFrame(
        {
            "Volatility": riscos,
            "Return": retornos_simulados,
            "Sharpe": sharpes,
        }
    )

    fig_fronteira = px.scatter(
        fronteira,
        x="Volatility",
        y="Return",
        color="Sharpe",
        color_continuous_scale="Purples",
    )

    fig_fronteira.add_trace(
        go.Scatter(
            x=[risco_max],
            y=[retorno_max],
            mode="markers",
            marker=dict(size=18, symbol="star", color="#6C4CE3"),
            name="Maximum Sharpe",
        )
    )

    fig_fronteira.add_trace(
        go.Scatter(
            x=[risco_min],
            y=[retorno_min],
            mode="markers",
            marker=dict(size=14, symbol="diamond", color="#171721"),
            name="Minimum Variance",
        )
    )

    configurar_figura(
        fig_fronteira,
        x_title="Annualized Volatility",
        y_title="Expected Annual Return",
    )

    st.plotly_chart(fig_fronteira, use_container_width=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Maximum-Sharpe Allocation")

        pesos_df = pd.DataFrame(
            {
                "Asset": treino.columns,
                "Weight": pesos_max_sharpe_array,
            }
        ).sort_values("Weight", ascending=True)

        fig_pesos = px.bar(
            pesos_df,
            x="Weight",
            y="Asset",
            orientation="h",
            text_auto=".1%",
        )

        fig_pesos.update_traces(marker_color="#6C4CE3")
        configurar_figura(
            fig_pesos,
            x_title="Portfolio Weight",
            y_title=None,
        )
        st.plotly_chart(fig_pesos, use_container_width=True)

    with col_right:
        st.markdown("### Out-of-Sample Sharpe")

        sharpe_oos_df = pd.DataFrame(
            {
                "Strategy": [
                    "Maximum Sharpe",
                    "Minimum Variance",
                    "Equal Weight",
                ],
                "Sharpe": [
                    resultado_max_sharpe["sharpe"],
                    resultado_min_variancia["sharpe"],
                    resultado_equal["sharpe"],
                ],
            }
        )

        fig_sharpe_oos = px.bar(
            sharpe_oos_df,
            x="Strategy",
            y="Sharpe",
            text_auto=".2f",
        )
        fig_sharpe_oos.update_traces(marker_color="#8171C9")
        configurar_figura(
            fig_sharpe_oos,
            x_title=None,
            y_title="Sharpe Ratio",
        )
        st.plotly_chart(fig_sharpe_oos, use_container_width=True)

    st.markdown("### Out-of-Sample Growth of R$ 1")
    st.caption(
        "The optimized weights are frozen after the training period and "
        "applied to unseen market data."
    )

    performance_plot = performance_oos.copy()
    performance_plot.index.name = "Date"

    perf_long = (
        performance_plot.reset_index()
        .melt(
            id_vars="Date",
            var_name="Strategy",
            value_name="Growth",
        )
    )

    fig_oos = px.line(
        perf_long,
        x="Date",
        y="Growth",
        color="Strategy",
    )

    configurar_figura(
        fig_oos,
        x_title=None,
        y_title="Growth of R$ 1",
    )

    st.plotly_chart(fig_oos, use_container_width=True)