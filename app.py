import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.portfolio import (
    carregar_dados,
    calcular_retornos,
    dividir_amostra,
    calcular_estatisticas,
    gerar_portfolios,
    otimizar_max_sharpe,
    otimizar_minima_variancia,
    estatisticas_portfolio,
    avaliar_portfolio,
    carteira_equal_weight,
    calcular_retornos_portfolio,
)

from src.risk import (
    var_parametrico,
    var_historico,
    expected_shortfall,
    backtest_var,
)

from src.stress import (
    carregar_periodo_stress,
    stress_historico,
    carregar_fator,
    calcular_beta_fator,
    stress_fator,
)

from src.derivatives import comparar_modelos
from src.ai_audit import auditar_resultado


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="QuantGuard AI",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #F7F8FC;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    h1, h2, h3 {
        color: #171721;
        font-family: Inter, sans-serif;
        letter-spacing: -0.4px;
    }

    p {
        color: #666674;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 28px;
        border-bottom: 1px solid #E5E5EC;
    }

    .stTabs [data-baseweb="tab"] {
        height: 52px;
        background: transparent;
        border: none;
        color: #747481;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        color: #6C4CE3 !important;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #ECECF2;
        padding: 20px 22px;
        border-radius: 16px;
        box-shadow: 0 5px 20px rgba(20, 20, 40, 0.04);
    }

    div[data-testid="stMetricLabel"] {
        color: #777786;
    }

    div[data-testid="stMetricValue"] {
        color: #171721;
        font-weight: 700;
    }

    div[data-testid="stPlotlyChart"] {
        background: white;
        border: 1px solid #ECECF2;
        border-radius: 18px;
        padding: 10px;
        box-shadow: 0 5px 20px rgba(20, 20, 40, 0.04);
    }

    .audit-card {
        background: white;
        border: 1px solid #ECECF2;
        border-radius: 16px;
        padding: 20px 22px;
        min-height: 150px;
        box-shadow: 0 5px 20px rgba(20, 20, 40, 0.04);
    }

    .audit-label {
        color: #777786;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1px;
    }

    .audit-score {
        color: #171721;
        font-size: 34px;
        font-weight: 800;
        margin: 6px 0;
    }

    .audit-pass {
        color: #16784A;
        font-weight: 700;
    }

    .audit-review {
        color: #9A6B00;
        font-weight: 700;
    }

    .audit-fail {
        color: #B42318;
        font-weight: 700;
    }

    .small-note {
        color: #858594;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
<div style="background: linear-gradient(120deg, #171721 0%, #27233A 100%); padding: 38px 42px; border-radius: 22px; margin-bottom: 28px;">
<div style="color: #A991FF; font-size: 13px; font-weight: 600; letter-spacing: 1.6px; margin-bottom: 10px;">QUANTITATIVE FINANCE × ARTIFICIAL INTELLIGENCE</div>
<div style="color: white; font-size: 48px; font-weight: 800; letter-spacing: -1.5px;">QuantGuard AI</div>
<div style="color: #CBC8D5; font-size: 17px; margin-top: 7px; max-width: 760px; line-height: 1.55;">A quantitative validation layer designed to benchmark and audit AI-generated financial analysis.</div>
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

TICKERS = [
    "PETR4.SA",
    "VALE3.SA",
    "ITUB4.SA",
    "WEGE3.SA",
]

DATA_INICIO = "2021-01-01"
DATA_FIM = "2026-08-01"
DATA_CORTE = "2024-12-31"

VALOR_CARTEIRA = 100_000
CONFIANCA = 0.95

# Estes valores ainda são DEMONSTRATIVOS.
# Substitua pelos outputs reais do agente de IA antes de publicar o projeto.
VALOR_IA_VAR_DEMO = 2_000
VALOR_IA_OPTION_DEMO = 9.80
PESOS_IA_DEMO = {
    "PETR4.SA": 0.25,
    "VALE3.SA": 0.25,
    "ITUB4.SA": 0.25,
    "WEGE3.SA": 0.25,
}


# ============================================================
# HELPERS VISUAIS
# ============================================================

def configurar_figura(fig, y_title=None, x_title=None):
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter", color="#626272"),
        margin=dict(l=30, r=30, t=30, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    fig.update_xaxes(
        gridcolor="#EEEEF3",
        zerolinecolor="#EEEEF3",
        title=x_title,
    )

    fig.update_yaxes(
        gridcolor="#EEEEF3",
        zerolinecolor="#EEEEF3",
        title=y_title,
    )

    return fig


def classe_css(classificacao):
    if classificacao == "PASS":
        return "audit-pass"
    if classificacao == "REVIEW":
        return "audit-review"
    return "audit-fail"


def auditar_portfolio_por_sharpe(sharpe_referencia, sharpe_ia):
    """
    Para alocação, Sharpe maior que o benchmark NÃO deve ser penalizado.
    Penalizamos apenas underperformance relativa ao benchmark quantitativo.
    """
    if sharpe_referencia == 0:
        return {
            "erro_percentual": 0.0,
            "score_numerico": 100.0,
            "classificacao": "PASS",
        }

    if sharpe_ia >= sharpe_referencia:
        erro = 0.0
    else:
        erro = (sharpe_referencia - sharpe_ia) / abs(sharpe_referencia)

    score = max(0.0, min(100.0, 100 * (1 - erro)))

    if erro <= 0.05:
        classificacao = "PASS"
    elif erro <= 0.15:
        classificacao = "REVIEW"
    else:
        classificacao = "FAIL"

    return {
        "erro_percentual": erro,
        "score_numerico": score,
        "classificacao": classificacao,
    }


# ============================================================
# MOTOR QUANTITATIVO
# ============================================================

with st.spinner("Loading market data and running quantitative models..."):
    # 1. Dados
    precos = carregar_dados(
        tickers=TICKERS,
        inicio=DATA_INICIO,
        fim=DATA_FIM,
    )

    retornos = calcular_retornos(precos)

    print("Última data do treino:", treino.index.max())
    print("Primeira data do teste:", teste.index.min())
    print("Datas em comum:", treino.index.intersection(teste.index))
    
    assert treino.index.max() < teste.index.min()
    assert len(treino.index.intersection(teste.index)) == 0

    treino, teste = dividir_amostra(
        retornos,
        data_corte=DATA_CORTE,
    )

    retorno_anual, cov_anual, volatilidade_anual = calcular_estatisticas(treino)

    # 2. Portfolio / Markowitz
    pesos_simulados, retornos_simulados, riscos, sharpes = gerar_portfolios(treino)

    max_sharpe = otimizar_max_sharpe(treino)
    min_variancia = otimizar_minima_variancia(treino)

    pesos_max_sharpe_array = max_sharpe.x
    pesos_min_variancia = min_variancia.x
    pesos_equal = carteira_equal_weight(treino)

    pesos_max_sharpe = pd.Series(
        pesos_max_sharpe_array,
        index=treino.columns,
    )

    retorno_max, risco_max, sharpe_max = estatisticas_portfolio(
        pesos_max_sharpe_array,
        treino,
    )

    retorno_min, risco_min, sharpe_min = estatisticas_portfolio(
        pesos_min_variancia,
        treino,
    )

    # 3. Out-of-sample
    resultado_max_sharpe = avaliar_portfolio(
        pesos_max_sharpe_array,
        teste,
    )

    resultado_min_variancia = avaliar_portfolio(
        pesos_min_variancia,
        teste,
    )

    resultado_equal = avaliar_portfolio(
        pesos_equal,
        teste,
    )

    retornos_oos_max = calcular_retornos_portfolio(
        teste,
        pesos_max_sharpe_array,
    )

    retornos_oos_min = calcular_retornos_portfolio(
        teste,
        pesos_min_variancia,
    )

    retornos_oos_equal = calcular_retornos_portfolio(
        teste,
        pesos_equal,
    )

    performance_oos = pd.DataFrame(
        {
            "Maximum Sharpe": (1 + retornos_oos_max).cumprod(),
            "Minimum Variance": (1 + retornos_oos_min).cumprod(),
            "Equal Weight": (1 + retornos_oos_equal).cumprod(),
        }
    )

    # 4. Retornos da carteira
    retornos_carteira_treino = calcular_retornos_portfolio(
        treino,
        pesos_max_sharpe_array,
    )

    retornos_carteira_teste = calcular_retornos_portfolio(
        teste,
        pesos_max_sharpe_array,
    )

    # 5. Risk Engine
    var_param = var_parametrico(
        retornos_carteira_treino,
        valor_carteira=VALOR_CARTEIRA,
        confianca=CONFIANCA,
    )

    var_hist = var_historico(
        retornos_carteira_treino,
        valor_carteira=VALOR_CARTEIRA,
        confianca=CONFIANCA,
    )

    es = expected_shortfall(
        retornos_carteira_treino,
        valor_carteira=VALOR_CARTEIRA,
        confianca=CONFIANCA,
    )

    backtest_param = backtest_var(
        retornos_carteira_teste,
        var_param,
        VALOR_CARTEIRA,
    )

    backtest_hist = backtest_var(
        retornos_carteira_teste,
        var_hist,
        VALOR_CARTEIRA,
    )

    # 6. Stress
    precos_covid = carregar_periodo_stress(
        TICKERS,
        inicio="2020-02-19",
        fim="2020-03-24",
    )

    resultado_covid = stress_historico(
        precos_covid,
        pesos_max_sharpe,
        valor_carteira=VALOR_CARTEIRA,
    )

    retornos_ibov = carregar_fator(
        "^BVSP",
        inicio="2021-01-01",
        fim="2025-01-01",
    )

    betas_ibov = calcular_beta_fator(
        treino,
        retornos_ibov,
    )

    stress_mercado = stress_fator(
        betas_ibov,
        pesos_max_sharpe,
        choque_fator=-0.20,
        valor_carteira=VALOR_CARTEIRA,
    )

    # 7. Derivativos
    resultado_derivativos = comparar_modelos(
        S=100,
        K=100,
        T=1,
        r=0.05,
        sigma=0.20,
        simulacoes=100_000,
    )

    # 8. AI Audit — valores demo
    auditoria_var = auditar_resultado(
        nome_teste="Historical VaR",
        valor_referencia=var_hist,
        valor_ia=VALOR_IA_VAR_DEMO,
    )

    pesos_ia_portfolio = pd.Series(PESOS_IA_DEMO).reindex(teste.columns)

    resultado_portfolio_ia = avaliar_portfolio(
        pesos_ia_portfolio,
        teste,
    )

    sharpe_quant = resultado_max_sharpe["sharpe"]
    sharpe_ia = resultado_portfolio_ia["sharpe"]

    auditoria_portfolio = auditar_portfolio_por_sharpe(
        sharpe_quant,
        sharpe_ia,
    )

    preco_black_scholes = resultado_derivativos["black_scholes"]

    auditoria_option = auditar_resultado(
        nome_teste="Option Pricing",
        valor_referencia=preco_black_scholes,
        valor_ia=VALOR_IA_OPTION_DEMO,
    )

    score_geral = (
        auditoria_var["score_numerico"]
        + auditoria_portfolio["score_numerico"]
        + auditoria_option["score_numerico"]
    ) / 3


# ============================================================
# TABS
# ============================================================

aba_portfolio, aba_risk, aba_stress, aba_derivatives, aba_ai = st.tabs(
    [
        "Portfolio",
        "Risk",
        "Stress Tests",
        "Derivatives",
        "AI Audit",
    ]
)


# ============================================================
# PORTFOLIO
# ============================================================

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


# ============================================================
# RISK
# ============================================================

with aba_risk:
    st.markdown("## Risk Engine")
    st.caption(
        "One-day portfolio risk estimated on the training sample and "
        "validated on unseen observations."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("PARAMETRIC VaR · 95%", f"R$ {var_param:,.0f}")
    col2.metric("HISTORICAL VaR · 95%", f"R$ {var_hist:,.0f}")
    col3.metric("EXPECTED SHORTFALL", f"R$ {es:,.0f}")
    col4.metric(
        "HIST. VaR VIOLATIONS",
        f"{backtest_hist['taxa_violacoes']:.2%}",
    )

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Return Distribution")

        distribuicao = pd.DataFrame(
            {"Portfolio Return": retornos_carteira_treino}
        )

        fig_dist = px.histogram(
            distribuicao,
            x="Portfolio Return",
            nbins=55,
        )
        fig_dist.update_traces(marker_color="#8171C9")

        limite_hist = -(var_hist / VALOR_CARTEIRA)

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

        st.plotly_chart(fig_dist, use_container_width=True)

    with col_right:
        st.markdown("### VaR Backtesting")

        perdas_teste = -retornos_carteira_teste * VALOR_CARTEIRA

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
                line=dict(color="#8171C9", width=1.5),
            )
        )

        violacoes_hist = perdas_teste > var_hist

        fig_backtest.add_trace(
            go.Scatter(
                x=perdas_teste.index[violacoes_hist],
                y=perdas_teste[violacoes_hist],
                mode="markers",
                name="VaR Violations",
                marker=dict(color="#B42318", size=7),
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

        st.plotly_chart(fig_backtest, use_container_width=True)

    st.caption(
        f"Expected violation rate at {CONFIANCA:.0%} confidence: "
        f"{1 - CONFIANCA:.0%}. "
        f"Observed historical-VaR violation rate: "
        f"{backtest_hist['taxa_violacoes']:.2%}."
    )


# ============================================================
# STRESS
# ============================================================

with aba_stress:
    st.markdown("## Stress Testing")
    st.caption(
        "Historical and factor-based scenarios applied to the "
        "maximum-Sharpe portfolio."
    )

    covid_impacto = resultado_covid["retorno_carteira"]
    ibov_impacto = stress_mercado["impacto_carteira"]

    col1, col2, col3 = st.columns(3)

    col1.metric("COVID-19 CRASH", f"{covid_impacto:.2%}")
    col2.metric("IBOVESPA -20%", f"{ibov_impacto:.2%}")
    col3.metric(
        "COVID LOSS",
        f"R$ {resultado_covid['perda']:,.0f}",
    )

    stress_df = pd.DataFrame(
        {
            "Scenario": ["COVID-19 Crash", "Ibovespa -20%"],
            "Portfolio Impact": [covid_impacto, ibov_impacto],
        }
    )

    fig_stress = px.bar(
        stress_df,
        x="Scenario",
        y="Portfolio Impact",
        text_auto=".1%",
    )
    fig_stress.update_traces(marker_color="#6C4CE3")

    configurar_figura(
        fig_stress,
        x_title=None,
        y_title="Portfolio Impact",
    )

    st.plotly_chart(fig_stress, use_container_width=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### COVID-19 Asset Impact")

        covid_assets = (
            resultado_covid["retornos_ativos"]
            .rename("Return")
            .reset_index()
        )
        covid_assets.columns = ["Asset", "Return"]

        fig_covid_assets = px.bar(
            covid_assets,
            x="Asset",
            y="Return",
            text_auto=".1%",
        )
        fig_covid_assets.update_traces(marker_color="#8171C9")

        configurar_figura(
            fig_covid_assets,
            x_title=None,
            y_title="Return During Scenario",
        )

        st.plotly_chart(fig_covid_assets, use_container_width=True)

    with col_right:
        st.markdown("### Market Betas")

        beta_df = betas_ibov.rename("Beta").reset_index()
        beta_df.columns = ["Asset", "Beta"]

        fig_beta = px.bar(
            beta_df,
            x="Asset",
            y="Beta",
            text_auto=".2f",
        )
        fig_beta.update_traces(marker_color="#A991FF")

        configurar_figura(
            fig_beta,
            x_title=None,
            y_title="Beta vs. Ibovespa",
        )

        st.plotly_chart(fig_beta, use_container_width=True)


# ============================================================
# DERIVATIVES
# ============================================================

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


# ============================================================
# AI AUDIT
# ============================================================

with aba_ai:
    st.markdown("## AI Financial Audit")
    st.caption(
        "AI-generated outputs are benchmarked against quantitative "
        "reference models. Current AI values are DEMO placeholders."
    )

    st.warning(
        "Demo mode: replace the placeholder AI outputs with real model "
        "responses before publishing the final project."
    )

    score_var = auditoria_var["score_numerico"]
    score_portfolio = auditoria_portfolio["score_numerico"]
    score_option = auditoria_option["score_numerico"]

    c1, c2, c3 = st.columns(3)

    with c1:
        css = classe_css(auditoria_portfolio["classificacao"])
        st.markdown(
            f"""
            <div class="audit-card">
                <div class="audit-label">PORTFOLIO ALLOCATION</div>
                <div class="audit-score">{score_portfolio:.1f}/100</div>
                <div class="{css}">{auditoria_portfolio["classificacao"]}</div>
                <div class="small-note">
                    Quant OOS Sharpe: {sharpe_quant:.3f}<br>
                    AI OOS Sharpe: {sharpe_ia:.3f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        css = classe_css(auditoria_var["classificacao"])
        st.markdown(
            f"""
            <div class="audit-card">
                <div class="audit-label">VALUE AT RISK</div>
                <div class="audit-score">{score_var:.1f}/100</div>
                <div class="{css}">{auditoria_var["classificacao"]}</div>
                <div class="small-note">
                    Quant benchmark: R$ {var_hist:,.0f}<br>
                    AI estimate: R$ {VALOR_IA_VAR_DEMO:,.0f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        css = classe_css(auditoria_option["classificacao"])
        st.markdown(
            f"""
            <div class="audit-card">
                <div class="audit-label">OPTION PRICING</div>
                <div class="audit-score">{score_option:.1f}/100</div>
                <div class="{css}">{auditoria_option["classificacao"]}</div>
                <div class="small-note">
                    Black–Scholes: R$ {preco_black_scholes:.4f}<br>
                    AI estimate: R$ {VALOR_IA_OPTION_DEMO:.4f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Reliability Overview")

    score_df = pd.DataFrame(
        {
            "Audit": ["Portfolio", "VaR", "Option Pricing"],
            "Score": [score_portfolio, score_var, score_option],
        }
    )

    fig_audit = px.bar(
        score_df,
        x="Audit",
        y="Score",
        text_auto=".1f",
        range_y=[0, 100],
    )
    fig_audit.update_traces(marker_color="#6C4CE3")

    fig_audit.add_hline(
        y=85,
        line_dash="dot",
        line_color="#9A6B00",
        annotation_text="Review threshold",
    )

    configurar_figura(
        fig_audit,
        x_title=None,
        y_title="Reliability Score",
    )

    st.plotly_chart(fig_audit, use_container_width=True)

    st.metric(
        "OVERALL AI RELIABILITY SCORE",
        f"{score_geral:.1f}/100",
    )

    if score_geral >= 95:
        st.success("PASS — quantitative outputs are closely aligned.")
    elif score_geral >= 85:
        st.warning("REVIEW — human financial review is recommended.")
    else:
        st.error("FAIL — human financial review is required.")

    st.caption(
        "Audit thresholds are project-defined validation rules, not an "
        "industry or regulatory standard."
    )
