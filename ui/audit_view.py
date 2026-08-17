"""Visualização do módulo de auditoria quantitativa."""

import pandas as pd
import streamlit as st
import plotly.express as px

from config import (
    VALOR_IA_VAR_DEMO,
    VALOR_IA_OPTION_DEMO,
)

from ui.charts import configurar_figura


def classe_css(classificacao):
    if classificacao == "PASS":
        return "audit-pass"

    if classificacao == "REVIEW":
        return "audit-review"

    return "audit-fail"


def render_audit(resultados):

    auditoria_var = resultados["auditoria_var"]
    auditoria_portfolio = resultados["auditoria_portfolio"]
    auditoria_option = resultados["auditoria_option"]

    sharpe_quant = resultados["sharpe_quant"]
    sharpe_ia = resultados["sharpe_ia"]

    preco_black_scholes = resultados["preco_black_scholes"]
    score_geral = resultados["score_geral"]

    # ========================================================
    # CABEÇALHO
    # ========================================================

    st.markdown("## AI Financial Audit")

    st.caption(
        "AI-generated outputs are benchmarked against quantitative "
        "reference models. Current AI values are DEMO placeholders."
    )

    st.warning(
        "Demo mode: replace the placeholder AI outputs with real model "
        "responses before publishing the final project."
    )

    # ========================================================
    # SCORES
    # ========================================================

    score_var = auditoria_var["score_numerico"]
    score_portfolio = auditoria_portfolio["score_numerico"]
    score_option = auditoria_option["score_numerico"]

    # ========================================================
    # CARDS
    # ========================================================

    c1, c2, c3 = st.columns(3)

    with c1:
        css = classe_css(
            auditoria_portfolio["classificacao"]
        )

        st.markdown(
            f"""
            <div class="audit-card">
                <div class="audit-label">
                    PORTFOLIO ALLOCATION
                </div>

                <div class="audit-score">
                    {score_portfolio:.1f}/100
                </div>

                <div class="{css}">
                    {auditoria_portfolio["classificacao"]}
                </div>

                <div class="small-note">
                    Quant OOS Sharpe: {sharpe_quant:.3f}
                    <br>
                    AI OOS Sharpe: {sharpe_ia:.3f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        css = classe_css(
            auditoria_var["classificacao"]
        )

        st.markdown(
            f"""
            <div class="audit-card">
                <div class="audit-label">
                    VALUE AT RISK
                </div>

                <div class="audit-score">
                    {score_var:.1f}/100
                </div>

                <div class="{css}">
                    {auditoria_var["classificacao"]}
                </div>

                <div class="small-note">
                    Quant benchmark:
                    R$ {auditoria_var["valor_referencia"]:,.0f}
                    <br>
                    AI estimate:
                    R$ {VALOR_IA_VAR_DEMO:,.0f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        css = classe_css(
            auditoria_option["classificacao"]
        )

        st.markdown(
            f"""
            <div class="audit-card">
                <div class="audit-label">
                    OPTION PRICING
                </div>

                <div class="audit-score">
                    {score_option:.1f}/100
                </div>

                <div class="{css}">
                    {auditoria_option["classificacao"]}
                </div>

                <div class="small-note">
                    Black–Scholes:
                    R$ {preco_black_scholes:.4f}
                    <br>
                    AI estimate:
                    R$ {VALOR_IA_OPTION_DEMO:.4f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ========================================================
    # RELIABILITY OVERVIEW
    # ========================================================

    st.markdown("### Reliability Overview")

    score_df = pd.DataFrame(
        {
            "Audit": [
                "Portfolio",
                "VaR",
                "Option Pricing",
            ],
            "Score": [
                score_portfolio,
                score_var,
                score_option,
            ],
        }
    )

    fig_audit = px.bar(
        score_df,
        x="Audit",
        y="Score",
        text_auto=".1f",
        range_y=[0, 100],
    )

    fig_audit.update_traces(
        marker_color="#6C4CE3"
    )

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

    st.plotly_chart(
        fig_audit,
        use_container_width=True,
    )

    # ========================================================
    # SCORE GERAL
    # ========================================================

    st.metric(
        "OVERALL AI RELIABILITY SCORE",
        f"{score_geral:.1f}/100",
    )

    if score_geral >= 95:
        st.success(
            "PASS — quantitative outputs are closely aligned."
        )

    elif score_geral >= 85:
        st.warning(
            "REVIEW — human financial review is recommended."
        )

    else:
        st.error(
            "FAIL — human financial review is required."
        )

    st.caption(
        "Audit thresholds are project-defined validation rules, "
        "not an industry or regulatory standard."
    )