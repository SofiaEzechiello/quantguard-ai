import streamlit as st

from src.engine import executar_modelos

from ui.theme import configurar_pagina, renderizar_header
from ui.portfolio_view import render_portfolio
from ui.risk_view import render_risk
from ui.stress_view import render_stress
from ui.derivatives_view import render_derivatives
from ui.audit_view import render_audit
from config import (
    TICKERS,
    DATA_INICIO,
    DATA_FIM,
    DATA_CORTE,
    VALOR_CARTEIRA,
    CONFIANCA,
    VALOR_IA_VAR_DEMO,
    VALOR_IA_OPTION_DEMO,
    PESOS_IA_DEMO,
)


configurar_pagina()
renderizar_header()


with st.spinner(
    "Loading market data and running quantitative models..."
):
    resultados = executar_modelos()


aba_portfolio, aba_risk, aba_stress, aba_derivatives, aba_ai = st.tabs(
    [
        "Portfolio",
        "Risk",
        "Stress Tests",
        "Derivatives",
        "AI Audit",
    ]
)


with aba_portfolio:
    render_portfolio(resultados)

with aba_risk:
    render_risk(resultados)

with aba_stress:
    render_stress(resultados)

with aba_derivatives:
    render_derivatives(resultados)

with aba_ai:
    render_audit(resultados)