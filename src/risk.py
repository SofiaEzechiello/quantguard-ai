"""risk.py - métricas e validação de risco do QuantGuard AI."""

import numpy as np

from scipy.special import xlogy
from scipy.stats import chi2, norm


# ============================================================
# VALUE AT RISK
# ============================================================

def var_parametrico(
    retornos,
    valor_carteira=100_000,
    confianca=0.95,
    horizonte=1,
):
    """
    Calcula o VaR paramétrico assumindo normalidade
    dos retornos.
    """

    mu = retornos.mean()
    sigma = retornos.std()

    quantil_retorno = (
        mu * horizonte
        + sigma
        * np.sqrt(horizonte)
        * norm.ppf(1 - confianca)
    )

    var = (
        -valor_carteira
        * quantil_retorno
    )

    return max(
        float(var),
        0.0,
    )


def var_historico(
    retornos,
    valor_carteira=100_000,
    confianca=0.95,
):
    """
    Calcula o VaR histórico a partir do quantil
    empírico dos retornos.
    """

    quantil_retorno = np.quantile(
        retornos,
        1 - confianca,
    )

    var = (
        -valor_carteira
        * quantil_retorno
    )

    return max(
        float(var),
        0.0,
    )


# ============================================================
# EXPECTED SHORTFALL
# ============================================================

def expected_shortfall(
    retornos,
    valor_carteira=100_000,
    confianca=0.95,
):
    """
    Calcula o Expected Shortfall histórico.

    Representa a perda média condicionada aos
    retornos que ultrapassam o limite do VaR.
    """

    limite = np.quantile(
        retornos,
        1 - confianca,
    )

    perdas_extremas = retornos[
        retornos <= limite
    ]

    if len(perdas_extremas) == 0:
        raise ValueError(
            "Não existem observações suficientes "
            "para calcular o Expected Shortfall."
        )

    es = (
        -valor_carteira
        * perdas_extremas.mean()
    )

    return max(
        float(es),
        0.0,
    )


# ============================================================
# KUPIEC — UNCONDITIONAL COVERAGE TEST
# ============================================================

def teste_kupiec(
    numero_violacoes,
    total_observacoes,
    confianca=0.95,
    nivel_significancia=0.05,
):
    """
    Executa o teste de cobertura incondicional de Kupiec.

    H0:
        a probabilidade observada de violações é
        compatível com a probabilidade esperada
        pelo nível de confiança do VaR.

    Para VaR de 95%:

        probabilidade esperada de violação = 5%.

    A estatística LR segue assintoticamente uma
    distribuição qui-quadrado com 1 grau de liberdade.
    """

    if total_observacoes <= 0:
        raise ValueError(
            "O número de observações deve ser positivo."
        )

    if not 0 < confianca < 1:
        raise ValueError(
            "A confiança deve estar entre 0 e 1."
        )

    if not 0 < nivel_significancia < 1:
        raise ValueError(
            "O nível de significância deve estar entre 0 e 1."
        )

    if (
        numero_violacoes < 0
        or numero_violacoes > total_observacoes
    ):
        raise ValueError(
            "Número de violações inválido."
        )

    probabilidade_esperada = (
        1 - confianca
    )

    probabilidade_observada = (
        numero_violacoes
        / total_observacoes
    )

    x = numero_violacoes
    n = total_observacoes

    # Log-verossimilhança sob H0:
    # taxa de violações = taxa esperada pelo VaR.

    log_likelihood_h0 = (
        xlogy(
            x,
            probabilidade_esperada,
        )
        + xlogy(
            n - x,
            1 - probabilidade_esperada,
        )
    )

    # Log-verossimilhança sob H1:
    # taxa de violações = frequência observada.

    log_likelihood_h1 = (
        xlogy(
            x,
            probabilidade_observada,
        )
        + xlogy(
            n - x,
            1 - probabilidade_observada,
        )
    )

    estatistica_lr = (
        -2
        * (
            log_likelihood_h0
            - log_likelihood_h1
        )
    )

    p_valor = chi2.sf(
        estatistica_lr,
        df=1,
    )

    rejeita_h0 = (
        p_valor
        < nivel_significancia
    )

    return {
        "estatistica_lr": float(
            estatistica_lr
        ),
        "p_valor": float(
            p_valor
        ),
        "taxa_esperada": float(
            probabilidade_esperada
        ),
        "taxa_observada": float(
            probabilidade_observada
        ),
        "nivel_significancia": float(
            nivel_significancia
        ),
        "rejeita_h0": bool(
            rejeita_h0
        ),
        "resultado": (
            "REJECT"
            if rejeita_h0
            else "DO NOT REJECT"
        ),
    }


# ============================================================
# BACKTEST DE VaR
# ============================================================

def backtest_var(
    retornos_teste,
    var,
    valor_carteira=100_000,
    confianca=0.95,
):
    """
    Realiza o backtest out-of-sample do VaR.

    Além de contar as violações, executa o teste
    de cobertura incondicional de Kupiec.
    """

    if len(retornos_teste) == 0:
        raise ValueError(
            "A amostra de teste está vazia."
        )

    perdas = (
        -retornos_teste
        * valor_carteira
    )

    violacoes = (
        perdas > var
    )

    numero_violacoes = int(
        violacoes.sum()
    )

    total_observacoes = len(
        retornos_teste
    )

    taxa_violacoes = (
        numero_violacoes
        / total_observacoes
    )

    kupiec = teste_kupiec(
        numero_violacoes=numero_violacoes,
        total_observacoes=total_observacoes,
        confianca=confianca,
    )

    return {
        "violacoes": numero_violacoes,
        "observacoes": total_observacoes,
        "taxa_violacoes": taxa_violacoes,
        "serie_violacoes": violacoes,
        "kupiec": kupiec,
    }