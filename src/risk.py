import numpy as np
from scipy.stats import norm


def var_parametrico(
    retornos,
    valor_carteira=100000,
    confianca=0.95,
    horizonte=1
):
    mu = retornos.mean()
    sigma = retornos.std()

    quantil_retorno = (
        mu * horizonte
        + sigma * np.sqrt(horizonte) * norm.ppf(1 - confianca)
    )

    var = -valor_carteira * quantil_retorno

    return max(var, 0)


def var_historico(
    retornos,
    valor_carteira=100000,
    confianca=0.95
):
    quantil_retorno = np.quantile(
        retornos,
        1 - confianca
    )

    var = -valor_carteira * quantil_retorno

    return max(var, 0)


def expected_shortfall(
    retornos,
    valor_carteira=100000,
    confianca=0.95
):
    limite = np.quantile(
        retornos,
        1 - confianca
    )

    perdas_extremas = retornos[
        retornos <= limite
    ]

    es = -valor_carteira * perdas_extremas.mean()

    return max(es, 0)


def backtest_var(
    retornos_teste,
    var,
    valor_carteira=100000
):
    perdas = -retornos_teste * valor_carteira

    violacoes = perdas > var

    numero_violacoes = int(violacoes.sum())
    total_observacoes = len(retornos_teste)

    taxa_violacoes = (
        numero_violacoes / total_observacoes
    )

    return {
        "violacoes": numero_violacoes,
        "observacoes": total_observacoes,
        "taxa_violacoes": taxa_violacoes,
        "serie_violacoes": violacoes
    }