"""portfolio.py - funções para análise e otimização de portfólio."""

import numpy as np
import pandas as pd
import scipy.optimize as optimization
import yfinance as yf


NUM_TRADING_DAYS = 252
NUM_PORTFOLIOS = 10_000


# ============================================================
# DADOS
# ============================================================

def carregar_dados(tickers, inicio, fim):
    """
    Carrega preços ajustados dos ativos via Yahoo Finance.
    """

    dados = yf.download(
        tickers,
        start=inicio,
        end=fim,
        auto_adjust=True,
        progress=False,
    )["Close"]

    if dados.empty:
        raise ValueError(
            "Nenhum dado de mercado foi retornado."
        )

    return dados


def calcular_retornos(precos):
    """
    Calcula retornos simples diários.
    """

    retornos = precos.pct_change().dropna()

    if retornos.empty:
        raise ValueError(
            "Não foi possível calcular os retornos."
        )

    return retornos


def dividir_amostra(
    retornos,
    data_corte="2025-01-01",
):
    """
    Divide a amostra temporalmente em treino e teste.

    Treino:
        datas anteriores à data de corte.

    Teste:
        datas iguais ou posteriores à data de corte.
    """

    data_corte = pd.Timestamp(data_corte)

    retornos = retornos.sort_index()

    treino = retornos[
        retornos.index < data_corte
    ]

    teste = retornos[
        retornos.index >= data_corte
    ]

    if treino.empty:
        raise ValueError(
            "A amostra de treino está vazia."
        )

    if teste.empty:
        raise ValueError(
            "A amostra de teste está vazia."
        )

    return treino, teste


# ============================================================
# ESTATÍSTICAS
# ============================================================

def calcular_estatisticas(retornos):
    """
    Calcula retorno esperado, matriz de covariância
    e volatilidade anualizada dos ativos.
    """

    retorno_anual = (
        retornos.mean()
        * NUM_TRADING_DAYS
    )

    cov_anual = (
        retornos.cov()
        * NUM_TRADING_DAYS
    )

    volatilidade_anual = (
        retornos.std()
        * np.sqrt(NUM_TRADING_DAYS)
    )

    return (
        retorno_anual,
        cov_anual,
        volatilidade_anual,
    )


def estatisticas_portfolio(
    pesos,
    retornos,
    taxa_livre_risco,
):
    """
    Calcula retorno, volatilidade e Sharpe Ratio
    anualizados de uma carteira.

    Sharpe:

        (Rp - Rf) / sigma_p

    onde:
        Rp = retorno anualizado da carteira
        Rf = taxa livre de risco anualizada
        sigma_p = volatilidade anualizada
    """

    pesos = np.asarray(
        pesos,
        dtype=float,
    )

    retorno_portfolio = (
        np.sum(
            retornos.mean()
            * pesos
        )
        * NUM_TRADING_DAYS
    )

    matriz_covariancia = (
        retornos.cov()
        * NUM_TRADING_DAYS
    )

    volatilidade_portfolio = np.sqrt(
        np.dot(
            pesos.T,
            np.dot(
                matriz_covariancia,
                pesos,
            ),
        )
    )

    if volatilidade_portfolio <= 0:
        raise ValueError(
            "A volatilidade da carteira deve ser positiva."
        )

    sharpe = (
        retorno_portfolio
        - taxa_livre_risco
    ) / volatilidade_portfolio

    return (
        retorno_portfolio,
        volatilidade_portfolio,
        sharpe,
    )


# ============================================================
# SIMULAÇÃO DE PORTFÓLIOS
# ============================================================

def gerar_portfolios(
    retornos,
    taxa_livre_risco,
    numero_portfolios=NUM_PORTFOLIOS,
):
    """
    Gera carteiras aleatórias long-only para visualizar
    o conjunto de oportunidades de portfólio.
    """

    retornos_portfolios = []
    riscos_portfolios = []
    sharpes = []
    pesos_portfolios = []

    numero_ativos = len(
        retornos.columns
    )

    rng = np.random.default_rng(42)

    for _ in range(numero_portfolios):

        pesos = rng.random(
            numero_ativos
        )

        pesos /= np.sum(pesos)

        retorno, risco, sharpe = (
            estatisticas_portfolio(
                pesos=pesos,
                retornos=retornos,
                taxa_livre_risco=taxa_livre_risco,
            )
        )

        pesos_portfolios.append(
            pesos
        )

        retornos_portfolios.append(
            retorno
        )

        riscos_portfolios.append(
            risco
        )

        sharpes.append(
            sharpe
        )

    return (
        np.array(pesos_portfolios),
        np.array(retornos_portfolios),
        np.array(riscos_portfolios),
        np.array(sharpes),
    )


# ============================================================
# OTIMIZAÇÃO
# ============================================================

def otimizar_max_sharpe(
    retornos,
    taxa_livre_risco,
):
    """
    Encontra a carteira long-only que maximiza
    o Sharpe Ratio.
    """

    numero_ativos = len(
        retornos.columns
    )

    pesos_iniciais = (
        np.ones(numero_ativos)
        / numero_ativos
    )

    def objetivo(pesos):

        _, _, sharpe = (
            estatisticas_portfolio(
                pesos=pesos,
                retornos=retornos,
                taxa_livre_risco=taxa_livre_risco,
            )
        )

        return -sharpe

    restricoes = {
        "type": "eq",
        "fun": lambda pesos: (
            np.sum(pesos) - 1
        ),
    }

    limites = tuple(
        (0, 1)
        for _ in range(numero_ativos)
    )

    resultado = optimization.minimize(
        objetivo,
        pesos_iniciais,
        method="SLSQP",
        bounds=limites,
        constraints=restricoes,
    )

    if not resultado.success:
        raise RuntimeError(
            "A otimização de Maximum Sharpe falhou: "
            f"{resultado.message}"
        )

    return resultado


def otimizar_minima_variancia(
    retornos,
    taxa_livre_risco,
):
    """
    Encontra a carteira long-only de mínima variância.

    A taxa livre de risco não afeta a solução
    de mínima variância, mas é utilizada para
    calcular corretamente o Sharpe da carteira.
    """

    numero_ativos = len(
        retornos.columns
    )

    pesos_iniciais = (
        np.ones(numero_ativos)
        / numero_ativos
    )

    def objetivo(pesos):

        _, volatilidade, _ = (
            estatisticas_portfolio(
                pesos=pesos,
                retornos=retornos,
                taxa_livre_risco=taxa_livre_risco,
            )
        )

        return volatilidade

    restricoes = {
        "type": "eq",
        "fun": lambda pesos: (
            np.sum(pesos) - 1
        ),
    }

    limites = tuple(
        (0, 1)
        for _ in range(numero_ativos)
    )

    resultado = optimization.minimize(
        objetivo,
        pesos_iniciais,
        method="SLSQP",
        bounds=limites,
        constraints=restricoes,
    )

    if not resultado.success:
        raise RuntimeError(
            "A otimização de Minimum Variance falhou: "
            f"{resultado.message}"
        )

    return resultado


# ============================================================
# AVALIAÇÃO
# ============================================================

def avaliar_portfolio(
    pesos,
    retornos,
    taxa_livre_risco,
):
    """
    Avalia uma carteira utilizando retorno,
    volatilidade e Sharpe Ratio anualizados.
    """

    retorno, volatilidade, sharpe = (
        estatisticas_portfolio(
            pesos=pesos,
            retornos=retornos,
            taxa_livre_risco=taxa_livre_risco,
        )
    )

    return {
        "retorno": retorno,
        "volatilidade": volatilidade,
        "sharpe": sharpe,
    }


def carteira_equal_weight(retornos):
    """
    Cria carteira igualmente ponderada.
    """

    numero_ativos = len(
        retornos.columns
    )

    pesos = (
        np.ones(numero_ativos)
        / numero_ativos
    )

    return pesos


def calcular_retornos_portfolio(
    retornos,
    pesos,
):
    """
    Calcula a série diária de retornos
    de uma carteira.
    """

    retornos_portfolio = (
        retornos.dot(pesos)
    )

    return retornos_portfolio