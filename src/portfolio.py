"""portfolio.py - funções para análise e otimização de portfólio"""

import numpy as np
import pandas as pd
import yfinance as yf
import scipy.optimize as optimization


# ============================================================
# DADOS
# ============================================================

def carregar_dados(tickers, inicio, fim):
    dados = yf.download(
        tickers,
        start=inicio,
        end=fim,
        auto_adjust=True
    )["Close"]

    return dados


def calcular_retornos(precos):
    retornos = precos.pct_change().dropna()
    return retornos


def dividir_amostra(retornos, data_corte="2025-01-01"):
    data_corte = pd.Timestamp(data_corte)

    retornos = retornos.sort_index()

    treino = retornos[retornos.index < data_corte]
    teste = retornos[retornos.index >= data_corte]

    if treino.empty:
        raise ValueError("A amostra de treino está vazia.")

    if teste.empty:
        raise ValueError("A amostra de teste está vazia.")

    return treino, teste


# ============================================================
# ESTATÍSTICAS
# ============================================================

def calcular_estatisticas(retornos):
    retorno_anual = retornos.mean() * 252
    cov_anual = retornos.cov() * 252
    volatilidade_anual = retornos.std() * np.sqrt(252)

    return retorno_anual, cov_anual, volatilidade_anual


NUM_TRADING_DAYS = 252
NUM_PORTFOLIOS = 10000


def estatisticas_portfolio(pesos, retornos, taxa_livre_risco=0.0):
    retorno_portfolio = (
        np.sum(retornos.mean() * pesos) * NUM_TRADING_DAYS
    )

    volatilidade_portfolio = np.sqrt(
        np.dot(
            pesos.T,
            np.dot(
                retornos.cov() * NUM_TRADING_DAYS,
                pesos
            )
        )
    )

    sharpe = (
        (retorno_portfolio - taxa_livre_risco)
        / volatilidade_portfolio
    )

    return retorno_portfolio, volatilidade_portfolio, sharpe


# ============================================================
# SIMULAÇÃO DE PORTFÓLIOS
# ============================================================

def gerar_portfolios(retornos, numero_portfolios=NUM_PORTFOLIOS):
    retornos_portfolios = []
    riscos_portfolios = []
    sharpes = []
    pesos_portfolios = []

    numero_ativos = len(retornos.columns)

    rng = np.random.default_rng(42)

    for _ in range(numero_portfolios):
        pesos = rng.random(numero_ativos)
        pesos /= np.sum(pesos)

        retorno, risco, sharpe = estatisticas_portfolio(
            pesos,
            retornos
        )

        pesos_portfolios.append(pesos)
        retornos_portfolios.append(retorno)
        riscos_portfolios.append(risco)
        sharpes.append(sharpe)

    return (
        np.array(pesos_portfolios),
        np.array(retornos_portfolios),
        np.array(riscos_portfolios),
        np.array(sharpes)
    )


# ============================================================
# OTIMIZAÇÃO
# ============================================================

def otimizar_max_sharpe(retornos, taxa_livre_risco=0.0):

    numero_ativos = len(retornos.columns)

    pesos_iniciais = np.ones(numero_ativos) / numero_ativos

    def objetivo(pesos):
        _, _, sharpe = estatisticas_portfolio(
            pesos,
            retornos,
            taxa_livre_risco
        )

        return -sharpe

    restricoes = {
        "type": "eq",
        "fun": lambda pesos: np.sum(pesos) - 1
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
        constraints=restricoes
    )

    return resultado


def otimizar_minima_variancia(retornos):

    numero_ativos = len(retornos.columns)

    pesos_iniciais = np.ones(numero_ativos) / numero_ativos

    def objetivo(pesos):
        _, volatilidade, _ = estatisticas_portfolio(
            pesos,
            retornos
        )

        return volatilidade

    restricoes = {
        "type": "eq",
        "fun": lambda pesos: np.sum(pesos) - 1
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
        constraints=restricoes
    )

    return resultado


# ============================================================
# AVALIAÇÃO
# ============================================================

def avaliar_portfolio(pesos, retornos):
    retorno, volatilidade, sharpe = estatisticas_portfolio(
        pesos,
        retornos
    )

    return {
        "retorno": retorno,
        "volatilidade": volatilidade,
        "sharpe": sharpe
    }


def carteira_equal_weight(retornos):
    numero_ativos = len(retornos.columns)

    pesos = np.ones(numero_ativos) / numero_ativos

    return pesos


def calcular_retornos_portfolio(retornos, pesos):
    retornos_portfolio = retornos.dot(pesos)

    return retornos_portfolio