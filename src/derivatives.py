import numpy as np
from scipy.stats import norm


# ============================================================
# BLACK-SCHOLES
# ============================================================

def black_scholes_call(S, K, T, r, sigma):
    d1 = (
        np.log(S / K)
        + (r + 0.5 * sigma ** 2) * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    preco = (
        S * norm.cdf(d1)
        - K * np.exp(-r * T) * norm.cdf(d2)
    )

    return preco


def black_scholes_put(S, K, T, r, sigma):
    d1 = (
        np.log(S / K)
        + (r + 0.5 * sigma ** 2) * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    preco = (
        K * np.exp(-r * T) * norm.cdf(-d2)
        - S * norm.cdf(-d1)
    )

    return preco


# ============================================================
# MONTE CARLO
# ============================================================

def monte_carlo_call(
    S,
    K,
    T,
    r,
    sigma,
    simulacoes=100000,
    seed=42
):
    rng = np.random.default_rng(seed)

    z = rng.standard_normal(simulacoes)

    preco_final = S * np.exp(
        (r - 0.5 * sigma ** 2) * T
        + sigma * np.sqrt(T) * z
    )

    payoff = np.maximum(
        preco_final - K,
        0
    )

    preco = (
        np.exp(-r * T)
        * np.mean(payoff)
    )

    return preco


def monte_carlo_put(
    S,
    K,
    T,
    r,
    sigma,
    simulacoes=100000,
    seed=42
):
    rng = np.random.default_rng(seed)

    z = rng.standard_normal(simulacoes)

    preco_final = S * np.exp(
        (r - 0.5 * sigma ** 2) * T
        + sigma * np.sqrt(T) * z
    )

    payoff = np.maximum(
        K - preco_final,
        0
    )

    preco = (
        np.exp(-r * T)
        * np.mean(payoff)
    )

    return preco


# ============================================================
# VALIDAÇÃO CRUZADA
# ============================================================

def comparar_modelos(
    S,
    K,
    T,
    r,
    sigma,
    simulacoes=100000
):
    preco_bs = black_scholes_call(
        S,
        K,
        T,
        r,
        sigma
    )

    preco_mc = monte_carlo_call(
        S,
        K,
        T,
        r,
        sigma,
        simulacoes
    )

    diferenca_absoluta = abs(
        preco_bs - preco_mc
    )

    diferenca_percentual = (
        diferenca_absoluta / preco_bs
    ) * 100

    return {
        "black_scholes": preco_bs,
        "monte_carlo": preco_mc,
        "diferenca_absoluta": diferenca_absoluta,
        "diferenca_percentual": diferenca_percentual
    }