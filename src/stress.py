import pandas as pd
import yfinance as yf


# ============================================================
# HISTORICAL STRESS TEST
# ============================================================

def carregar_periodo_stress(tickers, inicio, fim):
    precos = yf.download(
        tickers,
        start=inicio,
        end=fim,
        auto_adjust=True
    )["Close"]

    return precos


def stress_historico(
    precos,
    pesos,
    valor_carteira=100000
):
    # Retorno acumulado de cada ativo no período de stress
    retornos_ativos = (
        precos.iloc[-1] / precos.iloc[0] - 1
    )

    # Garante que os pesos estejam na mesma ordem dos ativos
    pesos = pesos.reindex(retornos_ativos.index)

    retorno_carteira = (
        retornos_ativos * pesos
    ).sum()

    valor_final = (
        valor_carteira * (1 + retorno_carteira)
    )

    perda = (
        valor_carteira - valor_final
    )

    return {
        "retornos_ativos": retornos_ativos,
        "retorno_carteira": retorno_carteira,
        "valor_final": valor_final,
        "perda": perda
    }


# ============================================================
# FACTOR STRESS TEST
# ============================================================

def carregar_fator(ticker, inicio, fim):
    dados = yf.download(
        ticker,
        start=inicio,
        end=fim,
        auto_adjust=True
    )["Close"]

    # yfinance pode devolver DataFrame mesmo com um único ticker
    if isinstance(dados, pd.DataFrame):
        dados = dados.iloc[:, 0]

    retornos = dados.pct_change(
        fill_method=None
    ).dropna()

    return retornos


def calcular_beta_fator(
    retornos_ativos,
    retornos_fator
):
    fator = retornos_fator.rename("FATOR")

    dados = retornos_ativos.join(
        fator,
        how="inner"
    )

    variancia_fator = dados["FATOR"].var()

    betas = {}

    for ativo in retornos_ativos.columns:
        covariancia = dados[ativo].cov(
            dados["FATOR"]
        )

        beta = (
            covariancia / variancia_fator
        )

        betas[ativo] = beta

    return pd.Series(betas)


def stress_fator(
    betas,
    pesos,
    choque_fator,
    valor_carteira=100000
):
    pesos = pesos.reindex(betas.index)

    impacto_ativos = (
        betas * choque_fator
    )

    impacto_carteira = (
        impacto_ativos * pesos
    ).sum()

    valor_final = (
        valor_carteira
        * (1 + impacto_carteira)
    )

    perda = (
        valor_carteira - valor_final
    )

    return {
        "impacto_ativos": impacto_ativos,
        "impacto_carteira": impacto_carteira,
        "valor_final": valor_final,
        "perda": perda
    }