"""risk_free.py - taxa livre de risco para o QuantGuard AI."""

import numpy as np
import pandas as pd
import requests


SERIE_CDI_ANUALIZADA = 4389
NUM_TRADING_DAYS = 252


def carregar_cdi(inicio, fim):
    """
    Carrega o CDI anualizado diário diretamente do
    Banco Central do Brasil.

    Fonte:
        SGS - Série 4389
        CDI anualizado em base de 252 dias úteis.

    Os valores são divulgados em percentual ao ano.
    """

    data_inicial = pd.Timestamp(inicio).strftime("%d/%m/%Y")
    data_final = pd.Timestamp(fim).strftime("%d/%m/%Y")

    url = (
        "https://api.bcb.gov.br/dados/serie/"
        f"bcdata.sgs.{SERIE_CDI_ANUALIZADA}/dados"
    )

    parametros = {
        "formato": "json",
        "dataInicial": data_inicial,
        "dataFinal": data_final,
    }

    resposta = requests.get(
        url,
        params=parametros,
        timeout=20,
    )

    resposta.raise_for_status()

    dados = resposta.json()

    if not dados:
        raise ValueError(
            "Nenhum dado do CDI foi retornado pelo Banco Central."
        )

    cdi = pd.DataFrame(dados)

    cdi["data"] = pd.to_datetime(
        cdi["data"],
        format="%d/%m/%Y",
    )

    cdi["valor"] = pd.to_numeric(
        cdi["valor"],
        errors="coerce",
    )

    cdi = (
        cdi
        .dropna(subset=["valor"])
        .set_index("data")
        .sort_index()
    )

    # O Banco Central fornece a série 4389
    # em percentual ao ano.
    #
    # Exemplo:
    # 13.90 -> 0.1390

    cdi["valor"] = (
        cdi["valor"] / 100
    )

    return cdi["valor"]


def calcular_taxa_livre_risco_anualizada(
    cdi_anual,
    indice_referencia,
):
    """
    Calcula uma taxa CDI anual equivalente
    para o período analisado.

    O CDI anualizado diário é convertido para
    sua taxa diária equivalente, composto durante
    o período e novamente anualizado em base 252.
    """

    if len(indice_referencia) == 0:
        raise ValueError(
            "O índice de referência não pode estar vazio."
        )

    taxas = cdi_anual.reindex(
        indice_referencia
    )

    taxas = taxas.ffill().bfill()

    if taxas.isna().any():
        raise ValueError(
            "Não foi possível alinhar o CDI "
            "às datas da amostra."
        )

    taxas_diarias = (
        (1 + taxas)
        ** (1 / NUM_TRADING_DAYS)
        - 1
    )

    fator_acumulado = np.prod(
        1 + taxas_diarias
    )

    numero_dias = len(
        taxas_diarias
    )

    taxa_anual_equivalente = (
        fator_acumulado
        ** (
            NUM_TRADING_DAYS
            / numero_dias
        )
        - 1
    )

    return float(
        taxa_anual_equivalente
    )