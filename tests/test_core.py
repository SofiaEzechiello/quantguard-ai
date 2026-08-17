"""Testes centrais do QuantGuard AI."""

import numpy as np
import pandas as pd

from src.ai_audit import (
    calcular_erro_percentual,
    calcular_score_numerico,
    classificar_resultado,
    auditar_portfolio_por_sharpe,
)

from src.derivatives import (
    black_scholes_call,
    monte_carlo_call,
)

from src.portfolio import (
    dividir_amostra,
    carteira_equal_weight,
    estatisticas_portfolio,
    otimizar_max_sharpe,
)

from src.risk import (
    var_parametrico,
    var_historico,
    expected_shortfall,
    teste_kupiec,
    backtest_var,
)

from src.risk_free import (
    calcular_taxa_livre_risco_anualizada,
)


# ============================================================
# FIXTURES / DADOS SINTÉTICOS
# ============================================================


def criar_retornos_sinteticos():
    """
    Cria uma pequena amostra determinística de retornos
    para os testes de portfólio.
    """

    indice = pd.date_range(
        start="2024-01-01",
        periods=12,
        freq="B",
    )

    retornos = pd.DataFrame(
        {
            "ATIVO_A": [
                0.010,
                0.005,
                -0.004,
                0.008,
                0.003,
                -0.002,
                0.009,
                0.004,
                -0.003,
                0.007,
                0.002,
                0.006,
            ],
            "ATIVO_B": [
                0.004,
                0.003,
                -0.002,
                0.005,
                0.002,
                -0.001,
                0.004,
                0.003,
                -0.001,
                0.005,
                0.002,
                0.004,
            ],
            "ATIVO_C": [
                0.012,
                -0.006,
                0.008,
                0.010,
                -0.004,
                0.009,
                -0.005,
                0.011,
                0.006,
                -0.003,
                0.010,
                0.007,
            ],
            "ATIVO_D": [
                0.002,
                0.001,
                0.000,
                0.003,
                0.001,
                -0.001,
                0.002,
                0.001,
                0.000,
                0.002,
                0.001,
                0.003,
            ],
        },
        index=indice,
    )

    return retornos


# ============================================================
# PORTFOLIO
# ============================================================


def test_equal_weight_soma_um():
    retornos = criar_retornos_sinteticos()

    pesos = carteira_equal_weight(
        retornos
    )

    assert np.isclose(
        pesos.sum(),
        1.0,
    )


def test_equal_weight_tem_mesmo_peso():
    retornos = criar_retornos_sinteticos()

    pesos = carteira_equal_weight(
        retornos
    )

    assert np.allclose(
        pesos,
        np.array(
            [0.25, 0.25, 0.25, 0.25]
        ),
    )


def test_divisao_temporal_sem_overlap():
    indice = pd.to_datetime(
        [
            "2024-12-30",
            "2024-12-31",
            "2025-01-02",
            "2025-01-03",
        ]
    )

    retornos = pd.DataFrame(
        {
            "ATIVO": [
                0.01,
                -0.01,
                0.02,
                0.01,
            ]
        },
        index=indice,
    )

    treino, teste = dividir_amostra(
        retornos,
        data_corte="2025-01-01",
    )

    assert treino.index.max() < teste.index.min()

    assert len(
        treino.index.intersection(
            teste.index
        )
    ) == 0


def test_cdi_reduz_sharpe():
    retornos = criar_retornos_sinteticos()

    pesos = carteira_equal_weight(
        retornos
    )

    _, _, sharpe_sem_rf = (
        estatisticas_portfolio(
            pesos=pesos,
            retornos=retornos,
            taxa_livre_risco=0.0,
        )
    )

    _, _, sharpe_com_rf = (
        estatisticas_portfolio(
            pesos=pesos,
            retornos=retornos,
            taxa_livre_risco=0.10,
        )
    )

    assert sharpe_com_rf < sharpe_sem_rf


def test_otimizacao_max_sharpe_converge():
    retornos = criar_retornos_sinteticos()

    resultado = otimizar_max_sharpe(
        retornos=retornos,
        taxa_livre_risco=0.10,
    )

    assert resultado.success

    assert np.isclose(
        resultado.x.sum(),
        1.0,
        atol=1e-6,
    )

    assert np.all(
        resultado.x >= -1e-8
    )

    assert np.all(
        resultado.x <= 1 + 1e-8
    )


# ============================================================
# CDI / RISK-FREE
# ============================================================


def test_taxa_livre_risco_constante():
    """
    Se o CDI anualizado for constantemente 10%,
    a taxa anual equivalente calculada deve permanecer
    próxima de 10%.
    """

    indice = pd.date_range(
        start="2024-01-01",
        periods=252,
        freq="B",
    )

    cdi = pd.Series(
        0.10,
        index=indice,
    )

    taxa = (
        calcular_taxa_livre_risco_anualizada(
            cdi_anual=cdi,
            indice_referencia=indice,
        )
    )

    assert np.isclose(
        taxa,
        0.10,
        atol=1e-10,
    )


# ============================================================
# RISK
# ============================================================


def test_var_parametrico_nao_negativo():
    retornos = pd.Series(
        [
            0.010,
            -0.020,
            0.005,
            -0.015,
            0.008,
            -0.030,
            0.012,
            -0.010,
        ]
    )

    var = var_parametrico(
        retornos,
        valor_carteira=100_000,
        confianca=0.95,
    )

    assert var >= 0


def test_var_historico_nao_negativo():
    retornos = pd.Series(
        [
            0.010,
            -0.020,
            0.005,
            -0.015,
            0.008,
            -0.030,
            0.012,
            -0.010,
        ]
    )

    var = var_historico(
        retornos,
        valor_carteira=100_000,
        confianca=0.95,
    )

    assert var >= 0


def test_expected_shortfall_maior_ou_igual_var_historico():
    retornos = pd.Series(
        [
            -0.050,
            -0.040,
            -0.030,
            -0.020,
            -0.015,
            -0.010,
            0.000,
            0.005,
            0.010,
            0.015,
            0.020,
            0.025,
            0.030,
            0.035,
            0.040,
            0.045,
            0.050,
            0.010,
            -0.005,
            0.020,
        ]
    )

    var = var_historico(
        retornos,
        valor_carteira=100_000,
        confianca=0.95,
    )

    es = expected_shortfall(
        retornos,
        valor_carteira=100_000,
        confianca=0.95,
    )

    assert es >= var


# ============================================================
# KUPIEC
# ============================================================


def test_kupiec_taxa_exata_nao_rejeita():
    resultado = teste_kupiec(
        numero_violacoes=5,
        total_observacoes=100,
        confianca=0.95,
    )

    assert np.isclose(
        resultado["taxa_esperada"],
        0.05,
    )

    assert np.isclose(
        resultado["taxa_observada"],
        0.05,
    )

    assert resultado["rejeita_h0"] is False

    assert resultado["p_valor"] > 0.05


def test_kupiec_excesso_extremo_rejeita():
    resultado = teste_kupiec(
        numero_violacoes=20,
        total_observacoes=100,
        confianca=0.95,
    )

    assert resultado["rejeita_h0"] is True

    assert resultado["p_valor"] < 0.05


def test_backtest_var_retorna_kupiec():
    retornos = pd.Series(
        [
            -0.01,
            -0.02,
            0.01,
            -0.03,
            0.005,
            -0.005,
            0.010,
            -0.025,
            0.002,
            -0.008,
        ]
    )

    resultado = backtest_var(
        retornos_teste=retornos,
        var=2_000,
        valor_carteira=100_000,
        confianca=0.95,
    )

    assert "kupiec" in resultado

    assert (
        resultado["observacoes"]
        == len(retornos)
    )


# ============================================================
# DERIVATIVOS
# ============================================================


def test_black_scholes_valor_conhecido():
    """
    Caso clássico:

        S = 100
        K = 100
        T = 1
        r = 5%
        sigma = 20%

    Call Black-Scholes ≈ 10.4506.
    """

    preco = black_scholes_call(
        S=100,
        K=100,
        T=1,
        r=0.05,
        sigma=0.20,
    )

    assert np.isclose(
        preco,
        10.4506,
        atol=0.001,
    )


def test_monte_carlo_proximo_black_scholes():
    preco_bs = black_scholes_call(
        S=100,
        K=100,
        T=1,
        r=0.05,
        sigma=0.20,
    )

    preco_mc = monte_carlo_call(
        S=100,
        K=100,
        T=1,
        r=0.05,
        sigma=0.20,
        simulacoes=100_000,
        seed=42,
    )

    erro_relativo = (
        abs(preco_mc - preco_bs)
        / preco_bs
    )

    assert erro_relativo < 0.02


# ============================================================
# AI AUDIT
# ============================================================


def test_erro_percentual():
    erro = calcular_erro_percentual(
        valor_referencia=100,
        valor_ia=105,
    )

    assert np.isclose(
        erro,
        0.05,
    )


def test_score_numerico_limitado():
    assert np.isclose(
        calcular_score_numerico(0.05),
        95.0,
    )

    assert np.isclose(
        calcular_score_numerico(2.0),
        0.0,
    )


def test_classificacao_auditoria():
    assert (
        classificar_resultado(0.03)
        == "PASS"
    )

    assert (
        classificar_resultado(0.10)
        == "REVIEW"
    )

    assert (
        classificar_resultado(0.30)
        == "FAIL"
    )


def test_portfolio_superior_ao_benchmark_nao_e_penalizado():
    resultado = (
        auditar_portfolio_por_sharpe(
            sharpe_referencia=1.0,
            sharpe_ia=1.20,
        )
    )

    assert np.isclose(
        resultado["score_numerico"],
        100.0,
    )

    assert (
        resultado["classificacao"]
        == "PASS"
    )