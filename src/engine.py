"""engine.py - motor de execução do QuantGuard AI."""

import pandas as pd

from config import (
    TICKERS,
    DATA_INICIO,
    DATA_FIM,
    DATA_CORTE,
    VALOR_CARTEIRA,
    CONFIANCA,
)

from src.portfolio import (
    carregar_dados,
    calcular_retornos,
    dividir_amostra,
    calcular_estatisticas,
    gerar_portfolios,
    otimizar_max_sharpe,
    otimizar_minima_variancia,
    estatisticas_portfolio,
    avaliar_portfolio,
    carteira_equal_weight,
    calcular_retornos_portfolio,
)

from src.risk_free import (
    carregar_cdi,
    calcular_taxa_livre_risco_anualizada,
)

from src.risk import (
    var_parametrico,
    var_historico,
    expected_shortfall,
    backtest_var,
)

from src.stress import (
    carregar_periodo_stress,
    stress_historico,
    carregar_fator,
    calcular_beta_fator,
    stress_fator,
)

from src.derivatives import comparar_modelos


def executar_modelos():

    # ========================================================
    # 1. DADOS DE MERCADO
    # ========================================================

    precos = carregar_dados(
        tickers=TICKERS,
        inicio=DATA_INICIO,
        fim=DATA_FIM,
    )

    retornos = calcular_retornos(
        precos
    )

    treino, teste = dividir_amostra(
        retornos,
        data_corte=DATA_CORTE,
    )

    # ========================================================
    # 2. VALIDAÇÃO DO SPLIT TEMPORAL
    # ========================================================

    if not treino.index.max() < teste.index.min():
        raise ValueError(
            "Erro na divisão da amostra: "
            "treino e teste estão sobrepostos."
        )

    if len(
        treino.index.intersection(
            teste.index
        )
    ) != 0:
        raise ValueError(
            "Erro na divisão da amostra: "
            "existem datas em comum."
        )

    # ========================================================
    # 3. CDI / TAXA LIVRE DE RISCO
    # ========================================================

    cdi = carregar_cdi(
        inicio=DATA_INICIO,
        fim=DATA_FIM,
    )

    taxa_livre_risco_treino = (
        calcular_taxa_livre_risco_anualizada(
            cdi_anual=cdi,
            indice_referencia=treino.index,
        )
    )

    taxa_livre_risco_teste = (
        calcular_taxa_livre_risco_anualizada(
            cdi_anual=cdi,
            indice_referencia=teste.index,
        )
    )

    # ========================================================
    # 4. ESTATÍSTICAS DOS ATIVOS
    # ========================================================

    (
        retorno_anual,
        cov_anual,
        volatilidade_anual,
    ) = calcular_estatisticas(
        treino
    )

    # ========================================================
    # 5. PORTFOLIO / MARKOWITZ
    # ========================================================

    (
        pesos_simulados,
        retornos_simulados,
        riscos,
        sharpes,
    ) = gerar_portfolios(
        retornos=treino,
        taxa_livre_risco=taxa_livre_risco_treino,
    )

    max_sharpe = otimizar_max_sharpe(
        retornos=treino,
        taxa_livre_risco=taxa_livre_risco_treino,
    )

    min_variancia = otimizar_minima_variancia(
        retornos=treino,
        taxa_livre_risco=taxa_livre_risco_treino,
    )

    pesos_max_sharpe_array = (
        max_sharpe.x
    )

    pesos_min_variancia = (
        min_variancia.x
    )

    pesos_equal = (
        carteira_equal_weight(
            treino
        )
    )

    pesos_max_sharpe = pd.Series(
        pesos_max_sharpe_array,
        index=treino.columns,
    )

    (
        retorno_max,
        risco_max,
        sharpe_max,
    ) = estatisticas_portfolio(
        pesos=pesos_max_sharpe_array,
        retornos=treino,
        taxa_livre_risco=taxa_livre_risco_treino,
    )

    (
        retorno_min,
        risco_min,
        sharpe_min,
    ) = estatisticas_portfolio(
        pesos=pesos_min_variancia,
        retornos=treino,
        taxa_livre_risco=taxa_livre_risco_treino,
    )

    # ========================================================
    # 6. OUT-OF-SAMPLE
    # ========================================================

    resultado_max_sharpe = (
        avaliar_portfolio(
            pesos=pesos_max_sharpe_array,
            retornos=teste,
            taxa_livre_risco=taxa_livre_risco_teste,
        )
    )

    resultado_min_variancia = (
        avaliar_portfolio(
            pesos=pesos_min_variancia,
            retornos=teste,
            taxa_livre_risco=taxa_livre_risco_teste,
        )
    )

    resultado_equal = (
        avaliar_portfolio(
            pesos=pesos_equal,
            retornos=teste,
            taxa_livre_risco=taxa_livre_risco_teste,
        )
    )

    retornos_oos_max = (
        calcular_retornos_portfolio(
            teste,
            pesos_max_sharpe_array,
        )
    )

    retornos_oos_min = (
        calcular_retornos_portfolio(
            teste,
            pesos_min_variancia,
        )
    )

    retornos_oos_equal = (
        calcular_retornos_portfolio(
            teste,
            pesos_equal,
        )
    )

    performance_oos = pd.DataFrame(
        {
            "Maximum Sharpe": (
                1 + retornos_oos_max
            ).cumprod(),

            "Minimum Variance": (
                1 + retornos_oos_min
            ).cumprod(),

            "Equal Weight": (
                1 + retornos_oos_equal
            ).cumprod(),
        }
    )

    # ========================================================
    # 7. RETORNOS DA CARTEIRA
    # ========================================================

    retornos_carteira_treino = (
        calcular_retornos_portfolio(
            treino,
            pesos_max_sharpe_array,
        )
    )

    retornos_carteira_teste = (
        calcular_retornos_portfolio(
            teste,
            pesos_max_sharpe_array,
        )
    )

    # ========================================================
    # 8. RISK ENGINE
    # ========================================================

    var_param = var_parametrico(
        retornos_carteira_treino,
        valor_carteira=VALOR_CARTEIRA,
        confianca=CONFIANCA,
    )

    var_hist = var_historico(
        retornos_carteira_treino,
        valor_carteira=VALOR_CARTEIRA,
        confianca=CONFIANCA,
    )

    es = expected_shortfall(
        retornos_carteira_treino,
        valor_carteira=VALOR_CARTEIRA,
        confianca=CONFIANCA,
    )

    backtest_param = backtest_var(
        retornos_carteira_teste,
        var_param,
        VALOR_CARTEIRA,
    )

    backtest_hist = backtest_var(
        retornos_carteira_teste,
        var_hist,
        VALOR_CARTEIRA,
    )

    # ========================================================
    # 9. STRESS TESTING
    # ========================================================

    precos_covid = carregar_periodo_stress(
        TICKERS,
        inicio="2020-02-19",
        fim="2020-03-24",
    )

    resultado_covid = stress_historico(
        precos_covid,
        pesos_max_sharpe,
        valor_carteira=VALOR_CARTEIRA,
    )

    retornos_ibov = carregar_fator(
        "^BVSP",
        inicio="2021-01-01",
        fim="2025-01-01",
    )

    betas_ibov = calcular_beta_fator(
        treino,
        retornos_ibov,
    )

    stress_mercado = stress_fator(
        betas_ibov,
        pesos_max_sharpe,
        choque_fator=-0.20,
        valor_carteira=VALOR_CARTEIRA,
    )

    # ========================================================
    # 10. DERIVATIVOS
    # ========================================================

    resultado_derivativos = (
        comparar_modelos(
            S=100,
            K=100,
            T=1,
            r=0.05,
            sigma=0.20,
            simulacoes=100_000,
        )
    )

    # ========================================================
    # 11. BENCHMARK PARA AI AUDIT
    # ========================================================

    preco_black_scholes = (
        resultado_derivativos[
            "black_scholes"
        ]
    )

    # ========================================================
    # 12. RESULTADOS
    # ========================================================

    return {

        # Dados
        "precos": precos,
        "retornos": retornos,
        "treino": treino,
        "teste": teste,

        # Taxa livre de risco
        "cdi": cdi,
        "taxa_livre_risco_treino": (
            taxa_livre_risco_treino
        ),
        "taxa_livre_risco_teste": (
            taxa_livre_risco_teste
        ),

        # Estatísticas
        "retorno_anual": retorno_anual,
        "cov_anual": cov_anual,
        "volatilidade_anual": (
            volatilidade_anual
        ),

        # Portfolio simulation
        "pesos_simulados": (
            pesos_simulados
        ),
        "retornos_simulados": (
            retornos_simulados
        ),
        "riscos": riscos,
        "sharpes": sharpes,

        # Maximum Sharpe
        "max_sharpe": max_sharpe,
        "pesos_max_sharpe": (
            pesos_max_sharpe
        ),
        "pesos_max_sharpe_array": (
            pesos_max_sharpe_array
        ),
        "retorno_max": retorno_max,
        "risco_max": risco_max,
        "sharpe_max": sharpe_max,

        # Minimum Variance
        "min_variancia": min_variancia,
        "pesos_min_variancia": (
            pesos_min_variancia
        ),
        "retorno_min": retorno_min,
        "risco_min": risco_min,
        "sharpe_min": sharpe_min,

        # Equal Weight
        "pesos_equal": pesos_equal,

        # Out-of-sample
        "resultado_max_sharpe": (
            resultado_max_sharpe
        ),
        "resultado_min_variancia": (
            resultado_min_variancia
        ),
        "resultado_equal": (
            resultado_equal
        ),
        "retornos_oos_max": (
            retornos_oos_max
        ),
        "retornos_oos_min": (
            retornos_oos_min
        ),
        "retornos_oos_equal": (
            retornos_oos_equal
        ),
        "performance_oos": (
            performance_oos
        ),

        # Risk
        "retornos_carteira_treino": (
            retornos_carteira_treino
        ),
        "retornos_carteira_teste": (
            retornos_carteira_teste
        ),
        "var_param": var_param,
        "var_hist": var_hist,
        "es": es,
        "backtest_param": (
            backtest_param
        ),
        "backtest_hist": (
            backtest_hist
        ),

        # Stress
        "resultado_covid": (
            resultado_covid
        ),
        "betas_ibov": betas_ibov,
        "stress_mercado": (
            stress_mercado
        ),

        # Derivativos
        "resultado_derivativos": (
            resultado_derivativos
        ),
        "preco_black_scholes": (
            preco_black_scholes
        ),
    }