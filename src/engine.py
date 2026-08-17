"""engine.py - motor de execução do backtest"""

with st.spinner("Loading market data and running quantitative models..."):
    # 1. Dados
    precos = carregar_dados(
        tickers=TICKERS,
        inicio=DATA_INICIO,
        fim=DATA_FIM,
    )

    retornos = calcular_retornos(precos)

    print("Última data do treino:", treino.index.max())
    print("Primeira data do teste:", teste.index.min())
    print("Datas em comum:", treino.index.intersection(teste.index))
    
    assert treino.index.max() < teste.index.min()
    assert len(treino.index.intersection(teste.index)) == 0

    treino, teste = dividir_amostra(
        retornos,
        data_corte=DATA_CORTE,
    )

    retorno_anual, cov_anual, volatilidade_anual = calcular_estatisticas(treino)

    # 2. Portfolio / Markowitz
    pesos_simulados, retornos_simulados, riscos, sharpes = gerar_portfolios(treino)

    max_sharpe = otimizar_max_sharpe(treino)
    min_variancia = otimizar_minima_variancia(treino)

    pesos_max_sharpe_array = max_sharpe.x
    pesos_min_variancia = min_variancia.x
    pesos_equal = carteira_equal_weight(treino)

    pesos_max_sharpe = pd.Series(
        pesos_max_sharpe_array,
        index=treino.columns,
    )

    retorno_max, risco_max, sharpe_max = estatisticas_portfolio(
        pesos_max_sharpe_array,
        treino,
    )

    retorno_min, risco_min, sharpe_min = estatisticas_portfolio(
        pesos_min_variancia,
        treino,
    )

    # 3. Out-of-sample
    resultado_max_sharpe = avaliar_portfolio(
        pesos_max_sharpe_array,
        teste,
    )

    resultado_min_variancia = avaliar_portfolio(
        pesos_min_variancia,
        teste,
    )

    resultado_equal = avaliar_portfolio(
        pesos_equal,
        teste,
    )

    retornos_oos_max = calcular_retornos_portfolio(
        teste,
        pesos_max_sharpe_array,
    )

    retornos_oos_min = calcular_retornos_portfolio(
        teste,
        pesos_min_variancia,
    )

    retornos_oos_equal = calcular_retornos_portfolio(
        teste,
        pesos_equal,
    )

    performance_oos = pd.DataFrame(
        {
            "Maximum Sharpe": (1 + retornos_oos_max).cumprod(),
            "Minimum Variance": (1 + retornos_oos_min).cumprod(),
            "Equal Weight": (1 + retornos_oos_equal).cumprod(),
        }
    )

    # 4. Retornos da carteira
    retornos_carteira_treino = calcular_retornos_portfolio(
        treino,
        pesos_max_sharpe_array,
    )

    retornos_carteira_teste = calcular_retornos_portfolio(
        teste,
        pesos_max_sharpe_array,
    )

    # 5. Risk Engine
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

    # 6. Stress
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

    # 7. Derivativos
    resultado_derivativos = comparar_modelos(
        S=100,
        K=100,
        T=1,
        r=0.05,
        sigma=0.20,
        simulacoes=100_000,
    )

    # 8. AI Audit — valores demo
    auditoria_var = auditar_resultado(
        nome_teste="Historical VaR",
        valor_referencia=var_hist,
        valor_ia=VALOR_IA_VAR_DEMO,
    )

    pesos_ia_portfolio = pd.Series(PESOS_IA_DEMO).reindex(teste.columns)

    resultado_portfolio_ia = avaliar_portfolio(
        pesos_ia_portfolio,
        teste,
    )

    sharpe_quant = resultado_max_sharpe["sharpe"]
    sharpe_ia = resultado_portfolio_ia["sharpe"]

    auditoria_portfolio = auditar_portfolio_por_sharpe(
        sharpe_quant,
        sharpe_ia,
    )

    preco_black_scholes = resultado_derivativos["black_scholes"]

    auditoria_option = auditar_resultado(
        nome_teste="Option Pricing",
        valor_referencia=preco_black_scholes,
        valor_ia=VALOR_IA_OPTION_DEMO,
    )

    score_geral = (
        auditoria_var["score_numerico"]
        + auditoria_portfolio["score_numerico"]
        + auditoria_option["score_numerico"]
    ) / 3