import numpy as np

from config import (
    TICKERS,
    DATA_INICIO,
    DATA_FIM,
    DATA_CORTE,
)

from src.portfolio import (
    carregar_dados,
    calcular_retornos,
    dividir_amostra,
    estatisticas_portfolio,
)

from src.risk_free import (
    carregar_cdi,
    calcular_taxa_livre_risco_anualizada,
)


print("\n=== QUANTGUARD CDI CHECK ===\n")

# ------------------------------------------------------------
# 1. Mercado
# ------------------------------------------------------------

precos = carregar_dados(
    tickers=TICKERS,
    inicio=DATA_INICIO,
    fim=DATA_FIM,
)

retornos = calcular_retornos(precos)

treino, teste = dividir_amostra(
    retornos,
    data_corte=DATA_CORTE,
)

# ------------------------------------------------------------
# 2. CDI
# ------------------------------------------------------------

cdi = carregar_cdi(
    inicio=DATA_INICIO,
    fim=DATA_FIM,
)

rf_treino = calcular_taxa_livre_risco_anualizada(
    cdi_anual=cdi,
    indice_referencia=treino.index,
)

rf_teste = calcular_taxa_livre_risco_anualizada(
    cdi_anual=cdi,
    indice_referencia=teste.index,
)

print(
    f"CDI anual equivalente · treino: "
    f"{rf_treino:.2%}"
)

print(
    f"CDI anual equivalente · teste: "
    f"{rf_teste:.2%}"
)

# ------------------------------------------------------------
# 3. Teste do Sharpe
# ------------------------------------------------------------

numero_ativos = len(treino.columns)

pesos = np.ones(numero_ativos) / numero_ativos

_, _, sharpe_sem_cdi = estatisticas_portfolio(
    pesos=pesos,
    retornos=treino,
    taxa_livre_risco=0.0,
)

_, _, sharpe_com_cdi = estatisticas_portfolio(
    pesos=pesos,
    retornos=treino,
    taxa_livre_risco=rf_treino,
)

print(
    f"\nSharpe com Rf = 0: "
    f"{sharpe_sem_cdi:.4f}"
)

print(
    f"Sharpe usando CDI: "
    f"{sharpe_com_cdi:.4f}"
)

# ------------------------------------------------------------
# 4. Validações
# ------------------------------------------------------------

assert 0 < rf_treino < 0.50, (
    "CDI de treino parece inválido."
)

assert 0 < rf_teste < 0.50, (
    "CDI de teste parece inválido."
)

assert not np.isclose(
    sharpe_sem_cdi,
    sharpe_com_cdi,
), (
    "O CDI não está alterando o Sharpe."
)

assert sharpe_com_cdi < sharpe_sem_cdi, (
    "O Sharpe com CDI deveria ser menor "
    "que o Sharpe calculado com Rf = 0."
)

print(
    "\n✅ CDI carregado corretamente."
)

print(
    "✅ CDI está alterando o Sharpe."
)

print(
    "✅ Etapa 3 validada.\n"
)