from src.risk import teste_kupiec


print("\n=== QUANTGUARD KUPIEC CHECK ===\n")


resultado = teste_kupiec(
    numero_violacoes=5,
    total_observacoes=100,
    confianca=0.95,
)

print(
    f"Expected violation rate: "
    f"{resultado['taxa_esperada']:.2%}"
)

print(
    f"Observed violation rate: "
    f"{resultado['taxa_observada']:.2%}"
)

print(
    f"Kupiec LR: "
    f"{resultado['estatistica_lr']:.4f}"
)

print(
    f"p-value: "
    f"{resultado['p_valor']:.4f}"
)

print(
    f"Decision: "
    f"{resultado['resultado']}"
)


assert (
    resultado["taxa_esperada"]
    == 0.05
)

assert (
    resultado["taxa_observada"]
    == 0.05
)

assert (
    resultado["rejeita_h0"]
    is False
)

print(
    "\n✅ Kupiec test funcionando."
)