def classe_css(classificacao):
    if classificacao == "PASS":
        return "audit-pass"
    if classificacao == "REVIEW":
        return "audit-review"
    return "audit-fail"


def auditar_portfolio_por_sharpe(sharpe_referencia, sharpe_ia):
    """
    Para alocação, Sharpe maior que o benchmark NÃO deve ser penalizado.
    Penalizamos apenas underperformance relativa ao benchmark quantitativo.
    """
    if sharpe_referencia == 0:
        return {
            "erro_percentual": 0.0,
            "score_numerico": 100.0,
            "classificacao": "PASS",
        }

    if sharpe_ia >= sharpe_referencia:
        erro = 0.0
    else:
        erro = (sharpe_referencia - sharpe_ia) / abs(sharpe_referencia)

    score = max(0.0, min(100.0, 100 * (1 - erro)))

    if erro <= 0.05:
        classificacao = "PASS"
    elif erro <= 0.15:
        classificacao = "REVIEW"
    else:
        classificacao = "FAIL"

    return {
        "erro_percentual": erro,
        "score_numerico": score,
        "classificacao": classificacao,
    }
