def calcular_erro_percentual(valor_referencia, valor_ia):
    if valor_referencia == 0:
        raise ValueError("O valor de referência não pode ser zero.")

    erro = (
        abs(valor_ia - valor_referencia)
        / abs(valor_referencia)
    )

    return erro


def calcular_score_numerico(erro):
    score = 100 * (1 - erro)

    return max(0, min(100, score))


def classificar_resultado(erro):
    if erro <= 0.05:
        return "PASS"

    elif erro <= 0.15:
        return "REVIEW"

    else:
        return "FAIL"


def auditar_resultado(
    nome_teste,
    valor_referencia,
    valor_ia
):
    erro = calcular_erro_percentual(
        valor_referencia,
        valor_ia
    )

    score = calcular_score_numerico(erro)

    classificacao = classificar_resultado(erro)

    return {
        "teste": nome_teste,
        "valor_referencia": valor_referencia,
        "valor_ia": valor_ia,
        "erro_percentual": erro,
        "score_numerico": score,
        "classificacao": classificacao
    }