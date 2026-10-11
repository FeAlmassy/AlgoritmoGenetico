# ordem: 4ª etapa, vem depois de selecao.py -> próxima: mutacao.py
"""Cruzamento (crossover) entre pai e mãe."""

import numpy as np


def cruzar_um_ponto(pai: np.ndarray, mae: np.ndarray, qtd_filhos: int = 3) -> np.ndarray:
    """
    Gera filhos cortando os pais em um ponto aleatório: o filho pega o
    começo do pai e o final da mãe. Cada filho tem seu próprio ponto de corte.

    Parâmetros:
        pai -> array (qtd_genes,) com o primeiro cromossomo escolhido
        mae -> array (qtd_genes,) com o segundo cromossomo escolhido
        qtd_filhos -> quantos filhos gerar

    Retorna:
        array (qtd_filhos, qtd_genes) com os filhos
    """

    if not isinstance(pai, np.ndarray) or not isinstance(mae, np.ndarray):
        raise TypeError("Os parâmetros pai e mae devem ser do tipo ndarray")

    # .ndim é o número de dimensões do array: 1 = lista simples (um cromossomo só)
    if pai.ndim != 1 or mae.ndim != 1:
        raise ValueError(
            f"pai e mae devem ser um cromossomo só (1 dimensão), "
            f"recebeu {pai.ndim} e {mae.ndim} dimensões"
        )

    if len(pai) != len(mae):
        raise ValueError(f"pai tem {len(pai)} genes, mas mae tem {len(mae)}")

    if len(pai) < 2:
        raise ValueError(
            f"pai e mae precisam de pelo menos 2 genes para ter onde cortar, têm {len(pai)}"
        )

    # bool é um tipo de int no Python (True == 1), então precisa barrar separado
    if not isinstance(qtd_filhos, int) or isinstance(qtd_filhos, bool):
        raise TypeError("qtd_filhos deve ser um número inteiro")

    if qtd_filhos <= 0:
        raise ValueError(f"qtd_filhos deve ser no mínimo 1, recebeu {qtd_filhos}")

    filhos = []
    for _ in range(qtd_filhos):
        # inteiro aleatório de 1 até len(pai)-1, nunca corta nas pontas
        corte = np.random.randint(1, len(pai))
        filho = pai.copy()           # começa igual ao pai
        filho[corte:] = mae[corte:]  # do corte pra frente pega os genes da mãe
        filhos.append(filho)

    return np.array(filhos)  # transforma a lista de filhos em uma matriz (qtd_filhos, qtd_genes)
