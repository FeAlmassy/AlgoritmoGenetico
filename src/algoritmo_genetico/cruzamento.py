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

    if len(pai) != len(mae):
        raise ValueError("pai e mae devem ter o mesmo número de genes")

    if qtd_filhos <= 0:
        raise ValueError(f"qtd_filhos deve ser no mínimo 1, recebeu {qtd_filhos}")

    filhos = []
    for _ in range(qtd_filhos):
        # inteiro aleatório de 1 até len(pai)-1, nunca corta nas pontas
        corte = np.random.randint(1, len(pai))
        filho = pai.copy()           # começa igual ao pai
        filho[corte:] = mae[corte:]  # do corte pra frente pega os genes da mãe
        filhos.append(filho)

    return np.array(filhos)  # transforma a lista de filhos em uma matriz
