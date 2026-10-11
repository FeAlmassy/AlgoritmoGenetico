# ordem: 4ª etapa, vem depois de selecao.py -> proxima: mutacao.py
"""Operadores de cruzamento,."""

import numpy as np

def cruzar_um_ponto(pai: np.ndarray, mae: np.ndarray, qtd_filhos: int = 3,) -> np.ndarray:
    """
    Gera filhos cortando os pais em um ponto aleatorio: o filho pega o
    comeco do pai e o final da mae. Cada filho tem seu proprio ponto de corte.

    Parametros:
        pai -> array (qtd_genes,) com o primeiro cromossomo escolhido
        mae -> array (qtd_genes,) com o segundo cromossomo escolhido
        qtd_filhos -> quantos filhos gerar

    Retorna:
        array (qtd_filhos, qtd_genes) com os filhos
    """

    if not isinstance(pai, np.ndarray) or not isinstance(mae, np.ndarray):
        raise TypeError("Os parametros pai e mae devem ser do tipo ndarray")

    # .ndim e o numero de dimensoes do array: 1 = lista simples (um cromossomo so)
    if pai.ndim != 1 or mae.ndim != 1:
        raise ValueError(f"pai e mae devem ser um cromossomo so (1 dimensao), recebeu {pai.ndim} e {mae.ndim} dimensoes")

    if len(pai) != len(mae):
        raise ValueError(f"pai tem {len(pai)} genes, mas mae tem {len(mae)}")

    if len(pai) < 2:
        raise ValueError(f"pai e mae precisam de pelo menos 2 genes para ter onde cortar, tem {len(pai)}")

    # bool e um tipo de int no python (True == 1), entao precisa barrar separado
    if not isinstance(qtd_filhos, int) or isinstance(qtd_filhos, bool):
        raise TypeError("qtd_filhos deve ser um numero inteiro")

    if qtd_filhos <= 0:
        raise ValueError(f"qtd_filhos deve ser no minimo 1, recebeu {qtd_filhos}")

    filhos = []
    for _ in range(qtd_filhos):
        corte = np.random.randint(1, len(pai))  # inteiro aleatorio de 1 ate len(pai)-1, nunca corta nas pontas
        filho = pai.copy()           # comeca igual ao pai
        filho[corte:] = mae[corte:]  # do corte pra frente pega os genes da mae
        filhos.append(filho)

    return np.array(filhos)  # transforma a lista de filhos em uma matriz (qtd_filhos, qtd_genes)
