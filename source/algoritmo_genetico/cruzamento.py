"""Operadores de cruzamento (crossover)."""

import numpy as np


def cruzar_um_ponto(
    pai: np.ndarray,
    mae: np.ndarray,
    qtd_filhos: int = 3,
) -> np.ndarray:
    """
    Crossover de 1 ponto, gerando `qtd_filhos` filhos com pontos
    de corte independentes.

    Retorna array (qtd_filhos, qtd_genes).
    """
    filhos = []
    for _ in range(qtd_filhos):
        corte = np.random.randint(1, len(pai))
        filho = np.concatenate([pai[:corte], mae[corte:]])
        filhos.append(filho)

    return np.array(filhos)