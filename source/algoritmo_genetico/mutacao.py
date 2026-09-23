"""Operadores de mutação."""

import numpy as np


def mutar(filhos: np.ndarray, qtd_genes_mutados: int = 1) -> np.ndarray:
    """
    Muta `qtd_genes_mutados` genes de cada filho, escolhidos sem reposição.
    Cada gene mutado é resorteado uniformemente em [-1, 1].

    filhos: array (N, qtd_genes)
    Retorna array mutado (modifica in-place e retorna).
    """
    filhos_mutados = filhos.copy()
    for filho in filhos_mutados:
        k = min(qtd_genes_mutados, len(filho))
        indices = np.random.choice(len(filho), size=k, replace=False)
        filho[indices] = -1 + 2 * np.random.rand(k)

    return filhos_mutados