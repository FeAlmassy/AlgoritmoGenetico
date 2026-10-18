# ordem: 5ª etapa, vem depois de cruzamento.py -> próxima: atualizacao.py
"""Mutação dos filhos."""

import numpy as np


def mutar(filhos: np.ndarray, qtd_genes_mutados: int = 1) -> np.ndarray:
    """
    Sorteia alguns genes de cada filho e troca por um valor novo aleatório
    entre -1 e 1. Não altera o array original, trabalha numa cópia.

    Parâmetros:
        filhos -> array com os filhos do cruzamento
        qtd_genes_mutados -> quantos genes trocar em cada filho

    Retorna:
        array com os filhos mutados
    """

    if not isinstance(filhos, np.ndarray):
        raise TypeError("O parâmetro filhos deve ser do tipo ndarray")

    if len(filhos) == 0:
        raise ValueError("filhos não pode ser vazio")

    if not isinstance(qtd_genes_mutados, int):
        raise TypeError("qtd_genes_mutados deve ser um número inteiro")

    qtd_genes = len(filhos[0])
    if qtd_genes_mutados < 0 or qtd_genes_mutados > qtd_genes:
        raise ValueError(f"qtd_genes_mutados deve estar entre 0 e {qtd_genes} (qtd de genes), recebeu {qtd_genes_mutados}")

    # .astype(float) faz uma cópia convertida pra float; se viesse int, o valor
    # sorteado (ex: 0.37) seria cortado pra 0 na hora de guardar
    filhos_mutados = filhos.astype(float)

    for filho in filhos_mutados:
        # sorteia qtd_genes_mutados posições diferentes entre 0 e qtd_genes-1
        indices = np.random.choice(qtd_genes, size=qtd_genes_mutados, replace=False)
        for indice in indices:
            filho[indice] = -1 + 2 * np.random.rand()

    return filhos_mutados
