# ordem: 5ª etapa, vem depois de cruzamento.py -> próxima: atualizacao.py
"""Mutação dos filhos."""

import numpy as np


def mutar(filhos: np.ndarray, qtd_genes_mutados: int = 1) -> np.ndarray:
    """
    Sorteia alguns genes de cada filho e troca por um valor novo aleatório
    entre -1 e 1. Não altera o array original, trabalha numa cópia.

    Parâmetros:
        filhos -> array (N, qtd_genes) com os filhos do cruzamento
        qtd_genes_mutados -> quantos genes trocar em cada filho (sem repetir gene)

    Retorna:
        array (N, qtd_genes) com os filhos mutados
    """

    if not isinstance(filhos, np.ndarray):
        raise TypeError("O parâmetro filhos deve ser do tipo ndarray")

    # .ndim é o número de dimensões do array: 2 = matriz (cada linha um filho)
    if filhos.ndim != 2:
        raise ValueError(
            f"filhos deve ser uma matriz (N, qtd_genes), recebeu {filhos.ndim} dimensões"
        )

    if len(filhos) == 0:
        raise ValueError("filhos não pode ser vazio")

    # bool é um tipo de int no Python (True == 1), então precisa barrar separado
    if not isinstance(qtd_genes_mutados, int) or isinstance(qtd_genes_mutados, bool):
        raise TypeError("qtd_genes_mutados deve ser um número inteiro")

    qtd_genes = len(filhos[0])
    if qtd_genes_mutados < 0 or qtd_genes_mutados > qtd_genes:
        raise ValueError(
            f"qtd_genes_mutados deve estar entre 0 e {qtd_genes} (qtd de genes), "
            f"recebeu {qtd_genes_mutados}"
        )

    # .astype(float) faz uma cópia convertida pra float; se viesse int, o valor
    # sorteado (ex: 0.37) seria cortado pra 0 na hora de guardar
    filhos_mutados = filhos.astype(float)

    for filho in filhos_mutados:
        # sorteia qtd_genes_mutados posições diferentes entre 0 e qtd_genes-1
        # (replace=False = sem repetir)
        indices = np.random.choice(qtd_genes, size=qtd_genes_mutados, replace=False)
        for indice in indices:
            # rand() dá entre 0 e 1, a conta leva pra entre -1 e 1
            filho[indice] = -1 + 2 * np.random.rand()

    return filhos_mutados
