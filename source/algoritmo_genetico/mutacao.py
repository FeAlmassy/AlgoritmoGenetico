# ordem: 5ª etapa, vem depois de cruzamento.py -> proxima: atualizar_crom.py
"""Operadores de mutação."""

import numpy as np


def mutar(filhos: np.ndarray, qtd_genes_mutados: int = 1) -> np.ndarray:
    """
    Sorteia alguns genes de cada filho e troca por um valor novo aleatorio
    entre -1 e 1. Nao altera o array original, trabalha numa copia.

    Parametros:
        filhos -> array (N, qtd_genes) com os filhos do cruzamento
        qtd_genes_mutados -> quantos genes trocar em cada filho (sem repetir gene)

    Retorna:
        array (N, qtd_genes) com os filhos mutados
    """

    if not isinstance(filhos, np.ndarray):
        raise TypeError("O parametro filhos deve ser do tipo ndarray")

    # .ndim e o numero de dimensoes do array: 2 = matriz (cada linha um filho)
    if filhos.ndim != 2:
        raise ValueError(f"filhos deve ser uma matriz (N, qtd_genes), recebeu {filhos.ndim} dimensoes")

    if len(filhos) == 0:
        raise ValueError("filhos nao pode ser vazio")

    # bool e um tipo de int no python (True == 1), entao precisa barrar separado
    if not isinstance(qtd_genes_mutados, int) or isinstance(qtd_genes_mutados, bool):
        raise TypeError("qtd_genes_mutados deve ser um numero inteiro")

    qtd_genes = len(filhos[0])
    if qtd_genes_mutados < 0 or qtd_genes_mutados > qtd_genes:
        raise ValueError(f"qtd_genes_mutados deve estar entre 0 e {qtd_genes} (qtd de genes), recebeu {qtd_genes_mutados}")

    # .astype(float) faz uma copia convertida pra float; se viesse int, o valor
    # sorteado (ex: 0.37) seria cortado pra 0 na hora de guardar
    filhos_mutados = filhos.astype(float)

    for filho in filhos_mutados:
        # sorteia qtd_genes_mutados posicoes diferentes entre 0 e qtd_genes-1 (replace=False = sem repetir)
        indices = np.random.choice(qtd_genes, size=qtd_genes_mutados, replace=False)
        for indice in indices:
            filho[indice] = -1 + 2 * np.random.rand()  # rand() da entre 0 e 1, a conta leva pra entre -1 e 1

    return filhos_mutados
