# ordem: 1ª etapa (início) -> próxima: fitness.py
"""Criação da população inicial de cromossomos."""

import numpy as np


def criar_cromossomos(qtd_cromossomos: int = 6, qtd_genes: int = 19,
    minimo: float = -1.0, maximo: float = 1.0,) -> np.ndarray:
    """
    Cria a população inicial: uma matriz onde cada linha é um cromossomo e
    cada coluna é um gene, com valores aleatórios entre mínimo e máximo.

    Parâmetros:
        qtd_cromossomos -> quantos cromossomos criar
        qtd_genes -> quantos genes cada cromossomo tem (features + 1 do bias)
        minimo -> menor valor que um gene pode ter (padrão -1)
        maximo -> maior valor que um gene pode ter (padrão +1)

    Retorna:
        array com a população de cromossomos criada
    """

    if not isinstance(qtd_cromossomos, int) or not isinstance(qtd_genes, int):
        raise TypeError("qtd_cromossomos e qtd_genes devem ser números inteiros")

    if qtd_cromossomos <= 0 or qtd_genes <= 0:
        raise ValueError("a função precisa de no mínimo 1 cromossomo e 1 gene")

    if not isinstance(minimo, (int, float)) or not isinstance(maximo, (int, float)):
        raise TypeError("minimo e maximo devem ser valores numéricos")

    if minimo >= maximo:
        raise ValueError(f"o valor mínimo {minimo} tem que ser menor que o valor máximo {maximo}")

    return np.random.uniform(minimo, maximo, size=(qtd_cromossomos, qtd_genes))
