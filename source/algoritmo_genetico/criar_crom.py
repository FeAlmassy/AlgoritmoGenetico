# ordem: 1ª etapa (inicio) -> proxima: fitness.py
"""criacao da populacao de cromossomos, 1 funcao"""

import numpy as np

def criar_cromossomos(qtd_cromossomos: int = 6, qtd_genes: int = 19, minimo: float = -1.0, maximo: float = 1.0) -> np.ndarray:
    """
    Cria a populacao inicial: uma matriz onde cada linha e um cromossomo e
    cada coluna e um gene, com valores aleatorios entre minimo e maximo.

    Parametros:
        qtd_cromossomos -> quantos cromossomos (linhas) criar
        qtd_genes -> quantos genes cada cromossomo tem (features + 1 do bias)
        minimo -> menor valor que um gene pode ter (padrao -1)
        maximo -> maior valor que um gene pode ter (padrao +1)

    Retorna:
        array (qtd_cromossomos, qtd_genes) com a populacao
    """

    if not isinstance(qtd_cromossomos, int) or not isinstance(qtd_genes, int):
        raise TypeError (f"a funcao precisa de valores numericos")
    
    if qtd_cromossomos <= 0 or qtd_genes <= 0:
        raise ValueError (f"a funcao precisa de no minimo 1 cromossomo ou 1 gene")

    if not isinstance(minimo, (int, float)) or not isinstance(maximo, (int, float)):
        raise TypeError (f"min e max precisam ser valores numericos")  

    if minimo >= maximo:
        raise ValueError (f"o valor minimo {minimo} tem que ser menor que o valor maximo {maximo}")

    return np.random.uniform(minimo, maximo, size=(qtd_cromossomos, qtd_genes))