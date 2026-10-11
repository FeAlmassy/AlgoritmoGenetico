# ordem: 6ª etapa, vem depois de mutacao.py -> proxima: fitness.py (nova geracao)
"""Criação e atualização da população de cromossomos"""

import math

import numpy as np

def atualizar_populacao(cromossomos: np.ndarray, vetor_fitnesses: np.ndarray, filhos: np.ndarray, fitnesses_filhos: np.ndarray, qtd_substituicoes: int = 2) -> np.ndarray:
    """
    Monta a nova geracao: tira os piores cromossomos da populacao e coloca
    os melhores filhos no lugar. O resto da populacao continua igual.

    Parametros:
        cromossomos -> array (N, qtd_genes) com a populacao atual
        vetor_fitnesses -> array (N,) com o fitness de cada cromossomo da populacao
        filhos -> array (M, qtd_genes) com os filhos mutados
        fitnesses_filhos -> array (M,) com o fitness de cada filho
        qtd_substituicoes -> quantos cromossomos trocar por filhos (no maximo M)

    Retorna:
        array (N, qtd_genes) com a nova populacao
    """

    for nome, valor in [("cromossomos", cromossomos), ("vetor_fitnesses", vetor_fitnesses),
                        ("filhos", filhos), ("fitnesses_filhos", fitnesses_filhos)]:
        if not isinstance(valor, np.ndarray):
            raise TypeError(f"O parametro {nome} deve ser do tipo ndarray")

    # .ndim e o numero de dimensoes do array: 2 = matriz, 1 = lista simples
    if cromossomos.ndim != 2 or filhos.ndim != 2:
        raise ValueError(
            f"cromossomos e filhos devem ser matrizes (linhas, qtd_genes), "
            f"recebeu {cromossomos.ndim} e {filhos.ndim} dimensoes"
        )

    if vetor_fitnesses.ndim != 1 or fitnesses_filhos.ndim != 1:
        raise ValueError(
            f"vetor_fitnesses e fitnesses_filhos devem ser vetores, "
            f"recebeu {vetor_fitnesses.ndim} e {fitnesses_filhos.ndim} dimensoes"
        )

    if len(cromossomos) != len(vetor_fitnesses):
        raise ValueError(f"cromossomos tem {len(cromossomos)} linhas, mas vetor_fitnesses tem {len(vetor_fitnesses)} valores")

    if len(filhos) != len(fitnesses_filhos):
        raise ValueError(f"filhos tem {len(filhos)} linhas, mas fitnesses_filhos tem {len(fitnesses_filhos)} valores")

    # .shape[1] e o numero de colunas da matriz, ou seja, a qtd de genes
    if cromossomos.shape[1] != filhos.shape[1]:
        raise ValueError(f"cromossomos tem {cromossomos.shape[1]} genes, mas filhos tem {filhos.shape[1]}")

    for fitness in list(vetor_fitnesses) + list(fitnesses_filhos):
        if math.isnan(fitness):
            raise ValueError("os fitnesses nao podem ter valores NaN")

    # bool e um tipo de int no python (True == 1), entao precisa barrar separado
    if not isinstance(qtd_substituicoes, int) or isinstance(qtd_substituicoes, bool):
        raise TypeError("qtd_substituicoes deve ser um numero inteiro")

    limite = min(len(cromossomos), len(filhos))
    if qtd_substituicoes < 0 or qtd_substituicoes > limite:
        raise ValueError(
            f"qtd_substituicoes deve estar entre 0 e {limite} "
            f"(menor valor entre populacao e filhos), recebeu {qtd_substituicoes}"
        )

    # lista de indices ordenada pelo fitness, do menor pro maior
    ordem_populacao = sorted(range(len(vetor_fitnesses)), key=lambda i: vetor_fitnesses[i])
    ordem_filhos = sorted(range(len(fitnesses_filhos)), key=lambda i: fitnesses_filhos[i])

    indices_piores = ordem_populacao[:qtd_substituicoes]                          # os primeiros sao os piores
    indices_melhores_filhos = ordem_filhos[len(ordem_filhos) - qtd_substituicoes:]  # os ultimos sao os melhores

    nova_populacao = cromossomos.copy()
    for i in range(qtd_substituicoes):
        nova_populacao[indices_piores[i]] = filhos[indices_melhores_filhos[i]]

    return nova_populacao
