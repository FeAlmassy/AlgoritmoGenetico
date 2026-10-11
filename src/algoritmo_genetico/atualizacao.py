# ordem: 6ª etapa, vem depois de mutacao.py -> próxima: fitness.py (nova geração)
"""Atualização da população com os melhores filhos."""

import math

import numpy as np


def atualizar_populacao(
    cromossomos: np.ndarray,
    vetor_fitnesses: np.ndarray,
    filhos: np.ndarray,
    fitnesses_filhos: np.ndarray,
    qtd_substituicoes: int = 2,
) -> np.ndarray:
    
    """
    Monta a nova geração: tira os piores cromossomos da população e coloca
    os melhores filhos no lugar. O resto da população continua igual.

    Parâmetros:
        cromossomos -> array (N, qtd_genes) com a população atual
        vetor_fitnesses -> array (N,) com o fitness de cada cromossomo da população
        filhos -> array (M, qtd_genes) com os filhos mutados
        fitnesses_filhos -> array (M,) com o fitness de cada filho
        qtd_substituicoes -> quantos cromossomos trocar por filhos (no máximo M)

    Retorna:
        array (N, qtd_genes) com a nova população
    """

    parametros = [
        ("cromossomos", cromossomos),
        ("vetor_fitnesses", vetor_fitnesses),
        ("filhos", filhos),
        ("fitnesses_filhos", fitnesses_filhos),
    ]
    for nome, valor in parametros:
        if not isinstance(valor, np.ndarray):
            raise TypeError(f"O parâmetro {nome} deve ser do tipo ndarray")

    if cromossomos.ndim != 2 or filhos.ndim != 2:
        raise ValueError(
            f"cromossomos e filhos devem ser matrizes (linhas, qtd_genes), "
            f"recebeu {cromossomos.ndim} e {filhos.ndim} dimensões"
        )

    if vetor_fitnesses.ndim != 1 or fitnesses_filhos.ndim != 1:
        raise ValueError(
            f"vetor_fitnesses e fitnesses_filhos devem ser vetores, "
            f"recebeu {vetor_fitnesses.ndim} e {fitnesses_filhos.ndim} dimensões"
        )

    if len(cromossomos) != len(vetor_fitnesses):
        raise ValueError(
            f"cromossomos tem {len(cromossomos)} linhas, "
            f"mas vetor_fitnesses tem {len(vetor_fitnesses)} valores"
        )

    if len(filhos) != len(fitnesses_filhos):
        raise ValueError(
            f"filhos tem {len(filhos)} linhas, "
            f"mas fitnesses_filhos tem {len(fitnesses_filhos)} valores"
        )

    if cromossomos.shape[1] != filhos.shape[1]:
        raise ValueError(
            f"cromossomos tem {cromossomos.shape[1]} genes, mas filhos tem {filhos.shape[1]}"
        )

    for fitness in list(vetor_fitnesses) + list(fitnesses_filhos):
        if math.isnan(fitness):
            raise ValueError("os fitnesses não podem ter valores NaN")

    if not isinstance(qtd_substituicoes, int) or isinstance(qtd_substituicoes, bool):
        raise TypeError("qtd_substituicoes deve ser um número inteiro")

    limite = min(len(cromossomos), len(filhos))
    if qtd_substituicoes < 0 or qtd_substituicoes > limite:
        raise ValueError(
            f"qtd_substituicoes deve estar entre 0 e {limite} "
            f"(menor valor entre população e filhos), recebeu {qtd_substituicoes}"
        )

    # lista de índices ordenada pelo fitness, do menor pro maior
    ordem_populacao = sorted(range(len(vetor_fitnesses)), key=lambda i: vetor_fitnesses[i])
    ordem_filhos = sorted(range(len(fitnesses_filhos)), key=lambda i: fitnesses_filhos[i])

    # os primeiros da população são os piores, os últimos dos filhos são os melhores
    indices_piores = ordem_populacao[:qtd_substituicoes]
    indices_melhores_filhos = ordem_filhos[len(ordem_filhos) - qtd_substituicoes:]

    nova_populacao = cromossomos.copy()
    for i in range(qtd_substituicoes):
        nova_populacao[indices_piores[i]] = filhos[indices_melhores_filhos[i]]

    return nova_populacao
