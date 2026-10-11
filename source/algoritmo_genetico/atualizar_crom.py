"""Criação e atualização da população de cromossomos."""

import numpy as np

def atualizar_populacao(cromossomos: np.ndarray, vetor_fitnesses: np.ndarray, filhos: np.ndarray, fitnesses_filhos: np.ndarray, qtd_substituicoes: int = 2) -> np.ndarray:
    """
    Steady-state com elitismo: substitui os N piores da população
    pelos N melhores filhos.
    """
    indices_melhores_filhos = np.argsort(fitnesses_filhos)[-qtd_substituicoes:]
    indices_piores = np.argsort(vetor_fitnesses)[:qtd_substituicoes]

    nova_populacao = cromossomos.copy()
    for i in range(qtd_substituicoes):
        idx_pior = int(indices_piores[i])
        idx_filho = int(indices_melhores_filhos[i])
        nova_populacao[idx_pior] = filhos[idx_filho]

    return nova_populacao






