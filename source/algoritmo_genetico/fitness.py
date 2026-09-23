"""Funções de fitness para classificação binária e ordinal."""

import numpy as np

def fitness_ordinal(
    cromossomos: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """
    Fitness Prandiano generalizado para K classes ordinais.
    fitness = produto dos recalls de cada classe.

    thresholds: array com (K-1) pontos de corte.
    y: inteiros de 0 a K-1.
    """
    K = len(thresholds) + 1
    totais = np.array([np.sum(y == k) for k in range(K)])

    lista_fitness = []
    for linha in cromossomos:
        bias = linha[0]
        genes = linha[1:]

        q = np.dot(X, genes) + bias
        hipotese = np.digitize(q, thresholds)

        fitness = 1.0
        for k in range(K):
            acertos_k = np.sum((hipotese == k) & (y == k))
            fitness *= acertos_k / totais[k]

        lista_fitness.append(fitness)

    return np.array(lista_fitness)


def fitness_percentual(vetor_fitnesses: np.ndarray) -> np.ndarray:
    """Normaliza fitnesses em probabilidades (soma = 1)."""
    soma = np.sum(vetor_fitnesses)
    if soma == 0:
        return np.ones(len(vetor_fitnesses)) / len(vetor_fitnesses)
    return vetor_fitnesses / soma