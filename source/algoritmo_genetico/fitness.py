"""Funções de fitness para classificação binária e ordinal."""

import numpy as np


def fitness_binario(
    cromossomos: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
    Fitness Prandiano para classificação binária (2 classes: 0 e 1).
    fitness = recall_classe_1 * recall_classe_0
    """
    total_positivos = np.sum(y == 1)
    total_negativos = np.sum(y == 0)

    lista_fitness = []
    for linha in cromossomos:
        bias = linha[0]
        genes = linha[1:]

        q = np.dot(X, genes) + bias
        hipotese = np.where(q >= 0, 1, 0)

        recall_pos = np.sum((hipotese == 1) & (y == 1)) / total_positivos
        recall_neg = np.sum((hipotese == 0) & (y == 0)) / total_negativos

        lista_fitness.append(recall_pos * recall_neg)

    return np.array(lista_fitness)


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