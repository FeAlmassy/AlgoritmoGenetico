"""Loop principal do algoritmo genético."""

from typing import Callable

import numpy as np

from .populacao import criar_cromossomos, atualizar_populacao
from .fitness import fitness_binario, fitness_ordinal, fitness_percentual
from .selecao import selecionar_pais_roleta
from .cruzamento import cruzar_um_ponto
from .mutacao import mutar


def rodar(
    X: np.ndarray,
    y: np.ndarray,
    qtd_cromossomos: int = 6,
    qtd_genes_mutados: int = 1,
    qtd_filhos: int = 3,
    geracoes: int = 100,
    fitness_alvo: float = 0.90,
    thresholds: np.ndarray | None = None,
    verbose: bool = True,
) -> dict:
    """
    Executa o algoritmo genético para classificação.

    Se `thresholds` for None, usa fitness binário (2 classes).
    Se `thresholds` for um array, usa fitness ordinal (K classes).

    Retorna dict com:
        - "cromossomo": melhor cromossomo encontrado
        - "fitness": fitness do melhor cromossomo
        - "geracao": geração em que parou
        - "thresholds": thresholds usados (None se binário)
    """
    qtd_genes = X.shape[1] + 1  # features + bias

    # escolhe a função de fitness
    if thresholds is None:
        def calcular(crom):
            return fitness_binario(crom, X, y)
    else:
        thresholds = np.asarray(thresholds)
        def calcular(crom):
            return fitness_ordinal(crom, X, y, thresholds)

    populacao = criar_cromossomos(qtd_cromossomos, qtd_genes)

    melhor_cromossomo = None
    melhor_fitness = 0.0
    geracao_final = 0

    for geracao in range(geracoes):
        fitnesses = calcular(populacao)
        percentuais = fitness_percentual(fitnesses)

        idx_melhor = int(np.argmax(fitnesses))
        fitness_atual = float(fitnesses[idx_melhor])

        if fitness_atual > melhor_fitness:
            melhor_fitness = fitness_atual
            melhor_cromossomo = populacao[idx_melhor].copy()

        geracao_final = geracao + 1

        if verbose:
            print(f"Geração {geracao_final:>4} | melhor fitness: {melhor_fitness:.4f}")

        if melhor_fitness >= fitness_alvo:
            if verbose:
                print(f"\nFitness alvo {fitness_alvo} atingido na geração {geracao_final}.")
            break

        pai, mae = selecionar_pais_roleta(populacao, percentuais)
        filhos = cruzar_um_ponto(pai, mae, qtd_filhos)
        filhos = mutar(filhos, qtd_genes_mutados)
        fitnesses_filhos = calcular(filhos)
        populacao = atualizar_populacao(
            populacao, fitnesses, filhos, fitnesses_filhos,
        )

    return {
        "cromossomo": melhor_cromossomo,
        "fitness": melhor_fitness,
        "geracao": geracao_final,
        "thresholds": thresholds,
    }


def prever(modelo: dict, X: np.ndarray) -> np.ndarray:
    """
    Usa o resultado de `rodar()` para classificar novos dados.

    X pode ser 1D (um registro) ou 2D (vários registros).
    Retorna array de inteiros com as classes previstas.
    """
    cromossomo = modelo["cromossomo"]
    thresholds = modelo["thresholds"]

    bias = cromossomo[0]
    genes = cromossomo[1:]

    if X.ndim == 1:
        X = X.reshape(1, -1)

    q = X @ genes + bias

    if thresholds is None:
        return np.where(q >= 0, 1, 0)
    else:
        return np.digitize(q, thresholds)