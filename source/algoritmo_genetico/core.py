"""Loop principal do algoritmo genético."""

from typing import Callable

import numpy as np

from .criar_crom import criar_cromossomos
from .atualizar_crom import atualizar_populacao
from .fitness import fitness_binario, fitness_ordinal, fitness_percentual
from .selecao import selecionar_pais_roleta
from .cruzamento import cruzar_um_ponto
from .mutacao import mutar


def rodar(X: np.ndarray, y: np.ndarray, qtd_cromossomos: int = 6, qtd_genes_mutados: int = 1,
    qtd_filhos: int = 3, geracoes: int = 100, fitness_alvo: float = 0.90, thresholds: np.ndarray | None = None,
    verbose: bool = True) -> dict:
    """
    Roda o algoritmo genetico completo: cria a populacao e repete
    fitness -> selecao -> cruzamento -> mutacao -> atualizacao ate atingir
    o fitness alvo ou acabar as geracoes.

    Parametros:
        X -> array (amostras, features) com os dados
        y -> vetor com a classe certa de cada amostra
        qtd_cromossomos -> tamanho da populacao
        qtd_genes_mutados -> quantos genes mudar em cada filho na mutacao
        qtd_filhos -> quantos filhos gerar por geracao
        geracoes -> numero maximo de geracoes
        fitness_alvo -> para quando o melhor fitness chegar nesse valor
        thresholds -> None para 2 classes (binario), ou array com K-1 pontos de corte para K classes (ordinal)
        verbose -> se True, mostra o melhor fitness de cada geracao

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

    return {"cromossomo": melhor_cromossomo, "fitness": melhor_fitness, "geracao": geracao_final, "thresholds": thresholds,}


def prever(modelo: dict, X: np.ndarray) -> np.ndarray:
    """
    Usa o melhor cromossomo encontrado por rodar() para classificar dados novos.

    Parametros:
        modelo -> dict retornado por rodar()
        X -> array (features,) com um registro ou (amostras, features) com varios

    Retorna:
        array de inteiros com a classe prevista de cada registro
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