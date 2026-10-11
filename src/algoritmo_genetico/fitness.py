# ordem: 2ª etapa, vem depois de criacao.py (e de atualizacao.py nas próximas gerações)
# -> próxima: selecao.py
"""Cálculo do fitness para classificação binária e ordinal, e normalização em percentuais."""

import numpy as np


def fitness_binario(cromossomos: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calcula a nota (fitness) de cada cromossomo para classificação binária
    (classes 0 e 1). A nota é o recall da classe 1 vezes o recall da classe 0,
    então só é alta se o cromossomo acerta bem as duas classes.

    Parâmetros:
        cromossomos -> array (N, qtd_genes), o primeiro gene é o bias e o resto são os pesos
        X -> array (amostras, features) com os dados
        y -> vetor com a classe certa de cada amostra (0 ou 1)

    Retorna:
        array (N,) com o fitness de cada cromossomo, entre 0 e 1
    """

    if not isinstance(cromossomos, np.ndarray):
        raise TypeError("O parâmetro cromossomos deve ser do tipo ndarray")

    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Os parâmetros X e y devem ser do tipo ndarray")

    if cromossomos.shape[1] - 1 != X.shape[1]:
        raise ValueError(
            f"cromossomos tem {cromossomos.shape[1] - 1} pesos (sem bias), "
            f"mas X tem {X.shape[1]} features"
        )

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X tem {X.shape[0]} amostras, mas y tem {y.shape[0]}")

    valores_unicos = np.unique(y)
    if not np.all(np.isin(valores_unicos, [0, 1])):
        raise ValueError(f"y deve conter apenas 0 e 1, encontrou {valores_unicos}")

    total_positivos = np.sum(y == 1)
    total_negativos = np.sum(y == 0)

    # verifica se o vetor y não tem só 0 ou só 1
    if total_positivos == 0 or total_negativos == 0:
        raise ValueError("y deve ter pelo menos uma amostra de cada classe (0 e 1)")

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
    Calcula a nota (fitness) de cada cromossomo para K classes ordinais
    (ex: baixo, médio, alto). Os thresholds dividem o resultado em faixas,
    cada faixa é uma classe. A nota é o produto dos recalls de todas as classes.

    Parâmetros:
        cromossomos -> array (N, qtd_genes), o primeiro gene é o bias e o resto são os pesos
        X -> array (amostras, features) com os dados
        y -> vetor com a classe certa de cada amostra (inteiros de 0 a K-1)
        thresholds -> array com K-1 pontos de corte, em ordem crescente

    Retorna:
        array (N,) com o fitness de cada cromossomo, entre 0 e 1
    """

    if not isinstance(cromossomos, np.ndarray):
        raise TypeError("O parâmetro cromossomos deve ser do tipo ndarray")

    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Os parâmetros X e y devem ser do tipo ndarray")

    if not isinstance(thresholds, np.ndarray):
        raise TypeError("O parâmetro thresholds deve ser do tipo ndarray")

    if cromossomos.shape[1] - 1 != X.shape[1]:
        raise ValueError(
            f"cromossomos tem {cromossomos.shape[1] - 1} pesos (sem bias), "
            f"mas X tem {X.shape[1]} features"
        )

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X tem {X.shape[0]} amostras, mas y tem {y.shape[0]}")

    if len(thresholds) == 0:
        raise ValueError("thresholds não pode ser vazio (precisa de pelo menos 1 ponto de corte)")

    if not np.all(np.diff(thresholds) > 0):
        raise ValueError(f"thresholds devem estar em ordem crescente, recebeu {thresholds}")

    K = len(thresholds) + 1

    valores_unicos = np.unique(y)
    if valores_unicos.min() < 0 or valores_unicos.max() >= K:
        raise ValueError(f"y deve conter inteiros entre 0 e {K - 1}, encontrou {valores_unicos}")

    totais = np.array([np.sum(y == k) for k in range(K)])

    if np.any(totais == 0):
        classes_vazias = np.where(totais == 0)[0].tolist()
        raise ValueError(
            f"y deve ter pelo menos uma amostra de cada classe. "
            f"Classes sem amostras: {classes_vazias}"
        )

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
    """
    Transforma os fitnesses em porcentagens que somam 1, para usar na roleta.
    Se todos os fitnesses forem 0, todos recebem a mesma chance.

    Parâmetros:
        vetor_fitnesses -> array (N,) com o fitness de cada cromossomo

    Retorna:
        array (N,) com a chance de cada cromossomo ser escolhido
    """
    soma = np.sum(vetor_fitnesses)
    if soma == 0:
        return np.ones(len(vetor_fitnesses)) / len(vetor_fitnesses)
    return vetor_fitnesses / soma
