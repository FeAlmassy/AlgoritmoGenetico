"""Funções para calcular o fitness para classificação binária e ordinal, 2 funcao"""

import numpy as np

def fitness_binario(cromossomos: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fitness para classificação binária (2 classes: 0 e 1)
    
    """

    if not isinstance(cromossomos, np.ndarray):
        raise TypeError("O parametro cromossomos deve ser do tipo ndarray")

    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Os parametros X e y devem ser do tipo ndarray")

    if cromossomos.shape[1] - 1 != X.shape[1]:
        raise ValueError(
            f"cromossomos tem {cromossomos.shape[1] - 1} pesos (sem bias), "
            f"mas X tem {X.shape[1]} features"
        )

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X tem {X.shape[0]} amostras, mas y tem {y.shape[0]}"
        )

    valores_unicos = np.unique(y)
    if not np.all(np.isin(valores_unicos, [0, 1])):
        raise ValueError(f"y deve conter apenas 0 e 1, encontrou {valores_unicos}")

    total_positivos = np.sum(y == 1)
    total_negativos = np.sum(y == 0)

    if total_positivos == 0 or total_negativos == 0: #verifica se o vetor y nao tem somente 0 ou 1
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





def fitness_ordinal(cromossomos: np.ndarray, X: np.ndarray, y: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Fitness Prandiano generalizado para K classes ordinais.
    fitness = produto dos recalls de cada classe.

    cromossomos -> array criado anteriormente com os genes
    X -> array dos dados (bias + features)
    y -> vetor das classes (inteiros de 0 a K-1)
    thresholds -> array com (K-1) pontos de corte, em ordem crescente
    """

    # --- Tipo ---
    if not isinstance(cromossomos, np.ndarray):
        raise TypeError("O parametro cromossomos deve ser do tipo ndarray")

    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Os parametros X e y devem ser do tipo ndarray")

    if not isinstance(thresholds, np.ndarray):
        raise TypeError("O parametro thresholds deve ser do tipo ndarray")

    # --- Shape ---
    if cromossomos.shape[1] - 1 != X.shape[1]:
        raise ValueError(
            f"cromossomos tem {cromossomos.shape[1] - 1} pesos (sem bias), "
            f"mas X tem {X.shape[1]} features"
        )

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X tem {X.shape[0]} amostras, mas y tem {y.shape[0]}"
        )

    if len(thresholds) == 0:
        raise ValueError("thresholds nao pode ser vazio (precisa de pelo menos 1 ponto de corte)")

    # --- Thresholds em ordem crescente (requisito do np.digitize) ---
    if not np.all(np.diff(thresholds) > 0):
        raise ValueError(f"thresholds devem estar em ordem crescente, recebeu {thresholds}")

    K = len(thresholds) + 1

    # --- Conteudo de y: inteiros entre 0 e K-1 ---
    valores_unicos = np.unique(y)
    if valores_unicos.min() < 0 or valores_unicos.max() >= K:
        raise ValueError(
            f"y deve conter inteiros entre 0 e {K - 1}, encontrou {valores_unicos}"
        )

    totais = np.array([np.sum(y == k) for k in range(K)])

    # --- Pelo menos uma amostra por classe (evita divisao por zero) ---
    if np.any(totais == 0):
        classes_vazias = np.where(totais == 0)[0].tolist()
        raise ValueError(
            f"y deve ter pelo menos uma amostra de cada classe. "
            f"Classes sem amostras: {classes_vazias}"
        )

    # --- Logica original ---
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