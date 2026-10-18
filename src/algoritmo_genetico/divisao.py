"""Divisão dos dados em treino e teste."""

import numpy as np


def _qtd_treino(qtd_amostras: int, proporcao_treino: float) -> int:
    """
    Calcula quantas amostras vão para o treino, garantindo pelo menos 1 no
    treino e, se houver 2 ou mais amostras, pelo menos 1 no teste.

    Parâmetros:
        qtd_amostras -> quantas amostras dividir
        proporcao_treino -> fração que vai para o treino (entre 0 e 1)

    Retorna:
        quantidade (int) de amostras do treino
    """
    qtd = int(round(qtd_amostras * proporcao_treino))
    if qtd_amostras >= 2:
        return min(max(qtd, 1), qtd_amostras - 1)
    return qtd_amostras


def dividir_treino_teste(X: np.ndarray, y: np.ndarray, proporcao_treino: float = 0.8,
    estratificar: bool = True, seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Separa os dados em treino e teste, embaralhando antes de cortar.
    Com estratificar=True a divisão é feita dentro de cada classe, então
    todas as classes aparecem no treino na mesma proporção dos dados originais.

    Parâmetros:
        X -> array (amostras, features) com os dados
        y -> vetor com a classe de cada amostra
        proporcao_treino -> fração dos dados que vai para o treino (padrão 0.8)
        estratificar -> se True, mantém a proporção de cada classe nos dois conjuntos
        seed -> semente para reprodutibilidade (None = aleatório a cada rodada)

    Retorna:
        tupla (X_train, X_test, y_train, y_test)
    """

    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Os parâmetros X e y devem ser do tipo ndarray")

    if X.ndim != 2:
        raise ValueError(f"X deve ser matriz 2D (amostras, features), recebeu {X.ndim} dimensões")

    if y.ndim != 1:
        raise ValueError(f"y deve ser vetor 1D, recebeu {y.ndim} dimensões")

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X tem {X.shape[0]} amostras, mas y tem {y.shape[0]} valores")

    if X.shape[0] < 2:
        raise ValueError("são necessárias pelo menos 2 amostras para dividir em treino e teste")

    if not isinstance(proporcao_treino, (int, float)) or isinstance(proporcao_treino, bool):
        raise TypeError("proporcao_treino deve ser um número")

    if proporcao_treino <= 0 or proporcao_treino >= 1:
        raise ValueError(f"proporcao_treino deve estar entre 0 e 1 (sem incluir), recebeu {proporcao_treino}")

    if not isinstance(estratificar, bool):
        raise TypeError("estratificar deve ser True ou False")

    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise TypeError("seed deve ser um número inteiro ou None")

    # gerador próprio, não mexe no np.random global
    gerador = np.random.default_rng(seed)

    if estratificar:
        # grupos = lista com os índices de cada classe
        grupos = [np.where(y == classe)[0] for classe in np.unique(y)]
    else:
        grupos = [np.arange(len(y))]

    indices_treino = []
    indices_teste = []
    for grupo in grupos:
        embaralhado = gerador.permutation(grupo)
        corte = _qtd_treino(len(embaralhado), proporcao_treino)
        indices_treino.extend(embaralhado[:corte])
        indices_teste.extend(embaralhado[corte:])

    # embaralha de novo pra as classes não ficarem agrupadas em sequência
    indices_treino = gerador.permutation(np.array(indices_treino, dtype=int))
    indices_teste = gerador.permutation(np.array(indices_teste, dtype=int))

    return X[indices_treino], X[indices_teste], y[indices_treino], y[indices_teste]
