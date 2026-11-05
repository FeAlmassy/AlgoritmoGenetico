"""Avaliação do modelo depois do treino."""

import numpy as np


def _classes_para_inteiros(y: np.ndarray, nome: str) -> np.ndarray:
    """
    Devolve as classes como inteiros. Aceita float com valor inteiro (ex: 1.0),
    que é como uma coluna de classes pode vir de uma planilha.

    Parâmetros:
        y -> vetor com as classes
        nome -> nome do parâmetro, para a mensagem de erro

    Retorna:
        vetor de inteiros
    """
    if np.issubdtype(y.dtype, np.integer):
        return y

    if np.issubdtype(y.dtype, np.floating) and np.isfinite(y).all() and np.all(y == np.floor(y)):
        return y.astype(int)

    raise TypeError(f"{nome} deve conter inteiros de 0 a K-1 (converta classes em texto antes)")


def matriz_confusao(y_real: np.ndarray, y_previsto: np.ndarray,
    nomes_classes: list[str] | None = None, verbose: bool = True) -> np.ndarray:
    """
    Monta a matriz de confusão: cada linha é a classe real, cada coluna é a
    classe prevista e a diagonal são os acertos.

    Parâmetros:
        y_real -> vetor com a classe certa de cada amostra (inteiros de 0 a K-1)
        y_previsto -> vetor com a classe prevista de cada amostra (saída de prever)
        nomes_classes -> nome de cada classe, na ordem de 0 a K-1 (None = usa os números).
                         Também define K, então garante a matriz completa mesmo se
                         alguma classe não aparecer nos dados
        verbose -> se True, mostra a matriz na tela

    Retorna:
        array (K, K) de inteiros com a matriz de confusão
    """

    if not isinstance(y_real, np.ndarray) or not isinstance(y_previsto, np.ndarray):
        raise TypeError("Os parâmetros y_real e y_previsto devem ser do tipo ndarray")

    if y_real.ndim != 1 or y_previsto.ndim != 1:
        raise ValueError(f"y_real e y_previsto devem ser vetores 1D, recebeu {y_real.ndim} e {y_previsto.ndim} dimensões")

    if len(y_real) != len(y_previsto):
        raise ValueError(f"y_real tem {len(y_real)} valores, mas y_previsto tem {len(y_previsto)}")

    if len(y_real) == 0:
        raise ValueError("y_real e y_previsto não podem ser vazios")

    y_real = _classes_para_inteiros(y_real, "y_real")
    y_previsto = _classes_para_inteiros(y_previsto, "y_previsto")

    if nomes_classes is not None and not isinstance(nomes_classes, list):
        raise TypeError("nomes_classes deve ser uma lista ou None")

    if nomes_classes is not None and len(nomes_classes) == 0:
        raise ValueError("nomes_classes não pode ser vazia")

    if not isinstance(verbose, bool):
        raise TypeError("verbose deve ser True ou False")

    menor_classe = min(y_real.min(), y_previsto.min())
    maior_classe = max(y_real.max(), y_previsto.max())

    if menor_classe < 0:
        raise ValueError(f"as classes devem ser inteiros a partir de 0, encontrou {menor_classe}")

    if nomes_classes is None:
        K = int(maior_classe) + 1
    else:
        K = len(nomes_classes)
        if maior_classe >= K:
            raise ValueError(f"nomes_classes tem {K} nomes, mas os dados têm a classe {maior_classe}")

    matriz = np.zeros((K, K), dtype=int)
    for real, previsto in zip(y_real, y_previsto):
        matriz[real, previsto] += 1

    if verbose:
        _mostrar_matriz(matriz, _nomes_para_mostrar(K, nomes_classes))

    return matriz


def avaliar(y_real: np.ndarray, y_previsto: np.ndarray,
    nomes_classes: list[str] | None = None, verbose: bool = True) -> dict:
    """
    Compara as previsões com as classes reais: calcula a acurácia, o fitness,
    o recall de cada classe e a matriz de confusão.

    Parâmetros:
        y_real -> vetor com a classe certa de cada amostra (inteiros de 0 a K-1)
        y_previsto -> vetor com a classe prevista de cada amostra (saída de prever)
        nomes_classes -> nome de cada classe, na ordem de 0 a K-1 (None = usa os números).
                         Também define K, então garante o resultado completo mesmo se
                         alguma classe não aparecer nos dados
        verbose -> se True, mostra o resultado na tela

    Retorna:
        dict com:
            "acuracia" -> fração das amostras que o modelo acertou
            "fitness" -> produto dos recalls (nan se alguma classe não tiver amostras)
            "recall" -> array com o recall de cada classe (nan se a classe não tiver amostras)
            "matriz" -> matriz de confusão, array (K, K) de inteiros
    """

    if not isinstance(verbose, bool):
        raise TypeError("verbose deve ser True ou False")

    # as outras validações já são feitas pela matriz_confusao
    matriz = matriz_confusao(y_real, y_previsto, nomes_classes, verbose=False)

    acertos = np.diag(matriz)
    totais = matriz.sum(axis=1)

    # classe sem amostras fica com nan: o recall dela não pode ser medido
    recall = np.full(len(matriz), np.nan)
    tem_amostras = totais > 0
    recall[tem_amostras] = acertos[tem_amostras] / totais[tem_amostras]

    acuracia = float(acertos.sum() / matriz.sum())
    fitness = float(np.prod(recall))

    if verbose:
        nomes = _nomes_para_mostrar(len(matriz), nomes_classes)

        print(f"acurácia: {acuracia:.2%}")
        if np.isnan(fitness):
            print("fitness: não calculado (há classe sem amostras)")
        else:
            print(f"fitness: {fitness:.4f}")

        print("\nrecall por classe:")
        for nome, acerto, total, valor in zip(nomes, acertos, totais, recall):
            if total == 0:
                print(f"  {nome}: sem amostras")
            else:
                print(f"  {nome}: {acerto}/{total} = {valor:.2%}")

        print("\nmatriz de confusão:")
        _mostrar_matriz(matriz, nomes)

    return {
        "acuracia": acuracia,
        "fitness": fitness,
        "recall": recall,
        "matriz": matriz,
    }


def _nomes_para_mostrar(K: int, nomes_classes: list[str] | None) -> list[str]:
    """
    Devolve o texto que representa cada classe na tela.

    Parâmetros:
        K -> quantidade de classes
        nomes_classes -> nome de cada classe, ou None para usar os números

    Retorna:
        lista com K textos
    """
    if nomes_classes is None:
        return [str(k) for k in range(K)]
    return [str(nome) for nome in nomes_classes]


def _mostrar_matriz(matriz: np.ndarray, nomes: list[str]) -> None:
    """
    Mostra a matriz de confusão na tela, com as colunas alinhadas.

    Parâmetros:
        matriz -> array (K, K) com a matriz de confusão
        nomes -> texto que representa cada classe
    """
    titulo = "real \\ previu"
    largura_nome = max(len(titulo), *(len(nome) for nome in nomes))
    largura_coluna = max(*(len(nome) for nome in nomes), len(str(matriz.max()))) + 2

    print(f"{titulo:>{largura_nome}}" + "".join(f"{nome:>{largura_coluna}}" for nome in nomes))
    for nome, linha in zip(nomes, matriz):
        print(f"{nome:>{largura_nome}}" + "".join(f"{valor:>{largura_coluna}}" for valor in linha))
