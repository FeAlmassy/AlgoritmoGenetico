"""Métricas para avaliar as previsões do modelo (mesmo padrão do scikit-learn)."""

import numpy as np


def _validar_vetores(y_real: np.ndarray, y_previsto: np.ndarray) -> None:
    """
    Confere que y_real e y_previsto podem ser comparados posição a posição.

    Parâmetros:
        y_real -> vetor com as classes verdadeiras
        y_previsto -> vetor com as classes previstas pelo modelo
    """

    if not isinstance(y_real, np.ndarray) or not isinstance(y_previsto, np.ndarray):
        raise TypeError("Os parâmetros y_real e y_previsto devem ser do tipo ndarray")

    if y_real.ndim != 1 or y_previsto.ndim != 1:
        raise ValueError(
            f"y_real e y_previsto devem ser vetores 1D, receberam {y_real.ndim}D e {y_previsto.ndim}D"
        )

    if y_real.shape[0] != y_previsto.shape[0]:
        raise ValueError(
            f"y_real tem {y_real.shape[0]} valores, mas y_previsto tem {y_previsto.shape[0]}"
        )

    if y_real.shape[0] == 0:
        raise ValueError("y_real e y_previsto estão vazios")


def acuracia(y_real: np.ndarray, y_previsto: np.ndarray) -> float:
    """
    Fração de previsões corretas (equivalente a accuracy_score do scikit-learn).

    Parâmetros:
        y_real -> vetor com as classes verdadeiras (ex.: y_test)
        y_previsto -> vetor com as classes previstas por prever()

    Retorna:
        float entre 0 e 1 (acertos / total)
    """

    _validar_vetores(y_real, y_previsto)

    return float(np.mean(y_real == y_previsto))


def matriz_confusao(y_real: np.ndarray, y_previsto: np.ndarray,
    classes: np.ndarray | None = None,
) -> np.ndarray:
    """
    Conta acertos e erros por classe (equivalente a confusion_matrix do scikit-learn).
    A linha i é a classe real e a coluna j é a classe prevista, então a célula
    [i, j] diz quantas amostras da classe i o modelo classificou como j.
    A diagonal são os acertos.

    Parâmetros:
        y_real -> vetor com as classes verdadeiras
        y_previsto -> vetor com as classes previstas por prever()
        classes -> ordem das classes nas linhas/colunas (None = todas que aparecem, em ordem crescente)

    Retorna:
        array (k, k) de inteiros, com k = número de classes
    """

    _validar_vetores(y_real, y_previsto)

    if classes is None:
        classes = np.unique(np.concatenate([y_real, y_previsto]))
    elif not isinstance(classes, np.ndarray):
        raise TypeError("O parâmetro classes deve ser do tipo ndarray ou None")

    # posição de cada classe na matriz
    indice = {classe: i for i, classe in enumerate(classes.tolist())}

    matriz = np.zeros((len(classes), len(classes)), dtype=int)
    for real, previsto in zip(y_real.tolist(), y_previsto.tolist()):
        if real in indice and previsto in indice:
            matriz[indice[real], indice[previsto]] += 1

    return matriz


def relatorio(y_real: np.ndarray, y_previsto: np.ndarray,
    nomes_classes: dict | None = None, casas: int = 2,
) -> str:
    """
    Resumo por classe com precisão, recall, F1 e suporte, mais a acurácia geral
    (equivalente a classification_report do scikit-learn).

        precisão -> das vezes que o modelo previu a classe, quantas estavam certas
        recall -> das amostras que eram da classe, quantas o modelo encontrou
        f1 -> média harmônica de precisão e recall
        suporte -> quantas amostras da classe existem em y_real

    Parâmetros:
        y_real -> vetor com as classes verdadeiras
        y_previsto -> vetor com as classes previstas por prever()
        nomes_classes -> dict {classe: nome} para exibir nomes no lugar dos números (opcional)
        casas -> casas decimais das métricas (padrão 2)

    Retorna:
        texto formatado; use print(relatorio(...)) para exibir
    """

    _validar_vetores(y_real, y_previsto)

    if nomes_classes is not None and not isinstance(nomes_classes, dict):
        raise TypeError("O parâmetro nomes_classes deve ser um dict ou None")

    classes = np.unique(np.concatenate([y_real, y_previsto]))
    matriz = matriz_confusao(y_real, y_previsto, classes)

    acertos = np.diag(matriz)
    previstos = matriz.sum(axis=0)  # coluna: quantas vezes o modelo previu cada classe
    suporte = matriz.sum(axis=1)    # linha: quantas amostras reais de cada classe

    # divisões por zero viram 0 (classe nunca prevista ou sem amostras), como no scikit-learn
    with np.errstate(divide="ignore", invalid="ignore"):
        precisao = np.where(previstos > 0, acertos / previstos, 0.0)
        recall = np.where(suporte > 0, acertos / suporte, 0.0)
        f1 = np.where(precisao + recall > 0, 2 * precisao * recall / (precisao + recall), 0.0)

    nomes = [
        str(nomes_classes.get(classe, classe)) if nomes_classes else str(classe)
        for classe in classes.tolist()
    ]

    total = int(suporte.sum())
    largura = max(len(nome) for nome in nomes + ["média ponderada"])
    num = f"{{:>10.{casas}f}}"

    linhas = [f"{'':>{largura}} {'precisão':>10} {'recall':>10} {'f1':>10} {'suporte':>10}", ""]

    for i, nome in enumerate(nomes):
        linhas.append(
            f"{nome:>{largura}} " + num.format(precisao[i]) + " " + num.format(recall[i])
            + " " + num.format(f1[i]) + f" {suporte[i]:>10}"
        )

    pesos = suporte / total
    linhas += [
        "",
        f"{'acurácia':>{largura}} {'':>10} {'':>10} " + num.format(acertos.sum() / total) + f" {total:>10}",
        f"{'média simples':>{largura}} " + num.format(precisao.mean()) + " " + num.format(recall.mean())
        + " " + num.format(f1.mean()) + f" {total:>10}",
        f"{'média ponderada':>{largura}} " + num.format((precisao * pesos).sum()) + " "
        + num.format((recall * pesos).sum()) + " " + num.format((f1 * pesos).sum()) + f" {total:>10}",
    ]

    return "\n".join(linhas)
