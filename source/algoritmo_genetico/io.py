"""Leitura de dados a partir de planilhas."""

import numpy as np
import pandas as pd


def ler_excel(caminho: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Lê uma planilha no formato padrão:
        - 1ª coluna: índice (ignorada)
        - colunas do meio: features
        - última coluna: classes (gabarito)

    Retorna (X, y) como arrays numpy.
    """
    df = pd.read_excel(caminho)

    X = df.iloc[:, 1:-1].values.astype(float)
    y = df.iloc[:, -1].values

    return X, y


def ler_csv(caminho: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Lê um CSV no formato padrão:
        - 1ª coluna: índice (ignorada)
        - colunas do meio: features
        - última coluna: classes (gabarito)

    Retorna (X, y) como arrays numpy.
    """
    df = pd.read_csv(caminho)

    X = df.iloc[:, 1:-1].values.astype(float)
    y = df.iloc[:, -1].values

    return X, y