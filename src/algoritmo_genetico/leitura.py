"""
Leitura de bases de dados padronizadas (.xlsx e .csv).

Layout esperado da planilha (veja exemplos/base_de_dados_para_exemplos/modelo_base.xlsx):
    primeira coluna -> identificador de cada amostra (lições)
    colunas do meio -> features
    última coluna -> classe
"""

import numpy as np
import pandas as pd


def _separar_colunas(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Separa a tabela em features e classes, ignorando a primeira coluna.

    Parâmetros:
        df -> tabela lida pelo pandas

    Retorna:
        tupla (features, classes): array (amostras, features) e vetor (amostras,)
    """
    array_features = df.iloc[:, 1:-1].to_numpy()
    array_classes = df.iloc[:, -1].to_numpy()
    return array_features, array_classes


def ler_csv(caminho: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Lê um arquivo .csv e separa as features das classes.
    Ignora a primeira coluna e usa a última como classe.

    Parâmetros:
        caminho -> caminho do arquivo .csv

    Retorna:
        tupla (features, classes): array (amostras, features) e vetor (amostras,)
    """
    return _separar_colunas(pd.read_csv(caminho))


def ler_excel(caminho: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Lê um arquivo .xlsx e separa as features das classes.
    Ignora a primeira coluna e usa a última como classe.

    Parâmetros:
        caminho -> caminho do arquivo .xlsx

    Retorna:
        tupla (features, classes): array (amostras, features) e vetor (amostras,)
    """
    return _separar_colunas(pd.read_excel(caminho))
