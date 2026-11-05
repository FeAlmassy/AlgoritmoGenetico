"""
Leitura de bases de dados padronizadas (.xlsx e .csv).

Layout esperado da planilha (veja exemplos/base_de_dados_para_exemplos/modelo_base.xlsx):
    primeira coluna -> identificador de cada amostra (lições)
    colunas do meio -> features
    última coluna -> classe (dados novos, sem classe, não têm essa coluna)
"""

import numpy as np
import pandas as pd


def _separar_colunas(df: pd.DataFrame, tem_classe: bool = True) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    Separa a tabela em features e classes, ignorando a primeira coluna.

    Parâmetros:
        df -> tabela lida pelo pandas
        tem_classe -> se True, a última coluna é a classe; se False, todas as
                      colunas depois da primeira são features

    Retorna:
        com tem_classe=True: tupla (features, classes), com um array
                             (amostras, features) e um vetor (amostras,)
        com tem_classe=False: array (amostras, features) só com as features
    """
    if not isinstance(tem_classe, bool):
        raise TypeError("tem_classe deve ser True ou False")

    if not tem_classe:
        return df.iloc[:, 1:].to_numpy()

    array_features = df.iloc[:, 1:-1].to_numpy()
    array_classes = df.iloc[:, -1].to_numpy()
    return array_features, array_classes


def ler_csv(caminho: str, tem_classe: bool = True) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    Lê um arquivo .csv e separa as features das classes.
    Ignora a primeira coluna e, por padrão, usa a última como classe.

    Parâmetros:
        caminho -> caminho do arquivo .csv
        tem_classe -> se True, a última coluna é a classe; use False para dados
                      novos, sem coluna de classe

    Retorna:
        com tem_classe=True: tupla (features, classes), com um array
                             (amostras, features) e um vetor (amostras,)
        com tem_classe=False: array (amostras, features) só com as features
    """
    return _separar_colunas(pd.read_csv(caminho), tem_classe)


def ler_excel(caminho: str, tem_classe: bool = True) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    Lê um arquivo .xlsx e separa as features das classes.
    Ignora a primeira coluna e, por padrão, usa a última como classe.

    Parâmetros:
        caminho -> caminho do arquivo .xlsx
        tem_classe -> se True, a última coluna é a classe; use False para dados
                      novos, sem coluna de classe

    Retorna:
        com tem_classe=True: tupla (features, classes), com um array
                             (amostras, features) e um vetor (amostras,)
        com tem_classe=False: array (amostras, features) só com as features
    """
    return _separar_colunas(pd.read_excel(caminho), tem_classe)
