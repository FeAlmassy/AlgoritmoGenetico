import pandas as pd
import numpy as np

'''
Funcao pra leitura de banco de dados padronizados .xlsx e .csv
caso nao esteja funcionando verificar o layout do banco de dados
primeira coluna deve ser somente com as licoes
ultima coluna somente com as classes
colunas intermediarias com as features
'''

def ler_csv(caminho):
    """
    Le um arquivo .csv e separa as features das classes.
    Ignora a primeira coluna e usa a ultima como classe.

    Parametros:
        caminho -> caminho do arquivo .csv

    Retorna:
        tupla (features, classes): array (amostras, features) e vetor (amostras,)
    """
    df = pd.read_csv(caminho)
    array_features = df.iloc[ : , 1:-1].to_numpy()
    array_classes = df.iloc[ : , -1].to_numpy()
    return array_features, array_classes

def ler_excel(caminho):
    """
    Le um arquivo .xlsx e separa as features das classes.
    Ignora a primeira coluna e usa a ultima como classe.

    Parametros:
        caminho -> caminho do arquivo .xlsx

    Retorna:
        tupla (features, classes): array (amostras, features) e vetor (amostras,)
    """
    df = pd.read_excel(caminho)
    array_features = df.iloc[ : , 1:-1].to_numpy()
    array_classes = df.iloc[ : , -1].to_numpy()
    return array_features, array_classes
