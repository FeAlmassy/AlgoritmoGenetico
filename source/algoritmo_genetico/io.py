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
    df = pd.read_csv(caminho)
    array_features = df.iloc[ : , 1:-1].to_numpy()
    array_classes = df.iloc[ : , -1].to_numpy()
    return array_features, array_classes

def ler_excel(caminho):
    df = pd.read_excel(caminho)
    array_features = df.iloc[ : , 1:-1].to_numpy()
    array_classes = df.iloc[ : , -1].to_numpy()
    return array_features, array_classes
