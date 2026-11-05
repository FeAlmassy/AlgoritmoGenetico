"""AlgoritmoGeneticoPy — classificação por algoritmo genético (Prandiano)."""

from .avaliacao import avaliar, matriz_confusao
from .divisao import dividir_treino_teste
from .leitura import ler_csv, ler_excel
from .nucleo import prever, rodar, treinar

__all__ = ["treinar", "rodar", "prever", "avaliar", "matriz_confusao", "ler_excel", "ler_csv", "dividir_treino_teste"]

__version__ = "1.0.0"
