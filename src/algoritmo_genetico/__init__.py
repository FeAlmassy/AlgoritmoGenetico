"""AlgoritmoGeneticoPy — classificação por algoritmo genético (Prandiano)."""

from .divisao import dividir_treino_teste
from .leitura import ler_csv, ler_excel
from .nucleo import prever, rodar

__all__ = ["rodar", "prever", "ler_excel", "ler_csv", "dividir_treino_teste"]

__version__ = "1.0.0"
