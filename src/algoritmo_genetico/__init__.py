"""AlgoritmoGeneticoPy — classificação por algoritmo genético (Prandiano)."""

from .divisao import dividir_treino_teste
from .leitura import ler_csv, ler_excel
from .metricas import acuracia, matriz_confusao, relatorio
from .nucleo import prever, rodar, treinar

# rodar fica fora do __all__: é só o nome antigo de treinar, mantido por compatibilidade
__all__ = [
    "treinar", "prever", "ler_excel", "ler_csv", "dividir_treino_teste",
    "acuracia", "matriz_confusao", "relatorio",
]

__version__ = "1.1.0"
