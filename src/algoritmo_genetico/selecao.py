# ordem: 3ª etapa, vem depois de fitness.py -> próxima: cruzamento.py
"""Seleção de pais."""

import math

import numpy as np


def _girar_roleta(percentual_fitnesses: np.ndarray) -> int:
    """
    Gira a roleta uma vez e devolve o índice do cromossomo sorteado.
    Sorteia um número entre 0 e 1 e vai somando as chances de cada
    cromossomo até a soma passar desse número.

    Parâmetros:
        percentual_fitnesses -> chances de cada cromossomo (somam 1)

    Retorna:
        índice (int) do cromossomo sorteado
    """

    sorteio = np.random.rand()  #num aleatorio entre 0 e 1

    soma = 0.0
    for indice, chance in enumerate(percentual_fitnesses):
        soma += chance
        if sorteio < soma:
            return indice

    # se a soma deu 0.9999999 por arredondamento e o sorteio caiu depois, fica com o último, claude ajudou
    return len(percentual_fitnesses) - 1


def selecionar_pais_roleta(cromossomos: np.ndarray, percentual_fitnesses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Escolhe um pai e uma mãe por roleta: quem tem fitness maior tem mais
    chance de ser sorteado. Pai e mãe podem sair o mesmo cromossomo.

    Parâmetros:
        cromossomos -> array com a população atual
        percentual_fitnesses -> array com a chance de cada um (saída de fitness_percentual)

    Retorna:
        tupla (pai, mae), cada um um array (qtd_genes,)
    """

    if not isinstance(cromossomos, np.ndarray) or not isinstance(percentual_fitnesses, np.ndarray):
        raise TypeError("Os parâmetros cromossomos e percentual_fitnesses devem ser do tipo ndarray")

    #ndim e o numero de dimensoes 
    if cromossomos.ndim != 2:
        raise ValueError(f"cromossomos deve ser uma matriz de duas dimensoes")

    if percentual_fitnesses.ndim != 1:
        raise ValueError(f"percentual_fitnesses deve ser um vetor")

    if len(cromossomos) == 0:
        raise ValueError("cromossomos não pode ser vazio")

    if len(cromossomos) != len(percentual_fitnesses):
        raise ValueError(f"cromossomos tem {len(cromossomos)} linhas mas percentual_fitnesses tem {len(percentual_fitnesses)} valores")

    #claude ajudou aqui
    for chance in percentual_fitnesses:
        if math.isnan(chance):
            raise ValueError("percentual_fitnesses não pode ter valores NaN")
        if chance < 0:
            raise ValueError(
                f"percentual_fitnesses não pode ter valores negativos, encontrou {chance}"
            )

    soma = sum(percentual_fitnesses)
    if abs(soma - 1) > 1e-6:  # tolerância pequena por causa de arredondamento de float
        raise ValueError(f"percentual_fitnesses deve somar 1, mas soma {soma}")

    indice_pai = _girar_roleta(percentual_fitnesses)
    indice_mae = _girar_roleta(percentual_fitnesses)

    return cromossomos[indice_pai], cromossomos[indice_mae]
