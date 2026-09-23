"""Métodos de seleção de pais."""

import numpy as np


def selecionar_pais_roleta(
    cromossomos: np.ndarray,
    percentual_fitnesses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Seleção por roleta: cromossomos com maior fitness têm
    maior probabilidade de serem escolhidos.
    """
    roleta_acumulada = np.cumsum(percentual_fitnesses)

    indice_pai = min(
        int(np.searchsorted(roleta_acumulada, np.random.rand())),
        len(cromossomos) - 1,
    )
    indice_mae = min(
        int(np.searchsorted(roleta_acumulada, np.random.rand())),
        len(cromossomos) - 1,
    )

    return cromossomos[indice_pai], cromossomos[indice_mae]