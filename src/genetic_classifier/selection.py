# order: step 3, comes after fitness.py -> next: crossover.py
"""Parent selection."""

import math

import numpy as np


def _spin_roulette(proportions: np.ndarray) -> int:
    """
    Spin the roulette once and return the index of the chromosome drawn.
    Draws a number between 0 and 1 and adds up the chances of each chromosome
    until the sum passes that number.

    Parameters
    ----------
    proportions : np.ndarray of shape (n_chromosomes,)
        Chance of each chromosome, adding up to 1 (output of fitness_proportions).

    Returns
    -------
    index : int
        Position (row) of the chromosome drawn, from 0 to n_chromosomes - 1.
    """

    draw = np.random.rand()  # random number between 0 and 1

    total = 0.0
    for index, chance in enumerate(proportions):
        total += chance
        if draw < total:
            return index

    # if rounding made the sum 0.9999999 and the draw landed above it, keep the last one
    return len(proportions) - 1


def select_parents_roulette(chromosomes: np.ndarray, proportions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Pick two parents by roulette: the higher a chromosome's fitness, the higher
    its chance of being drawn. Both parents can be the same chromosome.

    Parameters
    ----------
    chromosomes : np.ndarray of shape (n_chromosomes, n_genes)
        Current population.
    proportions : np.ndarray of shape (n_chromosomes,)
        Chance of each chromosome (output of fitness_proportions). No negative
        or NaN values, and they must add up to 1.

    Returns
    -------
    parents : tuple (parent1, parent2)
        The two chromosomes drawn, each an np.ndarray of shape (n_genes,).
        Usually unpacked straight away, as in the example.

    Examples
    --------
    >>> parent1, parent2 = select_parents_roulette(population, proportions)
    """

    if not isinstance(chromosomes, np.ndarray) or not isinstance(proportions, np.ndarray):
        raise TypeError("chromosomes and proportions must be ndarrays")

    # ndim is the number of dimensions
    if chromosomes.ndim != 2:
        raise ValueError("chromosomes must be a 2D matrix")

    if proportions.ndim != 1:
        raise ValueError("proportions must be a 1D vector")

    if len(chromosomes) == 0:
        raise ValueError("chromosomes cannot be empty")

    if len(chromosomes) != len(proportions):
        raise ValueError(f"chromosomes has {len(chromosomes)} rows but proportions has {len(proportions)} values")

    for chance in proportions:
        if math.isnan(chance):
            raise ValueError("proportions cannot contain NaN values")
        if chance < 0:
            raise ValueError(f"proportions cannot contain negative values, found {chance}")

    total = sum(proportions)
    if abs(total - 1) > 1e-6:  # small tolerance because of float rounding
        raise ValueError(f"proportions must add up to 1, but they add up to {total}")

    index_parent1 = _spin_roulette(proportions)
    index_parent2 = _spin_roulette(proportions)

    return chromosomes[index_parent1], chromosomes[index_parent2]
