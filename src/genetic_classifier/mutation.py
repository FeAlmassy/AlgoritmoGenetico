# order: step 5, comes after crossover.py -> next: update.py
"""Mutation of the offspring."""

import numpy as np


def mutate(offspring: np.ndarray, n_mutated_genes: int = 1) -> np.ndarray:
    """
    Pick some genes of each child at random and replace them with a new random
    value between -1 and 1. The input array is not changed: the function works
    on a copy.

    Parameters
    ----------
    offspring : np.ndarray of shape (n_offspring, n_genes)
        Children produced by the crossover. Cannot be empty.
    n_mutated_genes : int, default=1
        How many genes to replace in each child, from 0 to n_genes.

    Returns
    -------
    mutated_offspring : np.ndarray of shape (n_offspring, n_genes), dtype float
        Copy of the offspring with the mutations applied.

    Examples
    --------
    >>> mutated_offspring = mutate(offspring, n_mutated_genes=2)
    """

    if not isinstance(offspring, np.ndarray):
        raise TypeError("offspring must be an ndarray")

    if len(offspring) == 0:
        raise ValueError("offspring cannot be empty")

    if not isinstance(n_mutated_genes, int):
        raise TypeError("n_mutated_genes must be an integer")

    n_genes = len(offspring[0])
    if n_mutated_genes < 0 or n_mutated_genes > n_genes:
        raise ValueError(f"n_mutated_genes must be between 0 and {n_genes} (number of genes), got {n_mutated_genes}")

    # .astype(float) makes a float copy; with an int array the drawn value
    # (e.g. 0.37) would be truncated to 0 when stored
    mutated_offspring = offspring.astype(float)

    for child in mutated_offspring:
        # draws n_mutated_genes different positions between 0 and n_genes-1
        indices = np.random.choice(n_genes, size=n_mutated_genes, replace=False)
        for index in indices:
            child[index] = -1 + 2 * np.random.rand()

    return mutated_offspring
