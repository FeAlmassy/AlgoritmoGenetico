# order: step 4, comes after selection.py -> next: mutation.py
"""Crossover between two parents."""

import numpy as np


def one_point_crossover(parent1: np.ndarray, parent2: np.ndarray, n_offspring: int = 3) -> np.ndarray:
    """
    Create offspring by cutting the parents at a random point: each child takes
    the beginning of parent1 and the end of parent2. Each child gets its own
    cut point.

    Parameters
    ----------
    parent1 : np.ndarray of shape (n_genes,)
        First parent chosen (gives the genes before the cut).
    parent2 : np.ndarray of shape (n_genes,)
        Second parent chosen (gives the genes from the cut onwards). Must have
        the same number of genes as parent1, at least 2.
    n_offspring : int, default=3
        How many children to create. Must be >= 1.

    Returns
    -------
    offspring : np.ndarray of shape (n_offspring, n_genes)
        The children, one per row.

    Examples
    --------
    >>> offspring = one_point_crossover(parent1, parent2, n_offspring=16)
    """

    if not isinstance(parent1, np.ndarray) or not isinstance(parent2, np.ndarray):
        raise TypeError("parent1 and parent2 must be ndarrays")

    if len(parent1) != len(parent2):
        raise ValueError("parent1 and parent2 must have the same number of genes")

    if len(parent1) < 2:
        raise ValueError(f"parent1 and parent2 need at least 2 genes to have a cut point, got {len(parent1)}")

    if n_offspring <= 0:
        raise ValueError(f"n_offspring must be at least 1, got {n_offspring}")

    offspring = []
    for _ in range(n_offspring):
        # random integer from 1 to len(parent1)-1, never cuts at the ends
        cut = np.random.randint(1, len(parent1))
        child = parent1.copy()           # starts equal to parent1
        child[cut:] = parent2[cut:]      # from the cut onwards takes parent2's genes
        offspring.append(child)

    return np.array(offspring)  # turns the list of children into a matrix
