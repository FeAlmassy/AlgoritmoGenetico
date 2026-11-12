# order: step 1 (start) -> next: fitness.py
"""Creation of the initial population of chromosomes."""

import numpy as np


def create_chromosomes(n_chromosomes: int = 6, n_genes: int = 19,
    gene_min: float = -1.0, gene_max: float = 1.0) -> np.ndarray:
    """
    Create the initial population: a matrix where each row is a chromosome and
    each column is a gene, filled with random values between gene_min and gene_max.

    Parameters
    ----------
    n_chromosomes : int, default=6
        How many chromosomes to create (the population size). Must be >= 1.
    n_genes : int, default=19
        How many genes each chromosome has: one weight per feature plus one bias,
        so usually ``n_features + 1``. Must be >= 1.
    gene_min : float, default=-1.0
        Smallest value a gene can take.
    gene_max : float, default=1.0
        Largest value a gene can take. Must be greater than gene_min.

    Returns
    -------
    population : np.ndarray of shape (n_chromosomes, n_genes)
        The new population. Column 0 is the bias, the other columns are the weights.

    Examples
    --------
    >>> population = create_chromosomes(n_chromosomes=30, n_genes=X.shape[1] + 1)
    """

    if not isinstance(n_chromosomes, int) or not isinstance(n_genes, int):
        raise TypeError("n_chromosomes and n_genes must be integers")

    if n_chromosomes <= 0 or n_genes <= 0:
        raise ValueError("at least 1 chromosome and 1 gene are needed")

    if not isinstance(gene_min, (int, float)) or not isinstance(gene_max, (int, float)):
        raise TypeError("gene_min and gene_max must be numbers")

    if gene_min >= gene_max:
        raise ValueError(f"gene_min ({gene_min}) must be smaller than gene_max ({gene_max})")

    return np.random.uniform(gene_min, gene_max, size=(n_chromosomes, n_genes))
