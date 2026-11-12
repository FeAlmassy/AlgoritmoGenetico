# order: step 6, comes after mutation.py -> next: fitness.py (new generation)
"""Population update with the best offspring."""

import numpy as np


def update_population(chromosomes: np.ndarray, fitnesses: np.ndarray, offspring: np.ndarray,
    offspring_fitnesses: np.ndarray, n_replacements: int = 2) -> np.ndarray:
    """
    Build the next generation: remove the worst chromosomes of the population
    and put the best children in their place. The rest of the population stays
    the same (elitism).

    Parameters
    ----------
    chromosomes : np.ndarray of shape (n_chromosomes, n_genes)
        Current population.
    fitnesses : np.ndarray of shape (n_chromosomes,)
        Fitness of each chromosome of the population.
    offspring : np.ndarray of shape (n_offspring, n_genes)
        Mutated children (output of mutate).
    offspring_fitnesses : np.ndarray of shape (n_offspring,)
        Fitness of each child.
    n_replacements : int, default=2
        How many chromosomes to swap for children, from 0 to
        min(n_chromosomes, n_offspring).

    Returns
    -------
    new_population : np.ndarray of shape (n_chromosomes, n_genes)
        The population of the next generation (a new array, the input is not changed).

    Examples
    --------
    >>> new_population = update_population(population, fitnesses, offspring, offspring_fitnesses, n_replacements=8)
    """

    if not isinstance(chromosomes, np.ndarray):
        raise TypeError("chromosomes must be an ndarray")

    if not isinstance(fitnesses, np.ndarray):
        raise TypeError("fitnesses must be an ndarray")

    if not isinstance(offspring, np.ndarray):
        raise TypeError("offspring must be an ndarray")

    if not isinstance(offspring_fitnesses, np.ndarray):
        raise TypeError("offspring_fitnesses must be an ndarray")

    if chromosomes.ndim != 2 or offspring.ndim != 2:
        raise ValueError(f"chromosomes and offspring must be matrices (rows, n_genes), got {chromosomes.ndim} and {offspring.ndim} dimensions")

    if fitnesses.ndim != 1 or offspring_fitnesses.ndim != 1:
        raise ValueError(f"fitnesses and offspring_fitnesses must be vectors, got {fitnesses.ndim} and {offspring_fitnesses.ndim} dimensions")

    if len(chromosomes) != len(fitnesses):
        raise ValueError(f"chromosomes has {len(chromosomes)} rows, but fitnesses has {len(fitnesses)} values")

    if len(offspring) != len(offspring_fitnesses):
        raise ValueError(f"offspring has {len(offspring)} rows, but offspring_fitnesses has {len(offspring_fitnesses)} values")

    if chromosomes.shape[1] != offspring.shape[1]:
        raise ValueError(f"chromosomes have {chromosomes.shape[1]} genes, but offspring have {offspring.shape[1]} genes")

    if not isinstance(n_replacements, int):
        raise TypeError("n_replacements must be an integer")

    limit = min(len(chromosomes), len(offspring))
    if n_replacements < 0 or n_replacements > limit:
        raise ValueError(f"n_replacements must be between 0 and {limit} (smaller of population size and number of offspring), got {n_replacements}")

    # indices sorted by fitness, from lowest to highest
    population_order = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])
    offspring_order = sorted(range(len(offspring_fitnesses)), key=lambda i: offspring_fitnesses[i])

    # the first of the population are the worst, the last of the offspring are the best
    worst_indices = population_order[:n_replacements]
    best_offspring_indices = offspring_order[len(offspring_order) - n_replacements:]

    new_population = chromosomes.copy()
    for i in range(n_replacements):
        new_population[worst_indices[i]] = offspring[best_offspring_indices[i]]

    return new_population
