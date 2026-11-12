# order: step 2, comes after creation.py (and after update.py in later generations) -> next: selection.py
"""Fitness for binary and ordinal classification, and conversion into roulette probabilities."""

import numpy as np


def binary_fitness(chromosomes: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Score (fitness) each chromosome on a binary classification problem.
    The score is the recall of class 1 times the recall of class 0, so it is
    only high when the chromosome gets both classes right.

    Parameters
    ----------
    chromosomes : np.ndarray of shape (n_chromosomes, n_features + 1)
        Population to score. In each row the first gene is the bias and the
        rest are the weights.
    X : np.ndarray of shape (n_samples, n_features)
        Data, numbers only.
    y : np.ndarray of shape (n_samples,)
        True class of each sample, only 0 or 1. Both classes must appear.

    Returns
    -------
    fitnesses : np.ndarray of shape (n_chromosomes,), dtype float
        Fitness of each chromosome, between 0 and 1 (1 = every sample right).

    Examples
    --------
    >>> fitnesses = binary_fitness(population, X_train, y_train)
    """

    if not isinstance(chromosomes, np.ndarray):
        raise TypeError("chromosomes must be an ndarray")

    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be ndarrays")

    # number of weights (columns minus the bias) must match the number of features
    if chromosomes.shape[1] - 1 != X.shape[1]:
        raise ValueError(f"chromosomes have {chromosomes.shape[1] - 1} weights (without bias) but X has {X.shape[1]} features")

    # one class per sample
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} samples, but y has {y.shape[0]}")

    # np.unique lists the distinct values
    unique_values = np.unique(y)
    if not np.all(np.isin(unique_values, [0, 1])):
        raise ValueError(f"y must contain only 0 and 1, found {unique_values}")

    n_positives = np.sum(y == 1)
    n_negatives = np.sum(y == 0)

    # y cannot be only 0s or only 1s
    if n_positives == 0 or n_negatives == 0:
        raise ValueError("y must have at least one sample of each class (0 and 1)")

    fitness_list = []
    for chromosome in chromosomes:
        bias = chromosome[0]
        weights = chromosome[1:]

        scores = np.dot(X, weights) + bias
        predictions = np.where(scores >= 0, 1, 0)

        recall_pos = np.sum((predictions == 1) & (y == 1)) / n_positives
        recall_neg = np.sum((predictions == 0) & (y == 0)) / n_negatives

        fitness_list.append(recall_pos * recall_neg)

    return np.array(fitness_list)


def ordinal_fitness(chromosomes: np.ndarray, X: np.ndarray,
    y: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Score (fitness) each chromosome on a problem with K ordered classes
    (e.g. low, medium, high). The thresholds split the chromosome's score into
    ranges, one range per class. The fitness is the product of the recalls of
    all classes.

    Parameters
    ----------
    chromosomes : np.ndarray of shape (n_chromosomes, n_features + 1)
        Population to score. In each row the first gene is the bias and the
        rest are the weights.
    X : np.ndarray of shape (n_samples, n_features)
        Data, numbers only.
    y : np.ndarray of shape (n_samples,)
        True class of each sample, integers from 0 to K-1. Every class must appear.
    thresholds : np.ndarray of shape (K-1,)
        Cut points that separate the classes, in strictly increasing order.

    Returns
    -------
    fitnesses : np.ndarray of shape (n_chromosomes,), dtype float
        Fitness of each chromosome, between 0 and 1 (1 = every sample right).

    Examples
    --------
    >>> fitnesses = ordinal_fitness(population, X_train, y_train, np.array([-1.0, 1.0]))
    """

    if not isinstance(chromosomes, np.ndarray):
        raise TypeError("chromosomes must be an ndarray")

    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be ndarrays")

    if not isinstance(thresholds, np.ndarray):
        raise TypeError("thresholds must be an ndarray")

    # number of weights (columns minus the bias) must match the number of features
    if chromosomes.shape[1] - 1 != X.shape[1]:
        raise ValueError(f"chromosomes have {chromosomes.shape[1] - 1} weights (without bias) but X has {X.shape[1]} features")

    # one class per sample
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} samples, but y has {y.shape[0]}")

    if len(thresholds) == 0:
        raise ValueError("thresholds cannot be empty (at least 1 cut point is needed)")

    if not np.all(np.diff(thresholds) > 0):
        raise ValueError(f"thresholds must be in increasing order, got {thresholds}")

    K = len(thresholds) + 1

    if y.dtype.kind in "USO" and any(isinstance(value, str) for value in y):
        raise TypeError("y must contain integers from 0 to K-1; convert text classes to numbers before training")

    unique_values = np.unique(y)

    # a fractional or empty (NaN) value belongs to no class and would be silently ignored
    float_values = unique_values.astype(float)
    if not np.all(float_values == np.floor(float_values)):
        raise ValueError(f"y must contain integers between 0 and {K - 1}, found {unique_values}")

    if unique_values.min() < 0 or unique_values.max() >= K:
        raise ValueError(f"y must contain integers between 0 and {K - 1}, found {unique_values}")

    class_totals = np.array([np.sum(y == k) for k in range(K)])

    if np.any(class_totals == 0):
        empty_classes = np.where(class_totals == 0)[0].tolist()
        raise ValueError(
            f"y must have at least one sample of each class. "
            f"Classes without samples: {empty_classes}"
        )

    fitness_list = []
    for chromosome in chromosomes:
        bias = chromosome[0]
        weights = chromosome[1:]

        scores = np.dot(X, weights) + bias
        predictions = np.digitize(scores, thresholds)

        fitness = 1.0
        for k in range(K):
            hits_k = np.sum((predictions == k) & (y == k))
            fitness *= hits_k / class_totals[k]

        fitness_list.append(fitness)

    return np.array(fitness_list)


def fitness_proportions(fitnesses: np.ndarray) -> np.ndarray:
    """
    Turn the fitnesses into proportions that add up to 1, to be used as the
    chances of each chromosome in the roulette. If every fitness is 0, every
    chromosome gets the same chance.

    Parameters
    ----------
    fitnesses : np.ndarray of shape (n_chromosomes,)
        Fitness of each chromosome (output of binary_fitness or ordinal_fitness).
        Values must not be negative.

    Returns
    -------
    proportions : np.ndarray of shape (n_chromosomes,), dtype float
        Chance of each chromosome being picked, between 0 and 1, adding up to 1.

    Examples
    --------
    >>> proportions = fitness_proportions(fitnesses)
    """
    total = np.sum(fitnesses)
    if total == 0:
        return np.ones(len(fitnesses)) / len(fitnesses)
    return fitnesses / total
