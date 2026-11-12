"""Main loop of the genetic algorithm: training (train) and prediction (predict)."""

import numpy as np

from .creation import create_chromosomes
from .crossover import one_point_crossover
from .fitness import binary_fitness, fitness_proportions, ordinal_fitness
from .mutation import mutate
from .selection import select_parents_roulette
from .update import update_population


def _check_features(X: np.ndarray) -> None:
    """
    Check that X only holds valid numbers. Without this, an empty cell in the
    spreadsheet would silently turn into a wrong predicted class.

    Parameters
    ----------
    X : np.ndarray
        Data to check, of any shape.

    Returns
    -------
    None
        Raises TypeError if X has text or another non-numeric value, and
        ValueError if X has NaN or infinite values.
    """
    try:
        values = X.astype(float)
    except (TypeError, ValueError):
        raise TypeError("X must contain only numbers, but found text or another type of value") from None

    if not np.isfinite(values).all():
        raise ValueError("X has empty cells or invalid values (NaN or infinity), fix the data before continuing")


def train(
    X: np.ndarray,
    y: np.ndarray,
    n_chromosomes: int = 30,
    n_mutated_genes: int = 2,
    n_offspring: int = 16,
    n_replacements: int | None = None,
    n_generations: int = 1000,
    target_fitness: float = 1.0,
    thresholds: np.ndarray | None = None,
    gene_min: float = -1.0,
    gene_max: float = 1.0,
    seed: int | None = None,
    verbose: bool = True,
) -> dict:
    """
    Train a linear classifier with the full genetic algorithm: create the
    population and repeat fitness -> selection -> crossover -> mutation ->
    update until reaching target_fitness or running out of generations.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Training data, numbers only (no empty cells).
    y : np.ndarray of shape (n_samples,)
        True class of each sample: 0 or 1 for binary, integers from 0 to K-1
        for ordinal.
    n_chromosomes : int, default=30
        Population size.
    n_mutated_genes : int, default=2
        How many genes change in each child during mutation, from 0 to n_features + 1.
    n_offspring : int, default=16
        How many children are created per generation.
    n_replacements : int or None, default=None
        How many of the worst chromosomes are swapped for the best children, at
        most min(n_chromosomes, n_offspring). None = half of n_offspring.
    n_generations : int, default=1000
        Maximum number of generations.
    target_fitness : float, default=1.0
        Stops when the best fitness reaches this value, between 0 and 1
        (1.0 = only stops if every sample is right, so it usually runs every generation).
    thresholds : np.ndarray of shape (K-1,) or None, default=None
        None for 2 classes (binary), or K-1 cut points in increasing order for
        K classes (ordinal).
    gene_min : float, default=-1.0
        Smallest gene value in the initial population.
    gene_max : float, default=1.0
        Largest gene value in the initial population. Must be greater than gene_min.
    seed : int or None, default=None
        Random seed for reproducibility (None = different result every run).
    verbose : bool, default=True
        If True, prints the best fitness and accuracy of every generation.

    Returns
    -------
    model : dict
        The trained model, ready to be passed to predict(). Keys:

        - ``"chromosome"`` (np.ndarray of shape (n_features + 1,)): best
          chromosome found. Position 0 is the bias, the rest are the weights.
        - ``"fitness"`` (float): fitness of the best chromosome on the training data, 0 to 1.
        - ``"accuracy"`` (float): fraction of the samples of X that the best
          chromosome gets right, 0 to 1.
        - ``"generation"`` (int): generation where training stopped.
        - ``"thresholds"`` (np.ndarray or None): thresholds used (None if binary).
        - ``"history"`` (dict of lists of float, one value per generation):
          ``"best"`` = best fitness so far, ``"mean"`` = mean fitness of the
          population, ``"accuracy"`` = accuracy of the best chromosome so far.

    Examples
    --------
    >>> model = train(X_train, y_train, seed=0)
    >>> model["accuracy"]
    """

    if not isinstance(X, np.ndarray):
        raise TypeError("X must be an ndarray")

    if not isinstance(y, np.ndarray):
        raise TypeError("y must be an ndarray")

    if X.ndim != 2:
        raise ValueError(f"X must be a 2D matrix (n_samples, n_features), got {X.ndim} dimensions")

    if y.ndim != 1:
        raise ValueError(f"y must be a 1D vector, got {y.ndim} dimensions")

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} samples, but y has {y.shape[0]} values")

    if X.shape[1] == 0:
        raise ValueError("X has no features (at least 1 column is needed)")

    _check_features(X)

    if not isinstance(n_chromosomes, int):
        raise TypeError("n_chromosomes must be an integer")

    if n_chromosomes <= 0:
        raise ValueError(f"n_chromosomes must be >= 1, got {n_chromosomes}")

    if not isinstance(n_mutated_genes, int):
        raise TypeError("n_mutated_genes must be an integer")

    if n_mutated_genes < 0 or n_mutated_genes > X.shape[1] + 1:
        raise ValueError(f"n_mutated_genes must be between 0 and {X.shape[1] + 1} (number of genes), got {n_mutated_genes}")

    if not isinstance(n_offspring, int):
        raise TypeError("n_offspring must be an integer")

    if n_offspring <= 0:
        raise ValueError(f"n_offspring must be >= 1, got {n_offspring}")

    if n_replacements is None:
        n_replacements = min(max(n_offspring // 2, 1), n_chromosomes)

    if not isinstance(n_replacements, int):
        raise TypeError("n_replacements must be an integer or None")

    if n_replacements < 0 or n_replacements > min(n_chromosomes, n_offspring):
        raise ValueError(f"n_replacements must be between 0 and {min(n_chromosomes, n_offspring)} (smaller of n_chromosomes and n_offspring), got {n_replacements}")

    if not isinstance(n_generations, int):
        raise TypeError("n_generations must be an integer")

    if n_generations <= 0:
        raise ValueError(f"n_generations must be >= 1, got {n_generations}")

    if not isinstance(target_fitness, (int, float)):
        raise TypeError("target_fitness must be a number")

    if target_fitness < 0 or target_fitness > 1:
        raise ValueError(f"target_fitness must be between 0 and 1, got {target_fitness}")

    if not isinstance(gene_min, (int, float)):
        raise TypeError("gene_min must be a number")

    if not isinstance(gene_max, (int, float)):
        raise TypeError("gene_max must be a number")

    if gene_min >= gene_max:
        raise ValueError(f"gene_min ({gene_min}) must be smaller than gene_max ({gene_max})")

    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise TypeError("seed must be an integer or None")

    if not isinstance(verbose, bool):
        raise TypeError("verbose must be True or False")

    if thresholds is not None and not isinstance(thresholds, np.ndarray):
        raise TypeError("thresholds must be an ndarray or None")

    if seed is not None:
        np.random.seed(seed)

    # convert once: X @ weights with an integer X (e.g. output of read_csv) is ~6x slower than with float
    X = X.astype(float)

    n_genes = X.shape[1] + 1  # +1 for the bias

    if thresholds is None:
        def compute_fitness(chromosomes):
            return binary_fitness(chromosomes, X, y)
    else:
        thresholds = np.asarray(thresholds)

        def compute_fitness(chromosomes):
            return ordinal_fitness(chromosomes, X, y, thresholds)

    population = create_chromosomes(n_chromosomes, n_genes, gene_min, gene_max)

    best_chromosome = None
    best_fitness = -np.inf
    best_accuracy = 0.0
    final_generation = 0
    history = {"best": [], "mean": [], "accuracy": []}

    for generation in range(n_generations):
        fitnesses = compute_fitness(population)
        proportions = fitness_proportions(fitnesses)

        # best of the current generation
        best_index = int(np.argmax(fitnesses))
        current_fitness = float(fitnesses[best_index])

        # update the overall best, if it improved
        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_chromosome = population[best_index].copy()

            # accuracy = hits / total, only recomputed when the best changes
            y_pred = predict({"chromosome": best_chromosome, "thresholds": thresholds}, X)
            best_accuracy = float(np.mean(y_pred == y))

        final_generation = generation + 1
        history["best"].append(best_fitness)
        history["mean"].append(float(fitnesses.mean()))
        history["accuracy"].append(best_accuracy)

        if verbose:
            print(f"Generation {final_generation:>4} | best fitness: {best_fitness:.4f} | accuracy: {best_accuracy:.2%}")

        # stopping criterion: target fitness
        if best_fitness >= target_fitness:
            if verbose:
                print(f"\nTarget fitness {target_fitness} reached at generation {final_generation}.")
            break

        # no offspring in the last generation: they would never compete with the best
        if final_generation == n_generations:
            break

        # selection -> crossover -> mutation -> fitness -> replacement
        parent1, parent2 = select_parents_roulette(population, proportions)
        offspring = one_point_crossover(parent1, parent2, n_offspring)
        offspring = mutate(offspring, n_mutated_genes)
        offspring_fitnesses = compute_fitness(offspring)
        population = update_population(
            population, fitnesses,
            offspring, offspring_fitnesses,
            n_replacements,
        )

    return {
        "chromosome": best_chromosome,
        "fitness": best_fitness,
        "accuracy": best_accuracy,
        "generation": final_generation,
        "thresholds": thresholds,
        "history": history,
    }


def predict(model: dict, X: np.ndarray) -> np.ndarray:
    """
    Use the best chromosome found by train() to classify new data.

    Parameters
    ----------
    model : dict
        Model returned by train(). Needs the keys ``"chromosome"`` and ``"thresholds"``.
    X : np.ndarray of shape (n_features,) or (n_samples, n_features)
        One sample (1D) or several samples (2D), numbers only, with the same
        features used in training.

    Returns
    -------
    y_pred : np.ndarray of shape (n_samples,), dtype int
        Predicted class of each sample (shape (1,) when X is a single sample).

    Examples
    --------
    >>> y_pred = predict(model, X_test)
    """

    if not isinstance(model, dict):
        raise TypeError("model must be a dict (returned by train())")

    if "chromosome" not in model:
        raise ValueError("invalid model: missing the 'chromosome' key")

    if "thresholds" not in model:
        raise ValueError("invalid model: missing the 'thresholds' key")

    if model["chromosome"] is None:
        raise ValueError("model was not trained (chromosome is None)")

    if not isinstance(model["chromosome"], np.ndarray):
        raise TypeError("model['chromosome'] must be an ndarray")

    if model["thresholds"] is not None and not isinstance(model["thresholds"], np.ndarray):
        raise TypeError("model['thresholds'] must be an ndarray or None")

    if not isinstance(X, np.ndarray):
        raise TypeError("X must be an ndarray")

    if X.ndim not in (1, 2):
        raise ValueError(f"X must be 1D (one sample) or 2D (several samples), got {X.ndim}D")

    _check_features(X)

    chromosome = model["chromosome"]
    thresholds = model["thresholds"]

    bias = chromosome[0]
    weights = chromosome[1:]

    # a single sample (1D) becomes a (1, n_features) matrix so it works the same way
    if X.ndim == 1:
        X = X.reshape(1, -1)

    # the number of features must match the chromosome
    if X.shape[1] != len(weights):
        raise ValueError(
            f"X has {X.shape[1]} features, but the chromosome was trained with {len(weights)} features"
        )

    scores = np.dot(X, weights) + bias

    if thresholds is None:
        return np.where(scores >= 0, 1, 0)
    return np.digitize(scores, thresholds)
