"""
DRAFT (to be deleted): how the library would look with classes.

Idea: the step functions (create_chromosomes, mutate, ...) stay as they are,
because they are pure functions (arrays in -> arrays out). Only the *model*
becomes a class, following scikit-learn conventions without depending on it:

    - __init__ only stores the hyperparameters (no checks, no computation)
    - fit() validates, trains and returns self
    - what was learned ends with "_" (coef_, intercept_, history_ ...)
    - predict() and score()

So, later, plugging into scikit-learn is just changing
    class _GeneticLinearModel:
to
    class _GeneticLinearModel(BaseEstimator):
and deleting get_params / set_params.

Run it from the project root:
    python classes_draft.py
"""

import inspect
import sys

import numpy as np

sys.path.insert(0, "src")  # uses the library from this folder

from genetic_classifier.core import _check_features
from genetic_classifier.creation import create_chromosomes
from genetic_classifier.crossover import one_point_crossover
from genetic_classifier.fitness import binary_fitness, fitness_proportions, ordinal_fitness
from genetic_classifier.mutation import mutate
from genetic_classifier.selection import select_parents_roulette
from genetic_classifier.update import update_population


# ---------------------------------------------------------------------------
# Base class: the genetic algorithm loop, written only once
# ---------------------------------------------------------------------------

class _GeneticLinearModel:
    """
    Base of every model: a linear score (bias + weights · features) whose
    weights are found by a genetic algorithm. Not used directly: the subclasses
    say how to score a chromosome (_fitness) and how to turn the linear score
    into a prediction (_scores_to_prediction).

    Parameters
    ----------
    n_chromosomes : int, default=30
        Population size.
    n_mutated_genes : int, default=2
        How many genes change in each child during mutation.
    n_offspring : int, default=16
        How many children are created per generation.
    n_replacements : int or None, default=None
        How many of the worst chromosomes are swapped for the best children.
        None = half of n_offspring.
    n_generations : int, default=1000
        Maximum number of generations.
    target_fitness : float, default=1.0
        Stops when the best fitness reaches this value, between 0 and 1.
    gene_min : float, default=-1.0
        Smallest gene value in the initial population.
    gene_max : float, default=1.0
        Largest gene value in the initial population.
    seed : int or None, default=None
        Random seed for reproducibility.
    verbose : bool, default=False
        If True, prints the best fitness of every generation.

    Attributes (exist only after fit)
    ----------
    coef_ : np.ndarray of shape (n_features,)
        Weight of each feature.
    intercept_ : float
        Bias.
    fitness_ : float
        Fitness of the best chromosome on the training data.
    n_generations_run_ : int
        Generation where training stopped.
    history_ : dict of lists of float
        "best" and "mean" fitness per generation.
    n_features_in_ : int
        Number of features seen in fit.
    """

    def __init__(self, n_chromosomes=30, n_mutated_genes=2, n_offspring=16, n_replacements=None,
                 n_generations=1000, target_fitness=1.0, gene_min=-1.0, gene_max=1.0,
                 seed=None, verbose=False):
        # sklearn convention: only store, with the same names. No checks here.
        self.n_chromosomes = n_chromosomes
        self.n_mutated_genes = n_mutated_genes
        self.n_offspring = n_offspring
        self.n_replacements = n_replacements
        self.n_generations = n_generations
        self.target_fitness = target_fitness
        self.gene_min = gene_min
        self.gene_max = gene_max
        self.seed = seed
        self.verbose = verbose

    # --- what each subclass must define -----------------------------------

    def _fitness(self, chromosomes, X, y):
        """
        Score each chromosome. Defined by each subclass.

        Parameters
        ----------
        chromosomes : np.ndarray of shape (n_chromosomes, n_features + 1)
            Population to score (bias first).
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray of shape (n_samples,)
            Training targets.

        Returns
        -------
        fitnesses : np.ndarray of shape (n_chromosomes,)
            Fitness of each chromosome, >= 0, higher is better.
        """
        raise NotImplementedError

    def _scores_to_prediction(self, scores):
        """
        Turn the linear scores into the final prediction. Defined by each subclass.

        Parameters
        ----------
        scores : np.ndarray of shape (n_samples,)
            bias + weights · features of each sample.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Prediction of each sample.
        """
        raise NotImplementedError

    # --- shared code --------------------------------------------------------

    def fit(self, X, y):
        """
        Train the model with the genetic algorithm.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data, numbers only.
        y : np.ndarray of shape (n_samples,)
            Training targets (classes or values, depending on the model).

        Returns
        -------
        self : the model itself
            Already trained, so calls can be chained: model.fit(X, y).predict(X).

        Examples
        --------
        >>> clf = GeneticClassifier(seed=0).fit(X_train, y_train)
        """
        X, y = self._validate_data(X, y)
        n_replacements = self._validate_params(X)

        if self.seed is not None:
            np.random.seed(self.seed)

        X, y = self._prepare(X, y)  # hook: the regressor standardizes here

        population = create_chromosomes(self.n_chromosomes, X.shape[1] + 1, self.gene_min, self.gene_max)
        best_chromosome, best_fitness = None, -np.inf
        history = {"best": [], "mean": []}

        for generation in range(1, self.n_generations + 1):
            fitnesses = self._fitness(population, X, y)

            best_index = int(np.argmax(fitnesses))
            if fitnesses[best_index] > best_fitness:
                best_fitness = float(fitnesses[best_index])
                best_chromosome = population[best_index].copy()

            history["best"].append(best_fitness)
            history["mean"].append(float(fitnesses.mean()))

            if self.verbose:
                print(f"Generation {generation:>4} | best fitness: {best_fitness:.4f}")

            if best_fitness >= self.target_fitness or generation == self.n_generations:
                break

            parent1, parent2 = select_parents_roulette(population, fitness_proportions(fitnesses))
            offspring = mutate(one_point_crossover(parent1, parent2, self.n_offspring), self.n_mutated_genes)
            population = update_population(population, fitnesses, offspring,
                                           self._fitness(offspring, X, y), n_replacements)

        # learned attributes: end with "_"
        self.intercept_, self.coef_ = self._finish(best_chromosome[0], best_chromosome[1:])
        self.fitness_ = best_fitness
        self.n_generations_run_ = generation
        self.history_ = history
        self.n_features_in_ = X.shape[1]
        return self

    def decision_function(self, X):
        """
        Compute the linear score of each sample: intercept_ + coef_ · features.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features) or (n_features,)
            Several samples (2D) or one sample (1D).

        Returns
        -------
        scores : np.ndarray of shape (n_samples,), dtype float
            Linear score of each sample.
        """
        self._check_is_fitted()
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be an ndarray")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        _check_features(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but the model was trained with {self.n_features_in_}")
        return X.astype(float) @ self.coef_ + self.intercept_

    def predict(self, X):
        """
        Predict the target of each sample.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features) or (n_features,)
            Several samples (2D) or one sample (1D).

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Class (int) for the classifier, value (float) for the regressor.

        Examples
        --------
        >>> y_pred = clf.predict(X_test)
        """
        return self._scores_to_prediction(self.decision_function(X))

    # --- sklearn-style helpers (BaseEstimator gives these for free later) ---

    def get_params(self):
        """
        Return the hyperparameters passed to __init__.

        Returns
        -------
        params : dict
            Name -> value of each hyperparameter.
        """
        names = [p for p in inspect.signature(type(self).__init__).parameters if p != "self"]
        return {name: getattr(self, name) for name in names}

    def set_params(self, **params):
        """
        Change hyperparameters after creating the model (retrain with fit afterwards).

        Parameters
        ----------
        **params : any
            Hyperparameters to change, e.g. n_generations=500.

        Returns
        -------
        self : the model itself
        """
        valid = self.get_params()
        for name, value in params.items():
            if name not in valid:
                raise ValueError(f"invalid parameter {name!r} for {type(self).__name__}")
            setattr(self, name, value)
        return self

    def __repr__(self):
        """
        Text shown when printing the model: only the hyperparameters that differ from the default.

        Returns
        -------
        text : str
            E.g. "GeneticClassifier(n_generations=500, seed=0)".
        """
        defaults = {p.name: p.default for p in inspect.signature(type(self).__init__).parameters.values()}
        changed = [f"{k}={v!r}" for k, v in self.get_params().items() if not np.array_equal(v, defaults[k])]
        return f"{type(self).__name__}({', '.join(changed)})"

    # --- internal steps ----------------------------------------------------

    def _prepare(self, X, y):
        """
        Transform the data before training. Default: nothing.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)

        Returns
        -------
        X, y : tuple (np.ndarray, np.ndarray)
            Data used by the algorithm.
        """
        return X, y

    def _finish(self, bias, weights):
        """
        Convert the best chromosome into intercept_ and coef_. Default: as is.

        Parameters
        ----------
        bias : float
            Gene 0 of the best chromosome.
        weights : np.ndarray of shape (n_features,)
            Remaining genes of the best chromosome.

        Returns
        -------
        intercept, coef : tuple (float, np.ndarray of shape (n_features,))
        """
        return float(bias), weights

    def _validate_data(self, X, y):
        """
        Check X and y and convert X to float.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)

        Returns
        -------
        X, y : tuple (np.ndarray, np.ndarray)
            X as float, y unchanged.
        """
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X and y must be ndarrays")
        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("X must be 2D (n_samples, n_features) and y must be 1D (n_samples,)")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} samples, but y has {y.shape[0]} values")
        _check_features(X)
        return X.astype(float), y

    def _validate_params(self, X):
        """
        Check the hyperparameters (done in fit, not in __init__, as in sklearn).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data, used to know the number of genes.

        Returns
        -------
        n_replacements : int
            The value to use (resolves None to half of n_offspring).
        """
        for name in ("n_chromosomes", "n_offspring", "n_generations"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
        if not 0 <= self.n_mutated_genes <= X.shape[1] + 1:
            raise ValueError(f"n_mutated_genes must be between 0 and {X.shape[1] + 1}")
        if self.gene_min >= self.gene_max:
            raise ValueError("gene_min must be smaller than gene_max")
        n_replacements = self.n_replacements
        if n_replacements is None:
            n_replacements = min(max(self.n_offspring // 2, 1), self.n_chromosomes)
        if not 0 <= n_replacements <= min(self.n_chromosomes, self.n_offspring):
            raise ValueError("n_replacements must be between 0 and min(n_chromosomes, n_offspring)")
        return n_replacements

    def _check_is_fitted(self):
        """
        Raise an error if predict is called before fit.

        Returns
        -------
        None
        """
        if not hasattr(self, "coef_"):
            raise RuntimeError(f"this {type(self).__name__} is not fitted yet, call fit(X, y) first")


# ---------------------------------------------------------------------------
# Classifier: what the library already does today
# ---------------------------------------------------------------------------

class GeneticClassifier(_GeneticLinearModel):
    """
    Linear classifier (binary or ordinal) trained by a genetic algorithm.
    Fitness = product of the recalls of every class.

    Parameters
    ----------
    thresholds : np.ndarray of shape (K-1,) or None, default=None
        None for 2 classes (binary), or K-1 cut points in increasing order for
        K ordered classes (ordinal).
    **others
        Same as _GeneticLinearModel (n_chromosomes, n_generations, seed, ...).

    Examples
    --------
    >>> clf = GeneticClassifier(seed=0)
    >>> clf.fit(X_train, y_train)
    >>> y_pred = clf.predict(X_test)
    >>> accuracy = clf.score(X_test, y_test)
    """

    def __init__(self, thresholds=None, n_chromosomes=30, n_mutated_genes=2, n_offspring=16,
                 n_replacements=None, n_generations=1000, target_fitness=1.0,
                 gene_min=-1.0, gene_max=1.0, seed=None, verbose=False):
        super().__init__(n_chromosomes, n_mutated_genes, n_offspring, n_replacements, n_generations,
                         target_fitness, gene_min, gene_max, seed, verbose)
        self.thresholds = thresholds

    def _fitness(self, chromosomes, X, y):
        if self.thresholds is None:
            return binary_fitness(chromosomes, X, y)
        return ordinal_fitness(chromosomes, X, y, np.asarray(self.thresholds))

    def _scores_to_prediction(self, scores):
        if self.thresholds is None:
            return np.where(scores >= 0, 1, 0)
        return np.digitize(scores, self.thresholds)

    def score(self, X, y):
        """
        Accuracy of the model on (X, y).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data.
        y : np.ndarray of shape (n_samples,)
            True classes.

        Returns
        -------
        accuracy : float
            Fraction of samples predicted correctly, from 0 to 1.
        """
        return float(np.mean(self.predict(X) == y))


# ---------------------------------------------------------------------------
# Regressor: the new part
# ---------------------------------------------------------------------------

class GeneticRegressor(_GeneticLinearModel):
    """
    Linear regression trained by a genetic algorithm.
    Fitness = 1 / (1 + MSE), so it stays between 0 and 1 and higher is better.

    X and y are standardized (mean 0, std 1) before training, because the genes
    live around [-1, 1] and mutation only draws values in that range. At the
    end, coef_ and intercept_ are converted back to the original scale.

    Parameters
    ----------
    **others
        Same as _GeneticLinearModel (n_chromosomes, n_generations, seed, ...).

    Examples
    --------
    >>> reg = GeneticRegressor(n_generations=3000, seed=0)
    >>> reg.fit(X_train, y_train)
    >>> y_pred = reg.predict(X_test)
    >>> r2 = reg.score(X_test, y_test)
    """

    def _fitness(self, chromosomes, X, y):
        scores = X @ chromosomes[:, 1:].T + chromosomes[:, 0]  # (n_samples, n_chromosomes)
        mse = ((scores - y[:, None]) ** 2).mean(axis=0)
        return 1 / (1 + mse)

    def _scores_to_prediction(self, scores):
        return scores  # in regression the score is already the answer

    def _prepare(self, X, y):
        y = y.astype(float)
        self._x_mean, self._x_std = X.mean(axis=0), X.std(axis=0)
        self._x_std[self._x_std == 0] = 1.0  # constant column: avoid division by zero
        self._y_mean, self._y_std = y.mean(), (y.std() or 1.0)
        return (X - self._x_mean) / self._x_std, (y - self._y_mean) / self._y_std

    def _finish(self, bias, weights):
        # undo the standardization: y = y_mean + y_std * (b + sum(w * (x - x_mean) / x_std))
        coef = weights * self._y_std / self._x_std
        intercept = self._y_mean + self._y_std * bias - np.sum(coef * self._x_mean)
        return float(intercept), coef

    def score(self, X, y):
        """
        Coefficient of determination R² of the model on (X, y).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data.
        y : np.ndarray of shape (n_samples,)
            True values.

        Returns
        -------
        r2 : float
            1 = perfect, 0 = as good as always predicting the mean, < 0 = worse than that.
        """
        y_pred = self.predict(X)
        return float(1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from genetic_classifier import read_excel, train, train_test_split

    print("=== GeneticClassifier on binary_data_1.xlsx ===")
    X, y = read_excel("examples/example_1_binary/binary_data_1.xlsx")
    X_train, X_test, y_train, y_test = train_test_split(X, y, seed=0)

    clf = GeneticClassifier(n_generations=300, seed=0)
    print(clf)
    clf.fit(X_train, y_train)
    print("test accuracy:", round(clf.score(X_test, y_test), 4))
    print("stopped at generation:", clf.n_generations_run_)

    # same seed -> same result as the current function API
    model = train(X_train, y_train, n_generations=300, seed=0, verbose=False)
    print("same weights as train():", np.allclose(clf.coef_, model["chromosome"][1:]))

    print("\n=== GeneticRegressor vs least squares (MMQ) on synthetic data ===")
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 10, size=(200, 3))
    true_coef, true_intercept = np.array([2.0, -3.5, 0.8]), 50.0
    y = true_intercept + X @ true_coef + rng.normal(0, 1.0, size=200)

    reg = GeneticRegressor(n_generations=3000, seed=0).fit(X, y)

    # least squares: add a column of 1s for the intercept and solve directly
    X1 = np.column_stack([np.ones(len(X)), X])
    lstsq = np.linalg.lstsq(X1, y, rcond=None)[0]

    print(f"{'':>15}{'intercept':>11}{'coef':>30}")
    print(f"{'true':>15}{true_intercept:>11.3f}   {np.round(true_coef, 3)}")
    print(f"{'least squares':>15}{lstsq[0]:>11.3f}   {np.round(lstsq[1:], 3)}")
    print(f"{'genetic':>15}{reg.intercept_:>11.3f}   {np.round(reg.coef_, 3)}")
    print("R² genetic:", round(reg.score(X, y), 4))
