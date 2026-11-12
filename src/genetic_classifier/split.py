"""Splitting the data into training and test sets."""

import numpy as np


def _n_train(n_samples: int, train_size: float) -> int:
    """
    Compute how many samples go to the training set, keeping at least 1 in
    training and, when there are 2 or more samples, at least 1 in test.

    Parameters
    ----------
    n_samples : int
        How many samples to split.
    train_size : float
        Fraction that goes to training, between 0 and 1.

    Returns
    -------
    n_train : int
        Number of samples in the training set.
    """
    n = int(round(n_samples * train_size))
    if n_samples >= 2:
        return min(max(n, 1), n_samples - 1)
    return n_samples


def train_test_split(X: np.ndarray, y: np.ndarray, train_size: float = 0.8,
    stratify: bool = True, seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the data into a training set and a test set, shuffling before
    cutting. With stratify=True the split is done inside each class, so every
    class shows up in both sets in the same proportion as in the original data.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Data. Needs at least 2 samples.
    y : np.ndarray of shape (n_samples,)
        Class of each sample.
    train_size : float, default=0.8
        Fraction of the data that goes to training, strictly between 0 and 1.
    stratify : bool, default=True
        If True, keeps the proportion of each class in both sets.
    seed : int or None, default=None
        Random seed for reproducibility (None = different split every run).
        Uses its own generator, so it does not change np.random's global state.

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple of 4 np.ndarray
        X_train has shape (n_train, n_features), X_test (n_test, n_features),
        y_train (n_train,) and y_test (n_test,).

    Examples
    --------
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, seed=0)
    """

    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be ndarrays")

    if X.ndim != 2:
        raise ValueError(f"X must be a 2D matrix (n_samples, n_features), got {X.ndim} dimensions")

    if y.ndim != 1:
        raise ValueError(f"y must be a 1D vector, got {y.ndim} dimensions")

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} samples, but y has {y.shape[0]} values")

    if X.shape[0] < 2:
        raise ValueError("at least 2 samples are needed to split into training and test sets")

    if not isinstance(train_size, (int, float)) or isinstance(train_size, bool):
        raise TypeError("train_size must be a number")

    if train_size <= 0 or train_size >= 1:
        raise ValueError(f"train_size must be between 0 and 1 (exclusive), got {train_size}")

    if not isinstance(stratify, bool):
        raise TypeError("stratify must be True or False")

    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise TypeError("seed must be an integer or None")

    # own generator, does not touch the global np.random
    rng = np.random.default_rng(seed)

    if stratify:
        # groups = list with the indices of each class
        groups = [np.where(y == label)[0] for label in np.unique(y)]
    else:
        groups = [np.arange(len(y))]

    train_indices = []
    test_indices = []
    for group in groups:
        shuffled = rng.permutation(group)
        cut = _n_train(len(shuffled), train_size)
        train_indices.extend(shuffled[:cut])
        test_indices.extend(shuffled[cut:])

    # shuffle again so the classes are not grouped in sequence
    train_indices = rng.permutation(np.array(train_indices, dtype=int))
    test_indices = rng.permutation(np.array(test_indices, dtype=int))

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
