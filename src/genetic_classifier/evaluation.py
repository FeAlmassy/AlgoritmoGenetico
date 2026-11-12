"""Model evaluation after training."""

import numpy as np


def _labels_to_int(y: np.ndarray, name: str) -> np.ndarray:
    """
    Return the classes as integers. Accepts floats holding whole numbers
    (e.g. 1.0), which is how a class column may come from a spreadsheet.

    Parameters
    ----------
    y : np.ndarray of shape (n_samples,)
        Classes to convert.
    name : str
        Name of the parameter being converted, used in the error message.

    Returns
    -------
    y_int : np.ndarray of shape (n_samples,), dtype int
        The same classes as integers.
    """
    if np.issubdtype(y.dtype, np.integer):
        return y

    if np.issubdtype(y.dtype, np.floating) and np.isfinite(y).all() and np.all(y == np.floor(y)):
        return y.astype(int)

    raise TypeError(f"{name} must contain integers from 0 to K-1 (convert text classes first)")


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
    class_names: list[str] | None = None, verbose: bool = True) -> np.ndarray:
    """
    Build the confusion matrix: each row is the true class, each column is the
    predicted class, and the diagonal holds the correct predictions.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        True class of each sample, integers from 0 to K-1.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted class of each sample (output of predict).
    class_names : list of str or None, default=None
        Name of each class, in order from 0 to K-1 (None = use the numbers).
        It also sets K, so the matrix is complete even if some class does not
        appear in the data.
    verbose : bool, default=True
        If True, prints the matrix.

    Returns
    -------
    matrix : np.ndarray of shape (K, K), dtype int
        matrix[i, j] is how many samples of true class i were predicted as class j.

    Examples
    --------
    >>> matrix = confusion_matrix(y_test, y_pred, class_names=["poisonous", "edible"])
    """

    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("y_true and y_pred must be ndarrays")

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError(f"y_true and y_pred must be 1D vectors, got {y_true.ndim} and {y_pred.ndim} dimensions")

    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true has {len(y_true)} values, but y_pred has {len(y_pred)}")

    if len(y_true) == 0:
        raise ValueError("y_true and y_pred cannot be empty")

    y_true = _labels_to_int(y_true, "y_true")
    y_pred = _labels_to_int(y_pred, "y_pred")

    if class_names is not None and not isinstance(class_names, list):
        raise TypeError("class_names must be a list or None")

    if class_names is not None and len(class_names) == 0:
        raise ValueError("class_names cannot be empty")

    if not isinstance(verbose, bool):
        raise TypeError("verbose must be True or False")

    smallest_class = min(y_true.min(), y_pred.min())
    largest_class = max(y_true.max(), y_pred.max())

    if smallest_class < 0:
        raise ValueError(f"classes must be integers starting at 0, found {smallest_class}")

    if class_names is None:
        K = int(largest_class) + 1
    else:
        K = len(class_names)
        if largest_class >= K:
            raise ValueError(f"class_names has {K} names, but the data has class {largest_class}")

    matrix = np.zeros((K, K), dtype=int)
    for true, predicted in zip(y_true, y_pred):
        matrix[true, predicted] += 1

    if verbose:
        _print_matrix(matrix, _display_names(K, class_names))

    return matrix


def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
    class_names: list[str] | None = None, verbose: bool = True) -> dict:
    """
    Compare the predictions with the true classes: compute the accuracy, the
    fitness, the recall of each class and the confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        True class of each sample, integers from 0 to K-1.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted class of each sample (output of predict).
    class_names : list of str or None, default=None
        Name of each class, in order from 0 to K-1 (None = use the numbers).
        It also sets K, so the result is complete even if some class does not
        appear in the data.
    verbose : bool, default=True
        If True, prints the results.

    Returns
    -------
    results : dict
        Dictionary with the keys:

        - ``"accuracy"`` (float): fraction of samples predicted correctly, 0 to 1.
        - ``"fitness"`` (float): product of the recalls, 0 to 1 (nan if some
          class has no samples).
        - ``"recall"`` (np.ndarray of shape (K,), dtype float): recall of each
          class (nan for a class without samples).
        - ``"confusion_matrix"`` (np.ndarray of shape (K, K), dtype int): same
          as the output of confusion_matrix.

    Examples
    --------
    >>> results = evaluate(y_test, y_pred)
    >>> results["accuracy"]
    """

    if not isinstance(verbose, bool):
        raise TypeError("verbose must be True or False")

    # the other checks are already done by confusion_matrix
    matrix = confusion_matrix(y_true, y_pred, class_names, verbose=False)

    hits = np.diag(matrix)
    totals = matrix.sum(axis=1)

    # a class without samples gets nan: its recall cannot be measured
    recall = np.full(len(matrix), np.nan)
    has_samples = totals > 0
    recall[has_samples] = hits[has_samples] / totals[has_samples]

    accuracy = float(hits.sum() / matrix.sum())
    fitness = float(np.prod(recall))

    if verbose:
        names = _display_names(len(matrix), class_names)

        print(f"accuracy: {accuracy:.2%}")
        if np.isnan(fitness):
            print("fitness: not computed (a class has no samples)")
        else:
            print(f"fitness: {fitness:.4f}")

        print("\nrecall per class:")
        for name, hit, total, value in zip(names, hits, totals, recall):
            if total == 0:
                print(f"  {name}: no samples")
            else:
                print(f"  {name}: {hit}/{total} = {value:.2%}")

        print("\nconfusion matrix:")
        _print_matrix(matrix, names)

    return {
        "accuracy": accuracy,
        "fitness": fitness,
        "recall": recall,
        "confusion_matrix": matrix,
    }


def _display_names(K: int, class_names: list[str] | None) -> list[str]:
    """
    Return the text that represents each class on screen.

    Parameters
    ----------
    K : int
        Number of classes.
    class_names : list of str or None
        Name of each class, or None to use the numbers 0 to K-1.

    Returns
    -------
    names : list of str
        K texts, one per class.
    """
    if class_names is None:
        return [str(k) for k in range(K)]
    return [str(name) for name in class_names]


def _print_matrix(matrix: np.ndarray, names: list[str]) -> None:
    """
    Print the confusion matrix with aligned columns.

    Parameters
    ----------
    matrix : np.ndarray of shape (K, K)
        Confusion matrix.
    names : list of str
        Text that represents each class (K items).

    Returns
    -------
    None
        Only prints.
    """
    title = "true \\ predicted"
    name_width = max(len(title), *(len(name) for name in names))
    column_width = max(*(len(name) for name in names), len(str(matrix.max()))) + 2

    print(f"{title:>{name_width}}" + "".join(f"{name:>{column_width}}" for name in names))
    for name, row in zip(names, matrix):
        print(f"{name:>{name_width}}" + "".join(f"{value:>{column_width}}" for value in row))
