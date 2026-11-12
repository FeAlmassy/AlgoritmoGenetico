"""
Reading of standardized datasets (.xlsx and .csv).

Expected layout of the spreadsheet (see examples/template/template.xlsx):
    first column -> identifier of each sample (lessons)
    middle columns -> features
    last column -> class (new data, without labels, does not have this column)
"""

import numpy as np
import pandas as pd


def _split_columns(df: pd.DataFrame, has_labels: bool = True) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    Split a table into features and classes, ignoring the first column.

    Parameters
    ----------
    df : pd.DataFrame
        Table read by pandas, in the layout described at the top of this module.
    has_labels : bool, default=True
        If True, the last column is the class. If False, every column after the
        first one is a feature.

    Returns
    -------
    X, y : tuple (np.ndarray, np.ndarray), when has_labels=True
        X has shape (n_samples, n_features) with the features and y has shape
        (n_samples,) with the class of each sample.
    X : np.ndarray of shape (n_samples, n_features), when has_labels=False
        Only the features.
    """
    if not isinstance(has_labels, bool):
        raise TypeError("has_labels must be True or False")

    if not has_labels:
        return df.iloc[:, 1:].to_numpy()

    X = df.iloc[:, 1:-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()
    return X, y


def read_csv(path: str, has_labels: bool = True) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    Read a .csv file and split the features from the classes.
    Ignores the first column and, by default, uses the last one as the class.

    Parameters
    ----------
    path : str
        Path to the .csv file.
    has_labels : bool, default=True
        If True, the last column is the class. Use False for new data that has
        no class column.

    Returns
    -------
    X, y : tuple (np.ndarray, np.ndarray), when has_labels=True
        X has shape (n_samples, n_features) with the features and y has shape
        (n_samples,) with the class of each sample.
    X : np.ndarray of shape (n_samples, n_features), when has_labels=False
        Only the features.

    Examples
    --------
    >>> X, y = read_csv("data.csv")
    >>> X_new = read_csv("new_data.csv", has_labels=False)
    """
    return _split_columns(pd.read_csv(path), has_labels)


def read_excel(path: str, has_labels: bool = True) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    Read an .xlsx file (first sheet) and split the features from the classes.
    Ignores the first column and, by default, uses the last one as the class.

    Parameters
    ----------
    path : str
        Path to the .xlsx file.
    has_labels : bool, default=True
        If True, the last column is the class. Use False for new data that has
        no class column.

    Returns
    -------
    X, y : tuple (np.ndarray, np.ndarray), when has_labels=True
        X has shape (n_samples, n_features) with the features and y has shape
        (n_samples,) with the class of each sample.
    X : np.ndarray of shape (n_samples, n_features), when has_labels=False
        Only the features.

    Examples
    --------
    >>> X, y = read_excel("data.xlsx")
    >>> X_new = read_excel("new_data.xlsx", has_labels=False)
    """
    return _split_columns(pd.read_excel(path), has_labels)
