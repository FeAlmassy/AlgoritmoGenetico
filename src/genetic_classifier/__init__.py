"""genetic-classifier: linear classifier (binary and ordinal) trained by a genetic algorithm (Prandiano)."""

from .core import predict, train
from .evaluation import confusion_matrix, evaluate
from .reading import read_csv, read_excel
from .split import train_test_split

__all__ = ["train", "predict", "evaluate", "confusion_matrix", "read_excel", "read_csv", "train_test_split"]

__version__ = "3.0.0"
