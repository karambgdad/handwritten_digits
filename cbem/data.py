"""Read the repository's label-first MNIST CSV format without hidden preprocessing."""

from pathlib import Path

import numpy as np


def load_mnist_csv(path):
    """Load a headerless CSV: label, pixel_0, ..., pixel_783.

    Labels stay integers 0..9. Divide pixels by the known maximum 255 to put
    them in [0, 1]. This fixed transformation learns nothing from test data.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Dataset does not exist: {path}")
    rows = np.loadtxt(path, delimiter=",", ndmin=2)
    if rows.shape[0] == 0 or rows.shape[1] != 785:
        raise ValueError(f"{path}: expected nonempty rows with 785 columns.")
    if not np.isfinite(rows).all():
        raise ValueError(f"{path}: values must be finite.")
    labels, pixels = rows[:, 0], rows[:, 1:]
    if np.any(labels != np.floor(labels)) or np.any((labels < 0) | (labels > 9)):
        raise ValueError(f"{path}: labels must be integers from 0 to 9.")
    if np.any((pixels < 0) | (pixels > 255)):
        raise ValueError(f"{path}: pixels must be between 0 and 255.")
    return pixels / 255.0, labels.astype(np.int64)


def stratified_split(X, y, fraction=0.2, seed=42):
    """Reserve validation examples within each class, using training data only.

    A class needs at least two examples to appear in both partitions. With
    tiny data, rounding means the actual validation fraction may differ.
    """
    if not 0 < fraction < 1 or len(X) != len(y):
        raise ValueError("Require matching X/y lengths and 0 < fraction < 1.")
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for label in np.unique(y):
        indices = rng.permutation(np.flatnonzero(y == label))
        if len(indices) < 2:
            raise ValueError(f"Class {label} needs at least 2 examples for a split.")
        count = min(len(indices) - 1, max(1, round(len(indices) * fraction)))
        validation.extend(indices[:count])
        train.extend(indices[count:])
    if not train or not validation:
        raise ValueError("Cannot split an empty dataset.")
    train, validation = rng.permutation(train), rng.permutation(validation)
    return X[train], y[train], X[validation], y[validation]
