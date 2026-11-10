"""Dataset loading utilities.

Wraps scikit-learn's bundled digits dataset (1,797 hand-written digit
samples, 8x8 grayscale pixels each) so the rest of the package has one
place to go for training/test data, instead of depending directly on
`sklearn.datasets` everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Pixel intensities in the digits dataset range from 0 (background) to this
# value (brightest stroke pixel).
PIXEL_MAX_VALUE = 16

# Each sample is an 8x8 grayscale image.
IMAGE_SIZE = 8


@dataclass(frozen=True)
class DigitDataset:
    """A train/test split of the digits dataset.

    Attributes:
        x_train: Training features, shape (n_train, 64), one row per
            flattened 8x8 image, values in [0, PIXEL_MAX_VALUE].
        x_test: Test features, same shape convention as `x_train`.
        y_train: Training labels, shape (n_train,), integers in [0, 9].
        y_test: Test labels, same shape convention as `y_train`.
    """

    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def load_digit_dataset(test_size: float = 0.2, random_state: int = 42) -> DigitDataset:
    """Loads the digits dataset and splits it into train/test sets.

    Args:
        test_size: Fraction of samples to hold out for testing.
        random_state: Seed controlling the train/test split, so results are
            reproducible across runs.

    Returns:
        A `DigitDataset` with stratified train/test splits (stratified by
        digit label, so each split has a roughly even class distribution).
    """
    digits = load_digits()
    x_train, x_test, y_train, y_test = train_test_split(
        digits.data,
        digits.target,
        test_size=test_size,
        random_state=random_state,
        stratify=digits.target,
    )
    return DigitDataset(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
