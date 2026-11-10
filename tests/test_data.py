"""Tests for digitrecognizer.data."""

from digitrecognizer.data import load_digit_dataset


def test_load_digit_dataset_shapes_are_consistent():
    dataset = load_digit_dataset(test_size=0.2, random_state=42)
    assert dataset.x_train.shape[1] == 64
    assert dataset.x_test.shape[1] == 64
    assert dataset.x_train.shape[0] == dataset.y_train.shape[0]
    assert dataset.x_test.shape[0] == dataset.y_test.shape[0]


def test_load_digit_dataset_covers_all_ten_digits():
    dataset = load_digit_dataset()
    assert set(dataset.y_train.tolist()) | set(dataset.y_test.tolist()) == set(range(10))


def test_load_digit_dataset_is_reproducible_for_a_fixed_seed():
    first = load_digit_dataset(random_state=7)
    second = load_digit_dataset(random_state=7)
    assert (first.x_train == second.x_train).all()
    assert (first.y_test == second.y_test).all()
