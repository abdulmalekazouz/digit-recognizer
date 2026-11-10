"""Tests for digitrecognizer.model."""

import pytest

from digitrecognizer.data import load_digit_dataset
from digitrecognizer.model import get_or_train_model, load_model, save_model, train_model


@pytest.fixture(scope="module")
def dataset():
    return load_digit_dataset()


@pytest.fixture(scope="module")
def trained_model(dataset):
    return train_model(dataset)


def test_train_model_achieves_high_accuracy_on_held_out_data(dataset, trained_model):
    accuracy = trained_model.score(dataset.x_test, dataset.y_test)
    # The classic sklearn SVC digits example reaches ~97-99% accuracy;
    # require a looser bound so the test isn't brittle to minor changes.
    assert accuracy > 0.9


def test_save_and_load_round_trips_predictions(tmp_path, dataset, trained_model):
    model_path = tmp_path / "model.joblib"
    save_model(trained_model, model_path)

    loaded = load_model(model_path)
    original_predictions = trained_model.predict(dataset.x_test)
    loaded_predictions = loaded.predict(dataset.x_test)
    assert (original_predictions == loaded_predictions).all()


def test_load_model_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "does_not_exist.joblib")


def test_get_or_train_model_trains_once_and_reuses_saved_file(tmp_path):
    model_path = tmp_path / "model.joblib"
    assert not model_path.exists()

    first = get_or_train_model(model_path)
    assert model_path.exists()

    second = get_or_train_model(model_path)
    # Both come from the same saved file, so predictions must agree.
    dataset = load_digit_dataset()
    assert (first.predict(dataset.x_test) == second.predict(dataset.x_test)).all()
