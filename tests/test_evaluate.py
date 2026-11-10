"""Tests for digitrecognizer.evaluate."""

import pytest

from digitrecognizer.data import load_digit_dataset
from digitrecognizer.evaluate import evaluate_model, format_report
from digitrecognizer.model import train_model


@pytest.fixture(scope="module")
def dataset():
    return load_digit_dataset()


@pytest.fixture(scope="module")
def report(dataset):
    model = train_model(dataset)
    return evaluate_model(model, dataset)


def test_accuracy_matches_model_score(dataset, report):
    model = train_model(dataset)
    assert report.accuracy == pytest.approx(model.score(dataset.x_test, dataset.y_test), abs=1e-6)


def test_accuracy_is_high_for_this_well_separated_dataset(report):
    assert report.accuracy > 0.9


def test_confusion_matrix_is_ten_by_ten_and_sums_to_test_set_size(dataset, report):
    matrix = report.confusion_matrix
    assert len(matrix) == 10
    assert all(len(row) == 10 for row in matrix)
    assert sum(sum(row) for row in matrix) == len(dataset.y_test)


def test_confusion_matrix_diagonal_dominates_for_a_good_classifier(report):
    matrix = report.confusion_matrix
    diagonal_sum = sum(matrix[i][i] for i in range(10))
    total = sum(sum(row) for row in matrix)
    assert diagonal_sum / total > 0.9


def test_per_class_metrics_cover_all_ten_digits(report):
    assert set(report.per_class.keys()) == set(range(10))
    for metrics in report.per_class.values():
        assert 0.0 <= metrics.precision <= 1.0
        assert 0.0 <= metrics.recall <= 1.0
        assert 0.0 <= metrics.f1_score <= 1.0
        assert metrics.support > 0


def test_to_dict_is_json_serializable(report):
    import json

    serialized = json.dumps(report.to_dict())
    assert '"accuracy"' in serialized
    assert '"per_class"' in serialized


def test_format_report_includes_header_and_every_digit(report):
    text = format_report(report)
    assert "Accuracy" in text
    assert "Macro F1" in text
    for digit in range(10):
        assert str(digit) in text
