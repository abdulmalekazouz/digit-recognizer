"""Model evaluation metrics.

Wraps scikit-learn's standard classification metrics (accuracy, a
per-class precision/recall/F1 breakdown, and a confusion matrix) into one
structured, JSON-serializable report.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sklearn.metrics import confusion_matrix

from digitrecognizer.data import DigitDataset
from digitrecognizer.model import Classifier


@dataclass(frozen=True)
class ClassMetrics:
    """Precision/recall/F1 for a single digit class.

    Attributes:
        precision: Of the samples predicted as this digit, the fraction
            that were actually this digit.
        recall: Of the samples that were actually this digit, the
            fraction predicted as this digit.
        f1_score: The harmonic mean of precision and recall.
        support: The number of test samples with this true label.
    """

    precision: float
    recall: float
    f1_score: float
    support: int


@dataclass(frozen=True)
class EvaluationReport:
    """A full evaluation of a model against a held-out test set.

    Attributes:
        accuracy: Overall fraction of test samples classified correctly.
        macro_f1: The unweighted mean of each class's F1 score, treating
            every digit class as equally important regardless of how many
            test samples it has.
        per_class: Maps each digit label (0-9) to its `ClassMetrics`.
        confusion_matrix: A 10x10 matrix where `confusion_matrix[i][j]` is
            the number of samples whose true label is `i` and predicted
            label is `j`. A perfect classifier has nonzero values only on
            the diagonal.
    """

    accuracy: float
    macro_f1: float
    per_class: dict[int, ClassMetrics] = field(default_factory=dict)
    confusion_matrix: list[list[int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Converts this report into a plain, JSON-serializable dict.

        Returns:
            A dict with the same information as this dataclass, with
            `per_class` keyed by string digit labels (JSON object keys
            must be strings) and `ClassMetrics` expanded into plain dicts.
        """
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "per_class": {
                str(label): {
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1_score": metrics.f1_score,
                    "support": metrics.support,
                }
                for label, metrics in self.per_class.items()
            },
            "confusion_matrix": self.confusion_matrix,
        }


def evaluate_model(model: Classifier, dataset: DigitDataset) -> EvaluationReport:
    """Evaluates a trained model against a dataset's held-out test split.

    Args:
        model: A fitted classifier (as returned by
            `digitrecognizer.model.train_model`).
        dataset: The dataset to evaluate against; only `dataset.x_test`
            and `dataset.y_test` are used.

    Returns:
        An `EvaluationReport` summarizing accuracy, per-class metrics, and
        the confusion matrix.
    """
    predictions = model.predict(dataset.x_test)

    labels = sorted(set(dataset.y_test.tolist()))
    matrix = confusion_matrix(dataset.y_test, predictions, labels=labels)

    per_class: dict[int, ClassMetrics] = {}
    f1_scores: list[float] = []
    for i, label in enumerate(labels):
        true_positives = matrix[i, i]
        predicted_positives = matrix[:, i].sum()
        actual_positives = matrix[i, :].sum()

        precision = true_positives / predicted_positives if predicted_positives else 0.0
        recall = true_positives / actual_positives if actual_positives else 0.0
        f1_score = (
            2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        )

        per_class[label] = ClassMetrics(
            precision=precision, recall=recall, f1_score=f1_score, support=int(actual_positives)
        )
        f1_scores.append(f1_score)

    accuracy = (predictions == dataset.y_test).mean()
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    return EvaluationReport(
        accuracy=float(accuracy),
        macro_f1=float(macro_f1),
        per_class=per_class,
        confusion_matrix=matrix.tolist(),
    )


def format_report(report: EvaluationReport) -> str:
    """Renders an `EvaluationReport` as a human-readable text table.

    Args:
        report: The report to render.

    Returns:
        A multi-line string: overall accuracy and macro F1, followed by a
        per-class precision/recall/F1/support table.
    """
    lines = [
        f"Accuracy: {report.accuracy:.4f}",
        f"Macro F1: {report.macro_f1:.4f}",
        "",
        f"{'digit':>5}  {'precision':>9}  {'recall':>9}  {'f1':>9}  {'support':>7}",
    ]
    for label in sorted(report.per_class):
        metrics = report.per_class[label]
        lines.append(
            f"{label:>5}  {metrics.precision:>9.3f}  {metrics.recall:>9.3f}  "
            f"{metrics.f1_score:>9.3f}  {metrics.support:>7}"
        )
    return "\n".join(lines)
