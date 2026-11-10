"""Model training and persistence.

Trains a support vector classifier (the same architecture scikit-learn's
own digits-dataset example uses: an RBF kernel with a small gamma) and
saves/loads it with joblib, so training only has to happen once.
"""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

from digitrecognizer.data import DigitDataset, load_digit_dataset

DEFAULT_MODEL_PATH = Path("digit_model.joblib")

# A calibrated classifier, rather than SVC(probability=True), per
# scikit-learn's own deprecation of that constructor argument: wrapping an
# uncalibrated SVC in CalibratedClassifierCV is the supported way to get
# predict_proba from an SVM going forward.
Classifier = CalibratedClassifierCV


def train_model(dataset: DigitDataset) -> Classifier:
    """Trains a calibrated support vector classifier on a digit dataset.

    Args:
        dataset: The train/test split to fit on. Only the training
            portion (`dataset.x_train`, `dataset.y_train`) is used.

    Returns:
        A fitted `CalibratedClassifierCV` wrapping an RBF-kernel SVC
        (`gamma=0.001`), so predictions can also report a confidence score
        via `predict_proba`.
    """
    base_model = SVC(gamma=0.001, random_state=42)
    model = CalibratedClassifierCV(base_model, ensemble=False)
    model.fit(dataset.x_train, dataset.y_train)
    return model


def save_model(model: Classifier, path: Path | str = DEFAULT_MODEL_PATH) -> None:
    """Persists a trained model to disk.

    Args:
        model: The fitted model to save.
        path: Destination file path.
    """
    joblib.dump(model, path)


def load_model(path: Path | str = DEFAULT_MODEL_PATH) -> Classifier:
    """Loads a previously trained model from disk.

    Args:
        path: Path to a model file previously written by `save_model`.

    Returns:
        The deserialized model.

    Raises:
        FileNotFoundError: If no file exists at `path`.
    """
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"No model found at {model_path}; run training first")
    return joblib.load(model_path)


def get_or_train_model(path: Path | str = DEFAULT_MODEL_PATH) -> Classifier:
    """Loads a model from disk, or trains and saves a new one if missing.

    Convenient for callers (like the web UI) that just want a ready-to-use
    model without caring whether this is the first run.

    Args:
        path: Path to load the model from, and to save a freshly trained
            model to if none exists yet there.

    Returns:
        A fitted model, either loaded from disk or freshly trained.
    """
    model_path = Path(path)
    if model_path.exists():
        return load_model(model_path)

    dataset = load_digit_dataset()
    model = train_model(dataset)
    save_model(model, model_path)
    return model
