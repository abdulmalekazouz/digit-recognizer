# Digit Recognizer

A handwritten digit classifier built on scikit-learn's digits dataset
(1,797 samples of 8x8 grayscale digit images). Includes model training,
evaluation metrics, a prediction CLI, and a canvas-based Flask web UI
where you draw a digit with your mouse and get a live prediction.

Everything runs fully offline on a small, bundled dataset -- no large
downloaded models, no external API, no internet access required.

## Why this project

A small, classic, well-understood classification problem is a clear way
to demonstrate a full ML workflow end to end -- data loading, training,
evaluation, a saved/reloadable model, and a real inference path from raw
user input (mouse strokes on a canvas) to a prediction. Built
incrementally, one feature per branch/pull request.

## Installation

Requires Python 3.9+.

```bash
python -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

## Features

- **Model training.** Trains a calibrated support vector classifier (RBF
  kernel) on the digits dataset and persists it to disk with joblib.
- **Evaluation metrics.** Accuracy, macro F1, a per-class
  precision/recall/F1 breakdown, and a full confusion matrix, printable
  as text or exportable as JSON.
- **Image-to-prediction pipeline.** Converts an arbitrary image (any
  size, any mode, light- or dark-background) of a hand-drawn digit into
  the 8x8 feature vector the model expects -- auto-inverting and
  auto-cropping to the drawn content -- and predicts a digit with a
  confidence score.

## Project layout

```
src/digitrecognizer/
    data.py            Train/test split loader for the digits dataset
    model.py            Training, saving, and loading the classifier
    evaluate.py           Accuracy/precision/recall/F1/confusion-matrix metrics
    preprocessing.py        Arbitrary image -> 8x8 model-ready feature vector
    cli.py                    argparse-based CLI
tests/         pytest suite
```

## Usage

### CLI

```bash
digitrecognizer train --output digit_model.joblib
digitrecognizer evaluate --model digit_model.joblib --output report.json
digitrecognizer predict --model digit_model.joblib --image my_digit.png
```

## Testing

```bash
pytest --cov=digitrecognizer --cov-report=term-missing
ruff check src tests
```

## License

MIT -- see [LICENSE](LICENSE).
