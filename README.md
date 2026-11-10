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

## Project layout

```
src/digitrecognizer/
    data.py   Train/test split loader for the digits dataset
tests/        pytest suite
```

## Testing

```bash
pytest --cov=digitrecognizer --cov-report=term-missing
ruff check src tests
```

## License

MIT -- see [LICENSE](LICENSE).
