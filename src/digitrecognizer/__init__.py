"""digitrecognizer: a handwritten digit classifier.

Trains a support vector classifier on the scikit-learn digits dataset,
persists it to disk, and exposes it through a CLI and a canvas-based Flask
web UI. Everything runs fully offline on a small, bundled dataset -- no
large downloaded models and no network calls.
"""

__version__ = "0.1.0"
