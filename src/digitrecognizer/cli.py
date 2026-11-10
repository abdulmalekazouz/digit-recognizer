"""Command-line interface for digitrecognizer.

Examples:
    digitrecognizer train --output digit_model.joblib
    digitrecognizer evaluate --model digit_model.joblib --output report.json
    digitrecognizer predict --model digit_model.joblib --image my_digit.png
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from digitrecognizer.data import load_digit_dataset
from digitrecognizer.evaluate import evaluate_model, format_report
from digitrecognizer.model import DEFAULT_MODEL_PATH, load_model, save_model, train_model
from digitrecognizer.preprocessing import image_to_feature_vector, load_image


def _cmd_train(args: argparse.Namespace) -> int:
    """Handles the `train` subcommand.

    Trains a model on the digits dataset, reports its held-out test
    accuracy, and saves it to disk.

    Args:
        args: Parsed CLI arguments for the `train` subcommand.

    Returns:
        The process exit code (0 on success).
    """
    dataset = load_digit_dataset(test_size=args.test_size, random_state=args.random_state)
    model = train_model(dataset)
    accuracy = model.score(dataset.x_test, dataset.y_test)

    save_model(model, args.output)
    print(f"Trained model on {len(dataset.x_train)} samples.")
    print(f"Held-out test accuracy: {accuracy:.4f} ({len(dataset.x_test)} test samples)")
    print(f"Saved model to {args.output}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    """Handles the `evaluate` subcommand.

    Note that `--test-size`/`--random-state` should match the values used
    when the model was trained, since this re-derives the train/test
    split rather than saving it alongside the model; mismatched values
    would evaluate the model partly on data it was trained on.

    Args:
        args: Parsed CLI arguments for the `evaluate` subcommand.

    Returns:
        The process exit code (0 on success).
    """
    model = load_model(args.model)
    dataset = load_digit_dataset(test_size=args.test_size, random_state=args.random_state)
    report = evaluate_model(model, dataset)

    print(format_report(report))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, indent=2)
        print(f"\nWrote full report to {args.output}")
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    """Handles the `predict` subcommand.

    Args:
        args: Parsed CLI arguments for the `predict` subcommand.

    Returns:
        The process exit code (0 on success).
    """
    model = load_model(args.model)
    image = load_image(args.image)
    feature_vector = image_to_feature_vector(image).reshape(1, -1)

    prediction = int(model.predict(feature_vector)[0])
    confidence = float(model.predict_proba(feature_vector).max())

    print(f"Predicted digit: {prediction} (confidence: {confidence:.1%})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Builds the top-level argparse parser for the `digitrecognizer` CLI.

    Returns:
        A configured `ArgumentParser` with all subcommands registered.
    """
    parser = argparse.ArgumentParser(
        prog="digitrecognizer", description="Train and use a handwritten digit classifier."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model and save it to disk")
    train_parser.add_argument(
        "--output", default=str(DEFAULT_MODEL_PATH), help="Path to save the trained model to"
    )
    train_parser.add_argument(
        "--test-size", type=float, default=0.2, help="Fraction of data held out for evaluation"
    )
    train_parser.add_argument(
        "--random-state", type=int, default=42, help="Seed for the train/test split"
    )
    train_parser.set_defaults(func=_cmd_train)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate a saved model and print accuracy/confusion metrics"
    )
    evaluate_parser.add_argument(
        "--model", default=str(DEFAULT_MODEL_PATH), help="Path to a model saved by `train`"
    )
    evaluate_parser.add_argument("--output", help="Path to write the full JSON report to")
    evaluate_parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data held out for evaluation (must match training)",
    )
    evaluate_parser.add_argument(
        "--random-state", type=int, default=42, help="Seed for the train/test split (must match training)"
    )
    evaluate_parser.set_defaults(func=_cmd_evaluate)

    predict_parser = subparsers.add_parser(
        "predict", help="Predict the digit drawn in an image file"
    )
    predict_parser.add_argument(
        "--model", default=str(DEFAULT_MODEL_PATH), help="Path to a model saved by `train`"
    )
    predict_parser.add_argument("--image", required=True, help="Path to an image of a digit")
    predict_parser.set_defaults(func=_cmd_predict)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the `digitrecognizer` CLI.

    Args:
        argv: Command-line arguments, excluding the program name. Defaults
            to `sys.argv[1:]` when not provided.

    Returns:
        The process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
