"""Command-line interface for digitrecognizer.

Examples:
    digitrecognizer train --output digit_model.joblib
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from digitrecognizer.data import load_digit_dataset
from digitrecognizer.model import DEFAULT_MODEL_PATH, save_model, train_model


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
