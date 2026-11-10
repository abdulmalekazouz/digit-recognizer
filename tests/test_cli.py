"""Tests for digitrecognizer.cli."""

import json

from digitrecognizer.cli import main


def test_train_command_saves_model_and_reports_accuracy(tmp_path, capsys):
    output_path = tmp_path / "model.joblib"
    exit_code = main(["train", "--output", str(output_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert output_path.exists()
    assert "Held-out test accuracy" in captured.out
    assert "Saved model to" in captured.out


def test_evaluate_command_prints_report_and_writes_json(tmp_path, capsys):
    model_path = tmp_path / "model.joblib"
    main(["train", "--output", str(model_path)])
    capsys.readouterr()  # discard the train command's output

    report_path = tmp_path / "report.json"
    exit_code = main(
        ["evaluate", "--model", str(model_path), "--output", str(report_path)]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Accuracy" in captured.out
    assert report_path.exists()

    report_data = json.loads(report_path.read_text(encoding="utf-8"))
    assert "accuracy" in report_data
    assert "confusion_matrix" in report_data
