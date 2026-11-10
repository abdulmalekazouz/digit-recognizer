"""Tests for digitrecognizer.cli."""

from digitrecognizer.cli import main


def test_train_command_saves_model_and_reports_accuracy(tmp_path, capsys):
    output_path = tmp_path / "model.joblib"
    exit_code = main(["train", "--output", str(output_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert output_path.exists()
    assert "Held-out test accuracy" in captured.out
    assert "Saved model to" in captured.out
