import sys
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from unittest.mock import patch
from bert_classifier import cli


def test_main_runs_pipeline(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "bert-classifier", "--data-path", "foo.csv", "--epochs", "2",
    ])
    with patch("bert_classifier.cli.Pipeline") as pipeline_cls:
        cli.main()
        pipeline_cls.assert_called_once()
        cfg_arg = pipeline_cls.call_args[0][0]
        assert cfg_arg.data_path == "foo.csv"
        assert cfg_arg.epochs == 2


def test_main_rejects_unknown_optimizer(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["bert-classifier", "--optimizer", "Bogus"])
    with pytest.raises(SystemExit):
        cli.main()
