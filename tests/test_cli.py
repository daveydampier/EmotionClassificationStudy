"""Tests for the run_experiment command line (toy corpora; no network, no torch)."""

import csv
import importlib.util
from pathlib import Path

import pytest

from emotion_classification.config import PROJECT_ROOT

pytest.importorskip("sklearn")


def _load_cli():
    path = PROJECT_ROOT / "scripts" / "run_experiment.py"
    spec = importlib.util.spec_from_file_location("run_experiment_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cli = _load_cli()


def _config(*argv):
    return cli.config_from_args(cli.build_parser().parse_args(list(argv)))


def test_defaults_match_the_original_command_line():
    cfg = _config()
    assert cfg.models == ["logreg"] and cfg.dataset == "go_emotions"
    assert cfg.schema == "native" and cfg.features == "tfidf"
    assert cfg.eval_sets is None and cfg.seeds == [42]
    assert cfg.val_split == "validation"
    assert cfg.predictions_dir == Path("results") / "predictions"
    assert cfg.run_name == "go_emotions_native_tfidf"


def test_test_dataset_is_shorthand_for_one_eval_set():
    cfg = _config("--schema", "ekman6", "--test-dataset", "sem_eval_2018_task_1")
    assert cfg.eval_sets == ["sem_eval_2018_task_1:test"]


def test_eval_sets_and_test_dataset_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        _config("--eval-sets", "go_emotions:test", "--test-dataset", "sem_eval_2018_task_1")


def test_seeds_override_seed_and_limit_test_keeps_working():
    cfg = _config("--seed", "7", "--seeds", "1", "2", "--limit-test", "50")
    assert cfg.seeds == [1, 2] and cfg.limit_eval == 50
    assert _config("--limit-eval", "9").limit_eval == 9
    assert _config("--seed", "7").seeds == [7]


def test_saving_switches():
    assert _config("--no-predictions").predictions_dir is None
    assert _config("--no-save").predictions_dir is None
    assert _config("--val-split", "none").val_split is None
    assert _config("--out-dir", "elsewhere").predictions_dir == Path("elsewhere") / "predictions"


def test_main_writes_scorecards_and_predictions(toy_corpora, tmp_path, capsys):
    code = cli.main(["--models", "logreg", "naive_bayes", "--dataset", "toy",
                     "--eval-sets", "toy:test", "toy_b:test",
                     "--out-dir", str(tmp_path), "--run-name", "cli"])
    assert code == 0
    with (tmp_path / "cli_scorecard.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [(r["model"], r["dataset"]) for r in rows] == [
        ("logreg_tfidf", "toy"), ("logreg_tfidf", "toy->toy_b"),
        ("naive_bayes_tfidf", "toy"), ("naive_bayes_tfidf", "toy->toy_b"),
    ]
    # 2 models x (validation + 2 scored sets)
    assert len(list((tmp_path / "predictions" / "cli").glob("*.npz"))) == 6
    out = capsys.readouterr().out
    assert "macro_f1=" in out and "## Raw scorecard" in out
    # Models are ranked within an evaluation set, never across them.
    assert "## Normalized on toy (1 = best per axis)" in out
    assert "## Normalized on toy->toy_b (1 = best per axis)" in out


def test_main_multiseed_writes_a_summary(toy_corpora, tmp_path):
    cli.main(["--models", "naive_bayes", "--dataset", "toy", "--seeds", "1", "2",
              "--out-dir", str(tmp_path), "--run-name", "ms"])
    with (tmp_path / "ms_summary.csv").open(encoding="utf-8") as fh:
        rec = next(csv.DictReader(fh))
    assert rec["n_seeds"] == "2"


def test_main_no_save_writes_nothing(toy_corpora, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cli.main(["--models", "naive_bayes", "--dataset", "toy", "--no-save"])
    assert list(tmp_path.iterdir()) == []


def test_main_reports_configuration_errors_cleanly(toy_corpora):
    with pytest.raises(SystemExit) as err:
        cli.main(["--dataset", "toy", "--eval-sets", "toy:dev", "--no-save"])
    assert "no split" in str(err.value)


def test_main_rejects_a_misspelled_model_before_loading_data(toy_corpora):
    with pytest.raises(SystemExit) as err:
        cli.main(["--dataset", "toy", "--models", "logreg", "logregg", "--no-save"])
    assert "unknown model 'logregg'" in str(err.value)
    assert toy_corpora == {"toy": 0, "toy_b": 0}
