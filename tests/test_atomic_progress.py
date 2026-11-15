from pathlib import Path

import pytest

from fall_detect import io_utils
from fall_detect.smoke import synthetic_dataset
from fall_detect.training import experiment
from fall_detect.visualization.loss_viewer import read_loss_run


def locked_error():
    error = PermissionError("Windows file is temporarily open")
    error.winerror = 5
    return error


def test_atomic_replacement_retries_lock_without_exposing_partial_file(tmp_path, monkeypatch):
    path = tmp_path / "progress.json"
    path.write_bytes(b"old-complete")
    replace = io_utils.os.replace
    attempts, sleeps = [], []

    def sometimes_locked(source, destination):
        assert path.read_bytes() == b"old-complete"
        assert Path(source).read_bytes() == b"new-complete"
        attempts.append(1)
        if len(attempts) < 3:
            raise locked_error()
        replace(source, destination)

    monkeypatch.setattr(io_utils.os, "replace", sometimes_locked)
    monkeypatch.setattr(io_utils.time, "sleep", sleeps.append)
    io_utils.atomic_bytes(path, b"new-complete")
    assert sleeps == [0.01, 0.02]
    assert path.read_bytes() == b"new-complete" and not list(tmp_path.glob("*.part"))


@pytest.mark.parametrize("temporary_lock,expected_attempts", [(True, 9), (False, 1)])
def test_persistent_or_unrelated_io_errors_are_not_hidden(
    tmp_path, monkeypatch, temporary_lock, expected_attempts
):
    path = tmp_path / "progress.json"
    path.write_bytes(b"original")
    attempts = []

    def fail(*_args):
        attempts.append(1)
        raise locked_error() if temporary_lock else OSError("disk failure")

    monkeypatch.setattr(io_utils.os, "replace", fail)
    monkeypatch.setattr(io_utils.time, "sleep", lambda _seconds: None)
    with pytest.raises(OSError):
        io_utils.atomic_bytes(path, b"replacement")
    assert len(attempts) == expected_attempts
    assert path.read_bytes() == b"original" and not list(tmp_path.glob("*.part"))


def test_failed_training_keeps_history_and_never_claims_completion(tmp_path, config, monkeypatch):
    config["autoencoder"]["max_epochs"] = 2
    index = synthetic_dataset(tmp_path / "synthetic-features", config)
    run = tmp_path / "synthetic-failed"
    original = experiment.write_json

    def fail_on_second_epoch(path, value):
        if path.name == "training-progress.json" and len(value["history"]) == 2:
            raise locked_error()
        return original(path, value)

    monkeypatch.setattr(experiment, "write_json", fail_on_second_epoch)
    with pytest.raises(PermissionError):
        experiment.train(config, index, run, synthetic=True)
    report = read_loss_run(run)
    assert report["training"] == "failed" and len(report["history"]) == 1
    assert "不能當作可用模型" in report["warnings"][0]
    assert not (run / "autoencoder.pt").exists() and not (run / "training-artifacts.json").exists()
    with pytest.raises(io_utils.ContractError, match="exists"):
        experiment.train(config, index, run, synthetic=True)
    assert read_loss_run(run)["training"] == "failed"
