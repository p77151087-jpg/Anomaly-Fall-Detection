import csv
import io
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from fall_detect.io_utils import ContractError, sha256, write_csv, write_json
from fall_detect.visualization.loss_viewer import LossViewer, read_loss_run, validate_history


def completed_run(root):
    run = root / "synthetic-loss"
    rows = [
        {"epoch": 1, "train_loss": 0.8, "val_normal_loss": 0.6},
        {"epoch": 2, "train_loss": 0.4, "val_normal_loss": 0.5},
    ]
    write_csv(run / "training-history.csv", rows)
    write_json(
        run / "status.json",
        {
            "source_kind": "synthetic",
            "training": "complete",
            "epochs": 2,
            "best_val_normal_loss": 0.5,
        },
    )
    write_json(
        run / "frozen-artifacts.json",
        {"training-history.csv": sha256(run / "training-history.csv")},
    )
    return run, rows


def test_saved_history_verification_and_read_only_http(tmp_path):
    run, rows = completed_run(tmp_path)
    before = {p.name: sha256(p) for p in run.iterdir()}
    viewer = LossViewer(tmp_path, port=0, selected=run.name).start(open_browser=False)
    try:
        with urlopen(viewer.url) as response:
            assert "Loss 訓練曲線" in response.read().decode("utf-8")
        base = viewer.url.split("?")[0]
        with urlopen(base + "api/loss?run=" + run.name) as response:
            report = json.load(response)
        assert report["history"] == rows
        assert report["best"] == rows[1]
        assert report["source_kind"] == "synthetic" and report["history_verified"]
        with urlopen(base + "download/history.csv?run=" + run.name) as response:
            assert "attachment" in response.headers["Content-Disposition"]
            downloaded = list(csv.DictReader(io.StringIO(response.read().decode("utf-8-sig"))))
        assert validate_history(downloaded) == rows
        for route in ["api/loss?run=..%2Foutside", "status.json"]:
            with pytest.raises(HTTPError):
                urlopen(base + route)
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base + "api/runs", headers={"Host": "external.example"}))
        assert error.value.code == 403
        assert {p.name: sha256(p) for p in run.iterdir()} == before
        write_csv(run / "training-history.csv", [dict(rows[0], train_loss=123), rows[1]])
        with pytest.raises(ContractError, match="hash"):
            read_loss_run(run)
    finally:
        viewer.close()


@pytest.mark.parametrize("loss", [float("nan"), float("inf"), -0.1])
def test_invalid_loss_is_never_plotted(loss):
    with pytest.raises(ContractError):
        validate_history([{"epoch": 1, "train_loss": loss, "val_normal_loss": 1}])


def test_incomplete_history_and_source_mismatch(tmp_path):
    run, rows = completed_run(tmp_path)
    with pytest.raises(ContractError):
        validate_history([rows[1]])
    write_json(run / "status.json", {"source_kind": "real", "training": "running"})
    assert read_loss_run(run)["history"] == []  # Never show a stale completed CSV as live.
    write_json(run / "training-progress.json", {"source_kind": "synthetic", "history": rows})
    with pytest.raises(ContractError, match="來源"):
        read_loss_run(run)


def test_live_training_publishes_real_epoch_values(tmp_path, config, monkeypatch):
    from fall_detect.smoke import synthetic_dataset
    from fall_detect.training import experiment

    config["autoencoder"]["max_epochs"] = 3
    config["autoencoder"]["patience"] = 10
    index = synthetic_dataset(tmp_path / "features", config)
    run = tmp_path / "synthetic-live-training"
    observed = []
    original_write = experiment.write_json

    def observe(path, value):
        original_write(path, value)
        if path.name == "training-progress.json":
            report = read_loss_run(run)
            assert report["source_kind"] == "synthetic"
            assert report["training"] == "running"
            assert report["history"] == validate_history(value["history"])
            observed.append(report["history"])

    monkeypatch.setattr(experiment, "write_json", observe)
    experiment.train(config, index, run, synthetic=True)
    assert [len(rows) for rows in observed] == [1, 2, 3]
    final = read_loss_run(run)
    assert final["training"] == "complete" and final["history_verified"]
    assert final["history"] == observed[-1]
    assert final["counts"] == {} and final["warnings"] == []


def test_empty_directory_and_new_run_visible_without_server_restart(tmp_path):
    viewer = LossViewer(tmp_path, port=0).start(open_browser=False)
    try:
        with urlopen(viewer.url + "api/runs") as response:
            assert json.load(response)["runs"] == []
        run, _ = completed_run(tmp_path)
        with urlopen(viewer.url + "api/runs") as response:
            assert json.load(response)["runs"] == [run.name]
    finally:
        viewer.close()
