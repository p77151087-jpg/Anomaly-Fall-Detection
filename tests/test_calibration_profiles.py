import numpy as np
import pytest

from fall_detect.cli import main
from fall_detect.config import load_config
from fall_detect.io_utils import ContractError, read_json, sha256
from fall_detect.smoke import synthetic_dataset
from fall_detect.training import experiment
from fall_detect.visualization.loss_viewer import read_loss_run


def test_explicit_balanced_objective_keeps_unknown_in_denominators():
    rows = [
        {"sequence_id": str(i), "label": label, "sequence_score": score}
        for i, (label, score) in enumerate(
            [
                (0, 0.4),
                (0, 0.5),
                (0, None),
                (1, 0.3),
                (1, 0.35),
                (1, 0.6),
                (1, 0.7),
                (1, 0.8),
                (1, None),
            ]
        )
    ]
    baseline, _ = experiment.choose_threshold(rows)
    balanced, _ = experiment.choose_threshold(rows, "balanced_sequence_accuracy")
    assert baseline["metrics"]["TP"] == 5 and baseline["metrics"]["FP"] == 2
    assert balanced["threshold"] == 0.5
    assert balanced["metrics"]["TP"] == 3 and balanced["metrics"]["FP"] == 0
    assert balanced["metrics"]["balanced_sequence_accuracy"] == pytest.approx(0.5 * (3 / 6 + 2 / 3))
    assert balanced["metrics"]["n_unknown"] == 2
    with pytest.raises(ContractError, match="Unsupported"):
        experiment.choose_threshold(rows, "accuracy_typo")
    with pytest.raises(ContractError, match="each class"):
        experiment.choose_threshold(rows[:3], "balanced_sequence_accuracy")
    assert load_config()["calibration"]["objective"] == "sequence_fall_f1"


def test_derived_calibration_is_frozen_and_never_reads_test(tmp_path, config, monkeypatch):
    config["autoencoder"]["max_epochs"] = 2
    index = synthetic_dataset(tmp_path / "synthetic-features", config)
    source = experiment.train(config, index, tmp_path / "synthetic-original", synthetic=True)
    experiment.calibrate(source)
    before = {p.name: sha256(p) for p in source.iterdir() if p.is_file()}
    destination = tmp_path / "synthetic-balanced"
    original_cache_for = experiment.cache_for

    def validation_only(index, sid):
        assert "test" not in sid and "train" not in sid
        return original_cache_for(index, sid)

    monkeypatch.setattr(experiment, "cache_for", validation_only)
    experiment.derive_calibration_run(source, destination, "balanced_sequence_accuracy")
    experiment.calibrate(destination)
    for name in ("autoencoder.pt", "scaler.npz", "training-history.csv", "dataset-index.json"):
        assert sha256(destination / name) == before[name]
    assert {p.name: sha256(p) for p in source.iterdir() if p.is_file()} == before
    experiment.verify_artifacts(destination)
    a = experiment.Scorer(source)
    b = experiment.Scorer(destination)
    features = original_cache_for(a.index, "synthetic-val-normal").features
    np.testing.assert_array_equal(a.score(features), b.score(features))
    source_config = load_config(source / "resolved-config.yaml")
    alternative_config = load_config(destination / "resolved-config.yaml")
    alternative_config["calibration"]["objective"] = "sequence_fall_f1"
    assert source_config == alternative_config
    provenance = read_json(destination / "sources.json")["calibration_derivation"]
    assert provenance["training_repeated"] is False and provenance["test_used"] is False
    assert read_loss_run(destination)["warnings"][-1].endswith("沒有重複訓練。")
    with pytest.raises(ContractError, match="exists"):
        experiment.derive_calibration_run(source, destination, "balanced_sequence_accuracy")
    with pytest.raises(ContractError, match="different objective"):
        experiment.derive_calibration_run(source, tmp_path / "same", "sequence_fall_f1")


@pytest.mark.parametrize("option", ["--objective", "--output-run"])
def test_cli_rejects_in_place_objective_override(option):
    value = "balanced_sequence_accuracy" if option == "--objective" else "unused-run"
    assert main(["calibrate", "--run", "unused-source", option, value]) == 2
