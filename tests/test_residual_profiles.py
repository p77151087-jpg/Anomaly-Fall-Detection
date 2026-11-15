import numpy as np
import pytest
import torch

from fall_detect.io_utils import ContractError, read_json, sha256, write_json
from fall_detect.models.autoencoder import Autoencoder, reconstruction_scores
from fall_detect.smoke import synthetic_dataset
from fall_detect.training import experiment
from fall_detect.training.residual import derive_residual_run, weights_from_means


def test_weighting_rule_and_legacy_arithmetic():
    means = np.r_[np.ones(256), np.full(256, 3.0)]
    for mode in ("train_channel_mse", "train_stream_mse"):
        weights, floor = weights_from_means(means, mode)
        np.testing.assert_array_equal(weights, np.r_[np.full(256, 1.5), np.full(256, 0.5)])
        assert floor == 0.2
    weights, floor = weights_from_means([0, 1, 3], "train_channel_mse")
    assert floor == 0.2 and np.isfinite(weights).all() and weights.mean() == pytest.approx(1)
    model = Autoencoder(3).eval()
    x = np.asarray([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=np.float32)
    with torch.inference_mode():
        errors = torch.cat(
            [(torch.tensor(row[None]) - model(torch.tensor(row[None]))) ** 2 for row in x]
        )
    legacy = errors.mean(dim=1).numpy()
    np.testing.assert_array_equal(reconstruction_scores(model, x, "cpu", batch_size=1), legacy)
    expected = (errors * torch.tensor(weights)).mean(dim=1).numpy()
    np.testing.assert_array_equal(
        reconstruction_scores(model, x, "cpu", batch_size=1, residual_weights=weights), expected
    )


@pytest.mark.parametrize("weights", [[1, 2], [1, 0, 1], [1, -1, 1], [1, np.nan, 1]])
def test_invalid_weights_rejected(weights):
    with pytest.raises(ContractError, match="Residual weights"):
        reconstruction_scores(Autoencoder(3), np.ones((1, 3)), "cpu", residual_weights=weights)


@pytest.mark.parametrize("mode", ["train_channel_mse", "train_stream_mse"])
def test_fit_train_only_calibrate_validation_only_and_preserve_source(
    tmp_path, config, monkeypatch, mode
):
    config["autoencoder"]["max_epochs"] = 2
    index = synthetic_dataset(tmp_path / "features", config)
    source = experiment.train(config, index, tmp_path / "baseline", synthetic=True)
    experiment.calibrate(source)
    before = {p.name: sha256(p) for p in source.iterdir() if p.is_file()}
    original_cache = experiment.cache_for
    allowed = "train"
    accessed = []

    def guarded(index, sid):
        row = next(r for r in index["split_manifest"]["sequences"] if r["sequence_id"] == sid)
        assert row["split"] == allowed
        if allowed == "train":
            assert row["label"] == 0
        accessed.append(sid)
        return original_cache(index, sid)

    monkeypatch.setattr(experiment, "cache_for", guarded)
    target = derive_residual_run(source, tmp_path / "weighted", mode)
    profile = read_json(target / "residual-profile.json")
    assert set(accessed) == set(profile["feature_hashes"])
    assert not profile["test_used"] and not profile["validation_used_for_scales"]
    allowed = "validation"
    experiment.calibrate(target)
    scorer = experiment.Scorer(target)
    features = original_cache(scorer.index, "synthetic-val-normal").features
    scores = scorer.score(features)
    np.testing.assert_array_equal(
        scores, np.concatenate([scorer.score(row[None]) for row in features])
    )
    for name in ("autoencoder.pt", "scaler.npz", "training-history.csv", "dataset-index.json"):
        assert sha256(target / name) == before[name]
    assert {p.name: sha256(p) for p in source.iterdir() if p.is_file()} == before
    assert read_json(target / "status.json")["additional_epochs"] == 0
    assert "residual-profile.json" in experiment.verify_artifacts(target)
    assert scorer.threshold["residual_profile_sha256"] == sha256(target / "residual-profile.json")
    with pytest.raises(ContractError, match="exists"):
        derive_residual_run(source, target, mode)
    with pytest.raises(ContractError, match="baseline MSE"):
        experiment.train(scorer.config, index, tmp_path / "wrong", synthetic=True)
    profile["weights"][0] += 1
    write_json(target / "residual-profile.json", profile)
    with pytest.raises(ContractError, match="hash mismatch"):
        experiment.Scorer(target)
    # Even a rewritten manifest cannot make an inconsistent weighting recipe load.
    write_json(target / "frozen-artifacts.json", experiment.artifact_hashes(target))
    with pytest.raises(ContractError, match="weights/counts"):
        experiment.Scorer(target)


@pytest.mark.parametrize(
    "means, mode",
    [
        ([1, 2], "train_stream_mse"),
        ([1, -1], "train_channel_mse"),
        ([float("inf")], "train_channel_mse"),
    ],
)
def test_profile_input_rejection(means, mode):
    with pytest.raises(ContractError):
        weights_from_means(means, mode)
