import numpy as np
import pytest
import torch

from fall_detect.config import load_config, validate
from fall_detect.io_utils import ContractError, read_json
from fall_detect.smoke import synthetic_dataset
from fall_detect.training import experiment
from fall_detect.visualization.loss_viewer import read_loss_run


def test_uniform_sequence_mass_ignores_empty_sources_and_is_seeded():
    counts = {"short": 2, "empty": 0, "long": 10}
    sampler = experiment.training_sampler(counts, "sequences", torch.Generator().manual_seed(42))
    assert len(sampler) == 12 and sampler.replacement
    assert float(sampler.weights[:2].sum()) == pytest.approx(float(sampler.weights[2:].sum()))
    second = experiment.training_sampler(counts, "sequences", torch.Generator().manual_seed(42))
    samples = list(sampler)
    assert samples == list(second) and all(0 <= n < 12 for n in samples)
    assert experiment.training_sampler(counts, "windows", torch.Generator()) is None
    for invalid in ({}, {"empty": 0}, {"bad": -1}, {"bad": True}):
        with pytest.raises(ContractError):
            experiment.training_sampler(invalid, "sequences", torch.Generator())


@pytest.mark.parametrize("noise,decay", [(0.0, 0.0), (0.1, 0.001)])
def test_sequence_sampling_reads_only_normal_train_and_validation(
    tmp_path, config, monkeypatch, noise, decay
):
    config["autoencoder"].update(
        sampling="sequences", max_epochs=2, input_noise_std=noise, weight_decay=decay
    )
    path = synthetic_dataset(tmp_path / "synthetic-features", config)
    original = experiment.cache_for
    seen = []

    def normal_only(index, sid):
        source = next(s for s in index["split_manifest"]["sequences"] if s["sequence_id"] == sid)
        assert source["split"] in ("train", "validation") and source["label"] == 0
        seen.append(sid)
        return original(index, sid)

    monkeypatch.setattr(experiment, "cache_for", normal_only)
    run = experiment.train(config, path, tmp_path / "synthetic-balanced", synthetic=True)
    assert set(seen) == {"synthetic-train-a", "synthetic-train-b", "synthetic-val-normal"}
    sampling = read_json(run / "sources.json")["training_sampling"]
    assert sampling["train_normal_feature_counts"] == {
        "synthetic-train-a": 12,
        "synthetic-train-b": 12,
    }
    assert sampling["samples_per_epoch"] == 24 and sampling["replacement"]
    loss = read_loss_run(run)
    assert len(loss["history"]) == 2
    assert any("均衡抽樣" in warning for warning in loss["warnings"])
    features, _, counts = experiment.normal_matrix(experiment.load_index(path), "train", True)
    assert sum(counts.values()) == len(features)
    scaler = experiment.FeatureScaler.load(run / "scaler.npz")
    np.testing.assert_allclose(scaler.mean, features.astype(np.float64).mean(axis=0))


def test_default_sampling_unchanged_and_invalid_strategy_rejected():
    config = load_config()
    assert config["autoencoder"]["sampling"] == "windows"
    config["autoencoder"]["sampling"] = "test_balanced"
    with pytest.raises(ContractError, match="sampling"):
        validate(config)
