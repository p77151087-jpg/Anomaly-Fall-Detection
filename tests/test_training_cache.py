from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import observation

from fall_detect.data.cache import (
    load_feature_cache,
    load_pose_cache,
    save_feature_cache,
    save_pose_cache,
)
from fall_detect.io_utils import ContractError, read_json, sha256, write_json
from fall_detect.smoke import synthetic_dataset
from fall_detect.training.experiment import (
    Scorer,
    artifact_hashes,
    cache_for,
    calibrate,
    evaluate,
    load_index,
    train,
)
from fall_detect.training.scaler import FeatureScaler


def test_scaler_zero_variance_and_reload(tmp_path):
    x = np.array([[1, 2], [1, 4]], dtype=np.float32)
    scaler = FeatureScaler.fit(x)
    assert scaler.scale[0] == 1
    scaler.save(tmp_path / "scaler.npz")
    np.testing.assert_array_equal(
        scaler.transform(x), FeatureScaler.load(tmp_path / "scaler.npz").transform(x)
    )


def test_pose_cache_reload_and_metadata_rejection(tmp_path, skeleton):
    path = tmp_path / "synthetic-pose.npz"
    observations = [observation(i, skeleton[i]) for i in range(30)]
    meta = {"recipe_hash": "synthetic", "split_hash": "split"}
    save_pose_cache(path, observations, meta)
    loaded, _ = load_pose_cache(path, meta)
    np.testing.assert_array_equal(loaded[0][1][0].keypoints, skeleton[0])
    with pytest.raises(ContractError):
        load_pose_cache(path, {"recipe_hash": "wrong"})


def test_feature_cache_shape_recipe_rows_and_coverage(tmp_path, config):
    index_path = synthetic_dataset(tmp_path, config)
    index = load_index(index_path)
    cache = cache_for(index, "synthetic-test-unknown")
    from fall_detect.detection.events import score_sequence

    result, _, _ = score_sequence(cache, [], 1)
    assert result["prediction"] is None and result["window_coverage"] == 0
    assert result["n_scheduled_endpoints"] == 12
    path = Path(index["sequences"]["synthetic-train-a"]["feature_path"])
    with pytest.raises(ContractError, match="split_hash"):
        load_feature_cache(path, {"split_hash": "wrong"})
    cache = load_feature_cache(path)
    cache.records.pop()
    with pytest.raises(ContractError):
        save_feature_cache(tmp_path / "invalid.npz", cache)


@pytest.mark.parametrize("sampling", ["windows", "sequences"])
@pytest.mark.parametrize("noise,decay", [(0.0, 0.0), (0.1, 0.001)])
def test_training_test_isolation_save_reload_and_frozen_evaluation(
    tmp_path, config, sampling, noise, decay
):
    config["autoencoder"]["sampling"] = sampling
    config["autoencoder"].update(input_noise_std=noise, weight_decay=decay)
    config["autoencoder"]["max_epochs"] = 3
    config["autoencoder"]["patience"] = 2
    index_path = synthetic_dataset(tmp_path / "features", config)
    run = train(config, index_path, tmp_path / "synthetic-run-a", synthetic=True)
    with pytest.raises(ContractError, match="not calibrated"):
        Scorer(run)
    calibrate(run)
    first = Scorer(run)
    cache = cache_for(first.index, "synthetic-val-normal")
    expected = first.score(cache.features)
    np.testing.assert_allclose(Scorer(run).score(cache.features), expected, atol=1e-7)
    before = artifact_hashes(run)
    _, metrics = evaluate(run)
    assert artifact_hashes(run) == before and metrics["n_total"] == 3
    index = load_index(index_path)
    for sid, row in index["sequences"].items():
        if sid.startswith("synthetic-test"):
            changed = load_feature_cache(row["feature_path"])
            changed.features[:] += 1000
            save_feature_cache(row["feature_path"], changed)
    run_b = train(config, index_path, tmp_path / "synthetic-run-b", synthetic=True)
    calibrate(run_b)
    assert sha256(run / "scaler.npz") == sha256(run_b / "scaler.npz")
    a = torch.load(run / "autoencoder.pt", weights_only=True)["state_dict"]
    b = torch.load(run_b / "autoencoder.pt", weights_only=True)["state_dict"]
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert (
        read_json(run / "threshold.json")["threshold"]
        == read_json(run_b / "threshold.json")["threshold"]
    )
    evaluate(run)
    assert artifact_hashes(run) == before
    with pytest.raises(ContractError, match="already calibrated"):
        calibrate(run)
    threshold = read_json(run / "threshold.json")
    threshold["threshold"] = 999
    write_json(run / "threshold.json", threshold)
    with pytest.raises(ContractError, match="hash mismatch"):
        Scorer(run)
