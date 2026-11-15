"""Synthetic scope/leakage regressions; no UR recognition metrics."""

from types import SimpleNamespace

import numpy as np
import pytest
from conftest import observation

from fall_detect import pipeline
from fall_detect.config import recipe_fingerprint
from fall_detect.io_utils import ContractError, read_json, write_json
from fall_detect.smoke import synthetic_dataset
from fall_detect.training import experiment


def scoped_fixture(tmp_path, config):
    path = synthetic_dataset(tmp_path, config)
    index = read_json(path)
    index["preparation_scope"] = ["train", "validation"]
    for s in index["split_manifest"]["sequences"]:
        if s["split"] == "test":
            index["sequences"][s["sequence_id"]] = {"status": "not_prepared", "reason": "excluded"}
    write_json(path, index)
    return path, index


def test_scoped_train_calibrate_never_load_test_and_require_opt_in(tmp_path, config, monkeypatch):
    config["autoencoder"].update(max_epochs=2, patience=2)
    path, index = scoped_fixture(tmp_path / "features", config)
    with pytest.raises(ContractError, match="explicit"):
        experiment.train(config, path, tmp_path / "bad", synthetic=True)
    original = experiment.load_feature_cache
    accesses = []

    def guarded(path, *args, **kwargs):
        assert "synthetic-test" not in str(path)
        accesses.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(experiment, "load_feature_cache", guarded)
    run = experiment.train(
        config, path, tmp_path / "run", synthetic=True, train_validation_only=True
    )
    experiment.calibrate(run)
    assert len(accesses) == 5  # 2 TrainADL, 1 normalVal, 2 calibrationVal
    with pytest.raises(ContractError, match="outside explicit"):
        experiment.cache_for(index, "synthetic-test-fall")
    with pytest.raises(ContractError, match="unavailable"):
        experiment.evaluate(run, output=tmp_path / "forbidden")
    assert not (tmp_path / "forbidden").exists()
    assert read_json(run / "sources.json")["execution_scope"] == "train_validation_only"
    assert all(
        r["status"] == "not_prepared" for sid, r in index["sequences"].items() if "test" in sid
    )


@pytest.mark.parametrize(
    "change", ["missing_val_fall", "test_prepared", "test_old_cache", "smoke_scope"]
)
def test_scoped_mode_rejects_incomplete_validation_and_forged_test(tmp_path, config, change):
    path, index = scoped_fixture(tmp_path / "features", config)
    if change == "missing_val_fall":
        index["sequences"]["synthetic-val-fall"] = {"status": "not_selected"}
    elif change == "test_prepared":
        index["sequences"]["synthetic-test-fall"]["status"] = "prepared"
    elif change == "test_old_cache":
        index["sequences"]["synthetic-test-fall"]["feature_path"] = "old30.npz"
    else:
        index.pop("preparation_scope")
    write_json(path, index)
    with pytest.raises(ContractError):
        experiment.train(config, path, tmp_path / "bad", synthetic=True, train_validation_only=True)
    assert not (tmp_path / "bad").exists()


def test_prepare_scope_skips_test_source_and_retains_truthful_index(
    tmp_path, config, skeleton, monkeypatch
):
    _, index = scoped_fixture(tmp_path / "fixture", config)
    config["data"]["cache_dir"] = str(tmp_path / "new")
    config["skeleton"].update(
        window_size=20,
        temporal_context="short_20",
        stride=1,
        coordinate_normalization="per_frame_isotropic",
    )
    seen = []

    def identity(path, sync):
        assert "synthetic-test" not in str(path)
        seen.append(path.name)
        return {"source_hash": "fixture", "files": []}

    def read(path, **kwargs):
        assert "synthetic-test" not in str(path)
        for i in range(40):
            yield observation(i, skeleton[i % 30], sid=kwargs["sequence_id"])[0]

    backend = SimpleNamespace(
        reset=lambda: None,
        predict=lambda f: observation(
            f.sample_index, skeleton[f.sample_index % 30], sid=f.sequence_id
        )[1],
        forward_verified=False,
    )
    extractor = SimpleNamespace(
        feature_dim=512,
        load_report={"synthetic": True},
        forward_verified=False,
        extract_features=lambda *args: np.ones((1, 512), dtype=np.float32),
    )
    monkeypatch.setattr(pipeline, "dataset_split", lambda c: index["split_manifest"])
    monkeypatch.setattr(pipeline, "require_weights", lambda c: {})
    monkeypatch.setattr(pipeline, "upstream_commits", lambda c: {})
    monkeypatch.setattr(pipeline, "create_backends", lambda *args: (backend, extractor))
    monkeypatch.setattr(pipeline, "source_identity", identity)
    monkeypatch.setattr(pipeline, "read_source", read)
    _, prepared = pipeline.prepare(config, train_validation_only=True)
    assert len(seen) == 4
    assert prepared["preparation_scope"] == ["train", "validation"]
    assert all(
        r["status"] == "not_prepared" and "feature_path" not in r
        for sid, r in prepared["sequences"].items()
        if "test" in sid
    )
    assert not list((tmp_path / "new/features").glob("*test*"))
    with pytest.raises(ContractError, match="bounded smoke"):
        pipeline.prepare(config, selected_ids=["synthetic-train-a"], train_validation_only=True)
    per_frame = {
        **config,
        "skeleton": {**config["skeleton"], "coordinate_normalization": "per_frame"},
    }
    assert recipe_fingerprint(config, {}) != recipe_fingerprint(per_frame, {})
