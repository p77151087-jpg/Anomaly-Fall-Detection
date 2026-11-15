"""Shared-window normalization retains displacement; no synthetic recognition claims."""

from copy import deepcopy

import numpy as np
import pytest
from conftest import sample

from fall_detect.config import load_config, project_path, recipe_fingerprint, validate
from fall_detect.data.pose_reuse import validate_pose_reuse
from fall_detect.io_utils import ContractError, read_json
from fall_detect.models.stgcn import HumanFeatureExtractor
from fall_detect.models.weights import select_device
from fall_detect.skeleton.preprocessing import human_preprocess
from fall_detect.skeleton.windows import WindowBuilder


def test_shared_extents_retain_whole_body_translation_without_mutating_observations(skeleton):
    shifted = skeleton.copy()
    shifted[:, :, 1] += np.arange(30, dtype=np.float32)[:, None] * 3
    old, _ = human_preprocess(skeleton, (640, 480))
    old_shifted, _ = human_preprocess(shifted, (640, 480))
    np.testing.assert_allclose(old, old_shifted, atol=1e-6, rtol=0)
    preserved = shifted.copy()
    original, _ = human_preprocess(skeleton, (640, 480), normalization="per_window")
    actual, motion = human_preprocess(shifted, (640, 480), normalization="per_window")
    assert np.max(np.abs(original - actual)) > 0.1
    assert np.mean(motion[:, 1]) > 0.01
    assert actual[:, :2].min() >= -1 and actual[:, :2].max() <= 1
    np.testing.assert_array_equal(shifted, preserved)
    np.testing.assert_array_equal(motion, np.diff(actual[:, :2], axis=2))
    np.testing.assert_array_equal(actual[:, 2, :, :13], shifted[:, :, 2][None])


def test_window_normalization_is_explicit_and_isolates_cache(config):
    old = deepcopy(config)
    old["skeleton"].pop("coordinate_normalization")
    assert recipe_fingerprint(config, {}) == recipe_fingerprint(old, {})
    config["skeleton"]["coordinate_normalization"] = "per_window"
    validate(config)
    assert recipe_fingerprint(config, {}) != recipe_fingerprint(old, {})
    other = load_config(project_path("configs/ur-ae-decay-001.yaml"))
    other["skeleton"]["coordinate_normalization"] = "per_window"
    with pytest.raises(ContractError, match="Human"):
        validate(other)


def test_shared_extents_never_use_future_or_allow_degenerate_frames(config, skeleton):
    config["skeleton"]["coordinate_normalization"] = "per_window"
    config["skeleton"]["stride"] = 1

    def tensors(change_future):
        builder = WindowBuilder(config)
        rows = []
        for i in range(50):
            p = skeleton[i % 30].copy()
            if change_future and i > 35:
                p[:, :2] *= 5
            for w in builder.push(sample(i, p)):
                rows.append(
                    (
                        w.sample_index,
                        human_preprocess(w.keypoints, w.frame_size_wh, normalization="per_window"),
                    )
                )
        return rows

    for (index, a), (_, b) in zip(tensors(False), tensors(True)):
        if index <= 35:
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_array_equal(a[1], b[1])
    bad = skeleton.copy()
    bad[10, :, 0] = 7
    with pytest.raises(ContractError, match="degenerate_pose"):
        human_preprocess(bad, (640, 480), normalization="per_window")


def test_coordinate_pose_reuse_does_not_relax_observation_quality(config):
    manifest, weights = {"split_hash": "fixture"}, {"pose": "fixture"}
    index = {
        "source_kind": "real",
        "split_manifest": manifest,
        "preparation_config": deepcopy(config),
        "weights": {"pose": {"sha256": "fixture"}},
        "recipe_hash": recipe_fingerprint(config, weights),
    }
    config["skeleton"]["coordinate_normalization"] = "per_window"
    validate_pose_reuse(index, config, manifest, weights)
    config["quality"]["min_valid_joints"] = 8
    with pytest.raises(ContractError, match="only"):
        validate_pose_reuse(index, config, manifest, weights)


def test_real_pretrained_shared_extent_features_are_finite_and_frozen(skeleton):
    manifest = project_path("weights/sample-human-manifest.json")
    if not manifest.exists():
        pytest.skip("Real Human weights unavailable")
    resources = {r["role"]: r for r in read_json(manifest)["resources"]}
    if any(not project_path(r["local_path"]).exists() for r in resources.values()):
        pytest.skip("Real Human weights incomplete")
    config = load_config(project_path("configs/ur-sample-human-s1-001.yaml"))
    config["skeleton"]["coordinate_normalization"] = "per_window"
    extractor = HumanFeatureExtractor(validate(config), resources, select_device("cpu"))
    values = extractor.extract_features(skeleton, (640, 480))
    assert values.shape == (1, 512) and np.isfinite(values).all()
    assert extractor.forward_verified and extractor.load_report["strict"]
    assert all(not p.requires_grad for p in extractor.model.parameters())
