"""Causal short-context contracts; synthetic tests are not recognition evidence."""

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


@pytest.mark.parametrize("size", [15, 20])
def test_short_context_requires_explicit_opt_in_and_isolates_cache(config, size):
    historical = deepcopy(config)
    historical["skeleton"].pop("temporal_context")
    assert recipe_fingerprint(config, {}) == recipe_fingerprint(historical, {})
    config["skeleton"]["window_size"] = size
    with pytest.raises(ContractError, match="window_size"):
        validate(config)
    config["skeleton"]["temporal_context"] = f"short_{size}"
    validate(config)
    assert recipe_fingerprint(config, {}) != recipe_fingerprint(historical, {})
    other = load_config(project_path("configs/ur-ae-decay-001.yaml"))
    other["skeleton"].update(window_size=size, temporal_context=f"short_{size}")
    with pytest.raises(ContractError, match="Human"):
        validate(other)


@pytest.mark.parametrize("size", [15, 20])
def test_short_preprocess_keeps_observed_points_and_motion(skeleton, size):
    value = skeleton[-size:].copy()
    with pytest.raises(ContractError):
        human_preprocess(value, (640, 480))
    full, _ = human_preprocess(skeleton, (640, 480))
    points, motion = human_preprocess(value, (640, 480), window_size=size)
    np.testing.assert_array_equal(points, full[:, :, -size:])
    np.testing.assert_array_equal(motion, np.diff(points[:, :2], axis=2))
    np.testing.assert_array_equal(value, skeleton[-size:])


@pytest.mark.parametrize("size", [15, 20])
def test_short_windows_keep_quality_missing_and_causal_boundaries(config, skeleton, size):
    config["skeleton"].update(window_size=size, temporal_context=f"short_{size}", stride=1)

    def build(bad=0, missing=None, future=False):
        builder = WindowBuilder(config)
        rows = []
        for i in range(size * 3):
            points = skeleton[i % 30].copy()
            if i < bad:
                points[:, 2] = 0.1
            if future and i > size + 4:
                points[:, :2] *= 3
            rows.extend(builder.push(sample(i, None if i == missing else points)))
        return rows

    assert build(bad=size // 5)[0].quality == "valid"
    assert build(bad=size // 5 + 1)[0].quality == "low_confidence"
    rows = build(missing=size + 2)
    assert [r.sample_index for r in rows] == list(range(size - 1, size * 3))
    assert all(r.quality == "track_gap" for r in rows if size + 2 <= r.sample_index < size * 2 + 2)
    assert next(r for r in rows if r.sample_index == size * 2 + 2).quality == "valid"
    for before, after in zip(build(), build(future=True)):
        if before.sample_index <= size + 4:
            np.testing.assert_array_equal(before.keypoints, after.keypoints)
            assert before.start_ms == pytest.approx((before.sample_index - size + 1) * 1000 / 30)


def test_context_pose_reuse_still_rejects_quality_or_observation_changes(config):
    manifest, weights = {"split_hash": "fixture"}, {"pose": "test"}
    index = {
        "source_kind": "real",
        "split_manifest": manifest,
        "preparation_config": deepcopy(config),
        "weights": {"pose": {"sha256": "test"}},
        "recipe_hash": recipe_fingerprint(config, weights),
    }
    config["skeleton"].update(window_size=15, temporal_context="short_15", stride=1)
    validate_pose_reuse(index, config, manifest, weights)
    for group, key, value in [
        ("quality", "min_valid_joints", 8),
        ("pose", "use_track_proposals", True),
    ]:
        changed = deepcopy(config)
        changed[group][key] = value
        with pytest.raises(ContractError, match="only"):
            validate_pose_reuse(index, changed, manifest, weights)


@pytest.mark.parametrize("size", [15, 20])
def test_genuine_pretrained_human_supports_short_context(skeleton, size):
    if not project_path("weights/sample-human-manifest.json").exists():
        pytest.skip("Genuine Human resources unavailable")
    resources = {
        r["role"]: r
        for r in read_json(project_path("weights/sample-human-manifest.json"))["resources"]
    }
    if any(not project_path(r["local_path"]).exists() for r in resources.values()):
        pytest.skip("Genuine Human resources incomplete")
    config = load_config(project_path("configs/ur-sample-human-s1-001.yaml"))
    config["skeleton"].update(window_size=size, temporal_context=f"short_{size}")
    extractor = HumanFeatureExtractor(validate(config), resources, select_device("cpu"))
    values = extractor.extract_features(skeleton[-size:], (640, 480))
    assert values.shape == (1, 512) and np.isfinite(values).all()
    assert extractor.forward_verified and extractor.load_report["strict"]
    assert all(not p.requires_grad for p in extractor.model.parameters())
