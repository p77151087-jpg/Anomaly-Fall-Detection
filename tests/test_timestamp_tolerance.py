"""Synthetic risk fixtures only; these are not fall-detection accuracy evidence."""

from copy import deepcopy

import numpy as np
import pytest
from conftest import observation

from fall_detect.config import load_config, recipe_fingerprint, validate
from fall_detect.data.pose_reuse import validate_pose_reuse
from fall_detect.io_utils import ContractError
from fall_detect.skeleton.windows import CausalResampler, WindowBuilder


def stream(skeleton, tolerance, missing_index=None, future_change=False):
    sampler = CausalResampler(30, "source_interval", tolerance)
    samples = []
    for i in range(75):
        points = skeleton[i % 30].copy()
        if future_change and i >= 50:
            points[:, :2] += 100
        frame, poses = observation(i, points, timestamp=round(i * 1000 / 30))
        if i == missing_index:
            frame.source_missing = True
            poses = []
        samples.extend(sampler.push(frame, poses))
    return samples


def test_integer_ms_expiry_preserves_grid_real_poses_and_causality(config, skeleton):
    legacy = stream(skeleton, 0)
    corrected = stream(skeleton, 1)
    assert any(s.missing for s in legacy)
    assert not any(s.missing for s in corrected)
    assert [s.timestamp_ms for s in legacy] == [s.timestamp_ms for s in corrected]
    builder = WindowBuilder(config)
    assert all(w.quality == "valid" for s in corrected for w in builder.push(s))
    changed = stream(skeleton, 1, future_change=True)
    for a, b in zip(corrected, changed):
        for p in a.poses:
            assert p.timestamp_ms <= a.timestamp_ms + 1e-7
            np.testing.assert_array_equal(p.keypoints, skeleton[(p.source_frame_id - 1) % 30])
        if a.timestamp_ms < round(50 * 1000 / 30):
            assert a.source_frame_id == b.source_frame_id
            np.testing.assert_array_equal(a.poses[0].keypoints, b.poses[0].keypoints)


def test_missing_source_and_expired_observations_stay_unknown(config, skeleton):
    corrected = stream(skeleton, 1, missing_index=36)
    assert any(s.missing and not s.poses for s in corrected)
    builder = WindowBuilder(config)
    assert any(w.quality == "track_gap" for s in corrected for w in builder.push(s))
    sampler = CausalResampler(30, "source_interval", 1)
    frame, poses = observation(0, skeleton[0], timestamp=0)
    sampler.push(frame, poses)
    future, later = observation(1, skeleton[1], timestamp=200)
    samples = sampler.push(future, later)
    assert all(s.missing and not s.poses for s in samples if 35 < s.timestamp_ms < 200)


@pytest.mark.parametrize("value", [-1, 1.01, True, float("nan"), float("inf"), "1"])
def test_tolerance_is_bounded(value):
    config = load_config()
    config["skeleton"]["timestamp_tolerance_ms"] = value
    with pytest.raises(ContractError, match="tolerance"):
        validate(config)
    with pytest.raises(ContractError, match="tolerance"):
        CausalResampler(30, timestamp_tolerance_ms=value)


def test_legacy_fingerprint_and_pose_only_reuse_boundary():
    config = load_config("configs/openpose18-caucafall.yaml")
    weights = {"pose": "fixture"}
    legacy = deepcopy(config)
    legacy["skeleton"].pop("timestamp_tolerance_ms")
    assert recipe_fingerprint(config, weights) == recipe_fingerprint(legacy, weights)
    manifest = {"split_hash": "fixture"}
    index = {
        "source_kind": "real",
        "split_manifest": manifest,
        "preparation_config": legacy,
        "weights": {"pose": {"sha256": "fixture"}},
        "recipe_hash": recipe_fingerprint(legacy, weights),
    }
    config["skeleton"]["timestamp_tolerance_ms"] = 1
    assert recipe_fingerprint(config, weights) != index["recipe_hash"]
    validate_pose_reuse(index, config, manifest, weights)
    config["quality"]["min_valid_joints"] = 11
    with pytest.raises(ContractError, match="only"):
        validate_pose_reuse(index, config, manifest, weights)


def test_bounded_jitter_causal_hold_does_not_bridge_missing_or_future_frames(skeleton):
    sampler = CausalResampler(30, "bounded_jitter")
    frames = []
    for i, timestamp in enumerate([0, 31, 62, 109, 140, 171, 218]):
        frame, poses = observation(i, skeleton[i], timestamp=timestamp)
        frames.extend(sampler.push(frame, poses))
    assert not any(s.missing for s in frames)
    assert all(p.timestamp_ms <= s.timestamp_ms + 1e-7 for s in frames for p in s.poses)
    # A later arrival cannot extend a prior observation beyond the fixed 50 ms cap.
    frame, poses = observation(7, skeleton[7], timestamp=500)
    tail = sampler.push(frame, poses)
    assert all(s.missing for s in tail if 268 < s.timestamp_ms < 500)
    frame, poses = observation(8, skeleton[8], timestamp=531)
    frame.source_missing = True
    sampler.push(frame, [])
    frame, poses = observation(9, skeleton[9], timestamp=562)
    assert any(s.missing and not s.poses for s in sampler.push(frame, poses))


def test_bounded_jitter_respects_slower_source_cadence_and_recipe_change(skeleton):
    sampler = CausalResampler(30, "bounded_jitter")
    frames = []
    for i in range(12):
        frame, poses = observation(i, skeleton[i], timestamp=i * 1000 / 15)
        frame.source_fps = 15
        frames.extend(sampler.push(frame, poses))
    assert not any(s.missing for s in frames)
    cfg = load_config("configs/yolo11-caucafall.yaml")
    before = recipe_fingerprint(cfg, {})
    cfg["skeleton"]["resampling_hold_policy"] = "bounded_jitter"
    validate(cfg)
    assert recipe_fingerprint(cfg, {}) != before
