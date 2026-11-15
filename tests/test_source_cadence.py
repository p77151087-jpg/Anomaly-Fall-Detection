from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
from conftest import observation

from fall_detect.config import load_config, recipe_fingerprint, validate
from fall_detect.data.cache import load_pose_cache, save_pose_cache
from fall_detect.data.pose_reuse import reuse_pose_observations, validate_pose_reuse
from fall_detect.io_utils import ContractError, read_json, sha256, write_json
from fall_detect.skeleton.windows import CausalResampler, WindowBuilder


def sample_stream(skeleton, policy, fps=24.0003840061441, drop=(), future_change=False):
    resampler = CausalResampler(30, policy)
    samples = []
    for i in range(55):
        if i in drop:
            continue
        points = skeleton[i % 30].copy()
        if future_change and i > 30:
            points[:, :2] += 150
        frame, poses = observation(i, points, timestamp=i * 1000 / fps)
        frame.source_fps = fps
        samples.extend(resampler.push(frame, poses))
    return samples


def test_24fps_regression_and_no_invented_poses(config, skeleton):
    old = sample_stream(skeleton, "target_interval")
    fixed = sample_stream(skeleton, "source_interval")
    assert sum(s.missing for s in old) > 10
    assert not any(s.missing for s in fixed)
    builder = WindowBuilder(config)
    windows = [w for s in fixed for w in builder.push(s)]
    assert windows and all(w.quality == "valid" for w in windows)
    for sample in fixed:
        for pose in sample.poses:
            assert pose.timestamp_ms <= sample.timestamp_ms + 1e-7
            np.testing.assert_array_equal(pose.keypoints, skeleton[(pose.source_frame_id - 1) % 30])


def test_30fps_legacy_equivalence_and_real_gaps_remain_missing(config, skeleton):
    old = sample_stream(skeleton, "target_interval", fps=30)
    fixed = sample_stream(skeleton, "source_interval", fps=30)
    assert [(s.source_frame_id, s.missing) for s in old] == [
        (s.source_frame_id, s.missing) for s in fixed
    ]
    gapped = sample_stream(skeleton, "source_interval", drop=(10, 11))
    assert any(s.missing and not s.poses for s in gapped)
    builder = WindowBuilder(config)
    windows = [w for s in gapped for w in builder.push(s)]
    assert any(w.quality == "track_gap" for w in windows)
    frame, poses = observation(0, skeleton[0])
    frame.source_fps, frame.source_missing = 24, True
    missing = CausalResampler(30, "source_interval").push(frame, poses)[0]
    assert missing.missing and missing.poses == []


def test_source_hold_is_causal_and_future_timestamp_does_not_extend_validity(skeleton):
    first = sample_stream(skeleton, "source_interval")
    changed = sample_stream(skeleton, "source_interval", future_change=True)
    for a, b in zip(first, changed):
        if a.timestamp_ms <= 30 * 1000 / 24.0003840061441:
            assert a.source_frame_id == b.source_frame_id
            for x, y in zip(a.poses, b.poses):
                np.testing.assert_array_equal(x.keypoints, y.keypoints)
    resampler = CausalResampler(30, "source_interval")
    frame, poses = observation(0, skeleton[0])
    resampler.push(replace(frame, source_fps=24), poses)
    future, future_poses = observation(1, skeleton[1], timestamp=200)
    samples = resampler.push(replace(future, source_fps=1), future_poses)
    assert [s.missing for s in samples if 50 < s.timestamp_ms < 200] == [True] * 4


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf"), True, "24"])
def test_invalid_declared_fps_is_rejected(skeleton, fps):
    frame, poses = observation(0, skeleton[0])
    with pytest.raises(ContractError, match="Source FPS"):
        CausalResampler(30, "source_interval").push(replace(frame, source_fps=fps), poses)


def test_legacy_fingerprint_and_new_policy_isolation():
    config = load_config("configs/openpose18-le2i.yaml")
    weights = {"pose": "fixture", "stgcn": "fixture"}
    legacy = deepcopy(config)
    legacy["skeleton"].pop("resampling_hold_policy")
    assert recipe_fingerprint(config, weights) == recipe_fingerprint(legacy, weights)
    config["skeleton"]["resampling_hold_policy"] = "source_interval"
    assert recipe_fingerprint(config, weights) != recipe_fingerprint(legacy, weights)
    config["skeleton"]["resampling_hold_policy"] = "invented"
    with pytest.raises(ContractError):
        validate(config)


def test_pose_reuse_rejects_other_recipe_weight_and_split_changes(config):
    weights = {"pose": "fixture"}
    manifest = {"split_hash": "same-fixture"}
    index = {
        "source_kind": "real",
        "split_manifest": manifest,
        "preparation_config": deepcopy(config),
        "weights": {"pose": {"sha256": "fixture"}},
        "recipe_hash": recipe_fingerprint(config, weights),
    }
    config["skeleton"]["resampling_hold_policy"] = "source_interval"
    validate_pose_reuse(index, config, manifest, weights)
    changed = deepcopy(config)
    changed["pose"]["input_height"] += 32
    with pytest.raises(ContractError, match="only"):
        validate_pose_reuse(index, changed, manifest, weights)
    with pytest.raises(ContractError, match="split"):
        validate_pose_reuse(index, config, {"split_hash": "other"}, weights)
    with pytest.raises(ContractError, match="weight"):
        validate_pose_reuse(index, config, manifest, {"pose": "different"})


def test_pose_reuse_source_immutable_fps_roundtrip_and_rejects_mismatch(
    tmp_path, config, skeleton, monkeypatch
):
    meta = {
        "source_kind": "real",
        "source_hash": "fixture",
        "split_hash": "fixture",
        "sequence_id": "fixture",
        "split": "train",
        "recipe_hash": "old",
        "recipe_id": config["skeleton"]["recipe_id"],
        "limit_frames": None,
    }
    obs = [observation(i, skeleton[i], sid="fixture", timestamp=i * 41.666) for i in range(3)]
    src, dst = tmp_path / "source.npz", tmp_path / "new.npz"
    save_pose_cache(src, obs, meta)
    before = {p: sha256(p) for p in [src, src.with_suffix(".json")]}
    video = tmp_path / "fixture.avi"
    video.write_bytes(b"synthetic test stub, never a real-video result")
    from fall_detect.data import pose_reuse

    def reader(*args, **kwargs):
        yield replace(obs[0][0], source_fps=24.0003840061441)

    monkeypatch.setattr(pose_reuse, "read_source", reader)
    index = {
        "recipe_hash": "old",
        "sequences": {"fixture": {"status": "prepared", "pose_path": str(src)}},
    }
    metadata = {**meta, "recipe_hash": "new"}
    reused = reuse_pose_observations(index, "fixture", metadata, video, dst, config)
    loaded, _ = load_pose_cache(dst, metadata)
    assert [f.source_fps for f, _ in loaded] == [24.0003840061441] * 3
    assert sha256(dst) == sha256(src)
    for (a, poses), (b, other) in zip(obs, reused):
        assert a.timestamp_ms == b.timestamp_ms and poses[0].track_id == other[0].track_id
        np.testing.assert_array_equal(poses[0].keypoints, other[0].keypoints)
    assert {p: sha256(p) for p in before} == before
    with pytest.raises(ContractError, match="overwrite"):
        reuse_pose_observations(index, "fixture", metadata, video, src, config)
    with pytest.raises(ContractError, match="source_hash mismatch"):
        reuse_pose_observations(
            index, "fixture", {**metadata, "source_hash": "other"}, video, dst, config
        )
    old = read_json(src.with_suffix(".json"))
    for f in old["frames"]:
        f.pop("source_fps", None)
        f.pop("source_fps_estimated", None)
    write_json(src.with_suffix(".json"), old)
    assert load_pose_cache(src, meta)[0][0][0].source_fps is None
