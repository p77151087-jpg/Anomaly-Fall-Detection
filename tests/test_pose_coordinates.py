"""Synthetic risk checks, separate from the real frozen-model diagnosis."""

from copy import deepcopy

import numpy as np
import pytest

from fall_detect.config import load_config
from fall_detect.contracts import FrameRecord, PoseRecord
from fall_detect.evaluation.pose_coordinates import (
    BASE_RECIPE,
    DIAGNOSTIC,
    FRAMES,
    JOINT,
    apply_coordinates,
    intervention_recipe,
    validate_annotation,
    window_sources,
)
from fall_detect.io_utils import ContractError


@pytest.fixture
def inputs():
    annotation = {
        "schema_version": 1,
        "diagnostic_id": DIAGNOSTIC,
        "sequence_id": "fall-19",
        "split": "validation",
        "source_hash": "source",
        "base_recipe_hash": BASE_RECIPE,
        "pose_array_sha256": "pose",
        "pose_metadata_sha256": "meta",
        "author_kind": "ai_provisional",
        "human_confirmed": False,
        "points": [],
    }
    obs = []
    points = np.array([[200 + j * 10, 100 + j * 12, 0.8] for j in range(13)], dtype=np.float32)
    for fid in range(1, 71):
        timestamp = (fid - 1) * 1000 / 30
        frame = FrameRecord("fall-19", fid, fid - 1, timestamp, None, 640, 480)
        pose = PoseRecord(
            "fall-19",
            fid,
            fid - 1,
            timestamp,
            1,
            np.array([100, 50, 500, 450], dtype=np.float32),
            points.copy(),
            (640, 480),
            0.9,
        )
        obs.append((frame, [pose]))
        if fid in FRAMES:
            annotation["points"].append(
                {
                    "source_frame_id": fid,
                    "track_id": 1,
                    "joint_index": JOINT,
                    "joint_name": "left_elbow",
                    "timestamp_ms": timestamp,
                    "image_sha256": "rgb",
                    "frame_size_wh": [640, 480],
                    "original_xy": points[JOINT, :2].astype(float).tolist(),
                    "original_confidence": float(points[JOINT, 2]),
                    "visibility": "visible",
                    "xy": [300, 250],
                }
            )
    return obs, annotation


def test_only_xy_changed_original_untouched(inputs):
    obs, a = inputs
    result, applied = apply_coordinates(obs, a, a)
    assert applied == list(FRAMES)
    for (f, ps), (nf, nps) in zip(obs, result):
        assert f == nf
        np.testing.assert_array_equal(ps[0].bbox_xyxy, nps[0].bbox_xyxy)
        np.testing.assert_array_equal(ps[0].keypoints[:, 2], nps[0].keypoints[:, 2])
        assert ps[0].keypoints[JOINT, 0] == 230
        expected = ps[0].keypoints.copy()
        if f.source_frame_id in FRAMES:
            expected[JOINT, :2] = [300, 250]
        np.testing.assert_array_equal(nps[0].keypoints, expected)
        assert ps[0].keypoints is not nps[0].keypoints


@pytest.mark.parametrize(
    "key,value",
    [
        ("split", "test"),
        ("sequence_id", "fall-20"),
        ("base_recipe_hash", "other"),
        ("source_hash", "other"),
        ("pose_array_sha256", "other"),
        ("human_confirmed", "true"),
        ("author_kind", "oracle"),
        ("human_confirmed", True),
    ],
)
def test_reject_foreign_identity_and_fake_human(inputs, key, value):
    obs, ref = inputs
    a = deepcopy(ref)
    a[key] = value
    with pytest.raises(ContractError):
        apply_coordinates(obs, a, ref)


@pytest.mark.parametrize(
    "key,value",
    [
        ("source_frame_id", 43),
        ("source_frame_id", True),
        ("track_id", 2),
        ("joint_index", 4),
        ("timestamp_ms", 999),
        ("image_sha256", "different"),
        ("original_confidence", 1),
        ("xy", [640, 10]),
        ("xy", [0, -1]),
        ("xy", [float("nan"), 0]),
        ("xy", [True, 0]),
        ("xy", [float("inf"), 0]),
        ("visibility", "occluded"),
    ],
)
def test_reject_unlocked_or_invalid_points(inputs, key, value):
    obs, ref = inputs
    a = deepcopy(ref)
    a["points"][0][key] = value
    with pytest.raises(ContractError):
        apply_coordinates(obs, a, ref)


def test_duplicate_and_missing_frames_rejected(inputs):
    _, ref = inputs
    a = deepcopy(ref)
    a["points"][1] = deepcopy(a["points"][0])
    with pytest.raises(ContractError):
        validate_annotation(a, ref)
    with pytest.raises(ContractError):
        validate_annotation({**ref, "points": ref["points"][:-1]}, ref)


def test_uncertain_does_not_reconstruct(inputs):
    obs, ref = inputs
    a = deepcopy(ref)
    a["points"][0].update(visibility="not_visible_or_uncertain", xy=None)
    result, applied = apply_coordinates(obs, a, ref)
    assert 44 not in applied
    np.testing.assert_array_equal(result[43][1][0].keypoints, obs[43][1][0].keypoints)
    a["points"][0]["xy"] = [300, 250]
    with pytest.raises(ContractError):
        validate_annotation(a, ref)


def test_human_and_ai_recipes_are_separate(inputs):
    _, ref = inputs
    with pytest.raises(ContractError):
        validate_annotation(ref, ref, require_human=True)
    human = {
        **deepcopy(ref),
        "author_kind": "human_self_reported",
        "human_confirmed": True,
        "reviewer": "real reviewer",
        "reviewed_frame_ids": list(FRAMES),
    }
    validate_annotation(human, ref, require_human=True)
    assert intervention_recipe(human) != intervention_recipe(ref) != BASE_RECIPE
    human["reviewed_frame_ids"] = []
    with pytest.raises(ContractError):
        validate_annotation(human, ref, require_human=True)


def test_causal_source_membership_and_missing_track(inputs):
    obs, _ = inputs
    config = load_config()
    config["skeleton"]["stride"] = 1
    obs[45][1].clear()  # No future or adjacent source used to fill frame46.
    rows = window_sources(obs, config)
    for row in rows:
        assert max(row["source_frame_ids"], default=0) <= row["sample_index"] + 1
        if row["sample_index"] >= 45 and row["sample_index"] < 75:
            assert 46 not in row["source_frame_ids"]
            assert row["quality"] == "track_gap"


def test_apply_rejects_different_actual_pose(inputs):
    obs, a = inputs
    obs[43][1][0].keypoints[JOINT, 2] = 0.99
    with pytest.raises(ContractError):
        apply_coordinates(obs, a, a)


def test_entry_refuses_test_before_any_cache_io(monkeypatch):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "coordinate_diagnostic_test",
        Path(__file__).parents[1] / "scripts/pose_coordinate_diagnostic.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def forbidden(*args, **kwargs):
        raise AssertionError("Unapproved sequence caused cache I/O")

    monkeypatch.setattr(module, "load_feature_cache", forbidden)
    monkeypatch.setattr(module, "load_pose_cache", forbidden)
    with pytest.raises(ContractError):
        module.original_inputs("fall-01", {})
