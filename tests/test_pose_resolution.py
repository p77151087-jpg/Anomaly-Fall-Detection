"""Synthetic resolution/diagnosis contracts; no real human truth or accuracy claims."""

import copy

import numpy as np
import pytest

from fall_detect.config import load_config, recipe_fingerprint, validate
from fall_detect.contracts import FrameRecord, PoseRecord
from fall_detect.evaluation.pose_comparison import (
    assert_resolution_only,
    human_phase_summary,
    paired_pose_rows,
    resolution_config,
    scoped_source,
)
from fall_detect.io_utils import ContractError


def test_single_change_is_valid_distinct_recipe_and_does_not_mutate_base():
    base = load_config()
    before = copy.deepcopy(base)
    changed = resolution_config(
        base,
        "data/cache/synthetic-resolution",
        "data/cache/synthetic-resolution/manifests",
        "weights/synthetic-resolution.json",
    )
    assert base == before and validate(changed) == changed
    weights = {"detector": "a", "pose": "b", "stgcn": "c"}
    assert recipe_fingerprint(base, weights) != recipe_fingerprint(changed, weights)
    assert changed["quality"] == base["quality"] and changed["calibration"] == base["calibration"]
    assert changed["autoencoder"] == base["autoencoder"]


@pytest.mark.parametrize(
    "group,key,value",
    [
        ("quality", "min_valid_joints", 1),
        ("pose", "detector_size", 960),
        ("pose", "flip_tta", True),
        ("autoencoder", "latent_dim", 128),
        ("skeleton", "window_size", 20),
        ("pose", "input_height", 384),
    ],
)
def test_hidden_extra_tuning_is_rejected(group, key, value):
    base = load_config()
    changed = resolution_config(base, "synthetic-cache", "synthetic-manifest", "synthetic-weights")
    changed[group][key] = value
    with pytest.raises(ContractError):
        assert_resolution_only(base, changed)


@pytest.mark.parametrize("split,label", [("test", 0), ("test", 1), ("train", 1)])
def test_test_access_or_fall_train_forbidden(split, label):
    index = {
        "split_manifest": {
            "sequences": [{"sequence_id": "synthetic", "split": split, "label": label}]
        }
    }
    with pytest.raises(ContractError):
        scoped_source(index, "synthetic")


def fixture():
    points = np.zeros((13, 3), dtype=np.float32)
    points[:, 2] = 0.8
    observations, labels = [], []
    for i, t in enumerate([0, 33, 67], 1):
        f = FrameRecord("synthetic-validation", i, i - 1, t, None, 320, 240)
        poses = (
            []
            if i == 1
            else [
                PoseRecord(
                    f.sequence_id,
                    i,
                    i - 1,
                    t,
                    1,
                    np.array([0, 0, 10, 10]),
                    points.copy(),
                    (320, 240),
                    0.9,
                )
            ]
        )
        observations.append((f, poses))
        labels.append(
            {
                "source_frame_id": i,
                "timestamp_ms": t,
                "phase": "upright" if i < 3 else "transition",
                "pose_check": "wrong_or_drift",
            }
        )
    return observations, labels


def test_missing_stays_missing_and_original_human_judgment_does_not_label_new_pose():
    old, labels = fixture()
    new = copy.deepcopy(old)
    snapshot = copy.deepcopy(labels)
    quality = load_config()["quality"]
    rows = paired_pose_rows(old, new, quality, labels)
    assert rows[0]["new_pose_count"] == 0 and not rows[0]["new_good"]
    assert rows[1]["new_good"] and rows[1]["human_original_pose_check"] == "wrong_or_drift"
    assert all(r["human_new_pose_check"] is None for r in rows) and labels == snapshot


@pytest.mark.parametrize("change", ["time", "source", "size", "dropped"])
def test_misaligned_or_dropped_source_observations_rejected(change):
    old, _ = fixture()
    new = copy.deepcopy(old)
    if change == "time":
        new[1][0].timestamp_ms += 1
    elif change == "source":
        new[1][0].source_frame_id += 10
    elif change == "size":
        new[1][0].width += 1
    else:
        new.pop()
    with pytest.raises(ContractError):
        paired_pose_rows(old, new, load_config()["quality"])


def test_phase_endpoint_uses_past_observation_and_missing_phase_is_not_zero_score():
    old, labels = fixture()
    windows = [
        {
            "track_id": 1,
            "sample_index": 2,
            "available_at_ms": 66.666,
            "raw_score": None,
            "event_score": None,
            "state": "unknown",
        }
    ]
    result = human_phase_summary(windows, [], old, labels, load_config()["quality"])
    assert set(result) == {"upright"} and result["upright"]["valid_endpoints"] == 0
    assert result["upright"]["max_raw"] is None and result["upright"]["max_k3"] is None
    old[-1][1][0].keypoints[:, :2] = 9999
    assert human_phase_summary(windows, [], old, labels, load_config()["quality"]) == result
