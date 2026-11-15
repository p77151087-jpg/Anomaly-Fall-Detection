from copy import deepcopy

import numpy as np
import pytest
from conftest import sample

from fall_detect.config import recipe_fingerprint
from fall_detect.data.pose_reuse import validate_pose_reuse
from fall_detect.detection.events import EventState
from fall_detect.io_utils import ContractError
from fall_detect.skeleton.windows import WindowBuilder


def test_stride_only_reuse_preserves_rejection_of_quality_and_tracking_changes(config):
    weights, manifest = {"pose": "test"}, {"split_hash": "fixture"}
    index = {
        "source_kind": "real",
        "split_manifest": manifest,
        "preparation_config": deepcopy(config),
        "weights": {"pose": {"sha256": "test"}},
        "recipe_hash": recipe_fingerprint(config, weights),
    }
    config["skeleton"]["stride"] = 1
    validate_pose_reuse(index, config, manifest, weights)
    assert recipe_fingerprint(config, weights) != index["recipe_hash"]
    for group, key, value in [
        ("quality", "min_valid_joints", 8),
        ("pose", "track_similarity_joints", 2),
        ("skeleton", "window_size", 20),
    ]:
        changed = deepcopy(config)
        changed[group][key] = value
        with pytest.raises(ContractError, match="only"):
            validate_pose_reuse(index, changed, manifest, weights)


def test_dense_windows_match_sparse_and_still_reject_missing_pose(config, skeleton):
    dense_config = deepcopy(config)
    dense_config["skeleton"]["stride"] = 1
    dense, sparse = WindowBuilder(dense_config), WindowBuilder(config)
    observed = {}
    for i in range(70):
        current = sample(i, None if i == 48 else skeleton[i % 30])
        for window in dense.push(current):
            observed[(window.track_id, window.sample_index)] = window
        for window in sparse.push(current):
            other = observed[(window.track_id, window.sample_index)]
            assert window.quality == other.quality
            assert window.start_ms == other.start_ms and window.end_ms == other.end_ms
            if window.keypoints is not None:
                np.testing.assert_array_equal(window.keypoints, other.keypoints)
    assert all(w.quality == "track_gap" for w in observed.values() if w.sample_index >= 48)


@pytest.mark.parametrize("stride", [1, 2, 5])
def test_k3_confirmation_obeys_candidate_cadence_and_gap_reset(stride):
    events = EventState(0.5, consecutive_windows=3, stride=stride)
    states = [
        events.update("fixture", 0, 29 + i * stride, (29 + i * stride) * 1000 / 30, 0.6)
        for i in range(3)
    ]
    assert [s["state"] for s in states] == ["warming_up", "warming_up", "suspected_fall"]
    assert events.update("fixture", 0, 29 + 3 * stride, 2000, None)["state"] == "unknown"
    assert events.update("fixture", 0, 29 + 4 * stride, 2100, 0.9)["state"] == "warming_up"
