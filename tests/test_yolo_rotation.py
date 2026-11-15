from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from fall_detect.config import load_config, recipe_fingerprint, validate
from fall_detect.contracts import FrameRecord
from fall_detect.pose.yolo import YoloPoseBackend


def test_rotation_is_allowed_and_changes_recipe():
    config = load_config("configs/course-ur-balanced-001.yaml")
    weights = {"pose": "synthetic-pose-hash", "stgcn": "synthetic-stgcn-hash"}
    before = recipe_fingerprint(config, weights)
    candidate = deepcopy(config)
    candidate["pose"].update(rotation_angles=[0, 90, -90], rotation_policy="low_quality")
    validate(candidate)
    assert recipe_fingerprint(candidate, weights) != before


@pytest.mark.parametrize("real_joint_count, expected_calls", [(11, 3), (12, 1)])
def test_conditional_rotation_does_not_count_derived_neck(real_joint_count, expected_calls):
    config = load_config("configs/course-ur-balanced-001.yaml")
    config["pose"].update(rotation_angles=[0, 90, -90], rotation_policy="low_quality")
    points = np.zeros((18, 3), np.float32)
    # Eleven measured joints plus a derived neck must not suppress fallback.
    measured = [i for i in range(18) if i != 1][:real_joint_count]
    points[measured + [1]] = [20, 30, 0.9]
    calls = []

    class FakePose:
        last_id = 0

        def __init__(self, points, confidence):
            self.bbox = (0, 0, 50, 100)
            self.confidence = confidence
            self.id = 0

    backend = YoloPoseBackend.__new__(YoloPoseBackend)
    backend.config = config
    backend.pose_module = SimpleNamespace(Pose=FakePose, track_poses=lambda *a, **kw: None)
    backend.reset()

    def decode(image):
        calls.append(image.shape)
        return [(points.copy(), 0.9)]

    backend._decode = decode
    frame = FrameRecord("synthetic", 1, 0, 0, np.zeros((120, 160, 3), np.uint8), 160, 120)
    backend._predict(frame)
    assert len(calls) == expected_calls
    if expected_calls == 3:
        assert calls == [(120, 160, 3), (160, 120, 3), (160, 120, 3)]
