from copy import deepcopy

import numpy as np
import pytest

from fall_detect.contracts import PoseRecord
from fall_detect.recipes import YOLO_RECIPE
from fall_detect.visualization.simple import HEADER, draw_simple


@pytest.mark.parametrize("measured,drawn", [(6, False), (11, False), (12, True)])
def test_failed_pose_has_no_geometry_but_preserves_unknown_and_raw_input(measured, drawn):
    pixels = np.full((240, 320, 3), 80, dtype=np.uint8)
    points = np.zeros((18, 3), dtype=np.float32)
    points[:, 0] = np.arange(18) * 7 + 60
    points[:, 1] = np.arange(18) * 5 + 70
    indices = [i for i in range(18) if i != 1][:measured]
    points[indices + [1], 2] = 0.9  # Derived neck must not count as a measured joint.
    pose = PoseRecord(
        "fixture",
        1,
        0,
        0,
        2,
        np.array([30, 50, 200, 220]),
        points,
        (320, 240),
        0.9,
        layout_id="openpose18",
        recipe_id=YOLO_RECIPE,
    )
    states = {2: {"state": "unknown", "reason": "low_confidence", "raw_score": None}}
    before, original = points.copy(), deepcopy(states)
    result = draw_simple(
        pixels,
        [pose],
        states,
        2300,
        0.5,
        quality={"min_joint_confidence": 0.2, "min_valid_joints": 12},
    )
    assert bool(np.any(result[HEADER:] != pixels)) is drawn
    assert np.any(result[:HEADER])
    np.testing.assert_array_equal(points, before)
    assert states == original
    assert (pixels == 80).all()


def test_display_uses_configured_joint_quality_instead_of_a_new_threshold():
    pixels = np.zeros((240, 320, 3), dtype=np.uint8)
    points = np.column_stack((np.arange(13) * 7 + 30, np.arange(13) * 5 + 70, np.full(13, 0.3)))
    pose = PoseRecord("fixture", 1, 0, 0, 0, np.array([20, 30, 200, 200]), points, (320, 240), 0.9)
    states = {0: {"state": "unknown", "reason": "low_confidence", "raw_score": None}}
    output = draw_simple(
        pixels,
        [pose],
        states,
        1000,
        0.5,
        quality={"min_joint_confidence": 0.5, "min_valid_joints": 9},
    )
    np.testing.assert_array_equal(output[HEADER:], pixels)
