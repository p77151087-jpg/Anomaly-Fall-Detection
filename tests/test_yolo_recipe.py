import os
from copy import deepcopy

import numpy as np
import pytest
import torch

from fall_detect.config import load_config, recipe_fingerprint, validate
from fall_detect.io_utils import ContractError
from fall_detect.pose.yolo import COCO_TO_OPENPOSE, coco17_to_openpose18
from fall_detect.recipes import YOLO_RECIPE, recipe_spec
from fall_detect.skeleton.preprocessing import good_frame


def test_complete_joint_mapping_neck_and_missing_are_explicit():
    points = np.column_stack((np.arange(17) * 11, np.arange(17) * 7, np.ones(17))).astype(
        np.float32
    )
    points[5, 2], points[6, 2] = 0.4, 0.7
    original = points.copy()
    result = coco17_to_openpose18(points)
    np.testing.assert_array_equal(points, original)
    assert sorted(i for i in COCO_TO_OPENPOSE if i >= 0) == list(range(17))
    for destination, source in enumerate(COCO_TO_OPENPOSE):
        if source >= 0:
            np.testing.assert_array_equal(result[destination], points[source])
    np.testing.assert_array_equal(result[1, :2], points[[5, 6], :2].mean(axis=0))
    assert result[1, 2] == points[5, 2]
    points[5, 2] = 0
    result = coco17_to_openpose18(points)
    assert not result[1].any() and not result[5].any()
    np.testing.assert_array_equal(result[2], points[6])


def test_derived_neck_never_increases_quality_count():
    cfg = load_config("configs/yolo11-caucafall.yaml")
    spec = recipe_spec(cfg)
    points = np.ones((18, 3), np.float32)
    points[12:, 2] = 0  # 12 confident points including the derived neck = 11 real.
    assert good_frame(points, cfg["quality"], 18)
    assert not good_frame(points, cfg["quality"], 18, spec["derived_joints"])
    points[12, 2] = 1
    assert good_frame(points, cfg["quality"], 18, spec["derived_joints"])


def test_recipe_isolation_and_bad_pose_rejected():
    yolo = load_config("configs/yolo11-caucafall.yaml")
    assert yolo["skeleton"]["recipe_id"] == YOLO_RECIPE
    old = load_config("configs/openpose18-caucafall-timestamp.yaml")
    assert recipe_fingerprint(yolo, {}) != recipe_fingerprint(old, {})
    wrong = deepcopy(yolo)
    wrong["features"]["expected_dim"] = 512
    with pytest.raises(ContractError):
        validate(wrong)
    wrong = deepcopy(yolo)
    wrong["quality"]["min_valid_joints"] = 18
    with pytest.raises(ContractError):
        validate(wrong)
    with pytest.raises(ContractError):
        coco17_to_openpose18(np.zeros((13, 3)))
    bad = np.zeros((17, 3))
    bad[0, 2] = np.nan
    with pytest.raises(ContractError):
        coco17_to_openpose18(bad)


def test_real_yolo_strict_cpu_forward_and_missing_reset():
    from fall_detect.assets import require_weights
    from fall_detect.config import project_path
    from fall_detect.data.readers import read_source
    from fall_detect.models.weights import select_device
    from fall_detect.pose.yolo import YoloPoseBackend

    cfg = load_config("configs/yolo11-caucafall.yaml")
    if not project_path("weights/pretrained/yolo11/yolo11m-pose-state.pt").exists():
        pytest.skip("Optional real YOLO weights unavailable")
    backend = YoloPoseBackend(cfg, require_weights(cfg), select_device("cpu"))
    reader = read_source(project_path("data/raw/urfall/adl-05"), limit=1)
    frame = next(reader)
    reader.close()
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    poses = backend.predict(frame)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == visible_devices
    assert backend.load_report["strict"] and backend.load_report["state_keys"] == 691
    assert backend.forward_verified and poses
    assert all(p.recipe_id == YOLO_RECIPE and p.keypoints.shape == (18, 3) for p in poses)
    assert all(not p.requires_grad for p in backend.model.parameters())
    assert all(torch.isfinite(p).all() for p in backend.model.parameters())
    frame.source_missing = True
    frame.timestamp_ms += 33.3333
    assert backend.predict(frame) == [] and backend.previous == []
