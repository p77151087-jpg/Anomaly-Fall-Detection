from copy import deepcopy

import numpy as np
import pytest

from fall_detect.config import load_config, recipe_fingerprint, validate
from fall_detect.io_utils import ContractError
from fall_detect.pose.yolo import YoloPoseBackend


@pytest.mark.parametrize("measured,expected", [(11, [0.25, 0.1]), (12, [0.25])])
def test_fallback_uses_actual_joint_quality_and_current_image(measured, expected, monkeypatch):
    config = load_config("configs/course-ur-balanced-001.yaml")
    config["pose"]["fallback_detector_confidence"] = 0.1
    backend = YoloPoseBackend.__new__(YoloPoseBackend)
    backend.config = config
    points = np.zeros((18, 3), np.float32)
    points[[i for i in range(18) if i != 1][:measured] + [1]] = [20, 30, 0.9]
    image = np.zeros((64, 64, 3), np.uint8)
    calls = []

    def predict(source, confidence):
        assert source is image
        calls.append(confidence)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
        return [(points.copy(), 0.3)]

    backend._predict_candidates = predict
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    from os import environ

    result = backend._decode(image)
    assert calls == expected
    assert environ["CUDA_VISIBLE_DEVICES"] == "0"
    np.testing.assert_array_equal(result[0][0], points)


def test_disabled_fallback_preserves_legacy_recipe_and_rejects_invalid_values():
    config = load_config("configs/course-ur-balanced-001.yaml")
    old = deepcopy(config)
    old["pose"].pop("fallback_detector_confidence")
    assert recipe_fingerprint(config, {}) == recipe_fingerprint(old, {})
    config["pose"]["fallback_detector_confidence"] = 0.1
    validate(config)
    assert recipe_fingerprint(config, {}) != recipe_fingerprint(old, {})
    for invalid in [0.25, 0.5, -0.1, float("nan")]:
        config["pose"]["fallback_detector_confidence"] = invalid
        with pytest.raises(ContractError):
            validate(config)
