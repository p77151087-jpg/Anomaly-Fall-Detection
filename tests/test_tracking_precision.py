from copy import deepcopy
from types import SimpleNamespace

import pytest

from fall_detect.config import load_config, recipe_fingerprint
from fall_detect.data.track_reuse import identical_tracking_resets, validate_tracking_reuse
from fall_detect.io_utils import ContractError
from fall_detect.pose.lightweight import tracking_gap_exceeded


def configurations():
    old = load_config("configs/openpose18-le2i-timing.yaml")
    new = deepcopy(old)
    new["pose"]["track_gap_tolerance_ms"] = 1e-7
    return old, new


def observations(times, missing=False):
    return [(SimpleNamespace(timestamp_ms=t, source_missing=missing), []) for t in times]


def test_20fps_float_jitter_is_not_a_real_gap():
    old, new = configurations()
    assert tracking_gap_exceeded(100, 150.00000000000003, old)
    assert not tracking_gap_exceeded(100, 150.00000000000003, new)
    assert tracking_gap_exceeded(100, 150.001, new)
    assert tracking_gap_exceeded(100, 200, new)
    assert not tracking_gap_exceeded(None, 999, new)


def test_reuse_requires_equal_reset_decisions_for_every_observed_frame():
    old, new = configurations()
    assert identical_tracking_resets(observations([0, 41.6666667, 83.3333333, 125]), old, new)
    assert identical_tracking_resets(observations([0, 100, 200]), old, new)
    assert not identical_tracking_resets(observations([0, 50, 100, 150.00000000000003]), old, new)
    assert identical_tracking_resets(observations([0, 50, 100, 150.00000000000003], True), old, new)


def test_only_explicit_tolerance_change_can_reuse_and_old_hashes_remain_stable():
    old, new = configurations()
    weights = {"pose": {"sha256": "pose"}, "stgcn": {"sha256": "stgcn"}}
    hashes = {k: v["sha256"] for k, v in weights.items()}
    historical = deepcopy(old)
    historical["pose"].pop("track_gap_tolerance_ms")
    assert recipe_fingerprint(historical, hashes) == recipe_fingerprint(old, hashes)
    assert recipe_fingerprint(new, hashes) != recipe_fingerprint(old, hashes)
    index = {
        "weights": weights,
        "preparation_config": historical,
        "recipe_hash": recipe_fingerprint(old, hashes),
    }
    validate_tracking_reuse(index, new)
    changed = deepcopy(new)
    changed["quality"]["min_valid_joints"] -= 1
    with pytest.raises(ContractError):
        validate_tracking_reuse(index, changed)
    index["recipe_hash"] = "tampered"
    with pytest.raises(ContractError):
        validate_tracking_reuse(index, new)
