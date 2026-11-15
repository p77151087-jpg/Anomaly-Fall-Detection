"""Synthetic risk tests for bounded retirement and normal-only observed-joint augmentation."""

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    import retire_external_normal as retired
    import ur_train_shift as workflow
finally:
    sys.path.pop(0)

from fall_detect.io_utils import ContractError
from fall_detect.skeleton.preprocessing import human_preprocess


def points():
    values = np.ones((30, 13, 3), dtype=np.float32)
    values[:, :, 0] = 100 + np.arange(13)[None] * 15 + np.arange(30)[:, None] * 0.1
    values[:, :, 1] = 100 + np.arange(13)[None] * 20
    values[:, :, 2] = 0.9
    return values


def test_translation_is_removed_by_existing_isotropic_normalization():
    original = points()
    before = original.copy()
    a = human_preprocess(original, (640, 480), normalization="per_frame_isotropic")
    b = human_preprocess(
        workflow.global_shift(original, (640, 480)), (640, 480), normalization="per_frame_isotropic"
    )
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y, atol=2e-6, rtol=1e-6)
    np.testing.assert_array_equal(original, before)


def test_joint_shift_is_bounded_deterministic_and_preserves_confidence():
    original = points()
    before = original.copy()
    a = workflow.joint_shift(original, (640, 480), 42)
    np.testing.assert_array_equal(a, workflow.joint_shift(original, (640, 480), 42))
    np.testing.assert_array_equal(a[:, :, 2], original[:, :, 2])
    np.testing.assert_array_equal(original, before)
    delta = a[:, :, :2] - original[:, :, :2]
    assert np.max(np.abs(delta)) <= 2.40002
    assert np.max(np.abs(delta)) > 0.1
    np.testing.assert_allclose(delta, np.broadcast_to(delta[0], delta.shape), atol=3e-5)
    assert not np.array_equal(a[:, 0, :2] - original[:, 0, :2], a[:, 1, :2] - original[:, 1, :2])


def test_unreliable_or_cropped_joints_never_gain_fake_observations():
    original = points()
    original[3, 1, 2] = 0.1
    original[:, 2, 0] = -5
    original[:, 3, 0] = 645
    shifted = workflow.joint_shift(original, (640, 480), 42)
    np.testing.assert_array_equal(shifted[:, [1, 2, 3]], original[:, [1, 2, 3]])
    np.testing.assert_array_equal(shifted[:, :, 2], original[:, :, 2])


def test_all_unreliable_points_remain_unchanged():
    original = points()
    original[:, :, 2] = 0.1
    np.testing.assert_array_equal(workflow.joint_shift(original, (640, 480), 42), original)


@pytest.mark.parametrize("fraction", [0, -0.01, 0.02, float("nan")])
def test_rejects_unfrozen_or_excessive_shift(fraction):
    with pytest.raises(ContractError):
        workflow.joint_shift(points(), (640, 480), 42, fraction)


def test_rejects_nonfinite_or_non_observed_shape():
    bad = points()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ContractError):
        workflow.joint_shift(bad, (640, 480), 42)
    with pytest.raises(ContractError):
        workflow.joint_shift(np.zeros((30, 14, 3)), (640, 480), 42)


@pytest.mark.parametrize("relative", ["../outside", "."])
def test_archive_paths_cannot_move_workspace_or_escape_it(tmp_path, relative):
    with pytest.raises(ContractError):
        retired.bounded_path(tmp_path, relative)


def test_archive_mapping_is_exact_and_does_not_move_ur_data():
    mappings = {"data/raw/external": "data/retired/external"}
    assert (
        retired.mapped_path("data/raw/external/clip.mp4", mappings)
        == "data/retired/external/clip.mp4"
    )
    assert retired.mapped_path("data/raw/external-other/a", mappings) == "data/raw/external-other/a"
    assert retired.mapped_path("data/raw/urfall/a", mappings) == "data/raw/urfall/a"


def test_source_seed_is_reproducible_and_separates_windows():
    assert workflow.seed_for("adl-01", 0) == workflow.seed_for("adl-01", 0)
    assert workflow.seed_for("adl-01", 0) != workflow.seed_for("adl-01", 1)
