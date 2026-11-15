import importlib.util
from pathlib import Path

import numpy as np
import pytest

from fall_detect.io_utils import ContractError
from fall_detect.models.motion_novelty import MotionNovelty, fusion_scores, motion_trajectory
from fall_detect.skeleton.preprocessing import human_preprocess


def points():
    rng = np.random.default_rng(42)
    one = np.column_stack((rng.uniform(10, 100, (13, 2)), np.ones(13)))
    return np.tile(one, (30, 1, 1))


def test_centering_removes_moving_translation_but_motion_retains_order():
    original, moved = points(), points()
    moved[..., 1] += np.arange(30)[:, None] * 2
    a, _ = human_preprocess(original, (640, 480), normalization="per_frame_isotropic")
    b, _ = human_preprocess(moved, (640, 480), normalization="per_frame_isotropic")
    np.testing.assert_allclose(a, b, atol=1e-6)
    assert np.max(np.abs(motion_trajectory(moved) - motion_trajectory(original))) > 0.5
    assert not np.array_equal(motion_trajectory(moved), motion_trajectory(moved[::-1]))


def test_motion_retains_changing_scale_and_fixed_transform_invariance():
    raw = points()
    raw[..., :2] *= np.linspace(1, 1.6, 30)[:, None, None]
    transformed = raw.copy()
    transformed[..., :2] = transformed[..., :2] * 3 + [700, -230]
    np.testing.assert_allclose(motion_trajectory(raw), motion_trajectory(transformed), atol=1e-14)
    assert motion_trajectory(raw).reshape(30, 3)[-1, 2] == pytest.approx(np.log(1.6))


@pytest.mark.parametrize("kind", ["shape", "nan", "degenerate"])
def test_motion_rejects_invalid_original_window(kind):
    raw = points()
    if kind == "shape":
        raw = raw[:20]
    elif kind == "nan":
        raw[1, 2, 0] = np.nan
    else:
        raw[0, :, 0] = 0
    with pytest.raises(ContractError):
        motion_trajectory(raw)


def test_original_trajectory_does_not_mutate_pose():
    raw = points()
    before = raw.copy()
    motion_trajectory(raw)
    np.testing.assert_array_equal(raw, before)


def test_neighbors_train_only_exclude_entire_video_and_direction():
    values = np.zeros((12, 90))
    values[6:] = 3
    groups = np.array(["video-a"] * 6 + ["video-b"] * 6)
    model = MotionNovelty.fit(values, groups, np.zeros(12), 5)
    np.testing.assert_array_equal(model.score(values), 0)
    np.testing.assert_allclose(model.score(values, groups), 2)
    assert model.score(np.ones((1, 90)) * 30)[0] > model.score(values)[0]
    before = model.scaler.mean.copy()
    model.score(np.ones((2, 90)) * 100000)
    np.testing.assert_array_equal(model.scaler.mean, before)
    with pytest.raises(ContractError):
        MotionNovelty.fit(values, groups, np.array([1] + [0] * 11))
    with pytest.raises(ContractError):
        MotionNovelty.fit(values, np.array(["one"] * 12), np.zeros(12))


@pytest.mark.parametrize("ae_scale,motion_scale", [(0, 1), (1, -1), (np.inf, 1)])
def test_fusion_rejects_invalid_train_scales(ae_scale, motion_scale):
    with pytest.raises(ContractError):
        fusion_scores([1, 2], [2, 1], ae_scale, motion_scale)


def test_fusion_definition_and_no_extra_model_selection():
    np.testing.assert_array_equal(fusion_scores([2, 8], [9, 3], 4, 3), [3, 2])
    with pytest.raises(ContractError):
        fusion_scores([1, -1], [1, 2], 1, 1)


def test_scope_and_reconstruction_helpers_reject_test_and_incomplete(monkeypatch):
    # Import without executing stages; ur_normal_methods is already used by scoped tests.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "motion_workflow_test", Path(__file__).resolve().parents[1] / "scripts/ur_motion_novelty.py"
    )
    workflow = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workflow)
    index = {
        "source_kind": "real",
        "preparation_scope": ["train", "validation"],
        "split_manifest": {"sequences": []},
    }
    monkeypatch.setattr(workflow, "load_index", lambda _: index)
    with pytest.raises(ContractError):
        workflow.validated_index()
    with pytest.raises(ContractError):
        workflow.original_windows(
            {
                "sequences": {"test-x": {"status": "not_prepared"}},
                "split_manifest": {"sequences": [{"sequence_id": "test-x", "split": "test"}]},
                "preparation_scope": ["train", "validation"],
            },
            "test-x",
            {},
        )
