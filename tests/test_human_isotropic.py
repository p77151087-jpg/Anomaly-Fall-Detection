"""Geometry/recipe/casuality checks; synthetic tensors are not recognition evidence."""

import importlib.util
from copy import deepcopy

import numpy as np
import pytest

from fall_detect.config import load_config, project_path, recipe_fingerprint, validate
from fall_detect.data.pose_reuse import validate_pose_reuse
from fall_detect.io_utils import ContractError
from fall_detect.skeleton.preprocessing import human_preprocess
from fall_detect.visualization.action_status import supports_actions


def test_aspect_and_angles_preserved_confidence_unchanged(skeleton):
    before = skeleton.copy()
    pts, motion = human_preprocess(skeleton, (640, 480), normalization="per_frame_isotropic")
    original_span = np.ptp(skeleton[:, :, :2], axis=1)
    span = np.ptp(pts[0, :2, :, :13].transpose(1, 2, 0), axis=1)
    np.testing.assert_allclose(
        span[:, 0] / span[:, 1], original_span[:, 0] / original_span[:, 1], rtol=1e-6
    )
    np.testing.assert_array_equal(pts[0, 2, :, :13], skeleton[:, :, 2])
    np.testing.assert_array_equal(motion, np.diff(pts[:, :2], axis=2))
    np.testing.assert_array_equal(skeleton, before)
    assert np.max(np.abs(pts[:, :2])) <= 1.000001
    vector = skeleton[:, 2, :2] - skeleton[:, 1, :2]
    new_vector = pts[0, :2, :, 2].T - pts[0, :2, :, 1].T
    np.testing.assert_allclose(
        new_vector[:, 0] * vector[:, 1], new_vector[:, 1] * vector[:, 0], atol=1e-4
    )


def test_translation_uniform_scale_invariant_but_aspect_sensitive(skeleton):
    a, _ = human_preprocess(skeleton, (640, 480), normalization="per_frame_isotropic")
    uniform = skeleton.copy()
    uniform[:, :, :2] = uniform[:, :, :2] * 2 + 31
    b, _ = human_preprocess(uniform, (1280, 960), normalization="per_frame_isotropic")
    np.testing.assert_allclose(a, b, atol=1e-6)
    stretched = skeleton.copy()
    stretched[:, :, 0] *= 3
    c, _ = human_preprocess(stretched, (640, 480), normalization="per_frame_isotropic")
    assert np.max(np.abs(a - c)) > 0.05
    old, _ = human_preprocess(skeleton, (640, 480))
    old_stretched, _ = human_preprocess(stretched, (640, 480))
    np.testing.assert_allclose(old, old_stretched, atol=1e-6)


def test_original_modes_bit_exact_to_archived_source(skeleton):
    path = project_path(
        "docs/validation/human-isotropic-legacy-source/src/fall_detect/skeleton/preprocessing.py"
    )
    spec = importlib.util.spec_from_file_location("fall_detect.skeleton._before_isotropic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for mode in ("per_frame", "per_window"):
        expected = module.human_preprocess(skeleton, (640, 480), normalization=mode)
        actual = human_preprocess(skeleton, (640, 480), normalization=mode)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)


def test_new_recipe_is_explicit_pose_reuse_stays_strict(config):
    old = deepcopy(config)
    manifest = {"split_hash": "synthetic"}
    weights = {"pose": "synthetic"}
    index = {
        "source_kind": "real",
        "split_manifest": manifest,
        "preparation_config": old,
        "weights": {"pose": {"sha256": "synthetic"}},
        "recipe_hash": recipe_fingerprint(old, weights),
    }
    config["skeleton"]["coordinate_normalization"] = "per_frame_isotropic"
    validate(config)
    assert recipe_fingerprint(config, weights) != index["recipe_hash"]
    assert not supports_actions(config)
    validate_pose_reuse(index, config, manifest, weights)
    config["quality"]["min_valid_joints"] = 8
    with pytest.raises(ContractError, match="only"):
        validate_pose_reuse(index, config, manifest, weights)
    other = load_config(project_path("configs/ur-ae-decay-001.yaml"))
    other["skeleton"]["coordinate_normalization"] = "per_frame_isotropic"
    with pytest.raises(ContractError, match="Human"):
        validate(other)


def test_no_future_influence_and_degenerate_rejected(skeleton):
    original, _ = human_preprocess(skeleton, (640, 480), normalization="per_frame_isotropic")
    changed = skeleton.copy()
    changed[20:, :, :2] *= 4
    altered, _ = human_preprocess(changed, (640, 480), normalization="per_frame_isotropic")
    np.testing.assert_array_equal(original[:, :, :20], altered[:, :, :20])
    changed[5, :, 0] = 7
    with pytest.raises(ContractError, match="degenerate_pose"):
        human_preprocess(changed, (640, 480), normalization="per_frame_isotropic")
