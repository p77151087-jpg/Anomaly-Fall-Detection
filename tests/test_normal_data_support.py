"""Risk tests for explicit Train-only augmentation; these are synthetic evidence."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from fall_detect.io_utils import ContractError, sha256, write_npz
from fall_detect.training import experiment

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    spec = importlib.util.spec_from_file_location(
        "normal_data_support", SCRIPTS / "normal_data_support.py"
    )
    workflow = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workflow)
finally:
    sys.path.pop(0)


def test_reflection_is_involution_preserves_confidence_and_observed_input():
    rng = np.random.default_rng(8)
    points = rng.normal(size=(30, 13, 3)).astype(np.float32)
    original = points.copy()
    mirrored = workflow.mirror_human13(points, 640)
    np.testing.assert_allclose(workflow.mirror_human13(mirrored, 640), points, atol=4e-5)
    np.testing.assert_array_equal(mirrored[:, :, 2], points[:, workflow.MIRROR13, 2])
    np.testing.assert_array_equal(mirrored[:, :, 1], points[:, workflow.MIRROR13, 1])
    np.testing.assert_array_equal(points, original)
    np.testing.assert_allclose(mirrored[:, 1, 0], 639 - points[:, 2, 0])


def test_mirror_requires_observed_human13_shape_and_finite_input():
    with pytest.raises(ContractError):
        workflow.mirror_human13(np.zeros((30, 14, 3)), 640)
    bad = np.zeros((30, 13, 3))
    bad[0, 0, 0] = np.nan
    with pytest.raises(ContractError):
        workflow.mirror_human13(bad, 640)


def test_training_source_rejects_test_and_validation_derivatives():
    index = {
        "split_manifest": {
            "sequences": [
                {"sequence_id": "train", "split": "train", "label": 0},
                {"sequence_id": "val", "split": "validation", "label": 0},
                {"sequence_id": "test", "split": "test", "label": 0},
            ]
        }
    }
    assert workflow.input_source(index, "train", train=True)["label"] == 0
    with pytest.raises(ContractError):
        workflow.input_source(index, "val", train=True)
    with pytest.raises(ContractError):
        workflow.input_source(index, "test", train=True)


def test_augmented_matrix_exact_and_hash_guard_no_val_or_test_change(tmp_path, monkeypatch):
    array = np.arange(4 * 512, dtype=np.float32).reshape(4, 512)
    path = tmp_path / "train.npz"
    write_npz(path, features=array)
    index = {
        "recipe_hash": "recipe",
        "split_manifest": {
            "sequences": [
                {"sequence_id": "a", "split": "train", "label": 0},
                {"sequence_id": "b", "split": "validation", "label": 0},
                {"sequence_id": "c", "split": "test", "label": 0},
            ]
        },
    }
    profile = {
        "recipe_hash": "recipe",
        "sequences": {
            "a": {
                "combined_path": "train.npz",
                "combined_sha256": sha256(path),
                "combined_count": 4,
            }
        },
    }
    monkeypatch.setattr(workflow, "ROOT", tmp_path)
    monkeypatch.setattr(workflow, "ORIGINAL_MATRIX", lambda i, s, c: "original-validation")
    with workflow.train_matrix_inputs(index, profile=profile):
        actual, hashes, counts = experiment.normal_matrix(index, "train", return_counts=True)
        np.testing.assert_array_equal(actual, array)
        assert counts == {"a": 4} and hashes == {"a": sha256(path)}
        assert experiment.normal_matrix(index, "validation") == "original-validation"
        with pytest.raises(ContractError, match="No Test"):
            experiment.normal_matrix(index, "test")
        path.write_bytes(b"tampered")
        with pytest.raises(ContractError, match="hash mismatch"):
            experiment.normal_matrix(index, "train")


def test_training_matrix_rejects_fall_in_train(tmp_path, monkeypatch):
    index = {
        "split_manifest": {
            "sequences": [
                {"sequence_id": "bad-fall", "split": "train", "label": 1},
            ]
        }
    }
    with (
        workflow.train_matrix_inputs(index, selected={"bad-fall"}),
        pytest.raises(ContractError, match="Train ADL only"),
    ):
        experiment.normal_matrix(index, "train")
