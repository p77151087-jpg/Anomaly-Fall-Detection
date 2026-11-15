"""Synthetic risk tests: source isolation, unlabelled boundaries and provenance."""

import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fall_detect.io_utils import ContractError, sha256, write_npz
from fall_detect.training import experiment

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    import external_normal_data as acquisition
    import external_normal_train as workflow
    import prepare_external_normal as preparation
    import ur_annotation_review as annotation
finally:
    sys.path.pop(0)


@pytest.mark.parametrize("name", ["../escape.mp4", "C:/other/file.mp4", "/root/file", "a\\..\\b"])
def test_reject_unsafe_archive_paths(name):
    with pytest.raises(ContractError, match="Unsafe"):
        acquisition.safe_member(name)


def test_normal_selection_does_not_take_falls_or_subject_five_or_extra_trials():
    rows = [
        {"name": f"VideoDataset/ADL/SBJ_{s:02d}_LOC1/ACT{a}_F_1/clip.mp4"}
        for s in (1, 2, 3, 4, 6, 7, 8, 9)
        for a in (13, 18, 24)
    ]
    unwanted = [
        {"name": "VideoDataset/Fall/SBJ_01_LOC1/ACT13_F_1/fall.mp4"},
        {"name": "VideoDataset/ADL/SBJ_05_LOC1/ACT13_F_1/x.mp4"},
        {"name": "VideoDataset/ADL/SBJ_01_LOC1/ACT13_F_2/x.mp4"},
    ]
    assert acquisition.select_members(rows + unwanted) == rows


def test_remote_download_requires_exact_range_response_and_checks_crc():
    class Session:
        def get(self, *args, **kwargs):
            return SimpleNamespace(
                status_code=200, headers={}, content=b"whole file", raise_for_status=lambda: None
            )

    row = {
        "name": "VideoDataset/ADL/x.mp4",
        "offset": 20,
        "compressed": 5,
        "size": 5,
        "crc": zlib.crc32(b"video"),
        "method": 8,
    }
    with pytest.raises(ContractError, match="honor"):
        acquisition.remote_member(
            Session(), {"download_url": "https://example.invalid/source"}, row
        )


@pytest.mark.parametrize("split,label", [("test", 0), ("validation", 0), ("external_train", 1)])
def test_preparation_rejects_non_normal_or_existing_holdout(split, label):
    with pytest.raises(ContractError, match="external Train ADL"):
        preparation.metadata_for({"split": split, "label": label}, {})


def test_unlabelled_title_plot_and_cross_boundary_windows_never_enter_train():
    records = [
        SimpleNamespace(quality="valid", row_index=i, start_ms=s, end_ms=e)
        for i, (s, e) in enumerate(
            [(0, 1000), (900, 1500), (1000, 1800), (1500, 2000), (1800, 2200)]
        )
    ]
    records.append(
        SimpleNamespace(quality="low_confidence", row_index=None, start_ms=1200, end_ms=1700)
    )
    assert workflow.selected_rows(SimpleNamespace(records=records), [1000, 2000]) == [2, 3]


def test_phase_bounds_use_observed_source_times_and_keep_uncertainty():
    frames = [
        {"source_frame_id": i, "timestamp_ms": t}
        for i, t in [(5, 0), (9, 135), (15, 341), (20, 509)]
    ]
    actual = annotation.interval_from_frames(frames, (5, 9, 15, 20))
    assert actual["onset_uncertainty_ms"] == [0, 135]
    assert actual["settled_uncertainty_ms"] == [341, 509]
    with pytest.raises(ContractError):
        annotation.interval_from_frames(frames, (5, 9, 14, 20))


def test_external_real_subset_checks_hash_and_exact_original_rows(tmp_path, monkeypatch):
    original = np.arange(4 * 512, dtype=np.float32).reshape(4, 512)
    write_npz(tmp_path / "matrix.npz", features=original[1:3])
    (tmp_path / "raw.mp4").write_bytes(b"fixture raw; not a real model test")
    row = {
        "split": "external_train",
        "label": 0,
        "sequence_id": "external",
        "source_path": "raw.mp4",
        "source_sha256": sha256(tmp_path / "raw.mp4"),
        "matrix_path": "matrix.npz",
        "matrix_sha256": sha256(tmp_path / "matrix.npz"),
        "feature_path": "cache.npz",
        "curated_count": 2,
        "curated_row_indices": [1, 2],
        "curated_interval_ms": [1000, 2000],
    }
    cache = SimpleNamespace(
        features=original,
        records=[
            SimpleNamespace(quality="valid", row_index=i, start_ms=s, end_ms=e)
            for i, (s, e) in enumerate([(0, 1000), (1000, 1800), (1500, 2000), (1900, 2200)])
        ],
    )
    monkeypatch.setattr(workflow, "ROOT", tmp_path)
    monkeypatch.setattr(workflow, "load_feature_cache", lambda *a, **k: cache)
    np.testing.assert_array_equal(workflow.verified_array(row, "recipe"), original[1:3])
    (tmp_path / "matrix.npz").write_bytes(b"tampered")
    with pytest.raises(ContractError, match="changed"):
        workflow.verified_array(row, "recipe")


def test_mixed_training_isolation_keeps_validation_original_and_rejects_test(monkeypatch):
    original = np.zeros((3, 512), dtype=np.float32)
    external = np.ones((2, 512), dtype=np.float32)

    def matrix(index, split, return_counts=False):
        if split == "validation":
            return "original-validation"
        return original, {"ur": "urhash"}, {"ur": 3}

    monkeypatch.setattr(workflow, "ORIGINAL_MATRIX", matrix)
    monkeypatch.setattr(workflow, "verified_array", lambda *a: external)
    profile = {
        "recipe_hash": "recipe",
        "entries": [{"eligible": True, "sequence_id": "external", "matrix_sha256": "ehash"}],
    }
    with workflow.mixed_normal_inputs(profile):
        array, hashes, counts = experiment.normal_matrix({"recipe_hash": "recipe"}, "train", True)
        np.testing.assert_array_equal(array, np.concatenate([original, external]))
        assert hashes == {"ur": "urhash", "external": "ehash"} and counts == {
            "ur": 3,
            "external": 2,
        }
        assert experiment.normal_matrix({}, "validation") == "original-validation"
        with pytest.raises(ContractError, match="Test"):
            experiment.normal_matrix({}, "test")
        with pytest.raises(ContractError, match="recipes"):
            experiment.normal_matrix({"recipe_hash": "bad"}, "train")
