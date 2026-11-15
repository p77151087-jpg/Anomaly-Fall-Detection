"""Risks specific to read-only joint localization; no real Test data."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/localize_fall19_joints.py"
spec = importlib.util.spec_from_file_location("joint_localization", SCRIPT)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def frames():
    xy = np.arange(26, dtype=float).reshape(13, 2) + 100
    result = []
    for fid in (44, 45):
        result.append({
            "source_frame_id": fid,
            "timestamp_ms": (fid - 44) * 33.0,
            "track_id": 1,
            "keypoints": np.column_stack((xy + (fid - 44) * np.array((5, 8)), np.ones(13))).tolist(),
        })
    return result


def test_translation_is_separate_from_joint_residual():
    rows, bones = diagnostic.measurements(frames())
    assert len(rows) == 26 and len(bones) == 24
    assert all(r["step_px"] is None for r in rows[:13])
    assert all(r["step_px"] == pytest.approx(np.hypot(5, 8)) for r in rows[13:])
    assert all(r["median_motion_residual_px"] == 0 for r in rows[13:])
    # Confidence changes do not alter XY measurements or declare points incorrect.
    observations = frames()
    observations[1]["keypoints"][12][2] = 0.01
    altered, _ = diagnostic.measurements(observations)
    assert altered[-1]["median_motion_residual_px"] == 0
    assert altered[-1]["human_joint_confirmation"] == "not_provided_by_whole_pose_label"


def test_single_joint_jump_is_not_absorbed_as_person_translation():
    observations = frames()
    observations[1]["keypoints"][12][0] += 50
    before = copy.deepcopy(observations)
    rows, _ = diagnostic.measurements(observations)
    assert rows[-1]["median_motion_residual_px"] == pytest.approx(50)
    assert rows[-1]["human_joint_confirmation"] == "not_provided_by_whole_pose_label"
    assert observations == before


@pytest.mark.parametrize("field,value", [
    ("source_frame_id", 46), ("track_id", 2), ("timestamp_ms", 0),
])
def test_no_bridging_gap_track_or_reversed_time(field, value):
    observations = frames()
    observations[1][field] = value
    with pytest.raises(ValueError, match="No bridging"):
        diagnostic.measurements(observations)


def test_finite_coordinates_required():
    observations = frames()
    observations[1]["keypoints"][1][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        diagnostic.measurements(observations)


def test_test_metadata_rejected_before_array_or_image_access(monkeypatch):
    monkeypatch.setattr(diagnostic, "read", lambda _: {"sequence_id": "fall-19", "split": "test"})

    def forbidden(*args, **kwargs):
        pytest.fail("A Test array or image was accessed")

    monkeypatch.setattr(diagnostic.np, "load", forbidden)
    monkeypatch.setattr(diagnostic.Image, "open", forbidden)
    with pytest.raises(ValueError, match="Validation fall-19 only"):
        diagnostic.load_original()


def test_existing_output_refused_before_source_read(monkeypatch, tmp_path):
    monkeypatch.setattr(diagnostic, "OUT", tmp_path)
    monkeypatch.setattr(diagnostic, "load_original", lambda: pytest.fail("Source read before refusal"))
    with pytest.raises(ValueError, match="refuse overwrite"):
        diagnostic.inspect()


def test_existing_review_refused_before_any_loading(monkeypatch, tmp_path):
    (tmp_path / "review").mkdir()
    monkeypatch.setattr(diagnostic, "OUT", tmp_path)
    monkeypatch.setattr(diagnostic, "read", lambda _: pytest.fail("Loaded before refusal"))
    with pytest.raises(ValueError, match="refuse overwrite"):
        diagnostic.package(tmp_path / "synthetic.json")


def test_joint_mapping_matches_project_contract():
    from fall_detect.skeleton.preprocessing import HUMAN13_NAMES

    assert diagnostic.NAMES == HUMAN13_NAMES
    assert diagnostic.FRAMES == tuple(range(44, 52))
