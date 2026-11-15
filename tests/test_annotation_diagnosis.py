"""Synthetic annotation fixtures are not real human evidence."""

import copy

import pytest

from fall_detect.evaluation.annotation_diagnosis import diagnose
from fall_detect.io_utils import ContractError


def fixture():
    times = [0, 33, 67, 100]
    frames = [
        {
            "source_frame_id": i + 1,
            "timestamp_ms": t,
            "source_sha256": str(i),
            "poses": ([] if i == 0 else [{"track_id": 1, "current_good_pose": True}]),
        }
        for i, t in enumerate(times)
    ]
    labels = [
        {
            "source_frame_id": f["source_frame_id"],
            "timestamp_ms": f["timestamp_ms"],
            "source_sha256": f["source_sha256"],
            "phase": "upright" if i < 2 else "transition" if i == 2 else "ground",
            "visibility": "full",
            "pose_check": "matches_person" if i < 2 else "wrong_or_drift",
            "reviewed": True,
        }
        for i, f in enumerate(frames)
    ]
    annotation = {
        "split": "validation",
        "annotation_origin": "human_self_reported",
        "frames": labels,
        "onset_bounds_frame_ids": [3, 3],
        "impact_bounds_frame_ids": [4, 4],
    }
    rows = [
        {
            "track_id": 1,
            "sample_index": i,
            "available_at_ms": t,
            "quality": "valid",
            "raw_score": 0.1,
            "event_score": None if i < 2 else 0.1,
        }
        for i, t in enumerate([33, 66.666, 100])
    ]
    data = {"frames": frames, "windows": {"frozen": rows}, "models": {"frozen": {"threshold": 0.5}}}
    return annotation, data


def test_causal_join_does_not_move_future_transition_to_endpoint():
    a, d = fixture()
    before = copy.deepcopy((a, d))
    result = diagnose(a, d)
    row = result["window_rows"][1]
    assert (
        row["last_actual_source_frame_id"] == 2 and row["human_phase_at_last_source"] == "upright"
    )
    assert (a, d) == before


def test_wrong_confident_pose_and_absent_pose_warning_stay_descriptive():
    a, d = fixture()
    result = diagnose(a, d)
    assert result["wrong_but_confidence_good_frames"] == [3, 4]
    assert {w["code"] for w in result["semantic_warnings"]} == {
        "matches_person_without_cached_pose",
        "boundary_fields_equal_whole_phase_runs",
    }
    assert not result["used_for_training"] and not result["used_for_calibration"]
    assert not result["precise_onset_latency_computed"]


def test_unknown_is_kept_and_no_valid_ground_score_is_null():
    a, d = fixture()
    d["windows"]["frozen"][-1].update(quality="track_gap", raw_score=None, event_score=None)
    result = diagnose(a, d)["methods"]["frozen"]["ground"]
    assert result["scheduled"] == 1 and result["valid"] == 0 and result["coverage"] == 0
    assert result["max_raw"] is None and result["max_k3"] is None


def test_pose_from_other_track_does_not_establish_good_endpoint_pose():
    a, d = fixture()
    d["frames"][-1]["poses"][0]["track_id"] = 2
    row = diagnose(a, d)["window_rows"][-1]
    assert not row["current_track_pose_present"] and not row["current_track_good"]


@pytest.mark.parametrize("change", ["test", "ai", "source", "time", "empty"])
def test_non_validation_or_misaligned_source_rejected(change):
    a, d = fixture()
    if change == "test":
        a["split"] = "test"
    elif change == "ai":
        a["annotation_origin"] = "ai_visual_diagnostic"
    elif change == "source":
        a["frames"][0]["source_sha256"] = "wrong"
    elif change == "time":
        d["frames"][2]["timestamp_ms"] = 33
        a["frames"][2]["timestamp_ms"] = 33
    else:
        d["frames"] = []
    with pytest.raises(ContractError):
        diagnose(a, d)
