"""Synthetic fixtures verify amendment semantics; not human or real-model evidence."""

import copy

import pytest

from fall_detect.evaluation.annotation_amendment import amend_phase_annotation
from fall_detect.evaluation.annotation_diagnosis import diagnose
from fall_detect.io_utils import ContractError


def fixture():
    frames = [
        {
            "source_frame_id": i,
            "timestamp_ms": (i - 1) * 33,
            "source_sha256": str(i),
            "poses": [] if i < 3 else [{"track_id": 1, "current_good_pose": True}],
        }
        for i in range(1, 7)
    ]
    manifest = {"sequence_id": "fall-19", "split": "validation", "frames": frames}
    original = {
        "schema_version": 1,
        "sequence_id": "fall-19",
        "split": "validation",
        "manifest_sha256": "manifest-sha",
        "annotation_origin": "human_self_reported",
        "human_attested": True,
        "reviewer": "Synthetic reviewer",
        "frames": [
            {
                **{k: f[k] for k in ["source_frame_id", "timestamp_ms", "source_sha256"]},
                "reviewed": True,
                "phase": "upright" if i < 3 else "transition" if i < 5 else "ground",
                "visibility": "full",
                "pose_check": "wrong_or_drift" if i == 4 else "matches_person",
            }
            for i, f in enumerate(frames, 1)
        ],
        "onset_bounds_frame_ids": [3, 4],
        "impact_bounds_frame_ids": [5, 6],
    }
    data = {
        "frames": frames,
        "models": {"frozen": {"threshold": 0.5}},
        "windows": {
            "frozen": [
                {
                    "track_id": 1,
                    "available_at_ms": 99,
                    "quality": "valid",
                    "raw_score": 0.1,
                    "event_score": 0.1,
                }
            ]
        },
    }
    amendment = {
        "schema_version": 1,
        "origin": "human_text_clarification",
        "human_attested": True,
        "original_sha256": "original-sha",
        "range_semantics": "whole_posture_phases",
        "pose_corrections": [
            {"source_frame_id": 1, "before": "matches_person", "after": "missing"},
            {"source_frame_id": 2, "before": "matches_person", "after": "missing"},
        ],
    }
    return original, manifest, data, amendment


def apply(original, manifest, data, amendment):
    return amend_phase_annotation(
        original, manifest, "manifest-sha", "original-sha", data, amendment
    )


def test_only_explicit_pose_corrections_and_phase_meaning_change_without_mutating_inputs():
    o, m, d, a = fixture()
    snapshot = copy.deepcopy((o, m, d, a))
    result = apply(o, m, d, a)
    assert (o, m, d, a) == snapshot
    expected_frames = copy.deepcopy(o["frames"])
    for r in expected_frames[:2]:
        r["pose_check"] = "missing"
    assert result["frames"] == expected_frames
    assert result["phase_ranges_frame_ids"] == {"transition": [3, 4], "ground": [5, 6]}
    assert result["schema_version"] == 2 and o["schema_version"] == 1
    assert result["onset_bounds_frame_ids"] is None and result["impact_bounds_frame_ids"] is None
    assert not result["precise_time_ground_truth"]
    assert not result["used_for_training"] and not result["used_for_calibration"]


def test_diagnosis_resolves_warnings_and_preserves_frozen_method_results():
    o, m, d, a = fixture()
    before = diagnose(o, d)
    after = diagnose(apply(o, m, d, a), d)
    assert len(before["semantic_warnings"]) == 3 and not after["semantic_warnings"]
    assert before["methods"] == after["methods"]
    assert before["window_rows"] == after["window_rows"]
    assert after["wrong_but_confidence_good_frames"] == [4]
    assert not after["precise_onset_latency_computed"]


@pytest.mark.parametrize(
    "change",
    [
        "test",
        "ai",
        "not_attested",
        "wrong_sha",
        "event_bounds",
        "extra_pose",
        "cached_pose",
        "changed_source",
        "unreviewed",
        "other_sequence",
        "non_phase_bounds",
    ],
)
def test_non_attested_misaligned_or_unauthorized_amendments_rejected(change):
    o, m, d, a = fixture()
    if change == "test":
        o["split"] = "test"
    elif change == "ai":
        a["origin"] = "ai_visual_diagnostic"
    elif change == "not_attested":
        a["human_attested"] = False
    elif change == "wrong_sha":
        a["original_sha256"] = "wrong"
    elif change == "event_bounds":
        a["range_semantics"] = "event_uncertainty_bounds"
    elif change == "extra_pose":
        a["pose_corrections"].append({"source_frame_id": 3, "after": "missing"})
    elif change == "cached_pose":
        d["frames"][0]["poses"] = [{"track_id": 1, "current_good_pose": True}]
    elif change == "changed_source":
        d["frames"][0]["source_sha256"] = "changed"
    elif change == "unreviewed":
        o["frames"][0]["reviewed"] = False
    elif change == "other_sequence":
        o["sequence_id"] = m["sequence_id"] = "fall-20"
    else:
        o["onset_bounds_frame_ids"] = [3, 3]
    with pytest.raises(ContractError):
        apply(o, m, d, a)
