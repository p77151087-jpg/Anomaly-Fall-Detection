"""Synthetic safety checks; these test labels are never real human evidence."""

import copy

import pytest

from fall_detect.evaluation.frame_review import (
    annotation_template,
    latest_windows,
    validate_annotation,
)
from fall_detect.io_utils import ContractError


def window(sample, track, time, quality="valid"):
    return {"sample_index": sample, "track_id": track, "available_at_ms": time, "quality": quality}


def manifest():
    return {
        "schema_version": 1,
        "sequence_id": "synthetic-review",
        "split": "validation",
        "frames": [
            {
                "source_frame_id": i,
                "timestamp_ms": (i - 1) * 1000 / 30,
                "source_sha256": str(i) * 64,
            }
            for i in range(1, 5)
        ],
    }


def completed_fixture():
    m = manifest()
    a = annotation_template(m, "synthetic-manifest-hash")
    a.update(
        reviewer="synthetic-test-person",
        human_attested=True,
        onset_bounds_frame_ids=[1, 2],
        impact_bounds_frame_ids=[3, 4],
    )
    for f in a["frames"]:
        f.update(phase="uncertain", visibility="uncertain", pose_check="uncertain", reviewed=True)
    return m, a


def test_alignment_never_uses_future_or_backfills_first_frame():
    rows = [window(3, 1, 100), window(1, 1, 33.3), window(2, 1, 66.6)]
    assert latest_windows(rows, 0) == []
    assert latest_windows(rows, 50)[0]["sample_index"] == 1
    assert latest_windows(rows, 100)[0]["sample_index"] == 3
    assert rows[0] == window(3, 1, 100)


def test_track_gap_exposes_staleness_and_current_unknown():
    rows = [window(1, 1, 0), window(3, 2, 80, "unknown"), window(2, 1, 30, "low_quality")]
    actual = {r["track_id"]: r for r in latest_windows(rows, 120)}
    assert actual[1]["quality"] == "low_quality"
    assert actual[1]["age_ms"] == 90 and not actual[1]["fresh"]
    assert actual[2]["quality"] == "unknown" and actual[2]["fresh"]


@pytest.mark.parametrize("time", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_source_time_rejected(time):
    with pytest.raises(ContractError):
        latest_windows([], time)


def test_template_is_unreviewed_and_cannot_be_imported():
    m = manifest()
    a = annotation_template(m, "synthetic-manifest-hash")
    assert not a["human_attested"] and all(not f["reviewed"] for f in a["frames"])
    assert a["onset_bounds_frame_ids"] is None
    with pytest.raises(ContractError):
        validate_annotation(a, m, "synthetic-manifest-hash")


def test_valid_synthetic_attestation_stays_diagnostic_and_uncertain():
    m, a = completed_fixture()
    result = validate_annotation(a, m, "synthetic-manifest-hash")
    assert result["validated"]
    assert not result["identity_independently_verified"]
    assert not result["used_for_training"] and not result["used_for_calibration"]
    assert result["reviewed_frames"] == 4


@pytest.mark.parametrize(
    "key,value",
    [
        ("split", "test"),
        ("sequence_id", "other"),
        ("manifest_sha256", "changed"),
        ("human_attested", False),
        ("human_attested", 1),
        ("reviewer", ""),
        ("reviewer", " AI "),
        ("reviewer", "Codex"),
        ("reviewer", None),
        ("annotation_origin", "ai_visual_diagnostic"),
        ("annotation_origin", "unconfirmed_draft"),
        ("schema_version", 2),
        ("frames", None),
        ("frames", {}),
        ("onset_bounds_frame_ids", [2, 1]),
        ("onset_bounds_frame_ids", [False, 2]),
        ("onset_bounds_frame_ids", [1.0, 2]),
        ("impact_bounds_frame_ids", [3, 5]),
        ("impact_bounds_frame_ids", None),
        ("onset_bounds_frame_ids", [2, 4]),
    ],
)
def test_unconfirmed_ai_wrong_source_or_invalid_bounds_rejected(key, value):
    m, a = completed_fixture()
    a[key] = value
    with pytest.raises(ContractError):
        validate_annotation(a, m, "synthetic-manifest-hash")


@pytest.mark.parametrize(
    "key,value",
    [
        ("source_frame_id", 4),
        ("timestamp_ms", 900),
        ("source_sha256", "wrong"),
        ("reviewed", False),
        ("phase", "unreviewed"),
        ("visibility", "unreviewed"),
        ("pose_check", "unreviewed"),
    ],
)
def test_each_frame_identity_and_all_three_labels_checked(key, value):
    m, a = completed_fixture()
    a["frames"][0][key] = value
    with pytest.raises(ContractError):
        validate_annotation(a, m, "synthetic-manifest-hash")


def test_duplicates_wrong_order_or_non_mapping_rows_rejected():
    m, a = completed_fixture()
    for rows in [[a["frames"][0]] * 4, list(reversed(a["frames"])), [None] * 4, a["frames"][:3]]:
        bad = copy.deepcopy(a)
        bad["frames"] = rows
        with pytest.raises(ContractError):
            validate_annotation(bad, m, "synthetic-manifest-hash")


def test_test_manifest_cannot_import_even_attested_labels():
    m, a = completed_fixture()
    m["split"] = "test"
    with pytest.raises(ContractError):
        validate_annotation(a, m, "synthetic-manifest-hash")
