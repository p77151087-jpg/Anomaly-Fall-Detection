from copy import deepcopy

import pytest

from fall_detect.io_utils import ContractError
from scripts import shoulder_display_review_v2 as review


@pytest.fixture
def pair():
    clips = [
        {
            "sequence_id": sid,
            "video_sha256": "video-" + sid,
            "inherited_annotation": {"kind": "synthetic", "original": sid},
            "reference_sha256": "reference-" + sid,
        }
        for sid in review.CASES
    ]
    bundle = {"bundle_sha256": "synthetic-bundle", "clips": clips}
    annotation = {
        "schema_version": 1,
        "bundle_sha256": "synthetic-bundle",
        "review_scope": "shoulder_demo_display_only",
        "annotation_origin": "human_self_reported",
        "human_attested": True,
        "new_temporal_ground_truth": False,
        "draft": False,
        "reviewer": "synthetic fixture, not a real human review",
        "clips": [
            {
                "sequence_id": c["sequence_id"],
                "video_sha256": c["video_sha256"],
                "reference_sha256": c["reference_sha256"],
                "reviewed": True,
                "timing_relation": "uncertain",
                "subject_pose": "same_person_misaligned",
                "display_clarity": "slow_only",
                "notes": "synthetic",
            }
            for c in clips
        ],
        "inherited_annotations": [
            {
                "sequence_id": c["sequence_id"],
                "annotation": c["inherited_annotation"],
                "reference_sha256": c["reference_sha256"],
            }
            for c in clips
        ],
    }
    return annotation, bundle


def test_uncertain_and_negative_judgments_are_valid_and_keep_annotations(pair):
    annotation, bundle = pair
    old = deepcopy(bundle)
    assert review.validate_review(annotation, bundle)["clips"] == 4
    assert bundle == old
    annotation["clips"][0].update(
        timing_relation="before_fall", subject_pose="wrong_person", display_clarity="unclear"
    )
    assert review.validate_review(annotation, bundle)["valid"]


@pytest.mark.parametrize(
    "mutation",
    [
        "bundle",
        "draft",
        "attest",
        "scope",
        "gt",
        "missing",
        "duplicate",
        "blank",
        "unreviewed",
        "video",
        "reference",
        "inherited",
        "extra_inherited",
        "reviewer",
        "notes",
    ],
)
def test_bad_or_incomplete_exports_rejected(pair, mutation):
    a, b = pair
    if mutation == "bundle":
        a["bundle_sha256"] = "other"
    elif mutation == "draft":
        a["draft"] = True
    elif mutation == "attest":
        a["human_attested"] = False
    elif mutation == "scope":
        a["review_scope"] = "training"
    elif mutation == "gt":
        a["new_temporal_ground_truth"] = True
    elif mutation == "missing":
        a["clips"].pop()
    elif mutation == "duplicate":
        a["clips"][1] = deepcopy(a["clips"][0])
    elif mutation == "blank":
        a["clips"][0]["timing_relation"] = ""
    elif mutation == "unreviewed":
        a["clips"][0]["reviewed"] = False
    elif mutation == "video":
        a["clips"][0]["video_sha256"] = "wrong"
    elif mutation == "reference":
        a["clips"][0]["reference_sha256"] = "wrong"
    elif mutation == "inherited":
        a["inherited_annotations"][0]["annotation"] = {"changed": True}
    elif mutation == "extra_inherited":
        a["inherited_annotations"].append(a["inherited_annotations"][0])
    elif mutation == "reviewer":
        a["reviewer"] = " "
    else:
        a["clips"][0]["notes"] = "x" * 2001
    with pytest.raises(ContractError):
        review.validate_review(a, b)


@pytest.mark.parametrize("payload", [None, [], {}, {"schema_version": 1}])
def test_malformed_json_rejected(payload, pair):
    with pytest.raises(ContractError):
        review.validate_review(payload, pair[1])


def test_build_refuses_existing_destination_before_reading_data(tmp_path, monkeypatch):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("unchanged")
    monkeypatch.setattr(review, "OUT", output)
    monkeypatch.setattr(review, "QA", tmp_path / "qa")
    with pytest.raises(ContractError, match="already exists"):
        review.build()
    assert marker.read_text() == "unchanged"
