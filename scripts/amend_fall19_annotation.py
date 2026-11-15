"""Save/recheck the human phase clarification; no fitting, calibration, or Test."""

import argparse
import json
import shutil
from pathlib import Path

from fall_detect.evaluation.annotation_amendment import amend_phase_annotation
from fall_detect.evaluation.annotation_diagnosis import diagnose
from fall_detect.io_utils import ContractError, read_json, sha256, write_csv, write_json

ROOT = Path(__file__).resolve().parents[1]
TAG = "ur-fall19-human-amendment-20261013-001"
PARENT = "ur-fall19-human-review-20261012-001"
OLD = "ur-fall19-review-20261012-001"
DIGEST = "f7dbda11fcf42dbcf4f2a203ea0d0c22b70d7fa0b0c7915051be8cc540b88f3c"
E = ROOT / "docs/validation" / TAG
O = ROOT / "outputs" / TAG
IMPORTED = ROOT / "docs/validation" / OLD / "human-annotations" / DIGEST
MUTABLE = ["docs/current-progress.md", "docs/acceptance-results.md"]
SOURCES = [
    "scripts/amend_fall19_annotation.py",
    "src/fall_detect/evaluation/annotation_amendment.py",
    "tests/test_annotation_amendment.py",
    "src/fall_detect/evaluation/annotation_diagnosis.py",
    "src/fall_detect/evaluation/frame_review.py",
    "src/fall_detect/io_utils.py",
]


def check(condition, message):
    if not condition:
        raise ContractError(message)


def clarification():
    return {
        "schema_version": 1,
        "origin": "human_text_clarification",
        "human_attested": True,
        "source": "User's explicit answers in this conversation",
        "recorded_on": "2026-10-13",
        "original_sha256": DIGEST,
        "answers": {
            "range_meaning": "整段姿態階段",
            "frame_1_2_pose": "改為「沒有骨架」",
        },
        "range_semantics": "whole_posture_phases",
        "pose_corrections": [
            {"source_frame_id": 1, "before": "matches_person", "after": "missing"},
            {"source_frame_id": 2, "before": "matches_person", "after": "missing"},
        ],
        "precise_onset_or_impact_provided": False,
        "scope": "Validation diagnosis only; no independent identity or time GT certification",
    }


def amend(save=False):
    if save:
        check(not E.exists() and not O.exists(), "Preserve completed/attempted amendment")
    parent_dir = ROOT / "docs/validation" / PARENT
    parent_qa = parent_dir / "final-qa.json"
    qa = read_json(parent_qa)
    if save:
        # Carry forward the immutable chain, while archiving the two mutable docs.
        assets = dict(read_json(parent_dir / "before.json")["assets"])
        assets.update({n: h for n, h in qa["artifacts"].items() if n not in MUTABLE})
        assets[parent_qa.relative_to(ROOT).as_posix()] = sha256(parent_qa)
        for name, digest in assets.items():
            check(sha256(ROOT / name) == digest, f"Preserve previous artifact: {name}")
        for name in MUTABLE:
            check(sha256(ROOT / name) == qa["artifacts"][name], "Current contract changed")
        E.mkdir(parents=True)
        for name in MUTABLE:
            target = E / "prior-contract" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        inputs = [
            parent_qa,
            IMPORTED / "original-export.json",
            IMPORTED / "validation.json",
            ROOT / "outputs" / OLD / "source-manifest.json",
            ROOT / "outputs" / OLD / "review-data.json",
        ]
        write_json(
            E / "lock.json",
            {
                "source_hashes": {n: sha256(ROOT / n) for n in SOURCES},
                "input_hashes": {p.relative_to(ROOT).as_posix(): sha256(p) for p in inputs},
                "scope": "Human amendment and frozen Validation diagnosis only",
                "no_fitting": True,
                "no_calibration": True,
                "no_test": True,
            },
        )
        write_json(
            E / "before.json",
            {
                "assets": assets,
                "original_contracts": {n: qa["artifacts"][n] for n in MUTABLE},
            },
        )
    lock = read_json(E / "lock.json")
    for category in ["source_hashes", "input_hashes"]:
        for name, digest in lock[category].items():
            check(sha256(ROOT / name) == digest, f"Locked source/input changed: {name}")
    before = read_json(E / "before.json")
    for name, digest in before["assets"].items():
        check(sha256(ROOT / name) == digest, f"Previous artifact changed: {name}")
    for name, digest in before["original_contracts"].items():
        check(sha256(E / "prior-contract" / name) == digest, "Prior contract bytes changed")
    original = read_json(IMPORTED / "original-export.json")
    check(sha256(IMPORTED / "original-export.json") == DIGEST, "Original export changed")
    manifest_path = ROOT / "outputs" / OLD / "source-manifest.json"
    data = read_json(ROOT / "outputs" / OLD / "review-data.json")
    amendment = clarification()
    derived = amend_phase_annotation(
        original, read_json(manifest_path), sha256(manifest_path), DIGEST, data, amendment
    )
    old_diagnosis = diagnose(original, data)
    result = diagnose(derived, data)
    check(result["methods"] == old_diagnosis["methods"], "Frozen method results must not change")
    check(
        result["window_rows"] == old_diagnosis["window_rows"],
        "Clarification must not change window labels/scores/events",
    )
    check(not result["semantic_warnings"], "Unresolved source/semantic inconsistency")
    frames, windows = result.pop("frame_rows"), result.pop("window_rows")
    result.update(
        original_annotation_sha256=DIGEST,
        annotation_schema_version=2,
        human_boundary_semantics="whole_posture_phases",
        phase_ranges_frame_ids=derived["phase_ranges_frame_ids"],
        precise_onset_bounds_frame_ids=None,
        precise_impact_bounds_frame_ids=None,
        precise_time_ground_truth=False,
        semantic_clarifications_pending=False,
        corrected_pose_frame_ids=[1, 2],
        parent_artifacts_unchanged=len(before["assets"]),
        original_contracts_preserved=len(before["original_contracts"]),
        window_rows=len(windows),
        methods_and_window_rows_unchanged=True,
        identity_independently_verified=False,
        new_training_epochs=0,
        test_feature_reads=0,
        new_test_evaluations=0,
        default_model_changed=False,
        accuracy_improved=False,
        evidence_level="real_feature_smoke: human clarification aligned to frozen observations",
    )
    if save:
        O.mkdir(parents=True)
        write_json(O / "human-amendment.json", amendment)
        write_json(O / "derived-annotation.json", derived)
        write_csv(O / "human-frame-diagnosis.csv", frames)
        write_csv(O / "human-window-diagnosis.csv", windows)
    else:
        check(read_json(O / "human-amendment.json") == amendment, "Amendment record changed")
        check(read_json(O / "derived-annotation.json") == derived, "Derived annotation changed")
    result.update(
        amendment_sha256=sha256(O / "human-amendment.json"),
        derived_annotation_sha256=sha256(O / "derived-annotation.json"),
        frame_diagnosis_csv_sha256=sha256(O / "human-frame-diagnosis.csv"),
        window_diagnosis_csv_sha256=sha256(O / "human-window-diagnosis.csv"),
    )
    if save:
        write_json(O / "diagnosis.json", result)
        write_json(E / "diagnosis.json", result)
    else:
        check(read_json(O / "diagnosis.json") == result, "Published diagnosis changed")
        check(read_json(E / "diagnosis.json") == result, "Evidence diagnosis changed")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save", action="store_true", help="First immutable save; refuses overwrite"
    )
    amend(parser.parse_args().save)
