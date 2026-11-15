"""Import follow-up: frozen human-label diagnosis only; default read-only, no Test or fitting."""

import argparse
import json
import shutil
from pathlib import Path

from fall_detect.evaluation.annotation_diagnosis import diagnose
from fall_detect.evaluation.frame_review import validate_annotation
from fall_detect.io_utils import ContractError, read_json, sha256, write_csv, write_json

ROOT = Path(__file__).resolve().parents[1]
TAG = "ur-fall19-human-review-20261012-001"
OLD = "ur-fall19-review-20261012-001"
DIGEST = "f7dbda11fcf42dbcf4f2a203ea0d0c22b70d7fa0b0c7915051be8cc540b88f3c"
E = ROOT / "docs/validation" / TAG
O = ROOT / "outputs" / TAG
IMPORTED = ROOT / "docs/validation" / OLD / "human-annotations" / DIGEST
MUTABLE = ["docs/current-progress.md", "docs/acceptance-results.md"]
SOURCES = [
    "scripts/analyze_fall19_annotation.py",
    "src/fall_detect/evaluation/annotation_diagnosis.py",
    "tests/test_annotation_diagnosis.py",
    "src/fall_detect/evaluation/frame_review.py",
    "src/fall_detect/io_utils.py",
]


def check(condition, message):
    if not condition:
        raise ContractError(message)


def analyze(save=False):
    if save:
        check(not E.exists() and not O.exists(), "Preserve completed/attempted human diagnosis")
    parent_qa = ROOT / "docs/validation" / OLD / "final-qa.json"
    prior = read_json(parent_qa)["artifacts"]
    if save:
        for name, digest in prior.items():
            check(sha256(ROOT / name) == digest, f"Preserve parent artifact: {name}")
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
                "scope": "Validation diagnosis only",
                "no_fitting": True,
                "no_calibration": True,
                "no_test": True,
            },
        )
        write_json(
            E / "before.json",
            {
                "assets": {n: h for n, h in prior.items() if n not in MUTABLE},
                "original_contracts": {n: prior[n] for n in MUTABLE},
                "parent_qa_sha256": sha256(parent_qa),
            },
        )
    lock = read_json(E / "lock.json")
    for category in ["source_hashes", "input_hashes"]:
        for name, digest in lock[category].items():
            check(sha256(ROOT / name) == digest, f"Locked diagnosis input changed: {name}")
    annotation = read_json(IMPORTED / "original-export.json")
    check(sha256(IMPORTED / "original-export.json") == DIGEST, "Imported source bytes changed")
    manifest_path = ROOT / "outputs" / OLD / "source-manifest.json"
    validated = validate_annotation(annotation, read_json(manifest_path), sha256(manifest_path))
    report = diagnose(annotation, read_json(ROOT / "outputs" / OLD / "review-data.json"))
    frame_rows = report.pop("frame_rows")
    window_rows = report.pop("window_rows")
    before = read_json(E / "before.json")
    for name, digest in before["assets"].items():
        check(sha256(ROOT / name) == digest, f"Parent artifact changed: {name}")
    for name, digest in before["original_contracts"].items():
        check(sha256(E / "prior-contract" / name) == digest, "Prior contract bytes changed")
    report.update(
        annotation_sha256=DIGEST,
        imported_validation=validated,
        parent_artifacts_unchanged=len(before["assets"]),
        original_contracts_preserved=len(before["original_contracts"]),
        window_rows=len(window_rows),
        human_boundary_semantics="pending clarification; original ranges retained without reinterpretation",
        new_training_epochs=0,
        test_feature_reads=0,
        new_test_evaluations=0,
        evidence_level="real_feature_smoke: imported human self-report aligned with frozen scores",
    )
    if save:
        O.mkdir(parents=True)
        write_csv(O / "human-frame-diagnosis.csv", frame_rows)
        write_csv(O / "human-window-diagnosis.csv", window_rows)
        write_json(O / "diagnosis.json", report)
        write_json(E / "diagnosis.json", report)
    else:
        check(
            read_json(O / "diagnosis.json") == report,
            "Published diagnosis differs from recomputation",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save", action="store_true", help="Create first immutable report; refuses overwrite"
    )
    analyze(parser.parse_args().save)
