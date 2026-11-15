"""Read-only, hash-checked inputs for the course results notebook and film."""

import csv
import math
from pathlib import Path

from ..config import project_path
from ..io_utils import ContractError, read_json, sha256
from ..training.experiment import verify_artifacts
from .metrics import sequence_metrics


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def numeric_rows(path):
    integer = {"label", "prediction", "n_scheduled_endpoints", "n_valid_endpoints"}
    number = {"sequence_score", "first_alarm_ms", "window_coverage"}
    result = csv_rows(path)
    for row in result:
        for key in integer | number:
            if key in row:
                row[key] = (int if key in integer else float)(row[key]) if row[key] else None
    return result


def verify_metrics(rows, metrics):
    calculated = sequence_metrics(rows)
    for key, value in calculated.items():
        if metrics.get(key) != value:
            raise ContractError(f"Metric/denominator mismatch: {key}")
    return calculated


def verify_report(directory, run, split):
    directory, run = Path(directory), Path(run)
    metrics = read_json(directory / "metrics.json")
    artifacts = verify_artifacts(run)
    if artifacts != metrics["artifact_hashes"]:
        raise ContractError("Evaluation/model hash mismatch")
    rows = numeric_rows(directory / "sequence-results.csv")
    manifest = read_json(run / "split-manifest.json")
    planned = {r["sequence_id"]: r["label"] for r in manifest["sequences"] if r["split"] == split}
    if {r["sequence_id"]: r["label"] for r in rows} != planned:
        raise ContractError("Evaluation split membership mismatch")
    verify_metrics(rows, metrics)
    table = csv_rows(directory / "outcomes-with-unknown.csv")
    expected = [[int(r[k]) for k in ("normal", "suspected_fall", "unknown")] for r in table]
    if expected != metrics["outcomes_with_unknown"]:
        raise ContractError("Unknown outcome table mismatch")
    threshold = read_json(run / "threshold.json")["threshold"]
    windows = csv_rows(directory / "window-results.csv")
    events = csv_rows(directory / "events.csv")
    for row in rows:
        scoped = [w for w in windows if w["sequence_id"] == row["sequence_id"]]
        scores = [float(w["event_score"]) for w in scoped if w["event_score"]]
        score = max(scores) if scores else None
        prediction = int(score > threshold) if score is not None else None
        if score != row["sequence_score"] or prediction != row["prediction"]:
            raise ContractError("Sequence/window prediction mismatch")
        alarms = [e for e in events if e["sequence_id"] == row["sequence_id"]]
        if bool(alarms) != (prediction == 1):
            raise ContractError("Event/sequence alarm mismatch")
        if alarms and not math.isclose(
            float(alarms[0]["confirmed_start_ms"]), row["first_alarm_ms"], abs_tol=1e-8
        ):
            raise ContractError("First alarm timestamp mismatch")
    return {"metrics": metrics, "rows": rows, "events": events, "windows": windows}


def load_course_results():
    """Fail closed on missing/stale files. Never trains, calibrates or evaluates a model."""
    manifest = read_json(project_path("docs/validation/course-report-inputs.json"))
    for relative, expected in manifest["files"].items():
        path = project_path(relative)
        if not path.is_file() or sha256(path) != expected:
            raise ContractError(f"Course report input missing or changed: {relative}")
    freeze = read_json(project_path("docs/validation/course-before-test-freeze.json"))
    run = project_path("runs") / freeze["selected_run"]
    for candidate in freeze["candidates"]:
        if (
            verify_artifacts(project_path("runs") / candidate["run"])
            != candidate["frozen_artifacts"]
        ):
            raise ContractError("Frozen course candidate changed")
    test = verify_report(project_path("outputs/course-ur-test"), run, "test")
    validation = verify_report(project_path("outputs/course-ur-validation"), run, "validation")
    source = read_json(run / "sources.json")
    split = read_json(run / "split-manifest.json")
    train_ids = {
        r["sequence_id"] for r in split["sequences"] if r["split"] == "train" and r["label"] == 0
    }
    if set(source["train_feature_hashes"]) != train_ids:
        raise ContractError("Train ADL source membership mismatch")
    mixed_run = project_path("runs/caucafall-yolo11-balanced-001")
    mixed = read_json(project_path("outputs/refinement-test/metrics.json"))
    if verify_artifacts(mixed_run) != mixed["artifact_hashes"]:
        raise ContractError("Mixed comparison/model hash mismatch")
    mixed_ids = {
        r["sequence_id"]
        for r in read_json(mixed_run / "split-manifest.json")["sequences"]
        if r["split"] == "test" and r["dataset"] == "urfall"
    }
    mixed_rows = [
        r
        for r in numeric_rows(project_path("outputs/refinement-test/sequence-results.csv"))
        if r["sequence_id"] in mixed_ids
    ]
    verify_metrics(mixed_rows, mixed["by_dataset"]["urfall"])
    return {
        "run": run,
        "freeze": freeze,
        "test": test,
        "validation": validation,
        "sources": source,
        "split": split,
        "mixed_ur": mixed["by_dataset"]["urfall"],
        "history": {
            c["run"]: csv_rows(project_path("runs") / c["run"] / "training-history.csv")
            for c in freeze["candidates"]
        },
        "input_manifest": manifest,
    }
