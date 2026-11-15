"""Run the locked ADL-only AE comparison through the shared training APIs."""

import csv
from datetime import UTC, datetime

from fall_detect.config import load_config, project_path
from fall_detect.evaluation.metrics import sequence_metrics
from fall_detect.io_utils import ContractError, read_json, sha256, write_json
from fall_detect.training.experiment import calibrate, train, verify_artifacts


def progress(stage, **fields):
    record = {"stage": stage, "updated_at": datetime.now(UTC).isoformat(), **fields}
    write_json(project_path("docs/validation/refinement-workflow-progress.json"), record)
    print(record, flush=True)


def summarize(name):
    run = project_path("runs") / name
    frozen = verify_artifacts(run)
    calibration = read_json(run / "calibration.json")
    threshold = calibration["chosen"]["threshold"]
    # Calibration's raw sequence rows predate threshold selection. Recompute labels.
    rows = [
        {
            **{k: v for k, v in row.items() if k != "first_alarm_ms"},
            "prediction": None
            if row["sequence_score"] is None
            else int(row["sequence_score"] > threshold),
        }
        for row in calibration["sequences"]
    ]
    manifest = read_json(run / "split-manifest.json")
    datasets = {row["sequence_id"]: row["dataset"] for row in manifest["sequences"]}
    with (run / "training-history.csv").open(encoding="utf-8-sig", newline="") as stream:
        history = list(csv.DictReader(stream))
    best = min(history, key=lambda row: float(row["val_normal_loss"]))
    metrics = sequence_metrics(rows)
    expected = calibration["chosen"]["metrics"]
    if any(metrics[k] != expected[k] for k in ("TP", "TN", "FP", "FN", "n_unknown")):
        raise ContractError("Validation summary disagrees with frozen calibration")
    return {
        "run": name,
        "epochs": len(history),
        "best_epoch": int(best["epoch"]),
        "best_val_normal_loss": float(best["val_normal_loss"]),
        "threshold": threshold,
        "validation_metrics": metrics,
        "by_dataset_validation": {
            dataset: sequence_metrics([r for r in rows if datasets[r["sequence_id"]] == dataset])
            for dataset in sorted(set(datasets.values()))
        },
        "frozen_artifacts": frozen,
    }


def main():
    lock = read_json(project_path("docs/validation/refinement-lock.json"))
    index_path = project_path(lock["index_path"])
    if sha256(index_path) != lock["index_sha256"]:
        raise ContractError("Feature index changed after experiment lock")
    for name, expected in lock["preserved_runs"].items():
        if verify_artifacts(project_path("runs") / name) != expected:
            raise ContractError(f"Existing run changed: {name}")
    for candidate in lock["candidates"]:
        path = project_path(candidate["config"])
        if sha256(path) != candidate["config_sha256"]:
            raise ContractError("Candidate config changed after experiment lock")
        if (project_path("runs") / candidate["run"]).exists():
            raise ContractError("Candidate run already exists; do not overwrite or silently resume")
    rows = [summarize(lock["baseline_run"])]
    for candidate in lock["candidates"]:
        progress("training", run=candidate["run"], completed=rows)
        config = load_config(project_path(candidate["config"]), "cuda")
        run = project_path("runs") / candidate["run"]
        train(config, index_path, run)
        calibrate(run)
        rows.append(summarize(candidate["run"]))
        progress("candidate_complete", run=candidate["run"], completed=rows)
    chosen = max(
        rows,
        key=lambda row: (
            row["validation_metrics"]["operational_accuracy"],
            row["validation_metrics"]["end_to_end_fall_recall"],
            -row["validation_metrics"]["FP"],
        ),
    )
    for name, expected in lock["preserved_runs"].items():
        if verify_artifacts(project_path("runs") / name) != expected:
            raise ContractError(f"Existing run changed during experiment: {name}")
    freeze = {
        "created_at": datetime.now(UTC).isoformat(),
        "selection_split": "validation",
        "test_used_for_selection": False,
        "test_evaluated_in_this_script": False,
        "selected_run": chosen["run"],
        "candidates": rows,
        "lock_sha256": sha256(project_path("docs/validation/refinement-lock.json")),
    }
    write_json(project_path("docs/validation/refinement-before-test-freeze.json"), freeze)
    progress("comparison_complete_test_not_run", selected_run=chosen["run"], completed=rows)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        progress("failed", reason=f"{type(exc).__name__}: {exc}")
        raise
