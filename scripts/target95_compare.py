"""Run the pre-locked comparison via core APIs; never evaluate Test or modify old runs."""

import time
from datetime import UTC, datetime

from fall_detect.config import load_config, project_path
from fall_detect.io_utils import ContractError, read_json, sha256, write_json
from fall_detect.pipeline import prepare
from fall_detect.training.experiment import calibrate, train, verify_artifacts


def progress(stage, **fields):
    record = {"stage": stage, "updated_at": datetime.now(UTC).isoformat(), **fields}
    write_json(project_path("docs/validation/target95-workflow-progress.json"), record)
    print(record, flush=True)


def main():
    index_path = project_path("data/cache/caucafall-yolo11/dataset-index.json")
    deadline = time.monotonic() + 7200
    progress("waiting_for_real_preparation")
    while True:
        index = read_json(index_path)
        if "verification" in index:
            if any(r["status"] != "prepared" for r in index["sequences"].values()):
                raise ContractError("Full real preparation failed; no training started")
            break
        if time.monotonic() > deadline:
            raise ContractError("Preparation did not finish within the 2-hour supervision bound")
        time.sleep(3)
    progress("training_yolo", preparation_sha256=sha256(index_path))
    cfg = load_config(project_path("configs/yolo11-caucafall.yaml"), "cuda")
    first_run = project_path("runs/caucafall-yolo11-001")
    train(cfg, index_path, first_run)
    calibrate(first_run)
    progress("preparing_jitter_from_identical_observed_poses")
    jitter = load_config(project_path("configs/yolo11-caucafall-jitter.yaml"), "cuda")
    jitter_path, result = prepare(jitter, reuse_poses_index=index_path)
    if any(r["status"] != "prepared" for r in result["sequences"].values()):
        raise ContractError("Jitter preparation incomplete")
    progress("training_yolo_jitter", preparation_sha256=sha256(jitter_path))
    second_run = project_path("runs/caucafall-yolo11-jitter-001")
    train(jitter, jitter_path, second_run)
    calibrate(second_run)
    rows = []
    for name in [
        "caucafall-extension-001",
        "caucafall-timestamp-001",
        "caucafall-yolo11-001",
        "caucafall-yolo11-jitter-001",
    ]:
        run = project_path("runs") / name
        frozen = verify_artifacts(run)
        calibration = read_json(run / "calibration.json")
        rows.append({"run": name, "chosen": calibration["chosen"], "frozen_artifacts": frozen})
    chosen = max(
        rows,
        key=lambda r: (
            r["chosen"]["metrics"]["operational_accuracy"],
            r["chosen"]["metrics"]["end_to_end_fall_recall"],
            -r["chosen"]["metrics"]["FP"],
        ),
    )
    write_json(
        project_path("docs/validation/target95-before-test-freeze.json"),
        {
            "created_at": datetime.now(UTC).isoformat(),
            "selection_split": "validation",
            "test_used_for_selection": False,
            "selected_run": chosen["run"],
            "candidates": rows,
            "test_evaluated_in_this_script": False,
        },
    )
    progress(
        "comparison_complete_test_not_run",
        selected_run=chosen["run"],
        validation_metrics=chosen["chosen"]["metrics"],
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        progress("failed", error=f"{type(exc).__name__}: {exc}")
        raise
