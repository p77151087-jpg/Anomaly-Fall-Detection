"""Repair report metadata only; reuse frozen calibration and retain failed output."""

from unittest.mock import patch

import ur_pose_resolution as candidate

from fall_detect.io_utils import ContractError, read_json, sha256


def resume():
    candidate.verify_lock()
    run_hashes = {
        str(p.relative_to(candidate.RUN)): sha256(p)
        for p in candidate.RUN.rglob("*")
        if p.is_file()
    }
    status = read_json(candidate.RUN / "status.json")
    threshold = read_json(candidate.RUN / "threshold.json")
    if status["training"] != "complete" or status["calibration"] != "complete":
        raise ContractError("Only a completed frozen calibration can be reused")
    failed = candidate.E / "validation-report-failed-output"
    if not failed.is_dir() or candidate.O.exists():
        raise ContractError("Retain the failed output and require an empty new output path")
    if (candidate.E / "validation-report-repair.json").exists():
        raise ContractError("Preserve the completed report repair")
    original_report = candidate.evaluation_report

    def report(directory, metrics, rows, windows, events):
        metrics["protocol_id"] = threshold["protocol_id"]
        return original_report(directory, metrics, rows, windows, events)

    def frozen_calibration(run):
        if run != candidate.RUN:
            raise ContractError("Unexpected calibration target")
        if read_json(run / "threshold.json") != threshold:
            raise ContractError("Frozen threshold changed")

    with (
        patch.object(candidate, "evaluation_report", report),
        patch.object(candidate.experiment, "calibrate", frozen_calibration),
    ):
        candidate.validation()
    after = {
        str(p.relative_to(candidate.RUN)): sha256(p)
        for p in candidate.RUN.rglob("*")
        if p.is_file()
    }
    if after != run_hashes:
        raise ContractError("Report repair changed the frozen model/calibration run")
    candidate.save(
        candidate.E / "validation-report-repair.json",
        {
            "passed": True,
            "reason": "Missing protocol_id in report metadata",
            "change": "Add the original protocol_id to both report metadata dictionaries",
            "original_locked_source_unchanged": True,
            "model_and_calibration_run_unchanged": True,
            "additional_training": 0,
            "additional_calibrations": 0,
            "threshold": threshold["threshold"],
            "run_hashes": run_hashes,
            "failed_output_hashes": {
                str(p.relative_to(candidate.ROOT)): sha256(p)
                for p in failed.rglob("*")
                if p.is_file()
            },
            "repair_source_sha256": sha256(__file__),
        },
    )


if __name__ == "__main__":
    resume()
