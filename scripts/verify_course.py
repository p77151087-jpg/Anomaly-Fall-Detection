"""Verify course artifacts without retraining, calibration, or Test inference."""

import argparse
from datetime import UTC, datetime

import cv2
import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

from fall_detect.config import project_path
from fall_detect.detection.events import score_sequence
from fall_detect.evaluation.course_report import csv_rows, load_course_results
from fall_detect.io_utils import ContractError, read_json, sha256, write_json
from fall_detect.training.experiment import Scorer, cache_for, verify_artifacts
from fall_detect.visualization.video import demo

CASES = ("adl-04", "fall-02", "adl-20", "fall-19")


def verify():
    result = load_course_results()
    lock = read_json(project_path("docs/validation/course-experiment-lock.json"))
    for name, expected in lock["preserved_runs"].items():
        if verify_artifacts(project_path("runs") / name) != expected:
            raise ContractError(f"Historical run changed: {name}")
    for candidate in lock["candidates"]:
        if sha256(project_path(candidate["config"])) != candidate["config_sha256"]:
            raise ContractError("Locked config changed")
    if sha256(project_path(lock["reuse_index"])) != lock["reuse_index_sha256"]:
        raise ContractError("Historical dataset index changed")
    executed = read_json(project_path("docs/validation/course-notebook-execution.json"))
    if sha256(project_path("notebooks/course-results.ipynb")) != executed["notebook_sha256"]:
        raise ContractError("Executed notebook changed")
    print(
        f"PASS: {result['run'].name}; Test={result['test']['metrics']['n_total']}; {len(lock['preserved_runs'])} historical frozen runs unchanged"
    )
    return result


def integration():
    result = verify()
    report_path = project_path("docs/validation/course-demo-integration.json")
    if report_path.exists():
        raise ContractError("Integration evidence exists")
    run = result["run"]
    scorer = Scorer(run)
    reports, screenshots = [], {}
    for sid in CASES:
        folder = project_path("outputs/course-demo")
        before, after = folder / f"{sid}-debug.mp4", folder / f"{sid}-presentation.mp4"
        if not before.exists():
            demo(run, project_path(f"data/raw/urfall/{sid}"), before, "cuda", overlay_mode="debug")
        for suffix in (".csv", "-events.csv"):
            left = folder / f"{sid}-debug{suffix}"
            right = folder / f"{sid}-presentation{suffix}"
            if left.read_bytes() != right.read_bytes():
                raise ContractError(f"Rendering changed inference: {sid} {suffix}")
        before_meta, after_meta = (
            read_json(before.with_suffix(".json")),
            read_json(after.with_suffix(".json")),
        )
        if (
            before_meta["config"] != after_meta["config"]
            or before_meta["frames"] != after_meta["frames"]
        ):
            raise ContractError("Rendering changed config or timing")
        cache = cache_for(scorer.index, sid)
        _, windows, batch_events = score_sequence(
            cache, scorer.score(cache.features), scorer.threshold["threshold"]
        )
        logs = csv_rows(after.with_suffix(".csv"))
        shown = {(int(r["sample_index"]), int(r["track_id"])): r for r in logs if r["track_id"]}
        differences = []
        masked = 0
        for row in windows:
            if row["raw_score"] is None:
                continue
            visible = shown.get((row["sample_index"], row["track_id"]))
            if not visible or visible["raw_score"] == "":
                masked += 1
                continue
            differences.append(abs(float(visible["raw_score"]) - row["raw_score"]))
        if not differences or max(differences) > 1e-6:
            raise ContractError("Cached and fresh-inference scores differ")
        actual_events = csv_rows(folder / f"{sid}-presentation-events.csv")
        if len(actual_events) != len(batch_events):
            raise ContractError("Cached and fresh events differ")
        for actual, expected in zip(actual_events, batch_events):
            for key in ("confirmed_start_ms", "last_active_ms", "end_ms", "peak_score"):
                if abs(float(actual[key]) - expected[key]) > 1e-6:
                    raise ContractError(f"Event differs: {sid} {key}")
            if actual["end_reason"] != expected["end_reason"]:
                raise ContractError("Event end reason differs")
        invalid = [
            r for r in logs if r["state"] == "unknown" and (r["raw_score"] or r["event_score"])
        ]
        if invalid:
            raise ContractError("Unknown exposed old scores")
        for state in ("warming_up", "normal", "suspected_fall", "unknown"):
            if state in screenshots:
                continue
            matches = [
                r
                for r in logs
                if r["state"] == state and (state != "unknown" or float(r["timestamp_ms"]) > 1500)
            ]
            if not matches:
                continue
            row = matches[len(matches) // 2]
            cap = cv2.VideoCapture(str(after))
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(row["sample_index"]))
            ok, frame = cap.read()
            cap.release()
            if not ok:
                raise ContractError("Cannot decode state screenshot")
            path = folder / f"state-{state}.png"
            if path.exists():
                raise ContractError("Screenshot evidence exists")
            cv2.imencode(".png", frame)[1].tofile(path)
            screenshots[state] = {
                "sequence_id": sid,
                "sample_index": int(row["sample_index"]),
                "timestamp_ms": float(row["timestamp_ms"]),
                "path": str(path),
                "sha256": sha256(path),
            }
        reports.append(
            {
                "sequence_id": sid,
                "split": "validation",
                "frames": after_meta["frames"],
                "logs_byte_identical": True,
                "events_byte_identical": True,
                "fresh_scores_compared_with_cache": len(differences),
                "masked_valid_endpoints": masked,
                "max_abs_score_difference": max(differences),
                "unknown_stale_score_violations": 0,
                "decoded_verified": after_meta["decoded_verified"],
                "fps": after_meta["fps"],
                "states": sorted({r["state"] for r in logs}),
                "presentation_sha256": sha256(after),
                "logs_sha256": sha256(after.with_suffix(".csv")),
            }
        )
    if len(screenshots) != 4:
        raise ContractError("Missing real presentation state")
    verify()
    write_json(
        report_path,
        {
            "source_kind": "real_end_to_end",
            "created_at": datetime.now(UTC).isoformat(),
            "run": run.name,
            "cases": reports,
            "screenshots": screenshots,
            "rendering_only": True,
            "frozen_artifacts": verify_artifacts(run),
        },
    )
    print(report_path)


def notebook():
    verify()
    nb = nbformat.read(project_path("notebooks/course-results.ipynb"), as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    manager = KernelManager(
        kernel_name="course",
        kernel_spec_manager=KernelSpecManager(
            kernel_dirs=[str(project_path(".artifacts/course-kernels"))]
        ),
    )
    NotebookClient(
        nb, km=manager, timeout=180, resources={"metadata": {"path": str(project_path("."))}}
    ).execute()
    destination = (
        project_path("outputs")
        / f"course-notebook-check-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.ipynb"
    )
    nbformat.write(nb, destination)
    print(f"PASS: clean-kernel re-execution: {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", nargs="?", default="verify", choices=["verify", "integration", "notebook"]
    )
    args = parser.parse_args()
    globals()[args.stage]()
