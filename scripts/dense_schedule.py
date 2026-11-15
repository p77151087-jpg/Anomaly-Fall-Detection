"""Finite UR-only stride experiment; real feature parity and coarse timing checks."""

import argparse
from copy import deepcopy
from datetime import UTC, datetime

import numpy as np
import yaml
from ae_generalization import result

from fall_detect.config import load_config, project_path
from fall_detect.detection.events import score_sequence
from fall_detect.io_utils import ContractError, read_json, sha256, write_json
from fall_detect.pipeline import prepare
from fall_detect.training.experiment import (
    Scorer,
    cache_for,
    calibrate,
    evaluate,
    train,
    verify_artifacts,
)

ROOT = project_path(".")
EVIDENCE = ROOT / "docs/validation"
BASELINE = "ur-ae-decay-001"
PILOT_IDS = [
    "adl-05",
    "adl-24",
    "adl-38",
    "adl-04",
    "adl-20",
    "fall-02",
    "fall-09",
    "fall-19",
    "fall-29",
    "fall-25",
]
# Coarse, visually inspected intervals, not official onset/impact ground truth.
PHASES = {
    "fall-09": {"upright_until_ms": 4304, "fall_phase_ms": [4604, 5506]},
    "fall-19": {"upright_until_ms": 1167, "fall_phase_ms": [1468, 2369]},
    "fall-29": {"upright_until_ms": 1735, "fall_phase_ms": [2035, 2936]},
    "fall-25": {"upright_until_ms": 1568, "fall_phase_ms": [1868, 2802]},
}


def path(stage):
    return EVIDENCE / f"dense-schedule-{stage}.json"


def save(stage, record):
    if path(stage).exists():
        raise ContractError(f"Preserve existing evidence: {stage}")
    write_json(path(stage), record)


def lock():
    if path("lock").exists():
        raise ContractError("Experiment already locked")
    base = load_config(ROOT / f"runs/{BASELINE}/resolved-config.yaml")
    candidates = []
    for stride in [2, 1]:
        run = f"ur-stride{stride}-decay-001"
        config = deepcopy(base)
        config["skeleton"]["stride"] = stride
        config["data"]["cache_dir"] = f"data/cache/{run}"
        filename = ROOT / f"configs/{run}.yaml"
        if (
            filename.exists()
            or project_path(config["data"]["cache_dir"]).exists()
            or (ROOT / "runs" / run).exists()
        ):
            raise ContractError("Candidate already exists")
        filename.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf8")
        candidates.append(
            {"run": run, "config": str(filename.relative_to(ROOT)), "sha256": sha256(filename)}
        )
    source = ROOT / "data/cache/course-ur-yolo11/dataset-index.json"
    save(
        "lock",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "baseline": BASELINE,
            "source_index": str(source.relative_to(ROOT)),
            "source_index_sha256": sha256(source),
            "pilot_ids": PILOT_IDS,
            "candidates": candidates,
            "pilot_gate": "Common valid windows must match ST-GCN features within atol=1e-6 rtol=0; unchanged pose array hash; no originally evaluable pilot sequence becomes unknown",
            "fixed": "30-frame windows; confidence0.2/12 real joints/80% good frames; same observed poses and IDs; K3; sequence sampling; 128/32 AE; weight_decay0.001; noise0; seed42; max100 epochs/patience10",
            "selection": "Among timing-guard passing candidates and baseline: Validation operational_accuracy, F1, all-Fall recall, fewer FP, earlier candidate (baseline first)",
            "timing_guard": "At least one more diagnostic Fall sequence with an event starting inside its coarse fall phase; no more pre-upright-cutoff event starts than baseline; accuracy and F1 not below baseline",
            "coarse_visual_intervals": PHASES,
            "interval_limit": "Assistant visual engineering check on four Validation cases, coarse boundaries with unscored gaps, no official onset/impact annotation, not a timing Accuracy metric",
            "inspection_images": {
                sid: sha256(ROOT / f"outputs/short-window-diagnosis/{sid}.jpg") for sid in PHASES
            },
            "targets": {
                "operational_accuracy": 0.9,
                "sequence_accuracy": 0.9,
                "sequence_fall_f1": 0.87,
            },
            "stop": "Exactly two stride candidates and one fixed AE per candidate; no additional grid; no Test scoring before all Validation targets and timing guard",
            "historic_test_seen": True,
            "repeated_validation_not_independent": True,
            "preserved_runs": {
                p.parent.name: verify_artifacts(p.parent)
                for p in sorted((ROOT / "runs").glob("*/frozen-artifacts.json"))
            },
            "implementation_sha256": {
                f: sha256(ROOT / f)
                for f in [
                    "src/fall_detect/data/pose_reuse.py",
                    "src/fall_detect/config.py",
                    "src/fall_detect/pipeline.py",
                    "src/fall_detect/skeleton/windows.py",
                    "src/fall_detect/training/experiment.py",
                    "src/fall_detect/detection/events.py",
                    "scripts/dense_schedule.py",
                    "scripts/ae_generalization.py",
                ]
            },
        },
    )


def verify():
    record = read_json(path("lock"))
    if sha256(ROOT / record["source_index"]) != record["source_index_sha256"]:
        raise ContractError("Source index changed")
    for filename, digest in record["implementation_sha256"].items():
        if sha256(ROOT / filename) != digest:
            raise ContractError(f"Locked source changed: {filename}")
    for c in record["candidates"]:
        if sha256(ROOT / c["config"]) != c["sha256"]:
            raise ContractError("Locked config changed")
    for name, hashes in record["preserved_runs"].items():
        if verify_artifacts(ROOT / "runs" / name) != hashes:
            raise ContractError("Old frozen run changed")
    return record


def feature_parity(old, current, ids):
    rows = []
    for sid in ids:
        a, b = cache_for(old, sid), cache_for(current, sid)
        if sha256(project_path(old["sequences"][sid]["pose_path"])) != sha256(
            project_path(current["sequences"][sid]["pose_path"])
        ):
            raise ContractError("Reused pose array changed")
        old_features = {
            (r.track_id, r.sample_index): a.features[r.row_index]
            for r in a.records
            if r.row_index is not None
        }
        differences = []
        for r in b.records:
            key = (r.track_id, r.sample_index)
            if r.row_index is not None and key in old_features:
                difference = float(np.max(np.abs(old_features[key] - b.features[r.row_index])))
                if difference > 1e-6:
                    raise ContractError(f"Common feature changed: {sid} {key} {difference}")
                differences.append(difference)
        if not differences:
            raise ContractError("No shared valid window to verify")
        before, _, _ = score_sequence(a, np.zeros(len(a.features)), 1.0)
        after, _, _ = score_sequence(b, np.zeros(len(b.features)), 1.0)
        if before["prediction"] is not None and after["prediction"] is None:
            raise ContractError("Originally evaluable sequence became unknown")
        rows.append(
            {
                "sequence_id": sid,
                "before_features": len(a.features),
                "after_features": len(b.features),
                "shared_windows": len(differences),
                "max_abs_difference": max(differences),
                "pose_arrays_unchanged": True,
            }
        )
    return rows


def pilot():
    record = verify()
    if path("pilot").exists():
        raise ContractError("Pilot already completed")
    old = read_json(ROOT / record["source_index"])
    candidates = []
    for c in record["candidates"]:
        stage = "pilot-" + c["run"]
        if path(stage).exists():
            row = read_json(path(stage))
        else:
            index_path, index = prepare(
                load_config(ROOT / c["config"]),
                selected_ids=record["pilot_ids"],
                reuse_poses_index=ROOT / record["source_index"],
            )
            row = {
                "run": c["run"],
                "index": str(index_path),
                "index_sha256": sha256(index_path),
                "parity": feature_parity(old, index, record["pilot_ids"]),
                "gate_passed": True,
            }
            save(stage, row)
        candidates.append(row)
        print({"pilot": c["run"], "passed": row["gate_passed"]}, flush=True)
    save(
        "pilot",
        {
            "source_kind": "real_feature_smoke",
            "candidates": candidates,
            "test_scored": False,
            "accuracy_not_measured": True,
        },
    )
    verify()


def timing(run, phases):
    s = Scorer(ROOT / "runs" / run)
    rows, hits, early = [], 0, 0
    for sid, phase in phases.items():
        source = next(r for r in s.index["split_manifest"]["sequences"] if r["sequence_id"] == sid)
        if source["split"] != "validation":
            raise ContractError("Coarse timing audit is Validation only")
        cache = cache_for(s.index, sid)
        outcome, _, events = score_sequence(
            cache, s.score(cache.features), s.threshold["threshold"]
        )
        starts = [e["confirmed_start_ms"] for e in events]
        fall_hits = [
            t for t in starts if phase["fall_phase_ms"][0] <= t <= phase["fall_phase_ms"][1]
        ]
        early_events = [t for t in starts if t < phase["upright_until_ms"]]
        hits += bool(fall_hits)
        early += len(early_events)
        rows.append(
            {
                "sequence_id": sid,
                "coarse_phase": phase,
                "outcome": outcome,
                "events": events,
                "fall_phase_event_starts": fall_hits,
                "early_event_starts": early_events,
            }
        )
    return {
        "scope": "four coarse Validation checks, not formal temporal Accuracy",
        "fall_phase_sequences": hits,
        "early_events": early,
        "rows": rows,
    }


def training():
    record = verify()
    if path("validation").exists():
        raise ContractError("Training experiment already completed")
    pilot_result = read_json(path("pilot"))
    baseline = result(record["baseline"])
    baseline["timing"] = timing(baseline["run"], record["coarse_visual_intervals"])
    baseline["timing_guard_passed"] = True
    rows = [baseline]
    for c, p in zip(record["candidates"], pilot_result["candidates"]):
        if c["run"] != p["run"] or not p["gate_passed"] or sha256(p["index"]) != p["index_sha256"]:
            raise ContractError("Pilot gate/checksum failed")
        stage = "trained-" + c["run"]
        if path(stage).exists():
            row = read_json(path(stage))
            if verify_artifacts(ROOT / "runs" / c["run"]) != row["hashes"]:
                raise ContractError("Completed candidate changed")
        else:
            config = load_config(ROOT / c["config"])
            run = ROOT / "runs" / c["run"]
            if run.exists():
                raise ContractError("Preserve partial run; inspect before continuing")
            index_path, index = prepare(config, reuse_poses_index=ROOT / record["source_index"])
            if not all(r["status"] == "prepared" for r in index["sequences"].values()):
                raise ContractError("Incomplete real preparation")
            train(config, index_path, run)
            calibrate(run)
            row = result(c["run"])
            row["timing"] = timing(c["run"], record["coarse_visual_intervals"])
            b = baseline
            row["timing_guard_passed"] = (
                row["timing"]["fall_phase_sequences"] > b["timing"]["fall_phase_sequences"]
                and row["timing"]["early_events"] <= b["timing"]["early_events"]
                and row["validation"]["operational_accuracy"]
                >= b["validation"]["operational_accuracy"]
                and row["validation"]["sequence_fall_f1"] >= b["validation"]["sequence_fall_f1"]
            )
            save(stage, row)
        rows.append(row)
        print(
            {
                "run": row["run"],
                "validation": row["validation"],
                "timing_guard": row["timing_guard_passed"],
            },
            flush=True,
        )
    ranking = lambda r: (
        tuple(
            r["validation"][k]
            for k in ("operational_accuracy", "sequence_fall_f1", "end_to_end_fall_recall")
        )
        + (-r["validation"]["FP"],)
    )
    selected = max((r for r in rows if r["timing_guard_passed"]), key=ranking)
    targets = all(selected["validation"][k] >= v for k, v in record["targets"].items())
    save(
        "validation",
        {
            "source_kind": "real_end_to_end",
            "rows": rows,
            "selected": selected["run"],
            "raw_metric_winner": max(rows, key=ranking)["run"],
            "target_reached": targets,
            "test_scored": False,
            "coarse_timing_is_not_ground_truth_accuracy": True,
        },
    )
    verify()
    print({"selected": selected["run"], "target_reached": targets}, flush=True)


def test():
    verify()
    record = read_json(path("validation"))
    if not record["target_reached"]:
        raise ContractError("Validation targets not reached; no Test scoring")
    if path("before-test").exists():
        raise ContractError("Test already started; preserve results")
    run = ROOT / "runs" / record["selected"]
    save(
        "before-test",
        {"run": record["selected"], "hashes": verify_artifacts(run), "historical_test_seen": True},
    )
    destination, metrics = evaluate(run)
    save("test", {"output": str(destination), "metrics": metrics, "historical_test_seen": True})


def status():
    record = verify()
    print({"old_runs_preserved": len(record["preserved_runs"]), "verified": True})
    if path("validation").exists():
        report = read_json(path("validation"))
        for row in report["rows"]:
            if verify_artifacts(ROOT / "runs" / row["run"]) != row["hashes"]:
                raise ContractError("Result run changed")
            print(
                {
                    "run": row["run"],
                    "accuracy": row["validation"]["operational_accuracy"],
                    "F1": row["validation"]["sequence_fall_f1"],
                    "coarse_phase_hits": row["timing"]["fall_phase_sequences"],
                    "early_events": row["timing"]["early_events"],
                    "timing_guard": row["timing_guard_passed"],
                }
            )
        print(
            {
                "selected": report["selected"],
                "target_reached": report["target_reached"],
                "test_scored": path("test").exists(),
            }
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["lock", "pilot", "training", "test", "status"])
    globals()[parser.parse_args().stage]()
