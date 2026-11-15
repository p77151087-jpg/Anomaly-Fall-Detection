"""Bounded sample-derived Human pose / two-stream features / normal-only AE experiment."""

import argparse
from datetime import UTC, datetime

from ae_generalization import result
from dense_schedule import PHASES, PILOT_IDS, timing

from fall_detect.config import load_config, project_path
from fall_detect.io_utils import ContractError, read_json, sha256, write_json
from fall_detect.pipeline import prepare
from fall_detect.training.experiment import calibrate, evaluate, train, verify_artifacts

ROOT = project_path(".")
BASELINE = "ur-ae-decay-001"
RUNS = ["ur-sample-human-s5-001", "ur-sample-human-s1-001"]


def path(stage):
    return ROOT / f"docs/validation/sample-human-{stage}.json"


def save(stage, record):
    if path(stage).exists():
        raise ContractError(f"Preserve completed evidence: {stage}")
    write_json(path(stage), record)


def lock():
    if path("lock").exists():
        raise ContractError("Experiment already locked")
    files = [
        "src/fall_detect/pose/human.py",
        "src/fall_detect/models/stgcn.py",
        "src/fall_detect/skeleton/preprocessing.py",
        "src/fall_detect/skeleton/windows.py",
        "src/fall_detect/training/experiment.py",
        "src/fall_detect/pipeline.py",
        "src/fall_detect/config.py",
        "src/fall_detect/detection/events.py",
        "scripts/sample_human.py",
        "scripts/dense_schedule.py",
        "scripts/ae_generalization.py",
    ]
    save(
        "lock",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "baseline": BASELINE,
            "candidates": [
                {
                    "run": r,
                    "config": f"configs/{r}.yaml",
                    "sha256": sha256(ROOT / f"configs/{r}.yaml"),
                }
                for r in RUNS
            ],
            "pilot_ids": PILOT_IDS,
            "coarse_visual_intervals": PHASES,
            "hypothesis": "Human AlphaPose + matching 14-joint points/motion ST-GCN features may distinguish normal activity and falls better than Kinetics features; not yet established.",
            "fixed": "Original UR sequence split; Train ADL only; pretrained frozen 512D two-stream feature before classifier; confidence .2, 9/13 measured joints, 80% good frames; 30-frame causal windows; K3; 128/32 AE, sequence sampling, wd .001, seed42, max100 epochs, patience10.",
            "candidates_difference": "Stride5 control and stride1 sample-like scheduling only; no other candidates.",
            "pilot_gate": "Strict genuine weights, finite 512D features in at least two Train ADL, one Validation ADL and one Validation Fall; no Test pilot; feature extraction parity and no classifier outputs used by AE.",
            "selection": "Baseline plus candidates passing timing guard, ordered by Validation operational accuracy, F1, all-Fall recall, fewer FP; ties keep earlier baseline/candidate.",
            "timing_guard": "At least one more coarse fall-phase event-start case, no more early events, Accuracy and F1 no lower than baseline. Four diagnostic Validation cases only, not official timing ground truth.",
            "calibration": "Validation balanced_sequence_accuracy; unchanged K3 and unknown semantics.",
            "targets": {
                "operational_accuracy": 0.9,
                "sequence_accuracy": 0.9,
                "sequence_fall_f1": 0.87,
            },
            "stop": "Exactly two candidates; no further tuning or Test evaluation unless targets and timing guard pass.",
            "historic_test_seen": True,
            "repeated_validation_not_independent": True,
            "pretraining": "Human upstream Le2i seven-action supervised representation includes falls; only our AE/scaler are UR Train-normal-only.",
            "weights": read_json(ROOT / "weights/sample-human-manifest.json"),
            "source_index_sha256": sha256(ROOT / "data/cache/course-ur-yolo11/dataset-index.json"),
            "preserved_runs": {
                p.parent.name: verify_artifacts(p.parent)
                for p in sorted((ROOT / "runs").glob("*/frozen-artifacts.json"))
            },
            "preserved_configs": {
                str(p.relative_to(ROOT)): sha256(p) for p in (ROOT / "configs").glob("*.yaml")
            },
            "legacy_manifest_sha256": sha256(ROOT / "weights/manifest.json"),
            "implementation_sha256": {f: sha256(ROOT / f) for f in files},
        },
    )


def verify():
    record = read_json(path("lock"))
    for filename, digest in {
        **record["preserved_configs"],
        **record["implementation_sha256"],
    }.items():
        if sha256(ROOT / filename) != digest:
            raise ContractError(f"Locked file changed: {filename}")
    if sha256(ROOT / "weights/manifest.json") != record["legacy_manifest_sha256"]:
        raise ContractError("Legacy manifest changed")
    if (
        sha256(ROOT / "data/cache/course-ur-yolo11/dataset-index.json")
        != record["source_index_sha256"]
    ):
        raise ContractError("Old feature index changed")
    for entry in record["weights"]["resources"]:
        if sha256(ROOT / entry["local_path"]) != entry["sha256"]:
            raise ContractError("Sample weight changed")
    for name, hashes in record["preserved_runs"].items():
        if verify_artifacts(ROOT / "runs" / name) != hashes:
            raise ContractError(f"Old run changed: {name}")
    return record


def pilot():
    record = verify()
    if path("pilot").exists():
        raise ContractError("Pilot already completed")
    pilot_rows = []
    previous = None
    for candidate in record["candidates"]:
        stage = "pilot-" + candidate["run"]
        if path(stage).exists():
            row = read_json(path(stage))
        else:
            index_path, index = prepare(
                load_config(ROOT / candidate["config"]),
                selected_ids=record["pilot_ids"],
                reuse_poses_index=previous,
            )
            prepared = [
                s
                for s in index["split_manifest"]["sequences"]
                if s["sequence_id"] in record["pilot_ids"]
            ]
            counts = {}
            for s in prepared:
                data = index["sequences"][s["sequence_id"]]
                if data["status"] != "prepared":
                    raise ContractError(f"Real pilot failed: {data}")
                from fall_detect.training.experiment import cache_for

                counts[s["sequence_id"]] = {
                    "split": s["split"],
                    "label": s["label"],
                    "features": len(cache_for(index, s["sequence_id"]).features),
                }
            gate = all(
                sum(
                    v["split"] == split and v["label"] == label and v["features"] > 0
                    for v in counts.values()
                )
                >= minimum
                for split, label, minimum in [
                    ("train", 0, 2),
                    ("validation", 0, 1),
                    ("validation", 1, 1),
                ]
            )
            row = {
                "run": candidate["run"],
                "index": str(index_path),
                "index_sha256": sha256(index_path),
                "counts": counts,
                "gate_passed": gate,
                "model_load_reports": index["model_load_reports"],
            }
            save(stage, row)
        if sha256(row["index"]) != row["index_sha256"]:
            raise ContractError("Pilot index changed")
        pilot_rows.append(row)
        previous = project_path(row["index"])
    save(
        "pilot",
        {
            "source_kind": "real_feature_smoke",
            "rows": pilot_rows,
            "gate_passed": all(r["gate_passed"] for r in pilot_rows),
            "test_scored": False,
            "accuracy_not_measured": True,
        },
    )


def ranking(row):
    m = row["validation"]
    return (m["operational_accuracy"], m["sequence_fall_f1"], m["end_to_end_fall_recall"], -m["FP"])


def training():
    record = verify()
    if path("validation").exists():
        raise ContractError("Experiment already completed")
    pilot_record = read_json(path("pilot"))
    if not pilot_record["gate_passed"]:
        raise ContractError("Pilot gate failed; training stopped")
    for row in pilot_record["rows"]:
        if sha256(row["index"]) != row["index_sha256"]:
            raise ContractError("Pilot changed")
    baseline = result(BASELINE)
    baseline["timing"] = timing(BASELINE, record["coarse_visual_intervals"])
    baseline["timing_guard_passed"] = True
    rows, previous = [baseline], None
    for candidate in record["candidates"]:
        stage = "trained-" + candidate["run"]
        if path(stage).exists():
            row = read_json(path(stage))
            if verify_artifacts(ROOT / "runs" / row["run"]) != row["hashes"]:
                raise ContractError("Completed run changed")
        else:
            run = ROOT / "runs" / candidate["run"]
            if run.exists():
                raise ContractError("Partial run preserved; inspect before continuation")
            config = load_config(ROOT / candidate["config"])
            index_path, index = prepare(config, reuse_poses_index=previous)
            if not all(s["status"] == "prepared" for s in index["sequences"].values()):
                raise ContractError("Full feature preparation incomplete")
            train(config, index_path, run)
            calibrate(run)
            row = result(candidate["run"])
            row["index"] = str(index_path)
            row["timing"] = timing(candidate["run"], record["coarse_visual_intervals"])
            row["timing_guard_passed"] = (
                row["timing"]["fall_phase_sequences"] > baseline["timing"]["fall_phase_sequences"]
                and row["timing"]["early_events"] <= baseline["timing"]["early_events"]
                and row["validation"]["operational_accuracy"]
                >= baseline["validation"]["operational_accuracy"]
                and row["validation"]["sequence_fall_f1"]
                >= baseline["validation"]["sequence_fall_f1"]
            )
            save(stage, row)
        previous = project_path(row["index"])
        rows.append(row)
        print(
            {
                "run": row["run"],
                "validation": row["validation"],
                "timing_guard_passed": row["timing_guard_passed"],
            },
            flush=True,
        )
    selected = max((r for r in rows if r["timing_guard_passed"]), key=ranking)
    save(
        "validation",
        {
            "source_kind": "real_end_to_end",
            "rows": rows,
            "selected": selected["run"],
            "raw_metric_winner": max(rows, key=ranking)["run"],
            "target_reached": all(
                selected["validation"][k] >= value for k, value in record["targets"].items()
            ),
            "test_scored": False,
        },
    )
    verify()


def test():
    verify()
    report = read_json(path("validation"))
    if not report["target_reached"]:
        raise ContractError("Validation targets/timing guard not reached; no Test scoring")
    run = ROOT / "runs" / report["selected"]
    save(
        "before-test",
        {"run": report["selected"], "hashes": verify_artifacts(run), "historical_test_seen": True},
    )
    output, metrics = evaluate(run)
    save("test", {"output": str(output), "metrics": metrics, "historical_test_seen": True})


def status():
    record = verify()
    print({"old_runs_preserved": len(record["preserved_runs"]), "weights_verified": True})
    if path("validation").exists():
        report = read_json(path("validation"))
        for row in report["rows"]:
            if verify_artifacts(ROOT / "runs" / row["run"]) != row["hashes"]:
                raise ContractError("Result changed")
            print(
                {
                    "run": row["run"],
                    "validation": row["validation"],
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
