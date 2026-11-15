"""Three preregistered normal-only AE candidates; frozen Test gated on Validation."""

import argparse
from copy import deepcopy
from datetime import UTC, datetime

import yaml

from fall_detect.config import load_config, project_path
from fall_detect.io_utils import ContractError, read_json, sha256, write_json
from fall_detect.training.experiment import calibrate, evaluate, train, verify_artifacts

ROOT = project_path(".")
EVIDENCE = ROOT / "docs/validation"
BASELINE = "course-ur-balanced-001"
FILES = ["src/fall_detect/config.py", "src/fall_detect/training/experiment.py"]


def path(stage):
    return EVIDENCE / f"ae-generalization-{stage}.json"


def save(stage, value):
    if path(stage).exists():
        raise ContractError(f"Preserve existing evidence: {stage}")
    write_json(path(stage), value)


def lock():
    if path("lock").exists():
        raise ContractError("Experiment already locked")
    base = load_config(ROOT / f"configs/{BASELINE}.yaml")
    candidates = []
    for name, decay, noise in [("decay", 0.001, 0.0), ("denoise", 0.0, 0.1), ("both", 0.001, 0.1)]:
        run = f"ur-ae-{name}-001"
        filename = ROOT / f"configs/{run}.yaml"
        if filename.exists() or (ROOT / "runs" / run).exists():
            raise ContractError(f"Candidate exists: {run}")
        config = deepcopy(base)
        config["autoencoder"].update(weight_decay=decay, input_noise_std=noise)
        filename.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf8")
        candidates.append(
            {"run": run, "config": str(filename.relative_to(ROOT)), "sha256": sha256(filename)}
        )
    source = ROOT / "data/cache/course-ur-yolo11/dataset-index.json"
    save(
        "lock",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "diagnosis_sha256": sha256(EVIDENCE / "feature-diagnosis.json"),
            "source_index": str(source.relative_to(ROOT)),
            "source_index_sha256": sha256(source),
            "candidates": candidates,
            "baseline": BASELINE,
            "fixed": "UR-only; same Train ADL scaler/features; sequence sampling; 128/32 AE; seed42; max100 epochs; patience10; clean normal Validation checkpoint",
            "hypothesis": "Regularization may reduce generalization gap on normal Validation; not established as the cause or guaranteed improvement",
            "selection": "Validation operational_accuracy, Fall F1, all-Fall recall, fewer FP, earlier (baseline first)",
            "calibration": "Validation balanced_sequence_accuracy, fewer FP, higher threshold; unchanged K3 and unknown",
            "targets": {
                "operational_accuracy": 0.9,
                "sequence_accuracy": 0.9,
                "sequence_fall_f1": 0.87,
            },
            "stop": "Exactly these three candidates; no extra grid or Test scoring if Validation gate fails",
            "historic_test_seen": True,
            "repeated_validation_not_independent": True,
            "preserved_runs": {
                p.parent.name: verify_artifacts(p.parent)
                for p in sorted((ROOT / "runs").glob("*/frozen-artifacts.json"))
            },
            "implementation_sha256": {
                f: sha256(ROOT / f) for f in FILES + ["scripts/ae_generalization.py"]
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
    for candidate in record["candidates"]:
        if sha256(ROOT / candidate["config"]) != candidate["sha256"]:
            raise ContractError("Locked candidate changed")
    for name, hashes in record["preserved_runs"].items():
        if verify_artifacts(ROOT / "runs" / name) != hashes:
            raise ContractError("Old run changed")
    return record


def result(run):
    folder = ROOT / "runs" / run
    return {
        "run": run,
        "status": read_json(folder / "status.json"),
        "threshold": read_json(folder / "threshold.json")["threshold"],
        "validation": read_json(folder / "calibration.json")["chosen"]["metrics"],
        "hashes": verify_artifacts(folder),
    }


def training():
    record = verify()
    if path("validation").exists():
        raise ContractError("Completed experiment; preserve its results")
    rows = [result(record["baseline"])]
    for candidate in record["candidates"]:
        checkpoint = path(candidate["run"])
        if checkpoint.exists():
            row = read_json(checkpoint)
            if row != result(candidate["run"]):
                raise ContractError("Completed candidate changed")
        else:
            run = ROOT / "runs" / candidate["run"]
            if run.exists():
                raise ContractError(f"Partial run exists; inspect without overwriting: {run}")
            train(load_config(ROOT / candidate["config"]), ROOT / record["source_index"], run)
            calibrate(run)
            row = result(candidate["run"])
            save(candidate["run"], row)
        rows.append(row)
        print({"run": row["run"], "validation": row["validation"]}, flush=True)
    selected = max(
        rows,
        key=lambda r: (
            tuple(
                r["validation"][k]
                for k in ("operational_accuracy", "sequence_fall_f1", "end_to_end_fall_recall")
            )
            + (-r["validation"]["FP"],)
        ),
    )
    reached = all(selected["validation"][k] >= v for k, v in record["targets"].items())
    save(
        "validation",
        {
            "source_kind": "real_end_to_end",
            "rows": rows,
            "selected": selected["run"],
            "validation_target_reached": reached,
            "test_scored": False,
        },
    )
    verify()
    print({"selected": selected["run"], "validation_target_reached": reached}, flush=True)


def test():
    verify()
    results = read_json(path("validation"))
    if not results["validation_target_reached"]:
        raise ContractError("Validation gate failed; no Test scoring")
    if path("before-test").exists():
        raise ContractError("Test already started; inspect retained evidence")
    run = results["selected"]
    save(
        "before-test",
        {
            "selected": run,
            "frozen": verify_artifacts(ROOT / "runs" / run),
            "historical_test_seen": True,
        },
    )
    destination, metrics = evaluate(ROOT / "runs" / run)
    save(
        "test",
        {
            "selected": run,
            "output": str(destination),
            "metrics": metrics,
            "historical_test_seen": True,
        },
    )


def status():
    record = verify()
    print({"preserved_runs": len(record["preserved_runs"]), "verified": True})
    if path("validation").exists():
        data = read_json(path("validation"))
        for row in data["rows"]:
            if verify_artifacts(ROOT / "runs" / row["run"]) != row["hashes"]:
                raise ContractError("Result run changed")
            print({"run": row["run"], "validation": row["validation"]})
        print(
            {
                "selected": data["selected"],
                "target_reached": data["validation_target_reached"],
                "test_scored": path("test").exists(),
            }
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["lock", "training", "test", "status"])
    globals()[parser.parse_args().stage]()
