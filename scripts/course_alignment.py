"""Bounded course experiment, orchestrating the existing verified core APIs."""

import argparse
from datetime import UTC, datetime

import numpy as np
import yaml

from fall_detect.config import load_config, project_path
from fall_detect.data.splits import PROTOCOL
from fall_detect.detection.events import score_sequence
from fall_detect.evaluation.metrics import sequence_metrics
from fall_detect.io_utils import ContractError, read_json, sha256, write_json
from fall_detect.pipeline import prepare
from fall_detect.training.experiment import (
    Scorer,
    cache_for,
    calibrate,
    choose_threshold,
    evaluate,
    train,
    verify_artifacts,
)

EVIDENCE = project_path("docs/validation")
INDEX = project_path("data/cache/course-ur-yolo11/dataset-index.json")
MIXED = project_path("runs/caucafall-yolo11-balanced-001")


def save_new(name, payload):
    path = EVIDENCE / name
    if path.exists():
        raise ContractError(f"Evidence exists; preserve previous result: {path}")
    write_json(path, payload)


def diagnose():
    """Never load or score Test features to form improvement hypotheses."""
    scorer = Scorer(MIXED)
    rows = []
    for source in scorer.index["split_manifest"]["sequences"]:
        if source.get("dataset") != "urfall" or source["split"] == "test":
            continue
        sid = source["sequence_id"]
        cache = cache_for(scorer.index, sid)
        scores = scorer.score(cache.features)
        result, _, _ = score_sequence(cache, scores, scorer.threshold["threshold"])
        rows.append(
            {
                **source,
                **result,
                "n_features": len(cache.features),
                "n_samples": cache.metadata["n_samples"],
                "quality_counts": cache.metadata["quality_counts"],
                "track_count": len({r.track_id for r in cache.records if r.track_id is not None}),
                "feature_array_sha256": cache.metadata["array_sha256"],
                "raw_score_quantiles": np.quantile(scores, [0, 0.25, 0.5, 0.75, 1]).tolist()
                if len(scores)
                else None,
            }
        )
    val = [r for r in rows if r["split"] == "validation"]
    save_new(
        "course-ur-diagnosis.json",
        {
            "source_run": str(MIXED),
            "source_run_frozen": verify_artifacts(MIXED),
            "source_index_sha256": sha256(
                project_path("data/cache/caucafall-yolo11-jitter/dataset-index.json")
            ),
            "test_scored": False,
            "rows": rows,
            "validation_metrics_at_mixed_threshold": sequence_metrics(val),
            "ur_validation_only_threshold_diagnostics": {
                o: choose_threshold(val, o)[0]
                for o in ("sequence_fall_f1", "balanced_sequence_accuracy")
            },
        },
    )


def lock():
    diagnosis = read_json(EVIDENCE / "course-ur-diagnosis.json")
    if diagnosis["test_scored"] or any(r["split"] == "test" for r in diagnosis["rows"]):
        raise ContractError("Diagnosis must be Train/Validation only")
    destination = EVIDENCE / "course-experiment-lock.json"
    if destination.exists():
        raise ContractError("Experiment lock already exists")
    config = load_config(project_path("configs/caucafall-yolo11-balanced-001.yaml"))
    config["data"].update(
        dataset="urfall",
        raw_dir="data/raw/urfall",
        manifest_dir="data/manifests",
        cache_dir="data/cache/course-ur-yolo11",
        protocol_id=PROTOCOL,
    )
    config["autoencoder"].update(max_epochs=100, patience=10)
    candidates = []
    for suffix, sampling, latent in [
        ("baseline", "windows", 32),
        ("balanced", "sequences", 32),
        ("compact", "sequences", 16),
    ]:
        name = f"course-ur-{suffix}-001"
        path = project_path(f"configs/{name}.yaml")
        if path.exists() or project_path(f"runs/{name}").exists():
            raise ContractError(f"Candidate already exists: {name}")
        config["autoencoder"].update(sampling=sampling, latent_dim=latent)
        path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        candidates.append(
            {
                "run": name,
                "config": str(path.relative_to(project_path("."))),
                "config_sha256": sha256(path),
            }
        )
    preserved = {
        p.parent.name: verify_artifacts(p.parent)
        for p in sorted(project_path("runs").glob("*/frozen-artifacts.json"))
    }
    write_json(
        destination,
        {
            "created_at": datetime.now(UTC).isoformat(),
            "source_kind": "real",
            "diagnosis_sha256": sha256(EVIDENCE / "course-ur-diagnosis.json"),
            "reuse_index": "data/cache/caucafall-yolo11-jitter/dataset-index.json",
            "reuse_index_sha256": diagnosis["source_index_sha256"],
            "index": str(INDEX.relative_to(project_path("."))),
            "candidates": candidates,
            "preserved_runs": preserved,
            "calibration": "balanced_sequence_accuracy; unknown retained in class denominators",
            "selection": "Validation operational_accuracy, fall F1, fall recall, fewer FP, earlier candidate",
            "stop_rule": "Exactly three AE candidates, at most 100 epochs each, patience 10; no new candidates after Test",
            "hypotheses": [
                "UR-only scaler/AE may reduce mixed-domain mismatch",
                "Equal sequence sampling addresses 9-55-window Train length variation",
                "Latent16 may constrain reconstruction of anomalous features",
            ],
            "test_used_for_selection": False,
            "historical_test_previously_seen": True,
            "pose_feature_quality_timing_and_K_unchanged": True,
        },
    )


def verified_lock():
    record = read_json(EVIDENCE / "course-experiment-lock.json")
    for candidate in record["candidates"]:
        if sha256(project_path(candidate["config"])) != candidate["config_sha256"]:
            raise ContractError("Locked config changed")
    for name, expected in record["preserved_runs"].items():
        if verify_artifacts(project_path("runs") / name) != expected:
            raise ContractError(f"Preserved run changed: {name}")
    return record


def prepare_ur():
    record = verified_lock()
    old_path = project_path(record["reuse_index"])
    if sha256(old_path) != record["reuse_index_sha256"] or INDEX.exists():
        raise ContractError("Source index changed or destination already exists")
    old = read_json(old_path)
    _, new = prepare(
        load_config(project_path(record["candidates"][0]["config"]), "cuda"), reuse_index=old_path
    )
    checks = []
    old_sources = {r["sequence_id"]: r for r in old["split_manifest"]["sequences"]}
    for source in new["split_manifest"]["sequences"]:
        sid = source["sequence_id"]
        if any(source[k] != old_sources[sid][k] for k in ("split", "label", "camera", "modality")):
            raise ContractError("UR split changed")
        before, after = cache_for(old, sid), cache_for(new, sid)
        if not np.array_equal(before.features, after.features) or before.records != after.records:
            raise ContractError("Reused features or window records changed")
        checks.append(
            {
                "sequence_id": sid,
                "split": source["split"],
                "features": len(after.features),
                "source_hash": after.metadata["source_hash"],
                "array_sha256": after.metadata["array_sha256"],
                "arrays_records_equal": True,
            }
        )
    save_new(
        "course-cache-reuse.json",
        {
            "source_index_sha256": sha256(old_path),
            "index_sha256": sha256(INDEX),
            "recipe_unchanged": old["recipe_hash"] == new["recipe_hash"],
            "test_feature_access": "integrity equality only; no scores or selection",
            "checks": checks,
            "n_sequences": len(checks),
        },
    )


def summarize(run):
    calibration = read_json(run / "calibration.json")
    return {
        "run": run.name,
        "status": read_json(run / "status.json"),
        "threshold": calibration["chosen"]["threshold"],
        "validation_metrics": calibration["chosen"]["metrics"],
        "frozen_artifacts": verify_artifacts(run),
    }


def train_candidates():
    record = verified_lock()
    if sha256(INDEX) != read_json(EVIDENCE / "course-cache-reuse.json")["index_sha256"]:
        raise ContractError("Prepared index changed")
    if any(
        project_path(
            "runs",
        )
        .joinpath(c["run"])
        .exists()
        for c in record["candidates"]
    ):
        raise ContractError("Candidate directory exists; preserve failed and completed runs")
    rows = []
    for candidate in record["candidates"]:
        run = project_path("runs") / candidate["run"]
        write_json(
            EVIDENCE / "course-workflow-progress.json", {"stage": "training", "run": run.name}
        )
        train(load_config(project_path(candidate["config"]), "cuda"), INDEX, run)
        calibrate(run)
        rows.append(summarize(run))
        print(rows[-1], flush=True)
    chosen = max(
        rows,
        key=lambda r: (
            tuple(
                r["validation_metrics"][k]
                for k in ("operational_accuracy", "sequence_fall_f1", "end_to_end_fall_recall")
            )
            + (-r["validation_metrics"]["FP"],)
        ),
    )
    verified_lock()
    save_new(
        "course-before-test-freeze.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "selected_run": chosen["run"],
            "candidates": rows,
            "selection_split": "validation",
            "test_scored": False,
            "lock_sha256": sha256(EVIDENCE / "course-experiment-lock.json"),
        },
    )
    write_json(
        EVIDENCE / "course-workflow-progress.json",
        {"stage": "frozen_before_test", "selected": chosen["run"]},
    )


def test_selected():
    verified_lock()
    freeze = read_json(EVIDENCE / "course-before-test-freeze.json")
    for item in freeze["candidates"]:
        if verify_artifacts(project_path("runs") / item["run"]) != item["frozen_artifacts"]:
            raise ContractError("Candidate changed after selection")
    run = project_path("runs") / freeze["selected_run"]
    output, metrics = evaluate(run, output=project_path("outputs/course-ur-test"))
    save_new(
        "course-test-results.json",
        {
            "selected_run": run.name,
            "metrics": metrics,
            "output": str(output),
            "historical_test_previously_seen": True,
        },
    )
    verified_lock()
    print(metrics, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["diagnose", "lock", "prepare", "train", "test"])
    args = parser.parse_args()
    {
        "diagnose": diagnose,
        "lock": lock,
        "prepare": prepare_ur,
        "train": train_candidates,
        "test": test_selected,
    }[args.stage]()
