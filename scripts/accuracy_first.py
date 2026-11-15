"""Accuracy-first bounded follow-up. Presentation work remains on hold."""

import argparse
from datetime import UTC, datetime

import numpy as np
import yaml

from fall_detect.config import load_config, project_path
from fall_detect.detection.events import score_sequence
from fall_detect.io_utils import ContractError, read_json, sha256, write_json
from fall_detect.pipeline import prepare
from fall_detect.training.experiment import cache_for, calibrate, evaluate, train, verify_artifacts

DIR = project_path("docs/validation")
PREFIX = "accuracy-first-rotation"
PILOT_IDS = ["adl-05", "adl-24", "adl-04", "adl-20", "fall-02", "fall-18", "fall-19", "fall-29"]


def save(name, data):
    path = DIR / f"{PREFIX}-{name}.json"
    if path.exists():
        raise ContractError(f"Evidence exists: {path}")
    write_json(path, data)
    return path


def lock():
    if (DIR / f"{PREFIX}-lock.json").exists():
        raise ContractError("Experiment already locked")
    config = load_config(project_path("configs/course-ur-balanced-001.yaml"))
    config["data"]["cache_dir"] = "data/cache/accuracy-first-rotation"
    config["pose"].update(rotation_angles=[0, 90, -90], rotation_policy="low_quality")
    candidates = []
    for name, sampling, latent in [
        ("windows", "windows", 32),
        ("balanced", "sequences", 32),
        ("compact", "sequences", 16),
    ]:
        run = f"accuracy-ur-rotation-{name}-001"
        path = project_path(f"configs/{run}.yaml")
        if path.exists() or project_path(f"runs/{run}").exists():
            raise ContractError("Candidate already exists")
        config["autoencoder"].update(sampling=sampling, latent_dim=latent)
        path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        candidates.append(
            {"run": run, "config": str(path.relative_to(project_path("."))), "sha256": sha256(path)}
        )
    source = project_path("data/cache/course-ur-yolo11/dataset-index.json")
    save(
        "lock",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "pilot_ids": PILOT_IDS,
            "source_index": str(source.relative_to(project_path("."))),
            "source_index_sha256": sha256(source),
            "candidates": candidates,
            "pilot_gate": {
                "fall_feature_ratio_min": 1.3,
                "adl_feature_ratio_min": 0.9,
                "no_evaluable_regression": True,
            },
            "selection": "Validation operational_accuracy, F1, all-Fall recall, fewer FP, earlier candidate",
            "threshold": "Validation balanced_sequence_accuracy; fewer FP then higher threshold",
            "stop": "one pose candidate, exactly three AE candidates if pilot passes; max100/patience10; no Test unless Validation accuracy>=.9 and F1>=.87",
            "validation_targets": {
                "operational_accuracy": 0.9,
                "sequence_accuracy": 0.9,
                "sequence_fall_f1": 0.87,
            },
            "historical_test_seen": True,
            "presentation_updates_on_hold": True,
            "preserved_runs": {
                p.parent.name: verify_artifacts(p.parent)
                for p in sorted(project_path("runs").glob("*/frozen-artifacts.json"))
            },
        },
    )


def verify():
    record = read_json(DIR / f"{PREFIX}-lock.json")
    assert sha256(project_path(record["source_index"])) == record["source_index_sha256"]
    for row in record["candidates"]:
        assert sha256(project_path(row["config"])) == row["sha256"]
    for name, hashes in record["preserved_runs"].items():
        assert verify_artifacts(project_path("runs") / name) == hashes
    return record


def pilot():
    record = verify()
    if (DIR / f"{PREFIX}-pilot.json").exists():
        raise ContractError("Pilot evidence exists; preserve the completed experiment")
    _, index = prepare(
        load_config(project_path(record["candidates"][0]["config"]), "cuda"),
        selected_ids=record["pilot_ids"],
    )
    old = read_json(project_path(record["source_index"]))
    rows = []
    for sid in record["pilot_ids"]:
        source = next(s for s in index["split_manifest"]["sequences"] if s["sequence_id"] == sid)
        if source["split"] == "test":
            raise ContractError("Test cannot enter pose pilot")
        before, after = cache_for(old, sid), cache_for(index, sid)
        observations = []
        for cache in (before, after):
            outcome, windows, _ = score_sequence(cache, np.zeros(len(cache.features)), 1.0)
            observations.append(
                {
                    "features": len(cache.features),
                    "evaluable": outcome["prediction"] is not None,
                    "window_coverage": outcome["window_coverage"],
                    "legal_event_windows": sum(w["event_score"] is not None for w in windows),
                    "quality_counts": cache.metadata["quality_counts"],
                }
            )
        rows.append({**source, "before": observations[0], "after": observations[1]})
    ratios = {}
    for label, name in [(0, "ADL"), (1, "Fall")]:
        group = [r for r in rows if r["label"] == label]
        ratios[name] = sum(r["after"]["features"] for r in group) / sum(
            r["before"]["features"] for r in group
        )
    regression = any(r["before"]["evaluable"] and not r["after"]["evaluable"] for r in rows)
    passed = ratios["Fall"] >= 1.3 and ratios["ADL"] >= 0.9 and not regression
    save(
        "pilot",
        {
            "source_kind": "real_feature_smoke",
            "test_scored": False,
            "rows": rows,
            "feature_ratios": ratios,
            "evaluable_regression": regression,
            "gate_passed": passed,
            "accuracy_not_measured": True,
            "source_index_sha256": record["source_index_sha256"],
        },
    )
    print({"gate_passed": passed, "feature_ratios": ratios, "rows": rows}, flush=True)
    verify()


def training():
    record = verify()
    if (DIR / f"{PREFIX}-validation.json").exists():
        raise ContractError("Validation evidence exists; preserve the completed experiment")
    pilot = read_json(DIR / f"{PREFIX}-pilot.json")
    if not pilot["gate_passed"]:
        raise ContractError("Pilot gate failed; no full preparation or AE experiment")
    config = load_config(project_path(record["candidates"][0]["config"]), "cuda")
    full = project_path(config["data"]["cache_dir"]) / "dataset-index.json"
    if full.exists():
        raise ContractError("Full index exists; preserve previous execution")
    pilot_index = (
        project_path(config["data"]["cache_dir"]) / "real_feature_smoke/dataset-index.json"
    )
    index_path, _ = prepare(config, reuse_index=pilot_index)
    rows = []
    for candidate in record["candidates"]:
        run = project_path("runs") / candidate["run"]
        train(load_config(project_path(candidate["config"]), "cuda"), index_path, run)
        result = calibrate(run)
        rows.append(
            {
                "run": run.name,
                "status": read_json(run / "status.json"),
                "threshold": result["threshold"],
                "validation": read_json(run / "calibration.json")["chosen"]["metrics"],
                "hashes": verify_artifacts(run),
            }
        )
        print(rows[-1], flush=True)
    winner = max(
        rows,
        key=lambda r: (
            tuple(
                r["validation"][k]
                for k in ("operational_accuracy", "sequence_fall_f1", "end_to_end_fall_recall")
            )
            + (-r["validation"]["FP"],)
        ),
    )
    reached = all(winner["validation"][k] >= v for k, v in record["validation_targets"].items())
    save(
        "validation",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "source_kind": "real_end_to_end",
            "candidates": rows,
            "selected": winner["run"],
            "validation_target_reached": reached,
            "test_scored": False,
            "presentation_updates_on_hold": True,
        },
    )
    verify()
    print({"selected": winner["run"], "validation_target_reached": reached}, flush=True)


def test():
    record = verify()
    if (DIR / f"{PREFIX}-test.json").exists() or project_path(
        "outputs/accuracy-first-rotation-test"
    ).exists():
        raise ContractError("Test output exists; preserve the completed evaluation")
    result = read_json(DIR / f"{PREFIX}-validation.json")
    if not result["validation_target_reached"]:
        raise ContractError("Validation targets not met; Test stays unscored")
    candidate = next(c for c in result["candidates"] if c["run"] == result["selected"])
    run = project_path("runs") / candidate["run"]
    assert verify_artifacts(run) == candidate["hashes"]
    _, metrics = evaluate(run, output=project_path("outputs/accuracy-first-rotation-test"))
    reached = all(metrics[k] >= v for k, v in record["validation_targets"].items())
    save(
        "test",
        {
            "source_kind": "real_end_to_end",
            "metrics": metrics,
            "target_reached_under_engineering_protocol": reached,
            "official_protocol_unconfirmed": True,
        },
    )
    print(metrics)


def status():
    record = verify()
    path = DIR / f"{PREFIX}-pilot.json"
    result = read_json(path) if path.exists() else None
    print(
        {
            "preserved_frozen_runs": len(record["preserved_runs"]),
            "pilot_complete": result is not None,
            "pilot_gate_passed": result["gate_passed"] if result else None,
            "feature_ratios": result["feature_ratios"] if result else None,
            "rotation_runs_created": [
                c["run"] for c in record["candidates"] if (project_path("runs") / c["run"]).exists()
            ],
            "presentation_updates_on_hold": True,
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["lock", "pilot", "training", "test", "status"])
    args = parser.parse_args()
    globals()[args.stage]()
