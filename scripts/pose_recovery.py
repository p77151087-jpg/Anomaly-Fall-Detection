"""Bounded UR-only pose recovery experiment; no Test scoring below the Val gate."""

import argparse
from copy import deepcopy
from datetime import UTC, datetime

import numpy as np
import yaml

from fall_detect.config import load_config, project_path, recipe_fingerprint
from fall_detect.detection.events import score_sequence
from fall_detect.io_utils import ContractError, read_json, sha256, write_json
from fall_detect.pipeline import prepare
from fall_detect.training.experiment import cache_for, calibrate, train, verify_artifacts

ROOT = project_path(".")
EVIDENCE = ROOT / "docs/validation"
PREFIX = "pose-recovery"
IDS = ["adl-05", "adl-24", "adl-38", "adl-04", "adl-20", "fall-02", "fall-18", "fall-19", "fall-29"]


def path(stage):
    return EVIDENCE / f"{PREFIX}-{stage}.json"


def save(stage, record):
    if path(stage).exists():
        raise ContractError(f"Evidence already exists: {stage}")
    write_json(path(stage), record)


def lock():
    if path("lock").exists():
        raise ContractError("Experiment already locked")
    base = load_config(ROOT / "configs/course-ur-balanced-001.yaml")
    candidates = []
    for name, fallback in [("resolution960", 0.0), ("recovery960", 0.1)]:
        config = deepcopy(base)
        config["pose"].update(input_height=960, fallback_detector_confidence=fallback)
        config["data"]["cache_dir"] = f"data/cache/ur-{name}-001"
        filename = ROOT / f"configs/ur-{name}-001.yaml"
        if filename.exists() or project_path(config["data"]["cache_dir"]).exists():
            raise ContractError("Candidate exists")
        filename.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf8"
        )
        candidates.append(
            {"name": name, "config": str(filename.relative_to(ROOT)), "sha256": sha256(filename)}
        )
    source = ROOT / "data/cache/course-ur-yolo11/dataset-index.json"
    save(
        "lock",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "pilot_ids": IDS,
            "source_index": str(source.relative_to(ROOT)),
            "source_index_sha256": sha256(source),
            "diagnosis_sha256": sha256(EVIDENCE / "continuity-diagnosis.json"),
            "pose_candidates": candidates,
            "pilot_gate": {
                "fall_feature_ratio_min": 1.3,
                "adl_feature_ratio_min": 0.9,
                "no_evaluable_regression": True,
            },
            "pose_selection": "Among passing candidates: most Fall valid windows, then most ADL valid windows, then earlier candidate; inspect newly recovered Fall observations before full preparation",
            "ae_candidates": [
                {"name": "balanced", "sampling": "sequences", "latent_dim": 32},
                {"name": "compact", "sampling": "sequences", "latent_dim": 16},
                {"name": "windows", "sampling": "windows", "latent_dim": 32},
            ],
            "ae_limit": {"max_epochs": 100, "patience": 10},
            "ae_selection": "Validation operational_accuracy, F1, all-Fall recall, fewer FP, earlier candidate",
            "calibration": "Validation balanced_sequence_accuracy; fewer FP then higher threshold",
            "validation_targets": {
                "operational_accuracy": 0.9,
                "sequence_accuracy": 0.9,
                "sequence_fall_f1": 0.87,
            },
            "no_test_scoring_before_validation_target": True,
            "historic_test_seen": True,
            "presentation_on_hold": True,
            "preserved_runs": {
                p.parent.name: verify_artifacts(p.parent)
                for p in sorted((ROOT / "runs").glob("*/frozen-artifacts.json"))
            },
            "implementation_sha256": {
                f: sha256(ROOT / f)
                for f in ["src/fall_detect/config.py", "src/fall_detect/pose/yolo.py"]
            },
        },
    )


def verify(strict_source=False):
    record = read_json(path("lock"))
    assert sha256(ROOT / record["source_index"]) == record["source_index_sha256"]
    for candidate in record["pose_candidates"]:
        assert sha256(ROOT / candidate["config"]) == candidate["sha256"]
    for name, hashes in record["preserved_runs"].items():
        assert verify_artifacts(ROOT / "runs" / name) == hashes
    for filename, digest in record["implementation_sha256"].items():
        if strict_source or sha256(ROOT / filename) == digest:
            assert sha256(ROOT / filename) == digest
        else:
            archive = EVIDENCE / "pose-recovery-source" / filename.rsplit("/", 1)[-1]
            assert sha256(archive) == digest
    old = read_json(ROOT / record["source_index"])
    weights = {k: v["sha256"] for k, v in old["weights"].items()}
    assert (
        recipe_fingerprint(load_config(ROOT / "configs/course-ur-balanced-001.yaml"), weights)
        == old["recipe_hash"]
    )
    return record


def summarize(cache):
    result, windows, _ = score_sequence(cache, np.zeros(len(cache.features)), 1.0)
    return {
        "features": len(cache.features),
        "evaluable": result["prediction"] is not None,
        "legal_event_windows": sum(w["event_score"] is not None for w in windows),
        "coverage": result["window_coverage"],
        "quality_counts": cache.metadata["quality_counts"],
    }


def pilot():
    record = verify(strict_source=True)
    if path("pilot").exists():
        raise ContractError("Pilot exists")
    old = read_json(ROOT / record["source_index"])
    candidates = []
    for candidate in record["pose_candidates"]:
        checkpoint = path(f"pilot-{candidate['name']}")
        if checkpoint.exists():
            result = read_json(checkpoint)
        else:
            index_path, index = prepare(
                load_config(ROOT / candidate["config"], "cuda"), selected_ids=record["pilot_ids"]
            )
            rows = []
            for sid in record["pilot_ids"]:
                source = next(
                    s for s in index["split_manifest"]["sequences"] if s["sequence_id"] == sid
                )
                assert source["split"] != "test"
                rows.append(
                    {
                        **source,
                        "before": summarize(cache_for(old, sid)),
                        "after": summarize(cache_for(index, sid)),
                    }
                )
            totals = {}
            for label, name in [(0, "ADL"), (1, "Fall")]:
                group = [r for r in rows if r["label"] == label]
                before = sum(r["before"]["features"] for r in group)
                after = sum(r["after"]["features"] for r in group)
                totals[name] = {"before": before, "after": after, "ratio": after / before}
            regression = any(r["before"]["evaluable"] and not r["after"]["evaluable"] for r in rows)
            gate = (
                totals["Fall"]["ratio"] >= 1.3 and totals["ADL"]["ratio"] >= 0.9 and not regression
            )
            result = {
                **candidate,
                "index": str(index_path.relative_to(ROOT)),
                "index_sha256": sha256(index_path),
                "rows": rows,
                "totals": totals,
                "gate_passed": gate,
                "source_kind": "real_feature_smoke",
                "test_scored": False,
                "accuracy_not_measured": True,
            }
            save(f"pilot-{candidate['name']}", result)
        candidates.append(result)
        print({k: result[k] for k in ["name", "totals", "gate_passed"]}, flush=True)
    eligible = [c for c in candidates if c["gate_passed"]]
    winner = (
        max(eligible, key=lambda c: (c["totals"]["Fall"]["after"], c["totals"]["ADL"]["after"]))
        if eligible
        else None
    )
    save(
        "pilot",
        {
            "candidates": candidates,
            "selected": winner["name"] if winner else None,
            "source_kind": "real_feature_smoke",
            "test_scored": False,
        },
    )
    verify()


def training():
    record = verify(strict_source=True)
    if path("validation").exists():
        raise ContractError("Validation experiment already exists")
    pilot = read_json(path("pilot"))
    if pilot["selected"] is None:
        raise ContractError("No pose candidate passed; stop")
    candidate = next(c for c in pilot["candidates"] if c["name"] == pilot["selected"])
    qa = read_json(path("visual-check"))
    if qa.get("selected") != candidate["name"] or qa.get("passed") is not True:
        raise ContractError("Recovered observations need visual verification")
    assert sha256(ROOT / candidate["index"]) == candidate["index_sha256"]
    config = load_config(ROOT / candidate["config"], "cuda")
    index_path, _ = prepare(config, reuse_index=ROOT / candidate["index"])
    rows = []
    for item in record["ae_candidates"]:
        run = ROOT / "runs" / f"ur-{candidate['name']}-{item['name']}-001"
        current = deepcopy(config)
        current["autoencoder"].update(
            sampling=item["sampling"], latent_dim=item["latent_dim"], **record["ae_limit"]
        )
        if run.exists():
            raise ContractError(f"Preserve existing run: {run}")
        train(current, index_path, run)
        threshold = calibrate(run)
        rows.append(
            {
                "run": run.name,
                "threshold": threshold["threshold"],
                "status": read_json(run / "status.json"),
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
                for k in ["operational_accuracy", "sequence_fall_f1", "end_to_end_fall_recall"]
            )
            + (-r["validation"]["FP"],)
        ),
    )
    passed = all(winner["validation"][k] >= v for k, v in record["validation_targets"].items())
    save(
        "validation",
        {
            "source_kind": "real_end_to_end",
            "candidates": rows,
            "selected": winner["run"],
            "validation_target_reached": passed,
            "test_scored": False,
        },
    )
    verify()
    print({"selected": winner["run"], "validation_target_reached": passed}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["lock", "pilot", "training", "verify"])
    args = parser.parse_args()
    result = globals()[args.stage]()
    if args.stage == "verify":
        print({"preserved_runs": len(result["preserved_runs"]), "verified": True})
