"""Two preregistered causal Human temporal-context candidates, Train/Val only."""

import argparse
from copy import deepcopy
from datetime import UTC, datetime

import numpy as np
import yaml
from ae_generalization import result
from dense_schedule import PHASES, timing

from fall_detect.config import load_config, project_path, recipe_fingerprint
from fall_detect.detection.events import score_sequence
from fall_detect.evaluation.metrics import sequence_metrics
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
BASELINE = "ur-sample-human-s1-001"
INDEX = "data/cache/ur-sample-human-s1-001/dataset-index.json"
PILOT_IDS = [
    "adl-05",
    "adl-24",
    "adl-04",
    "adl-35",
    "fall-09",
    "fall-19",
    "fall-18",
    "fall-30",
    "fall-25",
    "fall-29",
]
INTERVALS = {
    **PHASES,
    "fall-18": {"upright_until_ms": 400, "fall_phase_ms": [800, 1601]},
    "fall-30": {"upright_until_ms": 800, "fall_phase_ms": [1334, 1902]},
}
MISSED = ("fall-19", "fall-18", "fall-30")


def path(stage):
    return ROOT / f"docs/validation/human-context-{stage}.json"


def save(stage, value):
    if path(stage).exists():
        raise ContractError(f"Preserve completed evidence: {stage}")
    write_json(path(stage), value)


def lock():
    if path("lock").exists():
        raise ContractError("Experiment already locked")
    base = load_config(ROOT / f"runs/{BASELINE}/resolved-config.yaml")
    candidates = []
    for size in (20, 15):
        run = f"ur-human-context{size}-001"
        config = deepcopy(base)
        config["skeleton"].update(window_size=size, temporal_context=f"short_{size}")
        config["data"]["cache_dir"] = f"data/cache/{run}"
        filename = ROOT / f"configs/{run}.yaml"
        if (
            filename.exists()
            or (ROOT / "runs" / run).exists()
            or project_path(config["data"]["cache_dir"]).exists()
        ):
            raise ContractError(f"Preserve existing candidate: {run}")
        filename.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf8")
        candidates.append({"run": run, "config": str(filename.relative_to(ROOT)), "size": size})
    files = [
        "src/fall_detect/config.py",
        "src/fall_detect/models/stgcn.py",
        "src/fall_detect/skeleton/preprocessing.py",
        "src/fall_detect/skeleton/windows.py",
        "src/fall_detect/data/pose_reuse.py",
        "src/fall_detect/pipeline.py",
        "src/fall_detect/data/cache.py",
        "src/fall_detect/detection/events.py",
        "src/fall_detect/training/experiment.py",
        "scripts/human_context.py",
        "scripts/ae_generalization.py",
        "scripts/dense_schedule.py",
        "tests/test_human_context.py",
    ]
    save(
        "lock",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "baseline": BASELINE,
            "candidates": candidates,
            "pilot_ids": PILOT_IDS,
            "coarse_intervals": INTERVALS,
            "diagnosis_sha256": sha256(ROOT / "docs/validation/human-ae-refinement-diagnosis.json"),
            "before_sha256": sha256(path("before")),
            "source_index": INDEX,
            "source_index_sha256": sha256(ROOT / INDEX),
            "hypothesis": "A 20/15-frame causal context may reduce normal-history dilution during early descent, before observed skeleton quality fails. Not established; shorter duration differs from upstream training.",
            "fixed": "Same observed poses, weights, 13-to-14 preprocessing, frozen pooled 512D features, original sequence split, target30fps/stride1, quality .2 / 9 real joints / .8 good frames, no gap filling, K3. AE128/32 sequence sampling wd.001 noise0 seed42 max100 patience10. Scaler/AE TrainADL only; clean normalVal checkpoint; Validation balanced sequence accuracy threshold.",
            "pilot_gate": "All10 pilot sequences retain a legal K3 score; ADL mean valid-endpoint coverage >=90% baseline; each of six coarse Fall phases has >=1 legal K3 window; summed legal phase windows >=80% baseline; exact observed-pose array hashes unchanged. Finite genuine512D features, no Test scoring.",
            "selection": "Baseline and eligible candidates: highest Validation operational Accuracy, then F1, all-Fall recall, fewer FP; ties keep baseline/earlier candidate.",
            "timing_guard": "At least baseline coarse phase-hit count, no additional early events, no Accuracy/F1 regression or extra unknown. Every newly detected formerly missed case must have an event start inside its prelocked coarse phase. Six Validation cases only, not official timing ground truth.",
            "targets": {
                "operational_accuracy": 0.9,
                "sequence_accuracy": 0.9,
                "sequence_fall_f1": 0.87,
            },
            "stop": "Exactly two contexts and one fixed AE each, max200epochs. Failed pilots are not trained. No extra sizes, seeds, threshold objectives or AE candidates after results. Test once on frozen selected run only after Validation targets/timing guard pass; no post-Test tuning.",
            "historical_test_seen": True,
            "repeated_validation_not_independent": True,
            "implementation_sha256": {name: sha256(ROOT / name) for name in files},
            "config_sha256": {c["config"]: sha256(ROOT / c["config"]) for c in candidates},
            "coarse_contact_sha256": {
                sid: sha256(ROOT / f"outputs/human-context-diagnosis/{sid}.jpg")
                for sid in ("fall-18", "fall-30")
            },
        },
    )


def verify():
    record = read_json(path("lock"))
    if sha256(path("before")) != record["before_sha256"]:
        raise ContractError("Initial snapshot changed")
    before = read_json(path("before"))
    files = {
        **record["implementation_sha256"],
        **record["config_sha256"],
        **before["configs"],
        **before["source_cache_files"],
        **before["manifests"],
    }
    for filename, digest in files.items():
        if sha256(ROOT / filename) != digest:
            raise ContractError(f"Preserved file changed: {filename}")
    if sha256(ROOT / INDEX) != record["source_index_sha256"]:
        raise ContractError("Source index changed")
    for run, hashes in before["old_runs"].items():
        if verify_artifacts(ROOT / "runs" / run) != hashes:
            raise ContractError(f"Old run changed: {run}")
    old_index = read_json(ROOT / INDEX)
    for r in old_index["weights"].values():
        if sha256(ROOT / r["local_path"]) != r["sha256"]:
            raise ContractError("Source weight changed")
    for name, digest in before["source_files"].items():
        if sha256(ROOT / "docs/validation/human-context-legacy-source" / name) != digest:
            raise ContractError("Historical source snapshot changed")
    config = load_config(ROOT / f"runs/{BASELINE}/resolved-config.yaml")
    if (
        recipe_fingerprint(config, {k: v["sha256"] for k, v in old_index["weights"].items()})
        != old_index["recipe_hash"]
    ):
        raise ContractError("Legacy30 recipe hash changed")
    return record


def pilot_stats(index):
    rows = []
    for source in index["split_manifest"]["sequences"]:
        sid = source["sequence_id"]
        if sid not in PILOT_IDS:
            continue
        if source["split"] == "test":
            raise ContractError("Pilot must exclude Test")
        cache = cache_for(index, sid)
        # Constant scores measure legal K3 availability only, never classification quality.
        out, windows, _ = score_sequence(cache, np.zeros(len(cache.features)), 1.0)
        phase = INTERVALS.get(sid)
        phase_windows = (
            sum(
                w["event_score"] is not None
                and phase["fall_phase_ms"][0] <= w["available_at_ms"] <= phase["fall_phase_ms"][1]
                for w in windows
            )
            if phase
            else None
        )
        rows.append(
            {
                **source,
                "valid_features": len(cache.features),
                "legal_k3": out["sequence_score"] is not None,
                "window_coverage": out["window_coverage"],
                "legal_phase_windows": phase_windows,
                "pose_sha256": sha256(project_path(index["sequences"][sid]["pose_path"])),
            }
        )
    return rows


def pilot():
    record = verify()
    if path("pilot").exists():
        raise ContractError("Pilot already completed")
    baseline = pilot_stats(read_json(ROOT / INDEX))
    rows = []
    for c in record["candidates"]:
        stage = "pilot-" + c["run"]
        if path(stage).exists():
            row = read_json(path(stage))
            if sha256(row["index"]) != row["index_sha256"]:
                raise ContractError("Pilot changed")
        else:
            index_path, index = prepare(
                load_config(ROOT / c["config"]),
                selected_ids=PILOT_IDS,
                reuse_poses_index=ROOT / INDEX,
            )
            stats = pilot_stats(index)
            original = {r["sequence_id"]: r for r in baseline}
            if any(r["pose_sha256"] != original[r["sequence_id"]]["pose_sha256"] for r in stats):
                raise ContractError("Observed poses changed")
            adl_ratio = np.mean([r["window_coverage"] for r in stats if r["label"] == 0]) / np.mean(
                [r["window_coverage"] for r in baseline if r["label"] == 0]
            )
            phase_ratio = sum(r["legal_phase_windows"] or 0 for r in stats) / sum(
                r["legal_phase_windows"] or 0 for r in baseline
            )
            gate = bool(
                all(r["legal_k3"] for r in stats)
                and adl_ratio >= 0.9
                and phase_ratio >= 0.8
                and all(r["legal_phase_windows"] > 0 for r in stats if r["label"] == 1)
            )
            row = {
                "run": c["run"],
                "index": str(index_path),
                "index_sha256": sha256(index_path),
                "stats": stats,
                "adl_coverage_ratio": float(adl_ratio),
                "fall_phase_legal_ratio": float(phase_ratio),
                "gate_passed": gate,
                "observed_pose_hashes_unchanged": True,
                "model_load_reports": index["model_load_reports"],
            }
            save(stage, row)
        rows.append(row)
        print(
            {
                "pilot": c["run"],
                "gate": row["gate_passed"],
                "adl_coverage_ratio": row["adl_coverage_ratio"],
                "phase_ratio": row["fall_phase_legal_ratio"],
            },
            flush=True,
        )
    save(
        "pilot",
        {
            "source_kind": "real_feature_smoke",
            "baseline": baseline,
            "rows": rows,
            "test_scored": False,
            "accuracy_not_measured": True,
        },
    )
    verify()


def predictions(run):
    s = Scorer(ROOT / "runs" / run)
    rows = []
    for source in s.index["split_manifest"]["sequences"]:
        if source["split"] != "validation":
            continue
        cache = cache_for(s.index, source["sequence_id"])
        out, _, events = score_sequence(cache, s.score(cache.features), s.threshold["threshold"])
        rows.append({**out, "label": source["label"], "events": events})
    return rows


def ranking(row):
    m = row["validation"]
    return m["operational_accuracy"], m["sequence_fall_f1"], m["end_to_end_fall_recall"], -m["FP"]


def training():
    record = verify()
    if path("validation").exists():
        raise ContractError("Completed experiment; preserve results")
    pilot_report = read_json(path("pilot"))
    baseline = result(BASELINE)
    baseline.update(timing=timing(BASELINE, INTERVALS), timing_guard_passed=True)
    rows = [baseline]
    skipped = []
    for c, pilot_row in zip(record["candidates"], pilot_report["rows"]):
        if sha256(pilot_row["index"]) != pilot_row["index_sha256"]:
            raise ContractError("Pilot index changed")
        if not pilot_row["gate_passed"]:
            skipped.append(c["run"])
            continue
        stage = "trained-" + c["run"]
        if path(stage).exists():
            row = read_json(path(stage))
            if verify_artifacts(ROOT / "runs" / c["run"]) != row["hashes"]:
                raise ContractError("Completed candidate changed")
        else:
            run = ROOT / "runs" / c["run"]
            if run.exists():
                raise ContractError("Partial run preserved; inspect before continuation")
            config = load_config(ROOT / c["config"])
            index_path, index = prepare(config, reuse_poses_index=ROOT / INDEX)
            if not all(r["status"] == "prepared" for r in index["sequences"].values()):
                raise ContractError("Full preparation incomplete")
            train(config, index_path, run)
            calibrate(run)
            row = result(c["run"])
            row.update(
                index=str(index_path),
                timing=timing(c["run"], INTERVALS),
                predictions=predictions(c["run"]),
            )
            for key, value in sequence_metrics(row["predictions"]).items():
                if row["validation"][key] != value:
                    raise ContractError(f"Recomputed Validation metric mismatch: {key}")
            new_hits_safe = all(
                r["outcome"]["prediction"] != 1 or bool(r["fall_phase_event_starts"])
                for r in row["timing"]["rows"]
                if r["sequence_id"] in MISSED
            )
            row["newly_detected_miss_timing_passed"] = new_hits_safe
            row["timing_guard_passed"] = (
                row["timing"]["fall_phase_sequences"] >= baseline["timing"]["fall_phase_sequences"]
                and row["timing"]["early_events"] <= baseline["timing"]["early_events"]
                and row["validation"]["operational_accuracy"]
                >= baseline["validation"]["operational_accuracy"]
                and row["validation"]["sequence_fall_f1"]
                >= baseline["validation"]["sequence_fall_f1"]
                and row["validation"]["n_unknown"] <= baseline["validation"]["n_unknown"]
                and new_hits_safe
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
    selected = max((r for r in rows if r["timing_guard_passed"]), key=ranking)
    save(
        "validation",
        {
            "source_kind": "real_end_to_end",
            "rows": rows,
            "pilot_rejected": skipped,
            "selected": selected["run"],
            "raw_metric_winner": max(rows, key=ranking)["run"],
            "target_reached": all(
                selected["validation"][k] >= v for k, v in record["targets"].items()
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
    verify()
    for stage in ("pilot", "validation", "test"):
        if path(stage).exists():
            r = read_json(path(stage))
            if stage == "pilot":
                print({"pilot": [{"run": v["run"], "passed": v["gate_passed"]} for v in r["rows"]]})
            elif stage == "validation":
                for row in r["rows"]:
                    if verify_artifacts(ROOT / "runs" / row["run"]) != row["hashes"]:
                        raise ContractError("Result changed")
                    print(
                        {
                            "run": row["run"],
                            "validation": row["validation"],
                            "timing_guard": row["timing_guard_passed"],
                        }
                    )
                print({"selected": r["selected"], "target_reached": r["target_reached"]})
            else:
                print({"historical_test_seen": True, "test": r["metrics"]})
    print(
        {
            "test_scored": path("test").exists(),
            "old_runs_preserved": len(read_json(path("before"))["old_runs"]),
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["lock", "pilot", "training", "test", "status"])
    globals()[parser.parse_args().stage]()
