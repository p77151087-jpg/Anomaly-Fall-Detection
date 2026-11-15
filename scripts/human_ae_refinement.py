"""Finite Human-feature AE refinement, with immutable Train/Validation diagnosis."""

import argparse
from collections import Counter
from copy import deepcopy
from datetime import UTC, datetime

import numpy as np
import yaml
from ae_generalization import result
from dense_schedule import PHASES, timing

from fall_detect.config import load_config, project_path
from fall_detect.data.cache import load_pose_cache
from fall_detect.detection.events import score_sequence
from fall_detect.evaluation.metrics import sequence_metrics
from fall_detect.io_utils import ContractError, read_json, sha256, write_json
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
CANDIDATES = [
    ("compact", 64, 8, 0.0),
    ("bottleneck", 128, 16, 0.0),
    ("wide", 256, 64, 0.0),
    ("denoise", 128, 32, 0.1),
]


def path(stage):
    return ROOT / f"docs/validation/human-ae-refinement-{stage}.json"


def save(stage, value):
    if path(stage).exists():
        raise ContractError(f"Preserve completed evidence: {stage}")
    write_json(path(stage), value)


def predictions(run):
    scorer = Scorer(ROOT / "runs" / run)
    rows = []
    for source in scorer.index["split_manifest"]["sequences"]:
        if source["split"] != "validation":
            continue
        cache = cache_for(scorer.index, source["sequence_id"])
        outcome, _, events = score_sequence(
            cache, scorer.score(cache.features), scorer.threshold["threshold"]
        )
        rows.append({**outcome, "label": source["label"], "events": events})
    return rows


def diagnose():
    if path("diagnosis").exists():
        raise ContractError("Diagnosis already completed")
    scorer = Scorer(ROOT / "runs" / BASELINE)
    rows = []
    for source in scorer.index["split_manifest"]["sequences"]:
        if source["split"] not in ("train", "validation"):
            continue
        sid = source["sequence_id"]
        cache = cache_for(scorer.index, sid)
        scores = scorer.score(cache.features)
        outcome, windows, events = score_sequence(cache, scores, scorer.threshold["threshold"])
        row = {
            **source,
            **outcome,
            "events": events,
            "feature_array_sha256": cache.metadata["array_sha256"],
            "feature_count": len(scores),
            "record_quality_counts": dict(Counter(r.quality for r in cache.records)),
            "raw_score_percentiles": np.percentile(scores, [0, 50, 90, 100]).tolist()
            if len(scores)
            else None,
        }
        if sid in ("fall-19", "fall-18", "fall-30"):
            observations, _ = load_pose_cache(scorer.index["sequences"][sid]["pose_path"], {})
            row["pose_timeline"] = [
                {
                    "frame": frame.source_frame_id,
                    "timestamp_ms": frame.timestamp_ms,
                    "poses": [
                        {
                            "track": p.track_id,
                            "valid_measured_joints": int((p.keypoints[:, 2] >= 0.2).sum()),
                        }
                        for p in poses
                    ],
                }
                for frame, poses in observations
            ]
            row["score_timeline"] = windows
            contact = ROOT / f"outputs/human-optimization-diagnosis/{sid}.jpg"
            row["contact_sheet"] = str(contact.relative_to(ROOT))
            row["contact_sha256"] = sha256(contact)
        rows.append(row)
    save(
        "diagnosis",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "source_kind": "real_end_to_end",
            "baseline": BASELINE,
            "threshold": scorer.threshold["threshold"],
            "rows": rows,
            "confirmed": [
                "All three missed Validation videos have evaluable low-score windows before pose loss; no sequence-level unknown.",
                "Visual contact sheets show partial body out of frame during descent/landing; pose confidence falls and/or track observations stop.",
                "fall-30 has one raw score above threshold, but not three consecutive legal high scores; K3 correctly prevents a confirmed event.",
                "The saved baseline threshold grid cannot exceed 20/23 operational accuracy.",
                "Existing fall-25/fall-29 demos show only 1/2 visible alert frames because per-frame quality can mask a valid 80%-quality window.",
            ],
            "hypotheses": [
                "A smaller latent space may preserve higher reconstruction error during early descent.",
                "A wider AE or denoising may generalize normal activities better and permit a useful Validation threshold.",
                "AE capacity cannot recover joints outside the camera image or create legal windows after track loss.",
            ],
            "test_scored": False,
            "formal_timing_ground_truth": False,
        },
    )


def lock():
    if path("lock").exists():
        raise ContractError("Experiment already locked")
    diagnosis = read_json(path("diagnosis"))
    if diagnosis["test_scored"]:
        raise ContractError("Diagnosis must exclude Test")
    base = load_config(ROOT / f"runs/{BASELINE}/resolved-config.yaml")
    candidates = []
    for name, hidden, latent, noise in CANDIDATES:
        run = f"ur-human-ae-{name}-001"
        filename = ROOT / f"configs/{run}.yaml"
        if filename.exists() or (ROOT / "runs" / run).exists():
            raise ContractError(f"Candidate already exists: {run}")
        config = deepcopy(base)
        config["autoencoder"].update(hidden_dim=hidden, latent_dim=latent, input_noise_std=noise)
        filename.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf8")
        candidates.append({"run": run, "config": str(filename.relative_to(ROOT))})
    files = [
        "scripts/human_ae_refinement.py",
        "scripts/ae_generalization.py",
        "scripts/dense_schedule.py",
        "src/fall_detect/config.py",
        "src/fall_detect/training/experiment.py",
        "src/fall_detect/training/scaler.py",
        "src/fall_detect/models/autoencoder.py",
        "src/fall_detect/detection/events.py",
        "src/fall_detect/evaluation/metrics.py",
    ]
    save(
        "lock",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "baseline": BASELINE,
            "diagnosis_sha256": sha256(path("diagnosis")),
            "source_index": INDEX,
            "source_index_sha256": sha256(ROOT / INDEX),
            "candidates": candidates,
            "fixed": "Same Human512D cache, raw sequence split, Train ADL-only scaler/AE; seed42, sequence sampling, weight decay .001, max100epochs/patience10, 30 frames/stride1, quality/K3 unchanged.",
            "checkpoint_selection": "Clean normal Validation reconstruction loss only.",
            "calibration": "Validation balanced_sequence_accuracy, fewer FP, higher threshold.",
            "selection": "Eligible baseline and candidates ranked by operational Accuracy, F1, all-Fall recall, fewer FP; exact tie keeps baseline/earlier candidate.",
            "timing_guard": "At least baseline's two coarse phase cases, zero early events on the same four diagnostic Validation clips; Accuracy/F1 no worse and unknown no greater. Not a formal timing metric.",
            "coarse_visual_intervals": PHASES,
            "targets": {
                "operational_accuracy": 0.9,
                "sequence_accuracy": 0.9,
                "sequence_fall_f1": 0.87,
            },
            "stop": "Exactly four candidates, at most400epochs. No additional candidates or seeds after results. Test only once for frozen selected model if all Validation targets and timing guard pass; no subsequent Test-driven adjustment.",
            "historical_test_seen": True,
            "repeated_validation_not_independent": True,
            "preserved_runs": {
                p.parent.name: verify_artifacts(p.parent)
                for p in sorted((ROOT / "runs").glob("*/frozen-artifacts.json"))
            },
            "locked_files": {
                str(p.relative_to(ROOT)): sha256(p)
                for p in [*(ROOT / "configs").glob("*.yaml"), *(ROOT / f for f in files)]
            },
            "source_model_hashes": {
                r["local_path"]: sha256(ROOT / r["local_path"])
                for r in read_json(ROOT / "weights/sample-human-manifest.json")["resources"]
            },
        },
    )


def verify():
    record = read_json(path("lock"))
    for filename, digest in {**record["locked_files"], **record["source_model_hashes"]}.items():
        if sha256(ROOT / filename) != digest:
            raise ContractError(f"Locked file changed: {filename}")
    if sha256(path("diagnosis")) != record["diagnosis_sha256"]:
        raise ContractError("Diagnosis changed")
    if sha256(ROOT / record["source_index"]) != record["source_index_sha256"]:
        raise ContractError("Source index changed")
    for run, hashes in record["preserved_runs"].items():
        if verify_artifacts(ROOT / "runs" / run) != hashes:
            raise ContractError(f"Old run changed: {run}")
    return record


def ranking(row):
    m = row["validation"]
    return m["operational_accuracy"], m["sequence_fall_f1"], m["end_to_end_fall_recall"], -m["FP"]


def training():
    record = verify()
    if path("validation").exists():
        raise ContractError("Completed experiment; preserve results")
    baseline = result(BASELINE)
    baseline["timing"] = timing(BASELINE, PHASES)
    baseline["timing_guard_passed"] = True
    rows = [baseline]
    for candidate in record["candidates"]:
        name = candidate["run"]
        stage = "trained-" + name
        if path(stage).exists():
            row = read_json(path(stage))
            if verify_artifacts(ROOT / "runs" / name) != row["hashes"]:
                raise ContractError("Completed candidate changed")
        else:
            run = ROOT / "runs" / name
            if run.exists():
                raise ContractError("Partial run preserved; inspect before continuation")
            train(load_config(ROOT / candidate["config"]), ROOT / INDEX, run)
            calibrate(run)
            row = result(name)
            row["timing"] = timing(name, PHASES)
            row["predictions"] = predictions(name)
            actual = sequence_metrics(row["predictions"])
            for key, value in actual.items():
                if row["validation"][key] != value:
                    raise ContractError(f"Recomputed Validation metric differs: {key}")
            row["timing_guard_passed"] = (
                row["timing"]["fall_phase_sequences"] >= baseline["timing"]["fall_phase_sequences"]
                and row["timing"]["early_events"] <= baseline["timing"]["early_events"]
                and row["validation"]["operational_accuracy"]
                >= baseline["validation"]["operational_accuracy"]
                and row["validation"]["sequence_fall_f1"]
                >= baseline["validation"]["sequence_fall_f1"]
                and row["validation"]["n_unknown"] <= baseline["validation"]["n_unknown"]
            )
            save(stage, row)
        rows.append(row)
        print(
            {
                "run": name,
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
    record = verify()
    print({"old_runs_preserved": len(record["preserved_runs"])})
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
    parser.add_argument("stage", choices=["diagnose", "lock", "training", "test", "status"])
    globals()[parser.parse_args().stage]()
