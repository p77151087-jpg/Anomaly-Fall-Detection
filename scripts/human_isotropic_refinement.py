"""Bounded Train/Validation-only refinement. Never evaluate a new candidate on Test."""

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from fall_detect.config import load_config, project_path
from fall_detect.detection.events import score_sequence
from fall_detect.evaluation.metrics import sequence_metrics
from fall_detect.io_utils import ContractError, read_json, sha256, write_csv, write_json
from fall_detect.training import experiment

ROOT = project_path(".")
PREFIX = "isotropic-refinement-20261012"
BASE = "ur-human-isotropic-001"
INDEX = ROOT / "runs" / BASE / "dataset-index.json"
ORIGINAL_CACHE_FOR = experiment.cache_for


def evidence(kind):
    return ROOT / "docs/validation" / f"{PREFIX}-{kind}.json"


def save_new(path, value):
    if path.exists():
        raise ContractError(f"Preserve completed evidence: {path}")
    write_json(path, value)


def allowed_source(index, sid):
    source = next(s for s in index["split_manifest"]["sequences"] if s["sequence_id"] == sid)
    if source["split"] not in ("train", "validation"):
        raise ContractError("This experiment forbids Test feature access")
    return source


@contextmanager
def train_validation_only(accesses):
    def guarded(index, sid):
        source = allowed_source(index, sid)
        accesses.append({"sequence_id": sid, "split": source["split"]})
        return ORIGINAL_CACHE_FOR(index, sid)

    with patch.object(experiment, "cache_for", side_effect=guarded):
        yield


def nearest_train(normalized, train_z, owners):
    distances, nearest = [], []
    train_norm = (train_z.astype(np.float64) ** 2).sum(axis=1)
    for start in range(0, len(normalized), 64):
        block = normalized[start : start + 64].astype(np.float64)
        squared = np.maximum(
            (block**2).sum(axis=1)[:, None] + train_norm - 2 * block @ train_z.T, 0
        )
        indices = squared.argmin(axis=1)
        distances.extend((squared[np.arange(len(block)), indices] / train_z.shape[1]).tolist())
        nearest.extend(owners[indices].tolist())
    return np.asarray(distances), nearest


def diagnose():
    if evidence("diagnosis").exists():
        raise ContractError("Diagnosis already completed; use status")
    scorer = experiment.Scorer(ROOT / "runs" / BASE)
    accesses, rows, arrays, owners = [], [], [], []
    with train_validation_only(accesses):
        for source in scorer.index["split_manifest"]["sequences"]:
            if source["split"] == "train":
                if source["label"] != 0:
                    raise ContractError("Train must contain ADL only")
                cache = experiment.cache_for(scorer.index, source["sequence_id"])
                arrays.append(scorer.scaler.transform(cache.features))
                owners.extend([source["sequence_id"]] * len(cache.features))
        train_z, owners = np.concatenate(arrays), np.asarray(owners)
        score_groups = {"train_adl": [], "validation_adl": [], "validation_fall": []}
        for source in scorer.index["split_manifest"]["sequences"]:
            if source["split"] not in ("train", "validation"):
                continue
            sid = source["sequence_id"]
            cache = experiment.cache_for(scorer.index, sid)
            scores = scorer.score(cache.features)
            out, windows, events = score_sequence(cache, scores, scorer.threshold["threshold"])
            row = {
                **source,
                **out,
                "n_features": len(scores),
                "mean_clean_score": float(scores.mean()) if len(scores) else None,
                "score_percentiles": np.percentile(scores, [0, 50, 90, 100]).tolist()
                if len(scores)
                else None,
                "quality_counts": dict(Counter(r.quality for r in cache.records)),
                "events": events,
                "feature_array_sha256": cache.metadata["array_sha256"],
            }
            group = (
                "train_adl"
                if source["split"] == "train"
                else ("validation_fall" if source["label"] else "validation_adl")
            )
            score_groups[group].extend(scores.tolist())
            if source["split"] == "validation" and len(scores):
                distance, nearest = nearest_train(
                    scorer.scaler.transform(cache.features), train_z, owners
                )
                row["nearest_train_mse_percentiles"] = np.percentile(
                    distance, [0, 50, 90, 100]
                ).tolist()
                row["nearest_train_sequence_counts"] = dict(Counter(nearest))
                if source["label"] == 1 and out["prediction"] == 0:
                    row["window_timeline"] = windows
            rows.append(row)
    val_rows = [r for r in rows if r["split"] == "validation"]
    calibration = read_json(ROOT / "runs" / BASE / "calibration.json")
    means = {k: float(np.mean(v)) for k, v in score_groups.items()}
    summaries = {
        k: {
            "n_windows": len(v),
            "mean": means[k],
            "percentiles": np.percentile(v, [0, 50, 90, 100]).tolist(),
        }
        for k, v in score_groups.items()
    }
    save_new(
        evidence("diagnosis"),
        {
            "created_at": datetime.now(UTC).isoformat(),
            "baseline": BASE,
            "evidence_tier": "real_feature_smoke",
            "scope": "Fresh AE inference on preserved genuine Train/Validation features; no new raw-video extraction",
            "threshold": scorer.threshold["threshold"],
            "rows": rows,
            "window_score_groups": summaries,
            "normal_clean_val_train_ratio": means["validation_adl"] / means["train_adl"],
            "validation_metrics": sequence_metrics(val_rows),
            "best_saved_threshold_operational_accuracy": max(
                c["metrics"]["operational_accuracy"] for c in calibration["candidates"]
            ),
            "feature_accesses": accesses,
            "test_feature_reads": 0,
            "new_test_evaluations": 0,
            "confirmed": [
                "Saved Test aggregate differs from Validation by two additional false negatives; both have 23 sequences and zero false positives/sequence unknowns.",
                "All Train examples are ADL. Current Validation missed falls have legal scores below the normal-controlled threshold.",
                "Overlapping stride-1 windows are correlated; thousands of windows are not thousands of independent videos.",
                "Historical Test was viewed and Validation has repeatedly selected experiments; neither is a fresh blind estimate.",
            ],
            "hypotheses": [
                "Normal reconstruction generalization and anomaly reconstruction capacity may depend on the new aspect-preserving feature distribution.",
                "Repeated Validation selection and a small sample can contribute to an optimistic Validation result; their exact causal contribution to the Test gap is not identified.",
                "Partial/out-of-frame joints and short usable descent context constrain anomaly evidence; no formal joint/timing ground truth is available.",
            ],
        },
    )
    print(
        {
            "means": means,
            "validation": sequence_metrics(val_rows),
            "threshold_ceiling": max(
                c["metrics"]["operational_accuracy"] for c in calibration["candidates"]
            ),
        }
    )


def frozen_files():
    """Hash preservation only: this reads no Test arrays into a model or diagnosis."""
    folders = [
        "runs",
        "configs",
        "data/manifests",
        "data/cache/ur-human-isotropic-001",
        "outputs/ur-human-isotropic-test-20261012-001",
        "src/fall_detect",
    ]
    files = {}
    for folder in folders:
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                files[path.relative_to(ROOT).as_posix()] = sha256(path)
    index = read_json(INDEX)
    for entry in index["weights"].values():
        path = project_path(entry["local_path"])
        files[path.relative_to(ROOT).as_posix()] = sha256(path)
    return files


def lock():
    if evidence("lock").exists():
        raise ContractError("Candidate lock already exists")
    diagnosis = read_json(evidence("diagnosis"))
    specifications = [
        ("ur-isotropic-compact-20261012-001", 64, 8, 0.0),
        ("ur-isotropic-wide-20261012-001", 256, 64, 0.0),
        ("ur-isotropic-denoise-20261012-001", 128, 32, 0.1),
    ]
    for name, _, _, _ in specifications:
        if (ROOT / "runs" / name).exists() or (ROOT / "configs" / f"{name}.yaml").exists():
            raise ContractError("All preregistered run/config IDs must be new")
    before = frozen_files()
    candidates = []
    for name, hidden, latent, noise in specifications:
        config = load_config(ROOT / "configs/ur-human-isotropic-001.yaml")
        config["autoencoder"].update(hidden_dim=hidden, latent_dim=latent, input_noise_std=noise)
        path = ROOT / "configs" / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        candidates.append(
            {
                "run": name,
                "config": path.relative_to(ROOT).as_posix(),
                "config_sha256": sha256(path),
                "hidden": hidden,
                "latent": latent,
                "input_noise_std": noise,
            }
        )
    save_new(
        evidence("lock"),
        {
            "created_at": datetime.now(UTC).isoformat(),
            "reference": BASE,
            "default_run_retained": "ur-sample-human-s1-001",
            "diagnosis_sha256": sha256(evidence("diagnosis")),
            "files_before": before,
            "source_sha256": sha256(Path(__file__)),
            "index_sha256": sha256(INDEX),
            "candidates": candidates,
            "rationale": "Normal Val reconstruction error is 2.03x clean Train; threshold ceiling is 21/23. These AE capacity/noise candidates have not been trained on the isotropic recipe. The old per_frame recipe trials are not reused as isotropic results.",
            "fixed": "Original split, frozen genuine isotropic 512D ST-GCN, all 3638 Train ADL windows for scaler/AE, sequence sampling, seed42, wd.001, clean normal Val checkpoint, balanced sequence accuracy Val threshold, K3 strict gt, same quality and window schedules.",
            "selection": "Compare to highest Validation reference21/23. Require strictly better overall Accuracy, F1 not lower, no additional FP or unknown, >=4 coarse phase hits and zero early events. Each recovered reference FN needs a current-good-pose alert in its preexisting coarse phase. Rank eligible by overall Accuracy, F1, overall recall, fewer FP; tie retains earlier fixed candidate. Keep original default if no candidate passes all guards.",
            "stop": "Exactly three candidates, <=100 epochs each, patience10, no extra seeds/variants/threshold objectives. Stop after all three regardless of outcomes. No Test evaluation this round, including a successful candidate.",
            "coarse_intervals": read_json(ROOT / "docs/validation/human-isotropic-lock.json")[
                "coarse_intervals"
            ],
            "reference_metrics": diagnosis["validation_metrics"],
            "reference_fn": [
                r["sequence_id"]
                for r in diagnosis["rows"]
                if r["split"] == "validation" and r["label"] == 1 and r["prediction"] == 0
            ],
            "historical_test_seen": True,
            "repeated_validation_not_independent": True,
            "no_test_ranking_or_calibration": True,
        },
    )
    print("Three candidates preregistered; old artifacts preserved", flush=True)


def verify_lock(record):
    if sha256(Path(__file__)) != record["source_sha256"]:
        raise ContractError("Experiment source changed after preregistration")
    if sha256(evidence("diagnosis")) != record["diagnosis_sha256"]:
        raise ContractError("Diagnosis changed after preregistration")
    for relative, expected in record["files_before"].items():
        if sha256(ROOT / relative) != expected:
            raise ContractError(f"Preserved artifact changed: {relative}")
    for candidate in record["candidates"]:
        if sha256(ROOT / candidate["config"]) != candidate["config_sha256"]:
            raise ContractError("Preregistered config changed")


def validation_result(name, record, export=True):
    from human_residual import visible_pose_keys

    scorer = experiment.Scorer(ROOT / "runs" / name)
    rows, all_windows, all_events, phases = [], [], [], []
    for source in scorer.index["split_manifest"]["sequences"]:
        if source["split"] != "validation":
            continue
        sid = source["sequence_id"]
        cache = experiment.cache_for(scorer.index, sid)
        out, windows, events = score_sequence(
            cache, scorer.score(cache.features), scorer.threshold["threshold"]
        )
        rows.append({**out, "label": source["label"]})
        all_windows.extend(windows)
        all_events.extend(events)
        if sid in record["coarse_intervals"]:
            allowed_source(scorer.index, sid)
            interval = record["coarse_intervals"][sid]
            lo, hi = interval["fall_phase_ms"]
            good = visible_pose_keys(scorer, sid)
            phases.append(
                {
                    "sequence_id": sid,
                    "phase_event_starts": [
                        e["confirmed_start_ms"]
                        for e in events
                        if lo <= e["confirmed_start_ms"] <= hi
                    ],
                    "early_event_starts": [
                        e["confirmed_start_ms"]
                        for e in events
                        if e["confirmed_start_ms"] < interval["normal_until_ms"]
                    ],
                    "visible_phase_alert_ms": [
                        w["available_at_ms"]
                        for w in windows
                        if w["state"] == "suspected_fall"
                        and lo <= w["available_at_ms"] <= hi
                        and (w["sample_index"], w["track_id"]) in good
                    ],
                }
            )
    metrics = sequence_metrics(rows)
    chosen = read_json(ROOT / "runs" / name / "calibration.json")["chosen"]["metrics"]
    if any(chosen[k] != v for k, v in metrics.items()):
        raise ContractError("Validation recomputation disagrees with calibration")
    recovered = [
        r["sequence_id"]
        for r in rows
        if r["sequence_id"] in record["reference_fn"] and r["prediction"] == 1
    ]
    visible_recovered = [
        p["sequence_id"]
        for p in phases
        if p["sequence_id"] in recovered and p["visible_phase_alert_ms"]
    ]
    ref = record["reference_metrics"]
    phase_hits = sum(bool(p["phase_event_starts"]) for p in phases)
    early = sum(len(p["early_event_starts"]) for p in phases)
    checks = {
        "accuracy_strictly_higher": metrics["operational_accuracy"] > ref["operational_accuracy"],
        "f1_not_lower": metrics["sequence_fall_f1"] >= ref["sequence_fall_f1"],
        "no_more_fp": metrics["FP"] <= ref["FP"],
        "no_more_unknown": metrics["n_unknown"] <= ref["n_unknown"],
        "phase_hits_not_lower": phase_hits >= 4,
        "no_early_events": early == 0,
        "recovered_fn_current_pose_alert": set(recovered) <= set(visible_recovered),
    }
    result = {
        "run": name,
        "metrics": metrics,
        "predictions": rows,
        "phases": phases,
        "recovered_fn": recovered,
        "visible_recovered_fn": visible_recovered,
        "phase_hits": phase_hits,
        "early_events": early,
        "adoption_checks": checks,
        "eligible": all(checks.values()),
        "threshold": scorer.threshold["threshold"],
        "status": read_json(ROOT / "runs" / name / "status.json"),
        "artifact_hashes": experiment.verify_artifacts(ROOT / "runs" / name),
    }
    if export:
        folder = ROOT / "outputs" / PREFIX / name
        if folder.exists():
            raise ContractError("Preserve previous Validation exports")
        folder.mkdir(parents=True)
        write_json(folder / "metrics.json", metrics)
        write_csv(folder / "sequences.csv", rows)
        write_csv(folder / "windows.csv", all_windows)
        write_csv(folder / "events.csv", all_events)
        result["exports"] = {p.relative_to(ROOT).as_posix(): sha256(p) for p in folder.iterdir()}
    return result


def rank(result):
    metrics = result["metrics"]
    return (
        metrics["operational_accuracy"],
        metrics["sequence_fall_f1"],
        metrics["end_to_end_fall_recall"],
        -metrics["FP"],
    )


def training():
    if evidence("validation").exists():
        raise ContractError("Completed round; no retraining or additional candidates")
    record = read_json(evidence("lock"))
    verify_lock(record)
    accesses, results = [], []
    with train_validation_only(accesses):
        for candidate in record["candidates"]:
            name = candidate["run"]
            result_path = evidence(f"result-{name}")
            if result_path.exists():
                experiment.verify_artifacts(ROOT / "runs" / name)
                results.append(read_json(result_path))
                continue
            run = ROOT / "runs" / name
            if run.exists():
                raise ContractError(
                    "Existing incomplete run needs an explicit audit; do not overwrite"
                )
            print(f"Training {name}", flush=True)
            experiment.train(load_config(ROOT / candidate["config"]), INDEX, run)
            experiment.calibrate(run)
            result = validation_result(name, record)
            save_new(result_path, result)
            results.append(result)
            print(
                {
                    "run": name,
                    "metrics": result["metrics"],
                    "adoption_checks": result["adoption_checks"],
                },
                flush=True,
            )
    eligible = [r for r in results if r["eligible"]]
    selected = max(eligible, key=rank)["run"] if eligible else None
    save_new(
        evidence("validation"),
        {
            "created_at": datetime.now(UTC).isoformat(),
            "lock_sha256": sha256(evidence("lock")),
            "evidence_tier": "real_feature_smoke",
            "reference": BASE,
            "results": results,
            "selected_candidate": selected,
            "highest_validation_candidate": max(results, key=rank)["run"],
            "default_run": "ur-sample-human-s1-001",
            "feature_accesses": accesses,
            "test_feature_reads": 0,
            "new_test_evaluations": 0,
            "formal_timing_ground_truth": False,
            "historical_test_seen": True,
            "repeated_validation_not_independent": True,
        },
    )
    verify_lock(record)
    print({"selected_candidate": selected, "test_evaluations": 0}, flush=True)


def status():
    record = read_json(evidence("lock"))
    verify_lock(record)
    result = read_json(evidence("validation")) if evidence("validation").exists() else None
    print(
        {
            "preserved_files": len(record["files_before"]),
            "complete": result is not None,
            "selected_candidate": result["selected_candidate"] if result else None,
            "new_test_evaluations": 0,
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["diagnose", "lock", "training", "status"])
    args = parser.parse_args()
    {"diagnose": diagnose, "lock": lock, "training": training, "status": status}[args.command]()


if __name__ == "__main__":
    main()
