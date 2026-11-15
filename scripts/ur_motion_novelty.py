"""One locked normal-motion candidate, with isolated Train/Validation execution."""

import argparse
import copy
import os
import shutil
from datetime import UTC, datetime

import joblib
import numpy as np
import psutil
import torch
from ur_normal_methods import BASE, phases_for

from fall_detect.config import project_path
from fall_detect.data.cache import load_pose_cache
from fall_detect.detection.events import score_sequence
from fall_detect.evaluation.metrics import sequence_metrics
from fall_detect.io_utils import (
    ContractError,
    fingerprint,
    read_json,
    sha256,
    write_json,
    write_npz,
)
from fall_detect.models.motion_novelty import MotionNovelty, fusion_scores, motion_trajectory
from fall_detect.models.normal_alternatives import calibrate_normal_scores, candidate_rank
from fall_detect.models.stgcn import HumanFeatureExtractor
from fall_detect.skeleton.windows import CausalResampler, WindowBuilder
from fall_detect.training.experiment import Scorer, cache_for, load_index
from fall_detect.visualization.reports import evaluation_report

ROOT = project_path(".")
TAG = "ur-motion-novelty-20261012-001"
OLD = "ur-normal-methods-20261012-001"
E = ROOT / "docs/validation" / TAG
C = ROOT / "data/cache" / TAG
O = ROOT / "outputs" / TAG
RUN = ROOT / "runs" / TAG
PLAN = ROOT / "docs/ur-motion-novelty-20261012-plan.md"
SPEC = {
    "id": TAG,
    "kind": "frozen_wide_ae_and_motion_normal_neighbors",
    "motion": "ordered30_center_delta_div_first_span_and_log_span_ratio_90d",
    "neighbors": 5,
    "train_motion_scale": "leave_complete_source_video_out_p95",
    "ae_scale": "original_train_scores_p95",
    "fusion": "max(ae/ae_train_p95,motion/motion_train_leave_video_out_p95)",
    "quality": "unchanged",
    "k": 3,
    "comparison": "gt",
    "selection": "overall_accuracy,fall_f1,all_fall_recall,fewer_fp,higher_threshold",
    "test": "unknown_not_prepared_not_evaluated",
    "augmentation": "none",
    "neural_epochs": 0,
}
IMPLEMENTATIONS = [
    "scripts/ur_motion_novelty.py",
    "scripts/ur_normal_methods.py",
    "src/fall_detect/models/motion_novelty.py",
    "src/fall_detect/models/normal_alternatives.py",
    "src/fall_detect/skeleton/preprocessing.py",
    "src/fall_detect/skeleton/windows.py",
    "src/fall_detect/training/scaler.py",
    "src/fall_detect/training/experiment.py",
    "src/fall_detect/models/stgcn.py",
    "src/fall_detect/data/cache.py",
    "src/fall_detect/detection/events.py",
    "src/fall_detect/evaluation/metrics.py",
    "src/fall_detect/visualization/reports.py",
    "src/fall_detect/io_utils.py",
]
MUTABLE = [
    "AGENTS.md",
    "Fall_Detection_Codex_Implementation_Spec.md",
    "docs/current-progress.md",
    "docs/acceptance-results.md",
    "docs/course-alignment-plan.md",
]


def save(path, value):
    if path.exists():
        raise ContractError(f"Preserve existing evidence: {path}")
    write_json(path, value)


def validated_index():
    index = load_index(C / "dataset-index.json")
    sources = index["split_manifest"]["sequences"]
    if index.get("preparation_scope") != ["train", "validation"] or index["source_kind"] != "real":
        raise ContractError("Real Train/Validation-only scope required")
    train = [s for s in sources if s["split"] == "train"]
    if len(train) != 24 or any(s["label"] != 0 for s in train):
        raise ContractError("Exactly original 24 normal Train videos required")
    if len([s for s in sources if s["split"] == "validation"]) != 23:
        raise ContractError("All original 23 Validation videos required")
    for s in sources:
        entry = index["sequences"][s["sequence_id"]]
        if s["split"] != "test" and entry["status"] != "prepared":
            raise ContractError("Incomplete Train/Validation")
    return index


def verify_lock():
    lock = read_json(E / "lock.json")
    if lock["spec"] != SPEC:
        raise ContractError("Locked candidate changed")
    for key in ["source_hashes", "input_hashes"]:
        for name, digest in lock[key].items():
            if sha256(ROOT / name) != digest:
                raise ContractError(f"Locked bytes changed: {name}")
    return lock


def inventory():
    if E.exists():
        raise ContractError("Inventory already exists")
    active = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = " ".join(proc.info["cmdline"] or [])
            if (
                proc.pid != os.getpid()
                and "python" in (proc.info["name"] or "").lower()
                and any(x in cmd for x in [" train ", "training", "ur_normal_methods.py ae"])
                and proc.pid != psutil.Process().ppid()
            ):
                active.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if active:
        raise ContractError(f"Concurrent research: {active}")
    E.mkdir(parents=True)
    contracts = {}
    for name in MUTABLE:
        destination = E / "prior-contract" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
        contracts[name] = sha256(ROOT / name)
    assets = {}
    for scope in ["runs", "configs", "data", "weights", "outputs"]:
        for path in sorted((ROOT / scope).rglob("*")):
            if path.is_file():
                assets[path.relative_to(ROOT).as_posix()] = sha256(path)
                if len(assets) % 5000 == 0:
                    print(f"Preserved {len(assets)} old files", flush=True)
    # Old experiment evidence/reports are immutable; current progress is archived separately.
    for scope in [ROOT / "docs/validation" / OLD]:
        for path in sorted(scope.rglob("*")):
            if path.is_file():
                assets[path.relative_to(ROOT).as_posix()] = sha256(path)
    for name in [
        "docs/ur-normal-methods-20261012.md",
        "docs/ur-normal-methods-20261012-plan.md",
        "docs/ur-normal-methods-20261012-augmentation.md",
    ]:
        assets[name] = sha256(ROOT / name)
    save(
        E / "before.json",
        {
            "assets": assets,
            "contracts": contracts,
            "active_training": active,
            "test_bytes_hash_only": True,
            "test_features_decoded": 0,
        },
    )
    print({"old_assets": len(assets), "active_training": active}, flush=True)


def original_windows(index, sid, config):
    cache = cache_for(index, sid)
    entry = index["sequences"][sid]
    observations, _ = load_pose_cache(
        project_path(entry["pose_path"]),
        {
            "sequence_id": sid,
            "recipe_hash": index["recipe_hash"],
            "source_hash": entry["source_hash"],
        },
    )
    original = {(r.sample_index, r.track_id): r for r in cache.records}
    found, raw, provenance = {}, {}, {}
    resampler = CausalResampler(30, config["skeleton"]["resampling_hold_policy"])
    builder = WindowBuilder(config)
    for frame, poses in observations:
        for sample in resampler.push(frame, poses):
            for window in builder.push(sample):
                key = (window.sample_index, window.track_id)
                if key not in original:
                    raise ContractError("Pose window endpoint differs from original features")
                rec = original[key]
                if (window.quality, window.start_ms, window.end_ms) != (
                    rec.quality,
                    rec.start_ms,
                    rec.end_ms,
                ):
                    raise ContractError("Original quality/timing changed")
                found[key] = True
                if rec.row_index is not None:
                    if window.quality != "valid" or window.keypoints is None:
                        raise ContractError("Motion cannot bypass original quality")
                    raw[rec.row_index] = window.keypoints.copy()
                    provenance[rec.row_index] = {
                        "row_index": rec.row_index,
                        "sample_index": rec.sample_index,
                        "track_id": rec.track_id,
                        "start_ms": rec.start_ms,
                        "end_ms": rec.end_ms,
                        "available_at_ms": rec.available_at_ms,
                        "frame_size_wh": list(window.frame_size_wh),
                        "quality": window.quality,
                    }
    if set(found) != set(original) or set(raw) != set(range(len(cache.features))):
        raise ContractError("Incomplete window reconstruction")
    return (
        cache,
        np.stack([raw[i] for i in range(len(raw))]),
        [provenance[i] for i in range(len(raw))],
    )


def prepare():
    if C.exists() or O.exists() or (E / "lock.json").exists():
        raise ContractError("Preparation already exists")
    read_json(E / "before.json")
    index = copy.deepcopy(load_index(ROOT / "data/cache" / OLD / "dataset-index.json"))
    C.mkdir(parents=True)
    O.mkdir(parents=True)
    copied = {}
    for source in index["split_manifest"]["sequences"]:
        if source["split"] == "test":
            continue
        entry = index["sequences"][source["sequence_id"]]
        for key, folder in [("feature_path", "features"), ("pose_path", "poses")]:
            oldpath = project_path(entry[key])
            newpath = C / folder / oldpath.name
            newpath.parent.mkdir(exist_ok=True)
            for oldfile, newfile in [
                (oldpath, newpath),
                (oldpath.with_suffix(".json"), newpath.with_suffix(".json")),
            ]:
                shutil.copyfile(oldfile, newfile)
                if sha256(oldfile) != sha256(newfile):
                    raise ContractError("Private copy differs")
                copied[newfile.relative_to(ROOT).as_posix()] = sha256(newfile)
            entry[key] = newpath.relative_to(ROOT).as_posix()
    save(C / "dataset-index.json", index)
    index = validated_index()
    reference = Scorer(ROOT / "runs" / BASE)
    torch.set_num_threads(4)
    extractor = HumanFeatureExtractor(reference.config, index["weights"], torch.device("cuda"))
    pilot = []
    for sid in ["adl-05", "adl-06", "fall-19"]:
        cache, raw, provenance = original_windows(index, sid, reference.config)
        record = provenance[0]
        actual = extractor.extract_features(raw[0], record["frame_size_wh"])[0]
        np.testing.assert_allclose(actual, cache.features[0], atol=1e-5, rtol=1e-5)
        trajectory = motion_trajectory(raw[0])
        pilot.append(
            {
                "sequence_id": sid,
                **record,
                "feature_max_abs": float(np.max(np.abs(actual - cache.features[0]))),
                "feature_dim": len(actual),
                "motion_dim": len(trajectory),
                "finite": bool(np.isfinite(trajectory).all()),
            }
        )
    if not extractor.forward_verified:
        raise ContractError("Genuine feature head parity incomplete")
    save(
        E / "pilot.json",
        {
            "strict_load": extractor.load_report,
            "head_parity": extractor.forward_verified,
            "samples": pilot,
            "evidence_tier": "real_feature_smoke",
        },
    )
    motion_recipe = fingerprint(
        {
            "spec": SPEC["motion"],
            "pose_recipe": index["recipe_hash"],
            "window_size": 30,
            "quality": reference.config["quality"],
            "provenance": "causal_original_same_track_window",
        }
    )
    counts = {}
    (C / "motion").mkdir()
    for source in index["split_manifest"]["sequences"]:
        if source["split"] == "test":
            continue
        sid = source["sequence_id"]
        cache, raw, provenance = original_windows(index, sid, reference.config)
        motion = np.stack([motion_trajectory(w) for w in raw])
        destination = C / "motion" / f"{sid}.npz"
        write_npz(destination, original_windows=raw, motion=motion)
        save(
            destination.with_suffix(".json"),
            {
                "sequence_id": sid,
                "split": source["split"],
                "motion_recipe_hash": motion_recipe,
                "array_sha256": sha256(destination),
                "pose_sha256": sha256(project_path(index["sequences"][sid]["pose_path"])),
                "rows": provenance,
            },
        )
        copied[destination.relative_to(ROOT).as_posix()] = sha256(destination)
        copied[destination.with_suffix(".json").relative_to(ROOT).as_posix()] = sha256(
            destination.with_suffix(".json")
        )
        counts[sid] = len(cache.features)
    old_lock = read_json(ROOT / "docs/validation" / OLD / "svm-lock.json")
    pinned = {
        **copied,
        (C / "dataset-index.json").relative_to(ROOT).as_posix(): sha256(C / "dataset-index.json"),
        PLAN.relative_to(ROOT).as_posix(): sha256(PLAN),
    }
    for path in (ROOT / "runs" / BASE).iterdir():
        if path.is_file():
            pinned[path.relative_to(ROOT).as_posix()] = sha256(path)
    for weight in index["weights"].values():
        pinned[weight["local_path"]] = sha256(ROOT / weight["local_path"])
    lock = {
        "locked_at": datetime.now(UTC).isoformat(),
        "spec": SPEC,
        "motion_recipe_hash": motion_recipe,
        "feature_recipe_hash": index["recipe_hash"],
        "method_recipe_hash": fingerprint(
            {
                "spec": SPEC,
                "motion_recipe": motion_recipe,
                "feature_recipe": index["recipe_hash"],
                "wide_ae_sha256": sha256(ROOT / "runs" / BASE / "autoencoder.pt"),
            }
        ),
        "input_hashes": pinned,
        "source_hashes": {n: sha256(ROOT / n) for n in IMPLEMENTATIONS},
        "counts": counts,
        "reference_metrics": old_lock["reference_metrics"],
        "reference_threshold": reference.threshold["threshold"],
        "coarse_intervals": old_lock["coarse_intervals"],
        "current_good_pose_keys": old_lock["current_good_pose_keys"],
        "test_feature_reads": 0,
        "test_evaluations": 0,
    }
    save(E / "lock.json", lock)
    save(
        ROOT / "configs" / f"{TAG}.json",
        {
            **SPEC,
            "motion_recipe_hash": motion_recipe,
            "method_recipe_hash": lock["method_recipe_hash"],
            "feature_recipe_hash": index["recipe_hash"],
        },
    )
    print(
        {
            "prepared": len(counts),
            "train_rows": sum(
                counts[s["sequence_id"]]
                for s in index["split_manifest"]["sequences"]
                if s["split"] == "train"
            ),
            "pilot": pilot,
        },
        flush=True,
    )


def fit():
    if RUN.exists():
        raise ContractError("Completed or attempted candidate must be preserved")
    lock = verify_lock()
    index = validated_index()
    RUN.mkdir(parents=True)
    save(RUN / "status.json", {"fit": "running"})
    arrays, caches, ae = {}, {}, {}
    reference = Scorer(ROOT / "runs" / BASE)
    for source in index["split_manifest"]["sequences"]:
        if source["split"] == "test":
            continue
        sid = source["sequence_id"]
        caches[sid] = cache_for(index, sid)
        with np.load(C / "motion" / f"{sid}.npz", allow_pickle=False) as values:
            arrays[sid] = values["motion"]
        ae[sid] = reference.score(caches[sid].features)
    train_ids = [
        s["sequence_id"] for s in index["split_manifest"]["sequences"] if s["split"] == "train"
    ]
    train = np.concatenate([arrays[sid] for sid in train_ids])
    groups = np.concatenate([np.repeat(sid, len(arrays[sid])) for sid in train_ids])
    model = MotionNovelty.fit(train, groups, np.zeros(len(train), dtype=int), neighbors=5)
    joblib.dump(model, RUN / "model.joblib")
    model.scaler.save(RUN / "motion-scaler.npz")
    train_motion = model.score(train, exclude_groups=groups)
    train_ae = np.concatenate([ae[sid] for sid in train_ids])
    ae_p95, motion_p95 = float(np.quantile(train_ae, 0.95)), float(np.quantile(train_motion, 0.95))
    save(
        RUN / "normalization.json",
        {
            "ae_p95": ae_p95,
            "motion_p95": motion_p95,
            "fit_sequences": train_ids,
            "fit_split": "train",
            "normal_only": True,
            "motion_exclusion": "entire_source_video",
        },
    )
    write_npz(
        RUN / "train-scale-scores.npz",
        ae=train_ae,
        motion_leave_video_out=train_motion,
        groups=groups,
    )
    motion = {sid: model.score(x) for sid, x in arrays.items()}
    fused = {sid: fusion_scores(ae[sid], motion[sid], ae_p95, motion_p95) for sid in arrays}
    for name, values in [("ae-scores", ae), ("motion-scores", motion), ("scores", fused)]:
        write_npz(RUN / f"{name}.npz", **values)
    rows = []
    for s in index["split_manifest"]["sequences"]:
        if s["split"] == "validation":
            sid = s["sequence_id"]
            row, _, _ = score_sequence(caches[sid], fused[sid], 0.0)
            rows.append({**row, "label": s["label"]})
    chosen, grid = calibrate_normal_scores(rows)
    save(
        RUN / "calibration.json",
        {
            "chosen": chosen,
            "candidates": grid,
            "split": "validation",
            "selection": SPEC["selection"],
        },
    )
    threshold = chosen["threshold"]
    save(
        RUN / "threshold.json",
        {
            "threshold": threshold,
            "comparison": "gt",
            "k": 3,
            "split": "validation",
            "method_recipe_hash": lock["method_recipe_hash"],
            "model_sha256": sha256(RUN / "model.joblib"),
            "score_direction": "larger_is_more_anomalous",
        },
    )
    rows, windows, events = [], [], []
    for s in index["split_manifest"]["sequences"]:
        if s["split"] == "validation":
            sid = s["sequence_id"]
            row, ws, es = score_sequence(caches[sid], fused[sid], threshold)
            rows.append({**row, "label": s["label"]})
            windows.extend(ws)
            events.extend(es)
    metrics = sequence_metrics(rows)
    if metrics != chosen["metrics"]:
        raise ContractError("Calibration/evaluation mismatch")
    phases = phases_for(rows, windows, events, lock)
    evaluation_report(
        O,
        {
            **metrics,
            "source_kind": "real_feature_smoke",
            "protocol_id": index["split_manifest"]["protocol_id"],
        },
        rows,
        windows,
        events,
    )
    fn = [r["sequence_id"] for r in rows if r["label"] == 1 and r["prediction"] == 0]
    fp = [r["sequence_id"] for r in rows if r["label"] == 0 and r["prediction"] == 1]
    guards = {
        "phase_hits_at_least_four": sum(bool(p["phase_event_starts"]) for p in phases) >= 4,
        "no_coarse_early_events": not any(p["early_event_starts"] for p in phases),
        "recovered_fall19_visible": "fall-19" in fn
        or any(p["sequence_id"] == "fall-19" and p["visible_phase_alert_ms"] for p in phases),
    }
    recognition = (
        candidate_rank(metrics, 0)[:-1] > candidate_rank(lock["reference_metrics"], 0)[:-1]
        and metrics["n_unknown"] <= lock["reference_metrics"]["n_unknown"]
        and metrics["window_coverage"] >= lock["reference_metrics"]["window_coverage"]
        and not fp
        and not any(sid != "fall-19" for sid in fn)
    )
    save(
        RUN / "method-config.json",
        {
            **SPEC,
            "method_recipe_hash": lock["method_recipe_hash"],
            "motion_recipe_hash": lock["motion_recipe_hash"],
            "feature_recipe_hash": lock["feature_recipe_hash"],
        },
    )
    save(RUN / "dataset-index.json", index)
    save(
        RUN / "sources.json",
        {
            "lock_sha256": sha256(E / "lock.json"),
            "fit_sequences": train_ids,
            "train_windows": len(train),
            "feature_recipe_hash": lock["feature_recipe_hash"],
            "motion_recipe_hash": lock["motion_recipe_hash"],
            "test_unknown": True,
            "new_neural_epochs": 0,
            "model_kind": "normal_neighbor_bank_fit",
            "baseline_model": BASE,
        },
    )
    save(
        RUN / "result.json",
        {
            "metrics": metrics,
            "threshold": threshold,
            "false_negatives": fn,
            "false_positives": fp,
            "phases": phases,
            "display_checks": guards,
            "recognition_adoptable": recognition,
            "deployment_checks_pass": recognition and all(guards.values()),
            "demo_changed": False,
            "test": "unknown_not_evaluated",
            "real_end_to_end": "not_run",
        },
    )
    write_json(
        RUN / "status.json",
        {
            "fit": "complete",
            "calibration": "complete",
            "validation": "complete",
            "test": "unknown_not_evaluated",
        },
    )
    save(RUN / "frozen-artifacts.json", {p.name: sha256(p) for p in RUN.iterdir() if p.is_file()})
    print(
        {
            "metrics": metrics,
            "fn": fn,
            "fp": fp,
            "scales": [ae_p95, motion_p95],
            "recognition_adoptable": recognition,
            "display_checks": guards,
        },
        flush=True,
    )


def status():
    lock = verify_lock()
    print(
        {
            "candidate": TAG,
            "method_recipe_hash": lock["method_recipe_hash"],
            "status": read_json(RUN / "status.json") if RUN.exists() else "not_fit",
            "result": read_json(RUN / "result.json") if (RUN / "result.json").exists() else None,
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["inventory", "prepare", "fit", "status"])
    globals()[parser.parse_args().stage]()
