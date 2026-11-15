"""Train/Validation-only failure-layer diagnostics; no causal ground truth claims."""

import argparse

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from ur_normal_methods import BASE, ROOT, E, O, phases_for, save, validated_index, verify_lock

from fall_detect.detection.events import score_sequence
from fall_detect.evaluation.metrics import sequence_metrics
from fall_detect.io_utils import ContractError, read_json, write_csv
from fall_detect.models.stgcn import HumanFeatureExtractor
from fall_detect.training.experiment import Scorer, cache_for, normal_matrix
from fall_detect.training.scaler import FeatureScaler
from fall_detect.visualization.reports import evaluation_report


def diagnose(stage):
    lock = verify_lock()
    target = E / f"diagnosis-{stage}.json"
    if target.exists():
        raise ContractError("Diagnosis exists; preserve evidence")
    index = validated_index()
    reference = Scorer(ROOT / "runs" / BASE)
    rows, windows, events = [], [], []
    raw_scores = {}
    for src in index["split_manifest"]["sequences"]:
        if src["split"] == "validation":
            sid = src["sequence_id"]
            cache = cache_for(index, sid)
            raw_scores[sid] = reference.score(cache.features)
            row, ws, es = score_sequence(cache, raw_scores[sid], reference.threshold["threshold"])
            rows.append({**row, "label": src["label"]})
            windows.extend(ws)
            events.extend(es)
    metrics = sequence_metrics(rows)
    if metrics != lock["reference_metrics"]:
        raise ContractError("Baseline Validation differs from frozen evidence")
    if stage == "stage1":
        destination = O / BASE
        destination.mkdir()
        evaluation_report(
            destination,
            {
                **metrics,
                "source_kind": "real_feature_smoke",
                "protocol_id": index["split_manifest"]["protocol_id"],
            },
            rows,
            windows,
            events,
        )
    train, _, counts = normal_matrix(index, "train", True)
    scaler = FeatureScaler.fit(train)
    nearest = NearestNeighbors(n_neighbors=1, algorithm="brute", metric="euclidean", n_jobs=4).fit(
        scaler.transform(train)
    )
    train_sids = np.concatenate([np.repeat(sid, n) for sid, n in counts.items()])
    distance, neighbors, normal_dist = {}, {}, []
    for src in index["split_manifest"]["sequences"]:
        if src["split"] == "validation":
            sid = src["sequence_id"]
            cache = cache_for(index, sid)
            d, n = nearest.kneighbors(scaler.transform(cache.features))
            distance[sid] = d[:, 0] / np.sqrt(512)
            neighbors[sid] = train_sids[n[:, 0]]
            if src["label"] == 0:
                normal_dist.extend(distance[sid].tolist())
    # Original per-frame classifier input, never applying its head to isotropic
    # features. This is a separate representation, not an anomaly prediction.
    original = Scorer(ROOT / "runs/ur-sample-human-s1-001")
    torch.set_num_threads(4)
    extractor = HumanFeatureExtractor(
        original.config, original.index["weights"], torch.device("cuda")
    )
    action_rows, action_by_key = [], {}
    for src in index["split_manifest"]["sequences"]:
        if src["split"] == "validation":
            sid = src["sequence_id"]
            cache = cache_for(original.index, sid)
            scores = extractor.action_scores(cache.features)
            for rec in cache.records:
                if rec.row_index is not None:
                    row = {
                        "sequence_id": sid,
                        "sample_index": rec.sample_index,
                        "track_id": rec.track_id,
                        "available_at_ms": rec.available_at_ms,
                        "class_index": int(scores[rec.row_index].argmax()),
                        "fall_down_score": float(scores[rec.row_index, 6]),
                        "is_falling_argmax": int(scores[rec.row_index].argmax()) == 6,
                    }
                    action_rows.append(row)
                    action_by_key[(sid, rec.sample_index, rec.track_id)] = row
    if stage == "stage1":
        write_csv(O / "independent-seven-class.csv", action_rows)
    results = read_json(E / "svm-results.json")["candidates"]
    if stage == "stage2":
        results += read_json(E / "ae-results.json")["candidates"]
    models = [
        {"spec": {"id": BASE}, "threshold": lock["reference_threshold"], "metrics": metrics}
    ] + results
    diagnostics = []
    for sid in ["fall-19", "fall-18", "fall-30", "fall-09"]:
        lo, hi = lock["coarse_intervals"][sid]["fall_phase_ms"]
        cache = cache_for(index, sid)
        phase_records = [r for r in cache.records if lo <= r.available_at_ms <= hi]
        legal = [r for r in phase_records if r.row_index is not None]
        independent = [
            r for r in action_rows if r["sequence_id"] == sid and lo <= r["available_at_ms"] <= hi
        ]
        matched = [
            w for w in windows if w["sequence_id"] == sid and lo <= w["available_at_ms"] <= hi
        ]
        comparisons = []
        for model in models:
            name = model["spec"]["id"]
            import csv

            with (O / name / "window-results.csv").open(encoding="utf-8-sig", newline="") as file:
                ws = [w for w in csv.DictReader(file) if w["sequence_id"] == sid]
            with (O / name / "sequence-results.csv").open(encoding="utf-8-sig", newline="") as file:
                sr = list(csv.DictReader(file))
            normal_scores = [
                float(r["sequence_score"]) for r in sr if r["label"] == "0" and r["sequence_score"]
            ]
            target_score = next(
                float(r["sequence_score"]) if r["sequence_score"] else None
                for r in sr
                if r["sequence_id"] == sid
            )
            phase = [w for w in ws if lo <= float(w["available_at_ms"]) <= hi]
            raw = [float(w["raw_score"]) for w in phase if w["raw_score"]]
            k3 = [float(w["event_score"]) for w in phase if w["event_score"]]
            threshold = model["threshold"]
            comparisons.append(
                {
                    "run": name,
                    "threshold": threshold,
                    "phase_raw_max": max(raw) if raw else None,
                    "phase_k3_max": max(k3) if k3 else None,
                    "phase_raw_crossing_count": sum(s > threshold for s in raw),
                    "phase_legal_k3": len(k3),
                    "sequence_score": target_score,
                    "minimum_adl_fp_if_threshold_recovers_this_sequence": sum(
                        s >= target_score for s in normal_scores
                    )
                    if target_score is not None
                    else None,
                    "falling_head_but_no_ae_alert_endpoints": sum(
                        action_by_key.get((sid, w["sample_index"], w["track_id"]), {}).get(
                            "is_falling_argmax", False
                        )
                        and w["state"] != "suspected_fall"
                        for w in matched
                    )
                    if name == BASE
                    else None,
                }
            )
        distances = [float(distance[sid][r.row_index]) for r in legal]
        diagnostics.append(
            {
                "sequence_id": sid,
                "coarse_phase_ms": [lo, hi],
                "quality_counts_full": cache.metadata["quality_counts"],
                "phase_scheduled_track_records": len(phase_records),
                "phase_valid_records": len(legal),
                "phase_good_pose_endpoints": sum(
                    (r.sample_index, r.track_id)
                    in {tuple(p) for p in lock["current_good_pose_keys"][sid]}
                    for r in phase_records
                ),
                "nearest_train_normal_distance_phase_median": float(np.median(distances))
                if distances
                else None,
                "nearest_train_normal_videos": sorted(
                    {str(neighbors[sid][r.row_index]) for r in legal}
                ),
                "original_per_frame_head_phase_falling_argmax_count": sum(
                    r["is_falling_argmax"] for r in independent
                ),
                "head_inputs_valid_count": len(independent),
                "models": comparisons,
            }
        )
    summary = {
        "baseline_accuracy": metrics["operational_accuracy"],
        "baseline_fn": [r["sequence_id"] for r in rows if r["label"] == 1 and r["prediction"] == 0],
        "svm_recovers_fall19": [
            r["spec"]["id"]
            for r in results
            if "fall-19" not in r["false_negatives"] and r["metrics"]["n_unknown"] == 0
        ],
        "normal_validation_nearest_distance_p50": float(np.median(normal_dist)),
        "normal_validation_nearest_distance_p95": float(np.quantile(normal_dist, 0.95)),
        "cause_boundary": "Legal windows and low score identify scoring/threshold behavior. Confidence, AI coarse phases and nearest distance do not establish accurate joints, causal representation loss or human onset truth.",
    }
    save(
        target,
        {
            "stage": stage,
            "summary": summary,
            "diagnostics": diagnostics,
            "reference_phases": phases_for(rows, windows, events, lock),
            "strict_classifier_load": extractor.load_report,
            "classifier_input": "original30 per_frame cached features, class6 Fall Down argmax; independent of isotropic anomaly input",
            "classifier_used_for_training": False,
            "test_feature_reads": 0,
            "test_evaluations": 0,
        },
    )
    print(summary, flush=True)
    for d in diagnostics:
        print(
            d["sequence_id"],
            "valid",
            d["phase_valid_records"],
            "FALLING",
            d["original_per_frame_head_phase_falling_argmax_count"],
            "nearest",
            d["nearest_train_normal_distance_phase_median"],
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["stage1", "stage2"])
    diagnose(parser.parse_args().stage)
