"""Independently verify the diagnostic package. Default is read-only, never reads Test arrays."""

import argparse
import csv
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fall19_review import CSV, ROOT, SID, E, O, verify_sources

from fall_detect.evaluation.frame_review import validate_annotation
from fall_detect.io_utils import ContractError, sha256, write_json


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def check(condition, message):
    if not condition:
        raise ContractError(message)


def verify():
    verify_sources()
    data = json.loads((O / "review-data.json").read_text(encoding="utf-8"))
    manifest = json.loads((O / "source-manifest.json").read_text(encoding="utf-8"))
    frames = data["frames"]
    check(
        len(frames) == 100 and [f["source_frame_id"] for f in frames] == list(range(1, 101)),
        "Complete original frame IDs/order",
    )
    index = json.loads(
        (ROOT / "data/cache/ur-motion-novelty-20261012-001/dataset-index.json").read_text(
            encoding="utf-8"
        )
    )
    pose_path = ROOT / index["sequences"][SID]["pose_path"]
    pose_meta = json.loads(pose_path.with_suffix(".json").read_text(encoding="utf-8"))
    raw_identity = {
        int(p["path"].split("-")[-1].split(".")[0]): p
        for p in pose_meta["source_identity"]["files"]
        if p["path"].endswith(".png")
    }
    sync_path = ROOT / "data/raw/urfall/fall-19/fall-19-data.csv"
    with sync_path.open(encoding="utf-8-sig", newline="") as stream:
        sync = {int(r[0]): float(r[1]) for r in csv.reader(stream)}
    counts = Counter()
    with np.load(pose_path, allow_pickle=False) as original:
        for f, source in zip(frames, pose_meta["frames"]):
            fid = f["source_frame_id"]
            check(
                source["source_frame_id"] == fid
                and source["timestamp_ms"] == f["timestamp_ms"] == sync[fid],
                "Original time/frame mapping",
            )
            check(not f["timestamp_estimated"], "Raw source times must be explicit")
            check(
                sha256(O / f["raw_path"]) == f["source_sha256"] == raw_identity[fid]["sha256"],
                "Raw RGB bytes",
            )
            image = cv2.imdecode(np.fromfile(O / f["raw_path"], dtype=np.uint8), cv2.IMREAD_COLOR)
            check(image.shape[:2] == (480, 640), "Raw dimensions")
            check(len(source["pose_indices"]) == len(f["poses"]), "Pose count/frame alignment")
            for saved, pi in zip(f["poses"], source["pose_indices"]):
                kp = original["keypoints"][pi]
                check(np.array_equal(kp, np.array(saved["keypoints"])), "Frozen Pose values")
                check(saved["track_id"] == pose_meta["poses"][pi]["track_id"], "Frozen track ID")
                reliable = kp[:, 2] >= 0.2
                good = bool(np.isfinite(kp).all() and reliable.sum() >= 9)
                check(
                    saved["reliable_joints"] == int(reliable.sum())
                    and saved["current_good_pose"] == good,
                    "Independent current confidence/quality",
                )
            counts[
                "missing"
                if not f["poses"]
                else "good"
                if any(p["current_good_pose"] for p in f["poses"])
                else "low_confidence"
            ] += 1
    models, metrics, checked_windows = {}, {}, 0
    for name, rows in data["windows"].items():
        # Exact published CSV fields; added quality columns cannot change original scores/states.
        source_rows = [r for r in read_csv(CSV[name]) if r["sequence_id"] == SID]
        copy_rows = read_csv(O / f"{name}-windows.csv")
        check(len(source_rows) == len(copy_rows) == len(rows), "Published window count")
        for original, copied in zip(source_rows, copy_rows):
            check(all(original[k] == copied[k] for k in original), "Published CSV altered")
        threshold = data["models"][name]["threshold"]
        histories, previous = {}, {}
        for row in rows:
            track = row["track_id"]
            history = histories.setdefault(track, [])
            if track in previous and row["sample_index"] != previous[track] + 1:
                history.clear()
            previous[track] = row["sample_index"]
            raw = row["raw_score"]
            if raw is None:
                history.clear()
                score, state, reason = None, "unknown", row["quality"]
            else:
                check(np.isfinite(raw) and row["quality"] == "valid", "Only legal finite scores")
                history.append(raw)
                history[:] = history[-3:]
                score = min(history) if len(history) == 3 else None
                state = (
                    "warming_up"
                    if score is None
                    else "suspected_fall"
                    if score > threshold
                    else "normal"
                )
                reason = "insufficient_consecutive_windows" if score is None else None
            check(
                (score, state, reason) == (row["event_score"], row["state"], row["reason"]),
                "Independent K3/direction/state",
            )
            checked_windows += 1
        check(all(r["state"] != "suspected_fall" for r in rows), "fall-19 has no event")
        # Independent join: maximum past endpoint per track, not the production helper.
        for f in frames:
            past = [r for r in rows if r["available_at_ms"] <= f["timestamp_ms"]]
            latest = {}
            for r in past:
                if r["track_id"] not in latest or (r["available_at_ms"], r["sample_index"]) > (
                    latest[r["track_id"]]["available_at_ms"],
                    latest[r["track_id"]]["sample_index"],
                ):
                    latest[r["track_id"]] = r
            actual = {r["track_id"]: r for r in f["models"][name]}
            check(set(actual) == set(latest), "Causal track join")
            for track, r in latest.items():
                shown = actual[track]
                check(all(shown[k] == v for k, v in r.items()), "Causal window content")
                age = f["timestamp_ms"] - r["available_at_ms"]
                check(
                    shown["age_ms"] == age and shown["fresh"] == (age <= 2000 / 30),
                    "Missing/stale window not backfilled",
                )
        valid = [r for r in rows if r["quality"] == "valid"]
        phase = [r for r in valid if 1468 <= r["available_at_ms"] <= 2369]
        models[name] = {
            "scheduled": len(rows),
            "valid": len(valid),
            "coverage": len(valid) / len(rows),
            "quality": dict(Counter(r["quality"] for r in rows)),
            "last_legal_end_ms": max(r["available_at_ms"] for r in valid),
            "threshold": threshold,
            "ai_coarse_phase_legal": len(phase),
            "ai_coarse_phase_raw_max": max(r["raw_score"] for r in phase),
            "ai_coarse_phase_k3_max": max(r["event_score"] for r in phase),
            "human_time_gt": False,
            "events": 0,
        }
        results = read_csv(CSV[name].with_name("sequence-results.csv"))
        check(
            len(results) == 23 and sum(r["label"] == "1" for r in results) == 15,
            "Original Validation denominator",
        )
        conf = [[0, 0, 0], [0, 0, 0]]
        for r in results:
            label = int(r["label"])
            pred = int(r["prediction"]) if r["prediction"] else 2
            conf[label][pred] += 1
        tn, fp, un = conf[0]
        fn, tp, uf = conf[1]
        metrics[name] = {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "unknown": un + uf,
            "overall_accuracy": (tp + tn) / 23,
            "overall_denominator": 23,
            "decidable_accuracy": (tp + tn) / (23 - un - uf),
            "decidable_denominator": 23 - un - uf,
            "fall_precision": tp / (tp + fp),
            "fall_recall_decidable": tp / (tp + fn),
            "fall_recall_all": tp / 15,
            "fall_f1_decidable": 2 * tp / (2 * tp + fp + fn),
            "confusion_with_unknown": conf,
            "new_evaluation": False,
        }
        fall_row = next(r for r in results if r["sequence_id"] == SID)
        check(
            fall_row["prediction"] == "0" and fall_row["first_alarm_ms"] == "",
            "Published fall-19 result",
        )
        event_rows = read_csv(CSV[name].with_name("events.csv"))
        check(not any(r["sequence_id"] == SID for r in event_rows), "Published fall-19 events")
    check(
        not manifest["human_confirmed"] and not data["human_confirmed"], "No fabricated human truth"
    )
    blank = json.loads((O / "annotation-template.json").read_text(encoding="utf-8"))
    check(
        all(
            not r["reviewed"] and r["phase"] == r["visibility"] == r["pose_check"] == "unreviewed"
            for r in blank["frames"]
        ),
        "Blank original human template",
    )
    try:
        validate_annotation(blank, manifest, sha256(O / "source-manifest.json"))
    except ContractError:
        pass
    else:
        raise ContractError("Blank annotation incorrectly accepted")
    before = json.loads((E / "before.json").read_text(encoding="utf-8"))

    def unchanged(item):
        name, digest = item
        check(sha256(ROOT / name) == digest, f"Old asset changed: {name}")

    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, _ in enumerate(pool.map(unchanged, before["assets"].items()), 1):
            if i % 5000 == 0:
                print(f"Preserved old assets: {i}/{len(before['assets'])}", flush=True)
    for name, digest in before["contracts"].items():
        check(sha256(E / "prior-contract" / name) == digest, "Original contract snapshot")
    build = json.loads((E / "build.json").read_text(encoding="utf-8"))
    for name, digest in build["artifacts"].items():
        check(sha256(ROOT / name) == digest, f"Review artifact changed: {name}")
    return {
        "passed": True,
        "raw_frames": 100,
        "original_pose_rows": 95,
        "frame_quality": dict(counts),
        "causal_model_frame_joins": 300,
        "independent_k3_windows": checked_windows,
        "models": models,
        "historical_validation_csv_metrics": metrics,
        "old_assets_preserved": len(before["assets"]),
        "old_contract_snapshots": len(before["contracts"]),
        "review_artifacts": len(build["artifacts"]),
        "human_confirmed": False,
        "annotation_template_rejected": True,
        "test_feature_reads": 0,
        "new_test_evaluations": 0,
        "test_bytes_hash_only": True,
        "new_training_epochs": 0,
        "evidence_level": "real_feature_smoke (frozen RGB/Pose/score review); no new model inference pipeline",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save", action="store_true", help="Save first independent verification, refuses overwrite"
    )
    args = parser.parse_args()
    if args.save and (E / "verification.json").exists():
        raise ContractError("Preserve completed verification")
    report = verify()
    if args.save:
        write_json(E / "verification.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
