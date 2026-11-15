"""Independent CSV metrics/K3/event and Train-only fitting audit; no Test loads."""

import argparse
import csv
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import ur_short20_isotropic as workflow
from human_isotropic_refinement import train_validation_only
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from fall_detect.io_utils import ContractError, read_json, sha256
from fall_detect.training import experiment
from fall_detect.training.scaler import FeatureScaler


def check(condition, message):
    if not condition:
        raise ContractError(message)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def reconstruct(windows, threshold, end_ms):
    histories, previous, active, completed = defaultdict(lambda: deque(maxlen=3)), {}, {}, []
    best = None
    for row in sorted(
        windows, key=lambda r: (int(r["sample_index"]), int(r["track_id"]) if r["track_id"] else -1)
    ):
        key = int(row["track_id"]) if row["track_id"] else None
        sample, now = int(row["sample_index"]), float(row["available_at_ms"])
        gap = key in previous and sample != previous[key] + 1
        missing = not row["raw_score"]
        if gap or missing:
            if key in active:
                completed.append(
                    {
                        **active.pop(key),
                        "end_ms": now,
                        "end_reason": "time_gap" if gap else row["reason"],
                    }
                )
            histories[key].clear()
        previous[key] = sample
        score = float(row["raw_score"]) if not missing else None
        event_score = None
        if score is not None:
            histories[key].append(score)
            if len(histories[key]) == 3:
                event_score = min(histories[key])
                best = event_score if best is None else max(best, event_score)
        check(
            (None if not row["event_score"] else float(row["event_score"])) == event_score,
            "Every rolling K3 mismatch",
        )
        alarm = event_score is not None and event_score > threshold
        expected_state = (
            "unknown"
            if missing
            else "suspected_fall"
            if alarm
            else "normal"
            if event_score is not None
            else "warming_up"
        )
        check(row["state"] == expected_state, "Independent window state mismatch")
        if alarm:
            if key not in active:
                active[key] = {
                    "sequence_id": row["sequence_id"],
                    "track_id": key,
                    "confirmed_start_ms": now,
                    "last_active_ms": now,
                    "peak_score": score,
                }
            else:
                active[key]["last_active_ms"] = now
                active[key]["peak_score"] = max(active[key]["peak_score"], score)
        elif key in active:
            completed.append({**active.pop(key), "end_ms": now, "end_reason": "low_score"})
    completed.extend({**e, "end_ms": end_ms, "end_reason": "sequence_end"} for e in active.values())
    return best, completed


def verify(write=False):
    lock = workflow.verify_lock()
    results = read_json(workflow.evidence("validation"))
    check(results["lock_sha256"] == sha256(workflow.evidence("lock")), "Lock changed")
    result = results["candidate"]
    run = workflow.ROOT / "runs" / workflow.RUN
    accesses = []
    with train_validation_only(accesses):
        index = experiment.load_index(workflow.INDEX)
        check(index["preparation_scope"] == ["train", "validation"], "Execution scope")
        check(index["recipe_hash"] == lock["recipe_hash"], "Recipe")
        check(index["split_manifest"]["split_hash"] == lock["split_hash"], "Original split")
        train_x, train_hashes = experiment.normal_matrix(index, "train")
        _, val_hashes = experiment.normal_matrix(index, "validation")
        expected = FeatureScaler.fit(train_x)
        scaler = FeatureScaler.load(run / "scaler.npz")
        np.testing.assert_array_equal(expected.mean, scaler.mean)
        np.testing.assert_array_equal(expected.scale, scaler.scale)
        sources = read_json(run / "sources.json")
        check(
            sources["train_feature_hashes"] == train_hashes and len(train_hashes) == 24,
            "Original24 TrainADL only",
        )
        check(
            sources["val_normal_feature_hashes"] == val_hashes and len(val_hashes) == 8,
            "Original8 normalVal checkpoint",
        )
        check(
            set(read_json(run / "threshold.json")["feature_hashes"])
            == {
                s["sequence_id"]
                for s in index["split_manifest"]["sequences"]
                if s["split"] == "validation"
            },
            "Original23 Validation calibration only",
        )
        check(experiment.verify_artifacts(run) == result["artifact_hashes"], "Frozen artifacts")
        folder = workflow.ROOT / "outputs" / workflow.PREFIX / workflow.RUN
        rows = read_csv(folder / "sequence-results.csv")
        windows = read_csv(folder / "window-results.csv")
        events = read_csv(folder / "events.csv")
        evaluable = [r for r in rows if r["prediction"]]
        truth, predicted = (
            [int(r["label"]) for r in evaluable],
            [int(r["prediction"]) for r in evaluable],
        )
        tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel().tolist()
        metrics = result["metrics"]
        check([tp, tn, fp, fn] == [metrics[k] for k in ("TP", "TN", "FP", "FN")], "Confusion")
        for key, value in {
            "sequence_accuracy": accuracy_score(truth, predicted),
            "sequence_fall_f1": f1_score(truth, predicted),
            "sequence_fall_precision": precision_score(truth, predicted),
            "sequence_fall_recall": recall_score(truth, predicted),
            "n_total": len(rows),
            "n_evaluable": len(evaluable),
            "n_unknown": len(rows) - len(evaluable),
            "operational_accuracy": (tp + tn) / len(rows),
            "end_to_end_fall_recall": tp / sum(int(r["label"]) == 1 for r in rows),
            "window_coverage": sum(int(r["n_valid_endpoints"]) for r in rows)
            / sum(int(r["n_scheduled_endpoints"]) for r in rows),
        }.items():
            check(metrics[key] == value, f"Independent {key}")
        grouped = defaultdict(list)
        for w in windows:
            grouped[w["sequence_id"]].append(w)
        reconstructed_events = []
        for row in rows:
            sid = row["sequence_id"]
            cache = experiment.cache_for(index, sid)
            best, rebuilt = reconstruct(grouped[sid], result["threshold"], cache.metadata["end_ms"])
            check(
                best == (float(row["sequence_score"]) if row["sequence_score"] else None),
                "K3 maximum",
            )
            check(
                (int(best > result["threshold"]) if best is not None else None)
                == (int(row["prediction"]) if row["prediction"] else None),
                "K3 decision",
            )
            check(
                (rebuilt[0]["confirmed_start_ms"] if rebuilt else None)
                == (float(row["first_alarm_ms"]) if row["first_alarm_ms"] else None),
                "First alarm",
            )
            reconstructed_events.extend(rebuilt)
        converted = [
            {
                k: int(v)
                if k == "track_id"
                else float(v)
                if k in ("confirmed_start_ms", "last_active_ms", "peak_score", "end_ms")
                else v
                for k, v in e.items()
            }
            for e in events
        ]
        order = lambda e: (e["sequence_id"], e["track_id"], e["confirmed_start_ms"])
        check(
            sorted(converted, key=order) == sorted(reconstructed_events, key=order),
            "Full event intervals/end reasons/peaks",
        )
        table = [
            [
                sum(int(r["label"]) == y and r["prediction"] == p for r in rows)
                for p in ("0", "1", "")
            ]
            for y in (0, 1)
        ]
        check(table == metrics["outcomes_with_unknown"], "2x3 unknown matrix")
        recomputed = workflow.validation_result(workflow.RUN, lock, export=False)
        for key in ("metrics", "phases", "recovered_fn", "visible_recovered_fn"):
            check(recomputed[key] == result[key], f"Recomputed {key}")
        history = read_csv(run / "training-history.csv")
        best_epoch = min(history, key=lambda r: float(r["val_normal_loss"]))
        check(
            float(best_epoch["val_normal_loss"]) == result["status"]["best_val_normal_loss"],
            "Best normalVal checkpoint",
        )
        check(len(history) <= 100, "Epoch limit")
        check(
            all(a["split"] != "test" for a in accesses + results["feature_accesses"]),
            "No Test feature reads",
        )
    test_rows = [
        index["sequences"][s["sequence_id"]]
        for s in index["split_manifest"]["sequences"]
        if s["split"] == "test"
    ]
    check(
        len(test_rows) == 23
        and all(
            r["status"] == "not_prepared" and "feature_path" not in r and "pose_path" not in r
            for r in test_rows
        ),
        "Truthful unprepared Test",
    )
    check(not list((workflow.CACHE / "features").glob("*test*")), "No Test files")
    check(
        len(list((workflow.CACHE / "features").glob("*.npz"))) == 47,
        "Exactly47 Train/Validation caches",
    )
    rejections = {}
    for stage in ("lock", "pilot", "prepare", "training"):
        try:
            getattr(workflow, stage if stage != "prepare" else "full_prepare")()
        except ContractError as exc:
            rejections[stage] = str(exc)
        else:
            raise ContractError("Completed mutation did not reject")
    output = {
        "passed": True,
        "evidence_tier": "real_feature_smoke",
        "metrics": metrics,
        "train_sequences": len(train_hashes),
        "train_windows": len(train_x),
        "normal_validation_sequences": len(val_hashes),
        "best_epoch": int(best_epoch["epoch"]),
        "events_independently_rebuilt": len(converted),
        "all_window_k3_and_states_match": True,
        "train_only_scaler_exact": True,
        "preserved_files": len(lock["preserved_files"]),
        "test_sequences_unprepared": 23,
        "test_feature_reads": 0,
        "test_evaluations": 0,
        "completed_mutations_rejected": rejections,
        "feature_accesses": accesses,
        "verifier_sha256": sha256(Path(__file__)),
    }
    if write:
        workflow.save("verification", output)
    workflow.verify_lock()
    print({k: v for k, v in output.items() if k != "feature_accesses"})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write", action="store_true")
    verify(p.parse_args().write)
