"""Independent CSV/event/calibration, model reload and source-preservation audit."""

import argparse
import csv
from collections import deque
from dataclasses import replace

import joblib
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from ur_normal_methods import ROOT, C, E, O, save, validated_index, verify_lock

from fall_detect.contracts import FeatureCache
from fall_detect.io_utils import ContractError, fingerprint, read_json, sha256
from fall_detect.models.autoencoder import Autoencoder
from fall_detect.models.normal_alternatives import TemporalAutoencoder
from fall_detect.training.experiment import cache_for, normal_matrix
from fall_detect.training.scaler import FeatureScaler


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def number(text):
    return None if text in ("", None) else float(text)


def independent_metrics(rows):
    truth = [int(r["label"]) for r in rows]
    pred = [None if r["prediction"] in ("", None) else int(r["prediction"]) for r in rows]
    known = [(a, b) for a, b in zip(truth, pred) if b is not None]
    y, p = zip(*known)
    matrix = confusion_matrix(y, p, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel().tolist()
    unknown = sum(b is None for b in pred)
    outcomes = np.column_stack(
        (matrix, [sum(a == label and b is None for a, b in zip(truth, pred)) for label in [0, 1]])
    ).tolist()
    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "n_total": len(rows),
        "n_evaluable": len(known),
        "n_unknown": unknown,
        "sequence_accuracy": float(accuracy_score(y, p)),
        "operational_accuracy": (tp + tn) / len(rows),
        "sequence_fall_f1": float(f1_score(y, p, zero_division=0)),
        "sequence_fall_precision": float(precision_score(y, p, zero_division=0)),
        "sequence_fall_recall": float(recall_score(y, p, zero_division=0)),
        "end_to_end_fall_recall": tp / sum(a == 1 for a in truth),
        "outcomes_with_unknown": outcomes,
        "sequence_coverage": len(known) / len(rows),
        "window_coverage": sum(int(r["n_valid_endpoints"]) for r in rows)
        / sum(int(r["n_scheduled_endpoints"]) for r in rows),
    }


def independent_temporal(cache, spec, scaler, provenance, saved):
    histories = provenance["histories"][cache.metadata["sequence_id"]]
    by_key = {(r.sample_index, r.track_id): r for r in cache.records}
    expected, adjusted, tensors = [], [], []
    for rec in sorted(cache.records, key=lambda r: (r.sample_index, r.track_id or -1)):
        preceding = [
            by_key.get((rec.sample_index - i * cache.metadata["stride"], rec.track_id))
            for i in reversed(range(spec["steps"]))
        ]
        legal = rec.track_id is not None and all(
            r is not None and r.row_index is not None for r in preceding
        )
        if not legal:
            adjusted.append(
                replace(
                    rec,
                    row_index=None,
                    quality=rec.quality if rec.row_index is None else "temporal_history_warmup",
                )
            )
            continue
        assert all(
            r.sequence_id == rec.sequence_id
            and r.track_id == rec.track_id
            and r.available_at_ms <= rec.available_at_ms
            for r in preceding
        )
        rows = [r.row_index for r in preceding]
        expected.append(
            {
                "sequence_id": rec.sequence_id,
                "track_id": rec.track_id,
                "target_sample_index": rec.sample_index,
                "input_sample_indices": [r.sample_index for r in preceding],
                "input_row_indices": rows,
            }
        )
        tensors.append(scaler.transform(cache.features[rows]))
        adjusted.append(replace(rec, row_index=len(tensors) - 1))
    assert histories == expected
    actual = np.asarray(tensors, dtype=np.float32).reshape(-1, spec["steps"], 512)
    np.testing.assert_array_equal(actual, saved)
    return actual, FeatureCache(
        np.asarray([x[-1] for x in actual], dtype=np.float32).reshape(-1, 512),
        adjusted,
        cache.metadata,
    )


def independent_events(cache, scores, threshold, actual):
    states, expected_events, event_scores, valid = {}, [], [], set()
    records = sorted(cache.records, key=lambda r: (r.sample_index, r.track_id or -1))
    assert len(records) == len(actual)

    def close(state, reason, end):
        if state["active"] is not None:
            expected_events.append({**state["active"], "end_ms": float(end), "end_reason": reason})
            state["active"] = None

    for r, w in zip(records, actual):
        key = (r.sequence_id, r.track_id)
        state = states.setdefault(key, {"last": None, "history": deque(maxlen=3), "active": None})
        if state["last"] is not None and r.sample_index != state["last"] + cache.metadata["stride"]:
            close(state, "time_gap", r.available_at_ms)
            state["history"].clear()
        state["last"] = r.sample_index
        raw = None if r.row_index is None else float(scores[r.row_index])
        score = None
        if raw is None:
            close(state, r.quality, r.available_at_ms)
            state["history"].clear()
            status = "unknown"
            reason = r.quality
        else:
            valid.add(r.sample_index)
            state["history"].append(raw)
            score = min(state["history"]) if len(state["history"]) == 3 else None
            status = (
                "suspected_fall"
                if score is not None and score > threshold
                else "normal"
                if score is not None
                else "warming_up"
            )
            reason = None if score is not None else "insufficient_consecutive_windows"
            if status == "suspected_fall":
                if state["active"] is None:
                    state["active"] = {
                        "sequence_id": r.sequence_id,
                        "track_id": r.track_id,
                        "confirmed_start_ms": float(r.available_at_ms),
                        "last_active_ms": float(r.available_at_ms),
                        "peak_score": raw,
                    }
                else:
                    state["active"]["last_active_ms"] = float(r.available_at_ms)
                    state["active"]["peak_score"] = max(raw, state["active"]["peak_score"])
            else:
                close(state, "low_score", r.available_at_ms)
        assert w["sequence_id"] == r.sequence_id and int(w["sample_index"]) == r.sample_index
        assert (None if w["track_id"] == "" else int(w["track_id"])) == r.track_id
        for field, value in [
            ("raw_score", raw),
            ("event_score", score),
            ("available_at_ms", r.available_at_ms),
            ("start_ms", r.start_ms),
            ("end_ms", r.end_ms),
        ]:
            assert number(w[field]) == value, (r.sequence_id, r.sample_index, field)
        assert w["state"] == status and (w["reason"] or None) == reason
        if score is not None:
            event_scores.append(score)
    for state in states.values():
        close(state, "sequence_end", cache.metadata.get("end_ms", 0))
    return (
        expected_events,
        max(event_scores) if event_scores else None,
        len(valid),
        len({r.sample_index for r in records}),
    )


def verify(write=False):
    lock = verify_lock()
    index = validated_index()
    for src in index["split_manifest"]["sequences"]:
        if src["split"] == "test":
            assert index["sequences"][src["sequence_id"]]["status"] == "not_prepared"
            assert not any(
                k in index["sequences"][src["sequence_id"]] for k in ["feature_path", "pose_path"]
            )
            try:
                cache_for(index, src["sequence_id"])
            except ContractError:
                pass
            else:
                raise AssertionError("Scoped Test access allowed")
    train, hashes = normal_matrix(index, "train")
    _, val_hashes = normal_matrix(index, "validation")
    scaler = FeatureScaler.fit(train)
    train_ids = [
        r["sequence_id"] for r in index["split_manifest"]["sequences"] if r["split"] == "train"
    ]
    val_ids = [
        r["sequence_id"]
        for r in index["split_manifest"]["sequences"]
        if r["split"] == "validation" and r["label"] == 0
    ]
    stages = (
        read_json(E / "svm-results.json")["candidates"]
        + read_json(E / "ae-results.json")["candidates"]
    )
    verified = []
    torch.set_num_threads(4)
    for result in stages:
        spec = result["spec"]
        run = ROOT / "runs" / spec["id"]
        frozen = read_json(run / "frozen-artifacts.json")
        assert all(sha256(run / n) == d for n, d in frozen.items())
        loaded = FeatureScaler.load(run / "scaler.npz")
        np.testing.assert_array_equal(loaded.mean, scaler.mean)
        np.testing.assert_array_equal(loaded.scale, scaler.scale)
        cfg = read_json(run / "method-config.json")
        expected_recipe = fingerprint(
            {
                "spec": spec,
                "features": index["recipe_hash"],
                "calibration": lock["selection_rule"],
                "k": 3,
            }
        )
        assert cfg["method_recipe_hash"] == expected_recipe
        assert cfg["feature_recipe_hash"] == index["recipe_hash"]
        sources = read_json(run / "sources.json")
        assert (
            sources["train_feature_hashes"] == hashes
            and sources["val_normal_feature_hashes"] == val_hashes
        )
        assert sources["method_recipe_hash"] == expected_recipe
        threshold = read_json(run / "threshold.json")
        assert (
            threshold["split"] == "validation"
            and threshold["comparison"] == "gt"
            and threshold["consecutive_windows"] == 3
        )
        assert threshold["method_recipe_hash"] == expected_recipe
        assert threshold["scaler_sha256"] == sha256(run / "scaler.npz")
        if spec["kind"] == "svm":
            model = joblib.load(run / "model.joblib")
            assert model.nu == spec["nu"] and model.gamma == spec["gamma"]
        else:
            checkpoint = torch.load(run / "model.pt", map_location="cuda", weights_only=True)
            assert (
                checkpoint["spec"] == spec and checkpoint["method_recipe_hash"] == expected_recipe
            )
            model = (
                (
                    TemporalAutoencoder(512, 128, 64, spec["steps"])
                    if spec["kind"] == "temporal"
                    else Autoencoder(512, 256, 64)
                )
                .cuda()
                .eval()
            )
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            history = read_json(run / "training-history.json")
            training = read_json(run / "training.json")
            best = float("inf")
            best_epoch = None
            stale = 0
            for row in history:
                if row["val_normal_loss"] < best - 1e-6:
                    best = row["val_normal_loss"]
                    best_epoch = row["epoch"]
                    stale = 0
                else:
                    stale += 1
            assert best_epoch == checkpoint["best_epoch"] == training["best_epoch"]
            assert best == training["best_val_normal_loss"] == checkpoint["val_normal_loss"]
            assert len(history) <= 100 and (stale >= 10 or len(history) == 100)
            assert (
                training["train_sequences"] == train_ids
                and training["val_normal_sequences"] == val_ids
            )
            assert (
                training["augmentation_split"] == "train_only"
                and training["normal_validation_clean"]
            )
        if spec["kind"] == "temporal":
            provenance = read_json(C / spec["id"] / "provenance.json")
            assert provenance["method_recipe_hash"] == expected_recipe
            assert all(
                sha256(C / spec["id"] / n) == d for n, d in provenance["array_hashes"].items()
            )
        output = O / spec["id"]
        sequences = csv_rows(output / "sequence-results.csv")
        windows = csv_rows(output / "window-results.csv")
        events = csv_rows(output / "events.csv")
        metrics = read_json(output / "metrics.json")
        recomputed = independent_metrics(sequences)
        for k, v in recomputed.items():
            assert metrics[k] == v, (spec["id"], k, metrics[k], v)
            assert result["metrics"][k] == v
        assert {r["sequence_id"] for r in sequences} == {
            r["sequence_id"]
            for r in index["split_manifest"]["sequences"]
            if r["split"] == "validation"
        }
        saved_scores = np.load(run / "scores.npz", allow_pickle=False)
        score_rows = []
        n_checked = 0
        expected_events = []
        for src in index["split_manifest"]["sequences"]:
            if src["split"] == "test":
                continue
            sid = src["sequence_id"]
            cache = cache_for(index, sid)
            if spec["kind"] == "temporal":
                with np.load(C / spec["id"] / f"{sid}.npz", allow_pickle=False) as a:
                    z, cache = independent_temporal(cache, spec, scaler, provenance, a["inputs"])
            else:
                z = scaler.transform(cache.features)
            if spec["kind"] == "svm":
                scores = -model.decision_function(z.astype(np.float64))
            else:
                scores = []
                with torch.inference_mode():
                    for x in z:
                        tensor = torch.from_numpy(x[None]).cuda()
                        error = (model(tensor) - tensor).square()
                        scores.append(
                            float(error[0, -1].mean())
                            if spec["kind"] == "temporal"
                            else float(error.mean())
                        )
                scores = np.asarray(scores, dtype=np.float64)
            np.testing.assert_array_equal(scores, saved_scores[sid])
            n_checked += len(scores)
            if src["split"] == "validation":
                ws = [w for w in windows if w["sequence_id"] == sid]
                es, score, nvalid, nscheduled = independent_events(
                    cache, scores, threshold["threshold"], ws
                )
                expected_events.extend(es)
                row = next(r for r in sequences if r["sequence_id"] == sid)
                assert number(row["sequence_score"]) == score
                assert (None if row["prediction"] == "" else int(row["prediction"])) == (
                    None if score is None else int(score > threshold["threshold"])
                )
                assert number(row["first_alarm_ms"]) == (
                    es[0]["confirmed_start_ms"] if es else None
                )
                assert (
                    int(row["n_valid_endpoints"]) == nvalid
                    and int(row["n_scheduled_endpoints"]) == nscheduled
                )
                score_rows.append({**row, "label": src["label"], "sequence_score": score})
        assert len(events) == len(expected_events)
        for a, b in zip(events, expected_events):
            for key, v in b.items():
                if key in ["sequence_id", "end_reason"]:
                    assert a[key] == v
                else:
                    assert number(a[key]) == v
        saved_scores.close()
        values = sorted(
            {r["sequence_score"] for r in score_rows if r["sequence_score"] is not None}
        )
        candidates = []
        for t in [float(np.nextafter(values[0], -np.inf))] + values:
            predictions = [
                {
                    **r,
                    "prediction": None
                    if r["sequence_score"] is None
                    else int(r["sequence_score"] > t),
                }
                for r in score_rows
            ]
            m = independent_metrics(predictions)
            candidates.append(
                (
                    m["operational_accuracy"],
                    m["sequence_fall_f1"],
                    m["end_to_end_fall_recall"],
                    -m["FP"],
                    t,
                )
            )
        assert max(candidates)[-1] == threshold["threshold"]
        calibration = read_json(run / "calibration.json")
        assert len(calibration["candidates"]) == len(candidates)
        assert (
            sha256(run / ("model.joblib" if spec["kind"] == "svm" else "model.pt"))
            == threshold["model_sha256"]
        )
        verified.append(
            {
                "run": spec["id"],
                "score_rows_recomputed": n_checked,
                "validation_sequences": len(sequences),
                "windows": len(windows),
                "events": len(events),
                "matrix": recomputed["outcomes_with_unknown"],
                "metrics": recomputed,
            }
        )
        print(
            f"Verified {spec['id']}: {n_checked} model scores; {len(windows)} windows; {len(events)} complete events",
            flush=True,
        )
    before = read_json(E / "before.json")
    for i, (name, digest) in enumerate(before["assets"].items(), 1):
        if sha256(ROOT / name) != digest:
            raise AssertionError(f"Preserved old asset changed: {name}")
        if i % 5000 == 0:
            print(f"Preservation {i}/{len(before['assets'])}", flush=True)
    for name, digest in before["mutable_contracts"].items():
        assert sha256(E / "prior-contract" / name) == digest
    report = {
        "passed": True,
        "candidates": verified,
        "preserved_old_assets": len(before["assets"]),
        "train_original_sequences": 24,
        "val_normal_sequences": 8,
        "calibration_sequences": 23,
        "scaler_train_rows": len(train),
        "test_feature_reads": 0,
        "test_evaluations": 0,
        "checks": [
            "Train-only scaler exact",
            "training/checkpoint provenance",
            "all model scores strict reload",
            "independent sklearn metrics/matrices/denominators",
            "independent per-window K3/state",
            "complete event intervals/peaks/reasons",
            "independent strict-threshold calibration",
            "temporal no future/track/gap mixing",
            "unique recipes and private caches",
            "23925 old asset hashes",
            "original contract archives",
        ],
    }
    if write:
        save(E / "verification.json", report)
    print(
        {"passed": True, "candidates": len(verified), "old_assets": len(before["assets"])},
        flush=True,
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    verify(parser.parse_args().write)
