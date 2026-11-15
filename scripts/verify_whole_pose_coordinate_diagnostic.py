"""Independent array/NumPy AE/manual causal-window and event verification, read-only."""

import argparse
import csv
import json
from dataclasses import asdict

import numpy as np
import torch

from fall_detect.config import ROOT, project_path
from fall_detect.data.cache import load_feature_cache, load_pose_cache
from fall_detect.evaluation.whole_pose_coordinates import ID
from fall_detect.io_utils import read_json, sha256

OUT = ROOT / "outputs" / ID
QA = ROOT / "docs/validation" / ID
CACHE = ROOT / "data/cache" / ID
BASE = ROOT / "runs/ur-isotropic-wide-20261012-001"


def verify():
    summary = read_json(OUT / "inference/summary.json")
    lock = read_json(QA / "lock.json")
    for name, expected in lock["files"].items():
        assert sha256(ROOT / name) == expected, name
    for name, expected in lock["reference_artifacts"].items():
        assert sha256(BASE / name) == expected, name
    old = read_json(QA / "old-assets-before.json")
    for name, expected in old.items():
        archived = QA / "prior-contracts" / name
        path = archived if archived.exists() else ROOT / name
        assert sha256(path) == expected, name
    annotation = read_json(OUT / "human-review-original.json")
    assert sha256(OUT / "human-review-original.json") == lock["source_sha256"]
    index = read_json(BASE / "dataset-index.json")
    # Select only fall-19. Test entries are never followed.
    src = index["sequences"]["fall-19"]
    original, pose_meta = load_pose_cache(project_path(src["pose_path"]), {"split": "validation"})
    derived, meta = load_pose_cache(CACHE / "derived-pose.npz", {"split": "validation"})
    assert meta["recipe_hash"] == lock["recipe_hash"] != pose_meta["recipe_hash"]
    points = {
        (f["source_frame_id"], p["joint_index"]): p
        for f in annotation["frames"]
        for p in f["points"]
    }
    applied = {key: p for key, p in points.items() if p["status"] == "mislocated_visible"}
    assert len(applied) == 6
    changed_values = 0
    for (f, ps), (g, qs) in zip(original, derived, strict=True):
        assert f == g
        for p, q in zip(ps, qs, strict=True):
            fields, other = asdict(p), asdict(q)
            fields.pop("keypoints")
            other.pop("keypoints")
            np.testing.assert_array_equal(fields.pop("bbox_xyxy"), other.pop("bbox_xyxy"))
            assert fields == other
            expected = p.keypoints.copy()
            for j in range(13):
                point = applied.get((f.source_frame_id, j))
                if point:
                    assert p.track_id == 1
                    expected[j, :2] = point["xy"]
                    changed_values += int(np.any(expected[j, :2] != p.keypoints[j, :2]))
            np.testing.assert_array_equal(expected, q.keypoints)
    assert changed_values == 6
    with np.load(BASE / "scaler.npz", allow_pickle=False) as z:
        mean, scale = z["mean"], z["scale"]
    ck = torch.load(BASE / "autoencoder.pt", map_location="cpu", weights_only=True)
    assert ck["architecture"] == "mlp-ae-v1" and (ck["hidden_dim"], ck["latent_dim"]) == (256, 64)
    weights = {k: v.numpy() for k, v in ck["state_dict"].items()}

    def manual_scores(features):
        scaled = ((features - mean) / scale).astype(np.float32)
        values = []
        for row in scaled:
            x = row[None].copy()
            for name in ("encoder.0", "encoder.2", "decoder.0", "decoder.2"):
                x = x @ weights[name + ".weight"].T + weights[name + ".bias"]
                if name != "decoder.2":
                    x = np.maximum(x, 0)
            values.append(float(np.mean((x - row[None]) ** 2)))
        return np.asarray(values)

    cached = load_feature_cache(project_path(src["feature_path"]), {"split": "validation"})
    sham = load_feature_cache(
        CACHE / "sham-features.npz", {"recipe_hash": pose_meta["recipe_hash"]}
    )
    intervention = load_feature_cache(
        CACHE / "intervention-features.npz", {"recipe_hash": lock["recipe_hash"]}
    )
    assert cached.records == sham.records == intervention.records
    np.testing.assert_array_equal(cached.features, sham.features)
    # Independent hold-last lookup, only past observations; no production window builder.
    sources = read_json(OUT / "inference/causal-window-sources.json")
    by_window = {(r["sample_index"], r["track_id"]): r["source_frame_ids"] for r in sources}
    past = []
    period = 1000 / 30
    for si in range(cached.metadata["n_samples"]):
        tick = original[0][0].timestamp_ms + si * period
        frame, poses = max(
            ((f, ps) for f, ps in original if f.timestamp_ms <= tick + 1e-7),
            key=lambda x: x[0].timestamp_ms,
        )
        present = tick - frame.timestamp_ms <= 1.5 * period + 1e-7 and not frame.source_missing
        past.append(
            (
                frame.source_frame_id if present else None,
                {p.track_id for p in poses} if present else set(),
            )
        )
    affected = 0
    for r in cached.records:
        ids = [
            fid
            for fid, tracks in past[r.sample_index - 29 : r.sample_index + 1]
            if r.track_id in tracks
        ]
        assert by_window[(r.sample_index, r.track_id)] == ids
        if r.row_index is not None:
            is_affected = bool({47, 49}.intersection(ids))
            affected += int(is_affected)
            if not is_affected:
                np.testing.assert_array_equal(
                    sham.features[r.row_index], intervention.features[r.row_index]
                )
    assert affected == summary["affected_legal_windows"]
    threshold = read_json(BASE / "threshold.json")["threshold"]
    assert threshold == summary["threshold"] and summary["comparison"] == "gt" and summary["k"] == 3
    score_count, window_count, event_count = 0, 0, 0
    stages = {}
    for name, cache in (("cached", cached), ("sham", sham), ("intervention", intervention)):
        scores = np.load(OUT / f"inference/{name}-scores.npy", allow_pickle=False)
        np.testing.assert_allclose(scores, manual_scores(cache.features), atol=1e-6, rtol=0)
        with (OUT / f"inference/{name}-windows.csv").open(encoding="utf-8-sig", newline="") as h:
            rows = list(csv.DictReader(h))
        records = sorted(cache.records, key=lambda r: (r.sample_index, r.track_id or -1))
        assert len(rows) == len(records)
        histories, last, active, events, k3 = {}, {}, {}, [], []

        def close(key, t, reason, active=active, events=events):
            if key in active:
                events.append({**active.pop(key), "end_ms": t, "end_reason": reason})

        for r, w in zip(records, rows, strict=True):
            key, t = r.track_id, r.available_at_ms
            assert (
                int(w["sample_index"]),
                float(w["available_at_ms"]),
                float(w["start_ms"]),
                float(w["end_ms"]),
            ) == (r.sample_index, t, r.start_ms, r.end_ms)
            hist = histories.setdefault(key, [])
            if key in last and r.sample_index != last[key] + cache.metadata["stride"]:
                close(key, t, "time_gap")
                hist.clear()
            last[key] = r.sample_index
            if r.row_index is None:
                assert (
                    w["raw_score"] == w["event_score"] == ""
                    and w["state"] == "unknown"
                    and w["reason"] == r.quality
                )
                close(key, t, r.quality)
                hist.clear()
                continue
            value = float(scores[r.row_index])
            hist.append(value)
            hist[:] = hist[-3:]
            assert float(w["raw_score"]) == value
            event = min(hist) if len(hist) == 3 else None
            if event is None:
                assert w["event_score"] == "" and w["state"] == "warming_up"
            else:
                assert float(w["event_score"]) == event
                k3.append(event)
                assert w["state"] == ("suspected_fall" if event > threshold else "normal")
            if event is not None and event > threshold:
                if key not in active:
                    active[key] = {
                        "sequence_id": r.sequence_id,
                        "track_id": key,
                        "confirmed_start_ms": t,
                        "last_active_ms": t,
                        "peak_score": value,
                    }
                else:
                    active[key]["last_active_ms"] = t
                    active[key]["peak_score"] = max(active[key]["peak_score"], value)
            else:
                close(key, t, "low_score")
        for key in list(active):
            close(key, cache.metadata["end_ms"], "sequence_end")
        assert events == read_json(OUT / f"inference/{name}-events.json")
        seq = max(k3) if k3 else None
        predicted = int(seq > threshold) if seq is not None else None
        result = summary["results"][name]
        assert seq == result["sequence_score"] and predicted == result["prediction"]
        assert result["first_alarm_ms"] == (events[0]["confirmed_start_ms"] if events else None)
        assert result["max_raw_score"] == float(scores.max())
        total = len({r.sample_index for r in records})
        legal = len({r.sample_index for r in records if r.row_index is not None})
        assert total == 71 and legal == 23 and result["window_coverage"] == legal / total
        assert result["raw_above_threshold"] == int((scores > threshold).sum())
        assert result["k3_above_threshold"] == sum(v > threshold for v in k3)
        assert result["n_events"] == len(events)
        stages[name] = {
            "prediction": predicted,
            "raw_max": float(scores.max()),
            "k3_max": seq,
            "legal": legal,
            "scheduled": total,
            "unknown_windows": len(records) - len(scores),
            "events": len(events),
            "first_alarm_ms": result["first_alarm_ms"],
            "single_video_confusion_matrix_true_normal_fall_pred_normal_fall": [
                [0, 0],
                [int(predicted == 0), int(predicted == 1)],
            ],
            "single_video_all_accuracy": int(predicted == 1) / 1,
            "single_video_decidable_denominator": int(predicted is not None),
            "single_video_all_denominator": 1,
            "single_video_unknown": int(predicted is None),
            "single_fall_recall": int(predicted == 1) / 1,
            "single_fall_f1": 1.0 if predicted == 1 else 0.0,
            "precision": 1.0 if predicted == 1 else None,
        }
        score_count += len(scores)
        window_count += len(rows)
        event_count += len(events)
    controls_count = 0
    for control in summary["controls"]:
        sid = control["sequence_id"]
        assert (sid, control["split"]) in {("adl-05", "train"), ("adl-04", "validation")}
        c = load_feature_cache(
            project_path(index["sequences"][sid]["feature_path"]), {"split": control["split"]}
        )
        selected = [r for r in c.records if r.row_index is not None][:3]
        values = np.array([c.features[r.row_index] for r in selected])
        np.testing.assert_allclose(control["scores"], manual_scores(values), atol=1e-6, rtol=0)
        assert control["feature_max_abs_delta"] == 0 and control["fit_called"] is False
        controls_count += len(values)
    phase = read_json(
        ROOT / "outputs/ur-fall19-human-amendment-20261013-001/derived-annotation.json"
    )
    phase_by_frame = {f["source_frame_id"]: f["phase"] for f in phase["frames"]}
    counts = {}
    for r in cached.records:
        frame, _ = max(
            ((f, ps) for f, ps in original if f.timestamp_ms <= r.available_at_ms + 1e-7),
            key=lambda x: x[0].timestamp_ms,
        )
        p = phase_by_frame[frame.source_frame_id]
        c = counts.setdefault(p, {"legal": 0, "scheduled": 0})
        c["scheduled"] += 1
        c["legal"] += int(r.row_index is not None)
    assert sum(p["scheduled"] for p in counts.values()) == 71
    assert sum(p["legal"] for p in counts.values()) == 23
    return {
        "id": ID,
        "pass": True,
        "method": "independent_numpy_linear_relu_mse_manual_hold_K3_events",
        "scores_verified": score_count,
        "windows_verified": window_count,
        "events_verified": event_count,
        "normal_control_scores_verified": controls_count,
        "changed_joint_xy_pairs": changed_values,
        "old_files_preserved": len(old),
        "affected_legal_windows": affected,
        "results": stages,
        "phase_counts_same_original_and_intervention": counts,
        "phase_basis": "whole human posture phase; latest causal source at window endpoint; no exact event GT",
        "test": "not_loaded_not_evaluated",
        "fit_calls": 0,
        "calibration_calls": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    args = parser.parse_args()
    value = verify()
    if args.output:
        from pathlib import Path

        with Path(args.output).open("x", encoding="utf-8") as h:
            json.dump(value, h, ensure_ascii=False, indent=2, allow_nan=False)
            h.write("\n")
    print(json.dumps(value, ensure_ascii=False, indent=2))
