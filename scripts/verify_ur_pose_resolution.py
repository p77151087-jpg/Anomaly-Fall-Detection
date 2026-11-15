"""Independent numerical/event audit of the single resolution experiment. No Test reads."""

import argparse
import csv
import json
from collections import defaultdict

import numpy as np
import torch

from fall_detect.config import project_path
from fall_detect.data.cache import load_pose_cache
from fall_detect.io_utils import ContractError, read_json, sha256, write_json
from fall_detect.models.autoencoder import Autoencoder
from fall_detect.skeleton.preprocessing import good_frame
from fall_detect.training import experiment

ROOT = project_path(".")
TAG = "ur-pose-resolution-20261013-001"
E = ROOT / "docs/validation" / TAG
O = ROOT / "outputs" / TAG
RUN = ROOT / "runs" / TAG


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def ratio(a, b):
    return a / b if b else None


def hand_metrics(rows):
    table = [[0, 0, 0], [0, 0, 0]]
    for r in rows:
        p = r["prediction"]
        table[r["label"]][2 if p is None else p] += 1
    tn, fp, un = table[0]
    fn, tp, uf = table[1]
    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "n_total": len(rows),
        "n_evaluable": tp + tn + fp + fn,
        "n_unknown": un + uf,
        "sequence_accuracy": ratio(tp + tn, tp + tn + fp + fn),
        "operational_accuracy": ratio(tp + tn, len(rows)),
        "sequence_fall_precision": (ratio(tp, tp + fp) if tp + fp else 0.0)
        if tp + tn + fp + fn
        else None,
        "sequence_fall_recall": ratio(tp, tp + fn),
        "sequence_fall_f1": (ratio(2 * tp, 2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
        if tp + tn + fp + fn
        else None,
        "end_to_end_fall_recall": ratio(tp, tp + fn + uf),
        "outcomes_with_unknown": table,
    }


def float_or_none(value):
    return float(value) if value != "" else None


def parse_windows(path):
    result = []
    for r in csv_rows(path):
        result.append(
            {
                "sequence_id": r["sequence_id"],
                "track_id": int(r["track_id"]) if r["track_id"] else None,
                "sample_index": int(r["sample_index"]),
                "window_id": int(r["window_id"]),
                **{
                    k: float_or_none(r[k])
                    for k in ["start_ms", "end_ms", "available_at_ms", "raw_score", "event_score"]
                },
                "state": r["state"],
                "reason": r["reason"] or None,
            }
        )
    return result


def independent_events(windows, stride, threshold, end_ms):
    history, previous, active = {}, {}, {}
    events = []
    scores = []

    def close(track, reason, time):
        if track in active:
            events.append({**active.pop(track), "end_ms": float(time), "end_reason": reason})

    for w in windows:
        track, sample, time, raw = (
            w["track_id"],
            w["sample_index"],
            w["available_at_ms"],
            w["raw_score"],
        )
        values = history.setdefault(track, [])
        if track in previous and sample != previous[track] + stride:
            values.clear()
            close(track, "time_gap", time)
        previous[track] = sample
        if raw is None:
            values.clear()
            close(track, w["reason"], time)
            assert w["state"] == "unknown" and w["event_score"] is None
            continue
        values.append(raw)
        del values[:-3]
        score = min(values) if len(values) == 3 else None
        assert score == w["event_score"]
        expected = (
            "warming_up" if score is None else "suspected_fall" if score > threshold else "normal"
        )
        assert w["state"] == expected
        assert w["reason"] == ("insufficient_consecutive_windows" if score is None else None)
        if score is not None:
            scores.append(score)
        if expected == "suspected_fall":
            if track not in active:
                active[track] = {
                    "sequence_id": w["sequence_id"],
                    "track_id": track,
                    "confirmed_start_ms": time,
                    "last_active_ms": time,
                    "peak_score": raw,
                }
            else:
                active[track]["last_active_ms"] = time
                active[track]["peak_score"] = max(active[track]["peak_score"], raw)
        else:
            close(track, "low_score", time)
    for track in history:
        close(track, "sequence_end", end_ms)
    return max(scores) if scores else None, events


def numerical_scores(run, features):
    ckpt = torch.load(run / "autoencoder.pt", map_location="cpu", weights_only=True)
    with np.load(run / "scaler.npz", allow_pickle=False) as scale:
        z = ((features - scale["mean"]) / scale["scale"]).astype(np.float32)
    model = Autoencoder(ckpt["input_dim"], ckpt["hidden_dim"], ckpt["latent_dim"])
    model.load_state_dict(ckpt["state_dict"], strict=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval().requires_grad_(False)
    result = []
    with torch.inference_mode():
        for row in z:
            x = torch.from_numpy(row[None]).to(device)
            result.append(float(((model(x) - x) ** 2).mean()))
    return np.array(result)


def audit_result(name, run, index):
    report = read_json(O / name / "metrics.json")
    threshold = read_json(run / "threshold.json")["threshold"]
    output = csv_rows(O / name / "sequence-results.csv")
    rows = [
        {
            "sequence_id": r["sequence_id"],
            "label": int(r["label"]),
            "prediction": int(r["prediction"]) if r["prediction"] else None,
        }
        for r in output
    ]
    assert len(rows) == 23 and len({r["sequence_id"] for r in rows}) == 23
    metrics = hand_metrics(rows)
    for k, value in metrics.items():
        assert report[k] == value, (name, k, report[k], value)
    confusion = csv_rows(O / name / "outcomes-with-unknown.csv")
    assert [
        [int(r[k]) for k in ["normal", "suspected_fall", "unknown"]] for r in confusion
    ] == metrics["outcomes_with_unknown"]
    windows = parse_windows(O / name / "window-results.csv")
    group = defaultdict(list)
    for w in windows:
        group[w["sequence_id"]].append(w)
    expected_events, calibration_rows = [], []
    scores_checked = 0
    for source in index["split_manifest"]["sequences"]:
        if source["split"] != "validation":
            continue
        sid = source["sequence_id"]
        cache = experiment.cache_for(index, sid)
        values = numerical_scores(run, cache.features)
        selected = group[sid]
        records = sorted(cache.records, key=lambda r: (r.sample_index, r.track_id or -1))
        assert len(records) == len(selected)
        for record, w in zip(records, selected, strict=True):
            assert w["sample_index"] == record.sample_index and w["track_id"] == record.track_id
            assert w["available_at_ms"] == record.available_at_ms and w["end_ms"] == record.end_ms
            if record.row_index is None:
                assert w["raw_score"] is None and w["reason"] == record.quality
            else:
                np.testing.assert_allclose(
                    w["raw_score"], values[record.row_index], atol=1e-7, rtol=1e-6
                )
                scores_checked += 1
        score, events = independent_events(
            selected, cache.metadata["stride"], threshold, cache.metadata["end_ms"]
        )
        expected_events.extend(events)
        official = next(r for r in output if r["sequence_id"] == sid)
        assert float_or_none(official["sequence_score"]) == score
        assert (int(score > threshold) if score is not None else None) == next(
            r["prediction"] for r in rows if r["sequence_id"] == sid
        )
        assert float_or_none(official["first_alarm_ms"]) == (
            events[0]["confirmed_start_ms"] if events else None
        )
        endpoints = {w["sample_index"] for w in selected}
        valid = {w["sample_index"] for w in selected if w["raw_score"] is not None}
        assert int(official["n_scheduled_endpoints"]) == len(endpoints)
        assert int(official["n_valid_endpoints"]) == len(valid)
        assert float(official["window_coverage"]) == len(valid) / len(endpoints)
        calibration_rows.append(
            {"sequence_id": sid, "label": source["label"], "sequence_score": score}
        )
    actual_events = []
    for r in csv_rows(O / name / "events.csv"):
        actual_events.append(
            {
                "sequence_id": r["sequence_id"],
                "track_id": int(r["track_id"]) if r["track_id"] else None,
                **{
                    k: float(r[k])
                    for k in ["confirmed_start_ms", "last_active_ms", "end_ms", "peak_score"]
                },
                "end_reason": r["end_reason"],
            }
        )
    assert actual_events == expected_events
    return {
        "metrics": metrics,
        "threshold": threshold,
        "scores_checked": scores_checked,
        "windows_checked": len(windows),
        "events_checked": len(expected_events),
        "calibration_rows": calibration_rows,
        "rows": rows,
    }


def temporal_diagnosis(name, index, config):
    """Phase/visibility diagnosis; no precise event-time GT or new-Pose human labels."""
    data = read_json(E / "validation.json")["baseline" if name == "baseline" else "candidate"]
    intervals = read_json(ROOT / "docs/validation/ur-normal-methods-20261012-001/svm-lock.json")[
        "coarse_intervals"
    ]
    labels = read_json(
        ROOT / "outputs/ur-fall19-human-amendment-20261013-001/derived-annotation.json"
    )["frames"]
    label_map = {r["source_frame_id"]: r for r in labels}
    phases = []
    human_summary = {}
    for sid, interval in intervals.items():
        observations, _ = load_pose_cache(project_path(index["sequences"][sid]["pose_path"]), {})

        def observed(time, track, observations=observations):
            frame, poses = max(
                (r for r in observations if r[0].timestamp_ms <= time),
                key=lambda r: r[0].timestamp_ms,
            )
            return frame, any(
                p.track_id == track and good_frame(p.keypoints, config["quality"]) for p in poses
            )

        windows = [w for w in data["windows"] if w["sequence_id"] == sid]
        events = [e for e in data["events"] if e["sequence_id"] == sid]
        lo, hi = interval["fall_phase_ms"]
        phase_events = [e for e in events if lo <= e["confirmed_start_ms"] <= hi]
        current_good = [
            e for e in phase_events if observed(e["confirmed_start_ms"], e["track_id"])[1]
        ]
        visible_alarm = [
            w
            for w in windows
            if lo <= w["available_at_ms"] <= hi
            and w["state"] == "suspected_fall"
            and observed(w["available_at_ms"], w["track_id"])[1]
        ]
        phases.append(
            {
                "sequence_id": sid,
                "interval": interval,
                "event_starts": [e["confirmed_start_ms"] for e in phase_events],
                "good_event_starts": [e["confirmed_start_ms"] for e in current_good],
                "good_alarm_endpoints": len(visible_alarm),
                "early_event_starts": [
                    e["confirmed_start_ms"] for e in events if e["confirmed_start_ms"] < lo
                ],
            }
        )
        if sid == "fall-19":
            grouped = defaultdict(list)
            for w in windows:
                frame, good = observed(w["available_at_ms"], w["track_id"])
                assert label_map[frame.source_frame_id]["timestamp_ms"] == frame.timestamp_ms
                grouped[label_map[frame.source_frame_id]["phase"]].append((w, good))
            for phase, rows in grouped.items():
                starts = [
                    e
                    for e in events
                    if label_map[
                        observed(e["confirmed_start_ms"], e["track_id"])[0].source_frame_id
                    ]["phase"]
                    == phase
                ]
                human_summary[phase] = {
                    "scheduled_endpoints": len({w["sample_index"] for w, _ in rows}),
                    "valid_endpoints": len(
                        {w["sample_index"] for w, _ in rows if w["raw_score"] is not None}
                    ),
                    "max_raw": max(
                        (w["raw_score"] for w, _ in rows if w["raw_score"] is not None),
                        default=None,
                    ),
                    "max_k3": max(
                        (w["event_score"] for w, _ in rows if w["event_score"] is not None),
                        default=None,
                    ),
                    "event_starts": [e["confirmed_start_ms"] for e in starts],
                    "good_event_starts": [
                        e["confirmed_start_ms"]
                        for e in starts
                        if observed(e["confirmed_start_ms"], e["track_id"])[1]
                    ],
                    "good_alarm_endpoints": sum(
                        w["state"] == "suspected_fall" and g for w, g in rows
                    ),
                }
                formal = data["fall19_human_phases"][phase]
                assert all(
                    formal[k] == v
                    for k, v in human_summary[phase].items()
                    if k != "good_event_starts"
                )
    return {
        "coarse_phases": phases,
        "coarse_event_movies": sum(bool(r["event_starts"]) for r in phases),
        "coarse_good_alarm_movies": sum(r["good_alarm_endpoints"] > 0 for r in phases),
        "coarse_early_events": sum(len(r["early_event_starts"]) for r in phases),
        "fall19_human_phases": human_summary,
        "precise_time_ground_truth": False,
    }


def verify():
    from ur_pose_resolution import verify_lock

    lock, config = verify_lock()
    index = experiment.load_index(RUN / "dataset-index.json")
    sources = index["split_manifest"]["sequences"]
    train = [s for s in sources if s["split"] == "train"]
    normal_val = [s for s in sources if s["split"] == "validation" and s["label"] == 0]
    assert len(train) == 24 and all(s["label"] == 0 for s in train) and len(normal_val) == 8
    assert index["preparation_scope"] == ["train", "validation"]
    test = [index["sequences"][s["sequence_id"]] for s in sources if s["split"] == "test"]
    assert len(test) == 23 and all(
        r["status"] == "not_prepared" and not any(k in r for k in ["feature_path", "pose_path"])
        for r in test
    )
    arrays = [experiment.cache_for(index, s["sequence_id"]).features for s in train]
    x = np.concatenate(arrays).astype(np.float64)
    expected_std = x.std(axis=0)
    expected_std[expected_std == 0] = 1
    with np.load(RUN / "scaler.npz", allow_pickle=False) as scaler:
        np.testing.assert_array_equal(scaler["mean"], x.mean(axis=0))
        np.testing.assert_array_equal(scaler["scale"], expected_std)
    provenance = read_json(RUN / "sources.json")
    assert set(provenance["train_feature_hashes"]) == {s["sequence_id"] for s in train}
    assert set(provenance["val_normal_feature_hashes"]) == {s["sequence_id"] for s in normal_val}
    history = csv_rows(RUN / "training-history.csv")
    best = min(history, key=lambda r: float(r["val_normal_loss"]))
    checkpoint_loss = np.concatenate(
        [
            numerical_scores(RUN, experiment.cache_for(index, s["sequence_id"]).features)
            for s in normal_val
        ]
    ).mean()
    np.testing.assert_allclose(
        checkpoint_loss, float(best["val_normal_loss"]), atol=1e-7, rtol=1e-6
    )
    baseline_run = ROOT / "runs/ur-isotropic-wide-20261012-001"
    baseline = audit_result(
        "baseline", baseline_run, experiment.load_index(baseline_run / "dataset-index.json")
    )
    candidate = audit_result("candidate", RUN, index)
    baseline_index = experiment.load_index(baseline_run / "dataset-index.json")
    baseline["temporal"] = temporal_diagnosis("baseline", baseline_index, config)
    candidate["temporal"] = temporal_diagnosis("candidate", index, config)
    paired = csv_rows(O / "paired-pose-frames.csv")
    pose_rows_checked = 0
    for source in sources:
        if source["split"] not in ("train", "validation"):
            continue
        sid = source["sequence_id"]
        old, _ = load_pose_cache(project_path(baseline_index["sequences"][sid]["pose_path"]), {})
        new, _ = load_pose_cache(project_path(index["sequences"][sid]["pose_path"]), {})
        chosen = [r for r in paired if r["sequence_id"] == sid]
        for row, (a, ap), (b, bp) in zip(chosen, old, new, strict=True):
            assert a.source_frame_id == b.source_frame_id == int(row["source_frame_id"])
            assert a.timestamp_ms == b.timestamp_ms == float(row["timestamp_ms"])
            assert int(row["old_pose_count"]) == len(ap) and int(row["new_pose_count"]) == len(bp)
            assert row["old_good"] == str(
                any(good_frame(p.keypoints, config["quality"]) for p in ap)
            )
            assert row["new_good"] == str(
                any(good_frame(p.keypoints, config["quality"]) for p in bp)
            )
            assert row["human_new_pose_check"] == ""
            pose_rows_checked += 1
    values = sorted(
        {
            r["sequence_score"]
            for r in candidate["calibration_rows"]
            if r["sequence_score"] is not None
        }
    )
    calibrated = []
    for threshold in [float(np.nextafter(values[0], -np.inf)), *values]:
        rows = [
            {
                **r,
                "prediction": int(r["sequence_score"] > threshold)
                if r["sequence_score"] is not None
                else None,
            }
            for r in candidate["calibration_rows"]
        ]
        m = hand_metrics(rows)
        rank = (0.5 * (m["TP"] / 15 + m["TN"] / 8), -m["FP"], threshold)
        calibrated.append((rank, threshold))
    assert max(calibrated)[1] == candidate["threshold"]
    before = read_json(E / "before.json")
    for i, (name, value) in enumerate(before["assets"].items(), 1):
        assert sha256(ROOT / name) == value, name
        if i % 5000 == 0:
            print(f"Verified {i} preserved assets", flush=True)
    for name, value in before["contracts"].items():
        assert sha256(E / "prior-contract" / name) == value
    for role, entry in index["weights"].items():
        assert sha256(project_path(entry["local_path"])) == lock["weight_hashes"][role]
    return {
        "passed": True,
        "evidence_level": "real_feature_smoke",
        "candidate": TAG,
        "baseline": baseline,
        "candidate_result": candidate,
        "train_videos": len(train),
        "train_windows": len(x),
        "normal_val_videos": len(normal_val),
        "epochs": len(history),
        "best_epoch": int(best["epoch"]),
        "best_normal_val_loss": float(best["val_normal_loss"]),
        "checkpoint_normal_val_loss_independent": float(checkpoint_loss),
        "threshold_candidates_checked": len(calibrated),
        "paired_pose_source_rows_checked": pose_rows_checked,
        "old_assets_preserved": len(before["assets"]),
        "old_contracts_archived": len(before["contracts"]),
        "test_unprepared": len(test),
        "test_feature_reads": 0,
        "test_image_reads": 0,
        "new_test_evaluations": 0,
        "precise_time_ground_truth": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    result = verify()
    path = E / "verification.json"
    if args.save:
        if path.exists():
            raise ContractError("Preserve existing verification")
        write_json(path, result)
    else:
        assert read_json(path) == result
    print(json.dumps(result, ensure_ascii=False, indent=2))
