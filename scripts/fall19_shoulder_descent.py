"""Evaluate one causal shoulder cue beside the frozen wide AE on Train/Validation."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from analyze_fall19_score_overlap import (
    BASE,
    HUMAN,
    HUMAN_ID,
    MOTION,
    MOTION_ID,
    QA,
    ROOT,
    VAL,
    WIDE,
    Inputs,
    require,
)

from fall_detect.config import load_config
from fall_detect.data.cache import load_pose_cache
from fall_detect.detection.events import EventState
from fall_detect.evaluation.metrics import sequence_metrics
from fall_detect.skeleton.preprocessing import good_frame
from fall_detect.skeleton.windows import CausalResampler
from fall_detect.visualization.cached_replay import CachedDisplayReplay

TAG = "ur-fall19-shoulder-descent-20261107-001"
OUT = ROOT / "outputs" / TAG
PLAN = ROOT / "docs/ur-fall19-shoulder-descent-20261107-plan.md"
TARGET = "fall-19"
LAG = 6
K = 3


def relative(path):
    return path.relative_to(ROOT).as_posix()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def save_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare():
    inputs = Inputs()
    frozen = inputs.json(WIDE / "frozen-artifacts.json")
    index = inputs.json(WIDE / "dataset-index.json", frozen["dataset-index.json"])
    threshold = inputs.json(WIDE / "threshold.json", frozen["threshold.json"])
    require(
        threshold["comparison"] == "gt" and threshold["consecutive_windows"] == K,
        "Unexpected wide threshold contract",
    )
    inputs.read(WIDE / "resolved-config.yaml", frozen["resolved-config.yaml"])
    config = load_config(WIDE / "resolved-config.yaml")
    skeleton = config["skeleton"]
    require(
        (
            skeleton["target_fps"],
            skeleton["stride"],
            skeleton["window_size"],
            skeleton["coordinate_normalization"],
        )
        == (30, 1, 30, "per_frame_isotropic"),
        "Unexpected wide timing or recipe",
    )
    archive = inputs.json(MOTION / "frozen-artifacts.json")
    source = inputs.json(MOTION / "sources.json", archive["sources.json"])
    copied = inputs.json(MOTION / "dataset-index.json", archive["dataset-index.json"])
    old_lock = inputs.json(QA / MOTION_ID / "lock.json", source["lock_sha256"])
    selected = [
        r for r in index["split_manifest"]["sequences"] if r["split"] in {"train", "validation"}
    ]
    require(
        Counter((r["split"], r["label"]) for r in selected)
        == {("train", 0): 24, ("validation", 0): 8, ("validation", 1): 15},
        "Unexpected split membership",
    )
    require(
        len({r["sequence_id"] for r in selected}) == 47
        and index["recipe_hash"] == threshold["recipe_hash"] == old_lock["feature_recipe_hash"],
        "Unexpected input recipe or duplicate sequence",
    )
    split_hash = index["split_manifest"]["split_hash"]
    require(split_hash == threshold["split_hash"], "Different split hashes")
    entries, metadata = {}, {}
    for row in selected:
        sid = row["sequence_id"]
        old, saved = index["sequences"][sid], copied["sequences"][sid]
        require(
            old["status"] == saved["status"] == "prepared"
            and old["source_hash"] == saved["source_hash"],
            f"Unexpected cache source: {sid}",
        )
        entry = {**row, "source_hash": old["source_hash"], "n_samples": old["n_samples"]}
        for key, kind in (("pose_path", "poses"), ("feature_path", "features")):
            p = (ROOT / old[key]).resolve()
            old_copy = ROOT / saved[key]
            require(
                p.parent == ROOT / "data/cache/ur-human-isotropic-001" / kind
                and p.name == old_copy.name,
                f"Unexpected cache path: {sid}",
            )
            suffixes = (".json", ".npz") if kind == "poses" else (".json",)
            for suffix in suffixes:
                inputs.read(
                    p.with_suffix(suffix),
                    old_lock["input_hashes"][relative(old_copy.with_suffix(suffix))],
                )
            entry[key] = relative(p if kind == "poses" else p.with_suffix(".json"))
        meta = inputs.json(ROOT / entry["feature_path"])
        require(
            (
                meta["sequence_id"],
                meta["split"],
                meta["recipe_hash"],
                meta["split_hash"],
                meta["source_hash"],
                meta["stride"],
            )
            == (sid, row["split"], index["recipe_hash"], split_hash, old["source_hash"], 1),
            "Original window metadata differs",
        )
        if row["split"] == "validation":
            require(
                meta["array_sha256"] == threshold["feature_hashes"][sid],
                "Original Validation feature digest differs",
            )
        entries[sid], metadata[sid] = entry, meta

    display_pins = inputs.json(QA / "ur-validation-display-20261105-001/inputs.json")["files"]
    original = {
        name: inputs.csv(VAL / f"{name}.csv", display_pins[relative(VAL / f"{name}.csv")])
        for name in ("windows", "sequences", "events")
    }
    original["metrics"] = inputs.json(
        VAL / "metrics.json", display_pins[relative(VAL / "metrics.json")]
    )
    human_pins = inputs.json(QA / HUMAN_ID / "final-qa.json")["artifacts"]
    path = HUMAN / "human-frame-diagnosis.csv"
    human_frames = {
        int(r["source_frame_id"]): r for r in inputs.csv(path, human_pins[relative(path)])
    }
    require(len(human_frames) == 100, "Unexpected fall-19 human frame record")
    timing = ROOT / "outputs/ur-validation-human-timing-20261105-001"
    timing_pins = inputs.json(timing / "artifact-manifest.json")["files"]
    human_intervals = inputs.json(timing / "comparisons.json", timing_pins["comparisons.json"])
    for p in (
        PLAN,
        Path(__file__),
        ROOT / "scripts/analyze_fall19_score_overlap.py",
        ROOT / "src/fall_detect/detection/events.py",
        ROOT / "src/fall_detect/data/cache.py",
        ROOT / "src/fall_detect/skeleton/windows.py",
        ROOT / "src/fall_detect/skeleton/preprocessing.py",
        ROOT / "src/fall_detect/visualization/cached_replay.py",
        ROOT / "src/fall_detect/visualization/sample_display.py",
        ROOT / "src/fall_detect/config.py",
        ROOT / "src/fall_detect/recipes.py",
        ROOT / "src/fall_detect/evaluation/metrics.py",
    ):
        inputs.read(p)
    recipe = {
        "lag_ticks": LAG,
        "fps": 30,
        "k": K,
        "score": "max(0, min(two shoulder dy) / lagged bbox height)",
        "eligible": "original legal window and both original confidence-good poses with both shoulder confidences passing and distinct causal source frames",
        "threshold": "max K3 over 24 Train ADL and 8 Validation ADL, strict gt",
        "fusion": "per-track OR after two independent K3 histories",
        "wide_recipe_hash": index["recipe_hash"],
        "wide_threshold": threshold["threshold"],
        "split_hash": split_hash,
    }
    lock = {
        "id": TAG,
        "locked_at_utc": datetime.now(UTC).isoformat(),
        "baseline": BASE,
        "recipe": recipe,
        "comparison_recipe_sha256": hashlib.sha256(
            json.dumps(recipe, sort_keys=True).encode()
        ).hexdigest(),
        "source_sha256": inputs.hashes,
        "allowed_sequences": entries,
        "legacy_intervals": old_lock["coarse_intervals"],
    }
    return lock, config, metadata, original, human_frames, human_intervals


def coherent_descent(previous, current, previous_box, quality):
    """Return an observed image-plane displacement or an explicit missing reason."""
    if not good_frame(previous, quality, 13) or not good_frame(current, quality, 13):
        return None, "anchor_confidence", None, None
    minimum = quality["min_joint_confidence"]
    if (previous[1:3, 2] < minimum).any() or (current[1:3, 2] < minimum).any():
        return None, "shoulder_confidence", None, None
    height = float(previous_box[3] - previous_box[1])
    if not np.isfinite(height) or height <= 0:
        return None, "invalid_reference_height", None, None
    delta = float((current[1:3, 1].astype(np.float64) - previous[1:3, 1]).min())
    return max(0.0, delta / height), None, delta, height


def derive(lock, config, metadata):
    cue, val_samples = {}, {}
    for sid, entry in lock["allowed_sequences"].items():
        expected = {
            "sequence_id": sid,
            "split": entry["split"],
            "recipe_hash": lock["recipe"]["wide_recipe_hash"],
            "split_hash": lock["recipe"]["split_hash"],
            "source_hash": entry["source_hash"],
        }
        observed, _ = load_pose_cache(ROOT / entry["pose_path"], expected)
        resampler = CausalResampler(30, config["skeleton"]["resampling_hold_policy"])
        samples = [s for frame, poses in observed for s in resampler.push(frame, poses)]
        require(
            len(samples) == entry["n_samples"]
            and [s.sample_index for s in samples] == list(range(len(samples))),
            f"Original sample grid differs: {sid}",
        )
        if entry["split"] == "validation":
            val_samples[sid] = samples
        lookup = {(s.sample_index, p.track_id): p for s in samples for p in s.poses}
        rows = []
        for w in sorted(
            metadata[sid]["records"], key=lambda r: (r["sample_index"], r["track_id"] or -1)
        ):
            t, track = w["sample_index"], w["track_id"]
            require(
                w["sequence_id"] == sid
                and w["split"] == entry["split"]
                and w["available_at_ms"] == samples[t].timestamp_ms,
                "Different original window timing",
            )
            frame = samples[t].source_frame_id
            ref_frame = samples[t - LAG].source_frame_id if t >= LAG else None
            previous, current = lookup.get((t - LAG, track)), lookup.get((t, track))
            score, delta, height = None, None, None
            reason = w["quality"]
            if w["row_index"] is not None:
                require(w["quality"] == "valid", "Original quality contract differs")
                if previous is None or current is None or frame is None or ref_frame is None:
                    reason = "anchor_track_gap"
                elif frame <= ref_frame:
                    reason = "no_distinct_causal_frame"
                else:
                    score, reason, delta, height = coherent_descent(
                        previous.keypoints, current.keypoints, previous.bbox_xyxy, config["quality"]
                    )
            rows.append(
                {
                    "sequence_id": sid,
                    "split": entry["split"],
                    "track_id": track,
                    "sample_index": t,
                    "available_at_ms": w["available_at_ms"],
                    "source_frame_id": frame,
                    "reference_source_frame_id": ref_frame,
                    "original_legal": w["row_index"] is not None,
                    "minimum_shoulder_delta_y_px": delta,
                    "reference_box_height_px": height,
                    "raw_score": score,
                    "missing_reason": reason,
                }
            )
        cue[sid] = rows
    return cue, val_samples


def score_branch(rows, threshold, end_ms):
    state = EventState(threshold, K, 1)
    output = []
    for row in rows:
        result = state.update(
            row["sequence_id"],
            row["track_id"],
            row["sample_index"],
            row["available_at_ms"],
            row["raw_score"],
            row["missing_reason"],
        )
        output.append({**row, **result})
    return output, state.finish(end_ms)


def normalize_wide(rows):
    out = defaultdict(list)
    for r in rows:
        row = dict(r)
        for field in ("sample_index", "track_id", "window_id"):
            row[field] = int(row[field]) if row[field] else None
        for field in ("available_at_ms", "raw_score", "event_score"):
            row[field] = float(row[field]) if row[field] else None
        row["reason"] = row["reason"] or None
        out[row["sequence_id"]].append(row)
    return out


def fuse_rows(wide, shoulder, wide_threshold, shoulder_threshold):
    require(len(wide) == len(shoulder), "Different timeline lengths")
    combined = []
    for a, b in zip(wide, shoulder, strict=True):
        for field in ("sequence_id", "track_id", "sample_index", "available_at_ms"):
            require(a[field] == b[field], "Wide and shoulder window alignment differs")
        require(
            (a["raw_score"] is not None) == b["original_legal"],
            "Original legal window mask differs",
        )
        require(
            b["event_score"] is None or a["event_score"] is not None,
            "Shoulder bypassed original K3 availability",
        )
        a_active, b_active = a["state"] == "suspected_fall", b["state"] == "suspected_fall"
        scores = [
            row["event_score"] / limit
            for row, limit in ((a, wide_threshold), (b, shoulder_threshold))
            if row["event_score"] is not None
        ]
        fused = max(scores, default=None)
        require(
            (fused is not None and fused > 1.0) == (a_active or b_active),
            "Fusion crossed strict threshold inconsistently",
        )
        combined.append(
            {
                "sequence_id": a["sequence_id"],
                "track_id": a["track_id"],
                "sample_index": a["sample_index"],
                "available_at_ms": a["available_at_ms"],
                "wide_raw": a["raw_score"],
                "wide_k3": a["event_score"],
                "shoulder_raw": b["raw_score"],
                "shoulder_k3": b["event_score"],
                "wide_active": a_active,
                "shoulder_active": b_active,
                "combined_k3_ratio": fused,
                "combined_state": "suspected_fall" if a_active or b_active else a["state"],
                "wide_reason": a["reason"],
                "shoulder_reason": b["reason"],
            }
        )
    return combined


def combined_events(rows, end_ms):
    active, last, events = {}, {}, []

    def close(track, time, reason):
        if track in active:
            events.append({**active.pop(track), "end_ms": time, "end_reason": reason})

    for row in rows:
        track, t, now = row["track_id"], row["sample_index"], row["available_at_ms"]
        if track in last and t != last[track] + 1:
            close(track, now, "time_gap")
        last[track] = t
        if row["combined_state"] != "suspected_fall":
            close(track, now, row["wide_reason"] or row["shoulder_reason"] or "low_score")
            continue
        event = active.setdefault(
            track,
            {
                "sequence_id": row["sequence_id"],
                "track_id": track,
                "confirmed_start_ms": now,
                "last_active_ms": now,
                "peak_relative_k3": 0.0,
                "wide_active_endpoints": 0,
                "shoulder_active_endpoints": 0,
            },
        )
        event["last_active_ms"] = now
        event["peak_relative_k3"] = max(event["peak_relative_k3"], row["combined_k3_ratio"])
        event["wide_active_endpoints"] += row["wide_active"]
        event["shoulder_active_endpoints"] += row["shoulder_active"]
    for track in list(active):
        close(track, end_ms, "sequence_end")
    return sorted(events, key=lambda e: (e["confirmed_start_ms"], e["track_id"]))


def replay_display(samples, wide, shoulder, events, config, thresholds):
    grouped = []
    for stream in (wide, shoulder):
        by_sample = defaultdict(list)
        for row in stream:
            by_sample[row["sample_index"]].append(row)
        grouped.append(by_sample)
    wide_replay, cue_replay = (CachedDisplayReplay(config, tau) for tau in thresholds)
    timeline = []
    for s in samples:
        _, wide_current = wide_replay.step(s, grouped[0].get(s.sample_index, []))
        _, cue_current = cue_replay.step(s, grouped[1].get(s.sample_index, []))
        a = {
            track for track, current in wide_current.items() if current["state"] == "suspected_fall"
        }
        b = {
            track for track, current in cue_current.items() if current["state"] == "suspected_fall"
        }
        timeline.append(
            {
                "sequence_id": s.sequence_id,
                "sample_index": s.sample_index,
                "source_frame_id": s.source_frame_id,
                "timestamp_ms": s.timestamp_ms,
                "wide_current_alert_tracks": json.dumps(sorted(a)),
                "shoulder_current_alert_tracks": json.dumps(sorted(b)),
                "combined_current_alert_tracks": json.dumps(sorted(a | b)),
                "wide_history_count": wide_replay.history.count,
                "combined_history_count": sum(
                    e["confirmed_start_ms"] <= s.timestamp_ms for e in events
                ),
            }
        )
    return timeline


def summarize_video(old, rows, events, display):
    peak = max(
        (r["combined_k3_ratio"] for r in rows if r["combined_k3_ratio"] is not None), default=None
    )
    valid = len({r["sample_index"] for r in rows if r["wide_raw"] is not None})
    total = len({r["sample_index"] for r in rows})
    require(
        (valid, total) == (int(old["n_valid_endpoints"]), int(old["n_scheduled_endpoints"])),
        "Original endpoint counts changed",
    )
    first = events[0]["confirmed_start_ms"] if events else None
    require(
        bool(events) == (peak is not None and peak > 1.0),
        "Combined events disagree with peak score",
    )
    current = [r for r in display if json.loads(r["combined_current_alert_tracks"])]
    require(
        all(
            set(json.loads(r["wide_current_alert_tracks"]))
            <= set(json.loads(r["combined_current_alert_tracks"]))
            for r in display
        ),
        "Original current alert lost",
    )
    return {
        "sequence_id": old["sequence_id"],
        "label": int(old["label"]),
        "wide_prediction": int(old["prediction"]) if old["prediction"] else None,
        "prediction": int(peak > 1.0) if peak is not None else None,
        "sequence_score": peak,
        "n_scheduled_endpoints": total,
        "n_valid_endpoints": valid,
        "wide_first_alarm_ms": float(old["first_alarm_ms"]) if old["first_alarm_ms"] else None,
        "first_alarm_ms": first,
        "combined_event_count": len(events),
        "wide_current_alert_frames": sum(
            bool(json.loads(r["wide_current_alert_tracks"])) for r in display
        ),
        "combined_current_alert_frames": len(current),
        "first_current_alert_ms": current[0]["timestamp_ms"] if current else None,
    }


def compare_timing(events, display, old_events, legacy, human_intervals):
    rows = []
    for sid, interval in legacy.items():
        lo, hi = interval["fall_phase_ms"]
        starts = [e["confirmed_start_ms"] for e in events if e["sequence_id"] == sid]
        before = [float(e["confirmed_start_ms"]) for e in old_events if e["sequence_id"] == sid]
        good = [
            r["timestamp_ms"]
            for r in display
            if r["sequence_id"] == sid
            and json.loads(r["combined_current_alert_tracks"])
            and lo <= r["timestamp_ms"] <= hi
        ]
        rows.append(
            {
                "sequence_id": sid,
                "wide_coarse_event_starts_ms": [t for t in before if lo <= t <= hi],
                "combined_coarse_event_starts_ms": [t for t in starts if lo <= t <= hi],
                "combined_coarse_early_ms": [t for t in starts if t < interval["normal_until_ms"]],
                "combined_current_alert_ms_in_coarse_phase": good,
            }
        )
    human = []
    for case in human_intervals:
        sid = case["sequence_id"]
        first = min(
            (e["confirmed_start_ms"] for e in events if e["sequence_id"] == sid), default=None
        )
        lo, hi = case["impact_bounds_ms"]
        relation = (
            "no_event"
            if first is None
            else "before_impact_interval"
            if first < lo
            else "after_impact_interval"
            if first > hi
            else "inside_impact_uncertainty"
        )
        human.append(
            {
                "sequence_id": sid,
                "impact_bounds_ms": [lo, hi],
                "wide_first_ms": case["first_alarm_ms"],
                "combined_first_ms": first,
                "relation": relation,
            }
        )
    return rows, human


def verify_hashes(expected, root):
    for path, digest in expected.items():
        require(sha(root / path) == digest, f"Frozen file changed: {path}")


def build():
    if OUT.exists():
        raise FileExistsError(f"Preserve existing experiment: {OUT}")
    lock, config, metadata, original, human_frames, human_intervals = prepare()
    OUT.mkdir(exist_ok=False)
    save_json(OUT / "lock.json", lock)
    cue, samples = derive(lock, config, metadata)
    normal = {}
    for sid, entry in lock["allowed_sequences"].items():
        if entry["label"] == 0:
            scored, _ = score_branch(cue[sid], 0.0, metadata[sid]["end_ms"])
            maximum = max(
                (r["event_score"] for r in scored if r["event_score"] is not None), default=None
            )
            require(maximum is not None, f"Normal sequence lacks a legal shoulder K3: {sid}")
            normal[sid] = {"split": entry["split"], "max_k3": maximum}
    require(
        Counter(v["split"] for v in normal.values()) == {"train": 24, "validation": 8},
        "Normal calibration scope differs",
    )
    threshold = max(v["max_k3"] for v in normal.values())
    require(threshold > 0, "Degenerate normal calibration")
    save_json(
        OUT / "normal-calibration.json",
        {"threshold": threshold, "normal_video_k3": normal, "comparison": "gt"},
    )
    original_windows = normalize_wide(original["windows"])
    wide_threshold = lock["recipe"]["wide_threshold"]
    old_sequences = {r["sequence_id"]: r for r in original["sequences"]}
    require(
        set(old_sequences) == set(samples) == set(original_windows) and len(samples) == 23,
        "Unexpected Validation output scope",
    )
    videos, windows, events, display, branch_rows = [], [], [], [], []
    for sid, metadata_rows in cue.items():
        shoulder, _ = score_branch(metadata_rows, threshold, metadata[sid]["end_ms"])
        branch_rows.extend(shoulder)
        if sid not in samples:
            continue
        old = original_windows[sid]
        merged = fuse_rows(old, shoulder, wide_threshold, threshold)
        found = combined_events(merged, metadata[sid]["end_ms"])
        screen = replay_display(
            samples[sid], old, shoulder, found, config, (wide_threshold, threshold)
        )
        videos.append(summarize_video(old_sequences[sid], merged, found, screen))
        windows.extend(merged)
        events.extend(found)
        display.extend(screen)
    metrics = sequence_metrics(videos)
    timing, manual = compare_timing(
        events, display, original["events"], lock["legacy_intervals"], human_intervals
    )
    target = next(r for r in videos if r["sequence_id"] == TARGET)
    target_events = [e for e in events if e["sequence_id"] == TARGET]
    target_display = [
        r
        for r in display
        if r["sequence_id"] == TARGET and json.loads(r["combined_current_alert_tracks"])
    ]
    first_display = target_display[0] if target_display else None
    phase = human_frames[first_display["source_frame_id"]] if first_display else None
    initial = original["metrics"]
    retained = [
        r for r in videos if r["label"] == 1 and r["wide_prediction"] == r["prediction"] == 1
    ]
    normal_retained = [
        r for r in videos if r["label"] == 0 and r["wide_prediction"] == r["prediction"] == 0
    ]
    hits = sum(bool(r["combined_coarse_event_starts_ms"]) for r in timing)
    target_phase = next(r for r in timing if r["sequence_id"] == TARGET)
    lo, hi = lock["legacy_intervals"][TARGET]["fall_phase_ms"]
    guard = {
        "fall19_in_human_transition_with_current_alert": phase is not None
        and phase["phase"] == "transition",
        "fall19_first_event_in_legacy_phase": target["first_alarm_ms"] is not None
        and lo <= target["first_alarm_ms"] <= hi,
        "fall19_current_alert_in_legacy_phase": bool(
            target_phase["combined_current_alert_ms_in_coarse_phase"]
        ),
        "all_14_wide_tp_preserved": len(retained) == 14,
        "all_8_wide_tn_preserved": len(normal_retained) == 8,
        "unknown_coverage_preserved": metrics["n_unknown"] == initial["n_unknown"]
        and metrics["window_coverage"] == initial["window_coverage"],
        "legacy_phase_at_least_4": hits >= 4,
        "no_legacy_normal_early_events": not any(r["combined_coarse_early_ms"] for r in timing),
    }
    first_row = next(
        (
            r
            for r in windows
            if r["sequence_id"] == TARGET
            and r["available_at_ms"] == target["first_alarm_ms"]
            and r["combined_state"] == "suspected_fall"
        ),
        None,
    )
    summary = {
        "id": TAG,
        "wide_metrics": initial,
        "combined_metrics": metrics,
        "normal_threshold": threshold,
        "normal_threshold_videos": [sid for sid, r in normal.items() if r["max_k3"] == threshold],
        "original_wide_threshold": wide_threshold,
        "fall19": {
            "sequence": target,
            "events": target_events,
            "first_event_window": first_row,
            "first_current_alert": first_display,
            "human_phase_at_first_current_alert": phase["phase"] if phase else None,
            "human_pose_at_first_current_alert": phase["pose_check"] if phase else None,
            "max_shoulder_k3": max(
                (
                    r["shoulder_k3"]
                    for r in windows
                    if r["sequence_id"] == TARGET and r["shoulder_k3"] is not None
                ),
                default=None,
            ),
        },
        "legacy_checks": timing,
        "manual_impact_checks": manual,
        "acceptance": guard,
        "accepted_on_this_validation": all(guard.values()),
        "preserved_wide_tp": len(retained),
        "preserved_wide_tn": len(normal_retained),
        "retained_wide_first_event_changes": [
            r["sequence_id"] for r in retained if r["wide_first_alarm_ms"] != r["first_alarm_ms"]
        ],
        "evidence_kind": "real_cached_pose_and_wide_scores_with_causal_shoulder_branch",
        "neural_training_epochs": 0,
        "test_evaluations": 0,
    }
    for name, rows in (
        ("branch-windows", branch_rows),
        ("validation-windows", windows),
        ("validation-events", events),
        ("validation-comparison", videos),
        ("display-grid", display),
    ):
        save_csv(OUT / f"{name}.csv", rows)
    save_json(OUT / "summary.json", summary)
    verify_hashes(lock["source_sha256"], ROOT)
    save_json(
        OUT / "artifact-manifest.json",
        {"files": {p.name: sha(p) for p in OUT.iterdir() if p.is_file()}},
    )
    print(
        json.dumps(
            {
                "output": str(OUT),
                "metrics": metrics,
                "fall19": summary["fall19"],
                "legacy_phase_hits": hits,
                "acceptance": guard,
                "earlier_wide_true_positives": summary["retained_wide_first_event_changes"],
            },
            ensure_ascii=False,
        )
    )


def verify():
    lock = json.loads((OUT / "lock.json").read_text(encoding="utf-8"))
    manifest = json.loads((OUT / "artifact-manifest.json").read_text(encoding="utf-8"))
    verify_hashes(lock["source_sha256"], ROOT)
    verify_hashes(manifest["files"], OUT)
    print(
        f"Verified {len(lock['source_sha256'])} frozen inputs and {len(manifest['files'])} outputs"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("build", "verify"))
    args = parser.parse_args()
    (build if args.stage == "build" else verify)()
