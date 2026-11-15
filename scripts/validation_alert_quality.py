"""Read existing Validation results and poses; never load or run an anomaly model."""

import argparse
import base64
import csv
import hashlib
import json
import math
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

from fall_detect.data.cache import load_pose_cache
from fall_detect.io_utils import ContractError
from fall_detect.skeleton.preprocessing import good_frame
from fall_detect.skeleton.windows import CausalResampler

ROOT = Path(__file__).resolve().parents[1]
TAG = "ur-validation-alert-quality-20261013-001"
OUT = ROOT / "outputs" / TAG
QA = ROOT / "docs/validation" / TAG
WIDE = ROOT / "outputs/isotropic-refinement-20261012/ur-isotropic-wide-20261012-001"
DEMO = ROOT / "outputs/human-isotropic-validation/ur-sample-human-s1-001"
RUN = ROOT / "runs/ur-isotropic-wide-20261012-001"
ANNOTATIONS = ROOT / "docs/annotations/ur-train-validation-20261012-v2"
HUMAN = ROOT / "outputs/ur-fall19-human-amendment-20261013-001/derived-annotation.json"
LEGACY = ROOT / "docs/validation/human-residual-lock.json"
PARENT = ROOT / "docs/validation/ur-fall19-whole-coordinate-20261013-001/final-qa.json"
SELECTED = ("fall-18", "fall-09", "fall-30")
MUTABLE = ("docs/current-progress.md", "docs/acceptance-results.md")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text("utf-8-sig"))


def save(path, value):
    with Path(path).open("x", encoding="utf8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def csv_read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def csv_save(path, rows):
    if not rows:
        raise ContractError("Cannot silently omit an empty report")
    with Path(path).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                    for k, v in row.items()
                }
            )


def number(value):
    if value is None or value == "":
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ContractError("Non-finite number")
    return result


def local_path(name, base=ROOT):
    # Every supplied cache/image path is data, not a command; reject escaping.
    result = (base / Path(name)).resolve()
    if not result.is_relative_to(base.resolve()):
        raise ContractError("Path escapes the allowed source directory")
    return result


def val_sources(index):
    rows = [r for r in index["split_manifest"]["sequences"] if r["split"] == "validation"]
    if len(rows) != 23 or len({r["sequence_id"] for r in rows}) != 23:
        raise ContractError("Expected 23 unique original Validation sequences")
    if Counter(r["label"] for r in rows) != {0: 8, 1: 15}:
        raise ContractError("Unexpected original Validation class denominator")
    for row in rows:
        if not re.fullmatch(r"(adl|fall)-\d{2}", row["sequence_id"]):
            raise ContractError("Unsafe sequence identity")
    return rows


def scoped_cache(index, source, kind):
    if source["split"] != "validation":
        raise ContractError("Only Validation may access a cache in this task")
    entry = index["sequences"][source["sequence_id"]]
    if entry["status"] != "prepared":
        raise ContractError("Existing Validation cache is not prepared")
    path = local_path(entry[f"{kind}_path"])
    if not path.is_relative_to((ROOT / "data/cache/ur-human-isotropic-001").resolve()):
        raise ContractError("Wrong baseline cache namespace")
    meta = read(path.with_suffix(".json"))
    for key, expected in {
        "sequence_id": source["sequence_id"],
        "split": "validation",
        "recipe_hash": index["recipe_hash"],
        "source_hash": entry["source_hash"],
        "split_hash": index["split_manifest"]["split_hash"],
    }.items():
        if meta.get(key) != expected:
            raise ContractError(f"Cache {key} mismatch")
    if meta["array_sha256"] != digest(path):
        raise ContractError("Existing cache checksum mismatch")
    return path, meta


def metrics(rows):
    table = [[0, 0, 0], [0, 0, 0]]
    for r in rows:
        table[int(r["label"])][2 if r["prediction"] is None else int(r["prediction"])] += 1
    tn, fp, un = table[0]
    fn, tp, uf = table[1]
    binary = tp + tn + fp + fn
    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "n_total": len(rows),
        "n_evaluable": binary,
        "n_unknown": un + uf,
        "sequence_accuracy": (tp + tn) / binary if binary else None,
        "operational_accuracy": (tp + tn) / len(rows) if rows else None,
        "sequence_fall_precision": tp / (tp + fp) if tp + fp else None,
        "sequence_fall_recall": tp / (tp + fn) if tp + fn else None,
        "end_to_end_fall_recall": tp / (tp + fn + uf) if tp + fn + uf else None,
        "sequence_fall_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "outcomes_with_unknown": table,
        "denominators": {
            "binary_accuracy": binary,
            "overall_accuracy": len(rows),
            "precision": tp + fp,
            "binary_recall": tp + fn,
            "overall_recall": tp + fn + uf,
            "binary_f1": 2 * tp + fp + fn,
            "overall_f1_unknown_as_miss": 2 * tp + fp + fn + uf,
        },
        "overall_f1_unknown_as_miss": 2 * tp / (2 * tp + fp + fn + uf)
        if 2 * tp + fp + fn + uf
        else None,
    }


def parse_windows(rows):
    out = []
    for r in rows:
        out.append(
            {
                **r,
                "track_id": int(r["track_id"]) if r["track_id"] else None,
                "sample_index": int(r["sample_index"]),
                "window_id": int(r["window_id"]),
                **{
                    k: number(r[k])
                    for k in ("start_ms", "end_ms", "available_at_ms", "raw_score", "event_score")
                },
                "reason": r["reason"] or None,
            }
        )
    return out


def audit_events(windows, threshold, end_ms, stride=1, k=3):
    """Independent replay of saved raw scores; no use of production EventState."""
    histories, previous, active, events = {}, {}, {}, []
    values = []

    def close(track, reason, time):
        if track in active:
            events.append({**active.pop(track), "end_ms": time, "end_reason": reason})

    for w in windows:
        t, s, time, raw = w["track_id"], w["sample_index"], w["available_at_ms"], w["raw_score"]
        if w["start_ms"] > w["end_ms"] or w["end_ms"] > time:
            raise ContractError("Noncausal saved window")
        h = histories.setdefault(t, [])
        if t in previous:
            if s <= previous[t]:
                raise ContractError("Duplicate or decreasing track endpoint")
            if s != previous[t] + stride:
                close(t, "time_gap", time)
                h.clear()
        previous[t] = s
        if raw is None:
            close(t, w["reason"] or "unknown", time)
            h.clear()
            score, state = None, "unknown"
        else:
            h.append(raw)
            del h[:-k]
            score = min(h) if len(h) == k else None
            state = (
                "warming_up"
                if score is None
                else "suspected_fall"
                if score > threshold
                else "normal"
            )
            if score is not None:
                values.append(score)
            if state == "suspected_fall":
                if t not in active:
                    active[t] = {
                        "sequence_id": w["sequence_id"],
                        "track_id": t,
                        "confirmed_start_ms": time,
                        "last_active_ms": time,
                        "peak_score": raw,
                    }
                else:
                    active[t]["last_active_ms"] = time
                    active[t]["peak_score"] = max(active[t]["peak_score"], raw)
            else:
                close(t, "low_score", time)
        if score != w["event_score"] or state != w["state"]:
            raise ContractError("Saved K3 state/score differs from independent replay")
    for t in histories:
        close(t, "sequence_end", end_ms)
    return max(values) if values else None, events


def in_phase(time, interval):
    return time is not None and interval is not None and interval[0] <= time <= interval[1]


def pose_records(observations, cfg):
    frames, samples = [], {}
    resampler = CausalResampler(
        cfg["skeleton"]["target_fps"],
        cfg["skeleton"]["resampling_hold_policy"],
        cfg["skeleton"]["timestamp_tolerance_ms"],
    )
    for frame, poses in observations:
        for p in poses or [None]:
            confident = (
                p.keypoints[:, 2] >= cfg["quality"]["min_joint_confidence"]
                if p is not None
                else np.zeros(13, bool)
            )
            inside = (
                (
                    (p.keypoints[:, 0] >= 0)
                    & (p.keypoints[:, 0] < frame.width)
                    & (p.keypoints[:, 1] >= 0)
                    & (p.keypoints[:, 1] < frame.height)
                )
                if p is not None
                else np.zeros(13, bool)
            )
            frames.append(
                {
                    "sequence_id": frame.sequence_id,
                    "split": "validation",
                    "source_frame_id": frame.source_frame_id,
                    "timestamp_ms": frame.timestamp_ms,
                    "timestamp_estimated": frame.timestamp_estimated,
                    "source_missing": frame.source_missing,
                    "track_id": p.track_id if p is not None else None,
                    "has_pose": p is not None,
                    "stored_pose_quality": p.quality if p is not None else "missing",
                    "confidence_good": bool(good_frame(p.keypoints, cfg["quality"]))
                    if p is not None
                    else False,
                    "confident_joints": int(confident.sum()),
                    "confident_out_of_bounds_joints": int((confident & ~inside).sum()),
                }
            )
        for sample in resampler.push(frame, poses):
            samples[sample.sample_index] = {
                "sequence_id": sample.sequence_id,
                "sample_index": sample.sample_index,
                "timestamp_ms": sample.timestamp_ms,
                "source_frame_id": sample.source_frame_id,
                "missing": sample.missing,
                "repeated": sample.repeated,
                "good_tracks": [
                    p.track_id for p in sample.poses if good_frame(p.keypoints, cfg["quality"])
                ],
                "tracks": [p.track_id for p in sample.poses],
            }
    return frames, samples


def validate_review(annotation, bundle):
    if (
        annotation.get("schema_version") != 1
        or annotation.get("bundle_sha256") != bundle["bundle_sha256"]
    ):
        raise ContractError("Review bundle/schema mismatch")
    if (
        annotation.get("annotation_origin") != "human_self_reported"
        or annotation.get("human_attested") is not True
    ):
        raise ContractError("Human self-report attestation is required")
    if not isinstance(annotation.get("reviewer"), str) or not annotation["reviewer"].strip():
        raise ContractError("Reviewer is required")
    rows = annotation.get("clips", [])
    clips = {c["sequence_id"]: c for c in bundle["clips"]}
    if len(rows) != 3 or {c.get("sequence_id") for c in rows} != set(clips):
        raise ContractError("Exactly the three selected clips are required")
    for row in rows:
        clip = clips[row["sequence_id"]]
        ids = {f["source_frame_id"] for f in clip["frames"]}
        if row.get("reviewed") is not True:
            raise ContractError("Every clip must be reviewed")
        for key in ("onset", "impact"):
            mark = row.get(key, {})
            if mark.get("status") not in {"bounded", "uncertain", "outside_clip"}:
                raise ContractError("Choose bounded/uncertain/outside_clip for each timing mark")
            bounds = mark.get("bounds_frame_ids")
            if mark["status"] == "bounded":
                if (
                    not isinstance(bounds, list)
                    or len(bounds) != 2
                    or any(type(x) is not int or x not in ids for x in bounds)
                    or bounds[0] > bounds[1]
                ):
                    raise ContractError("Bounds must be two ordered observed source frame IDs")
            elif bounds is not None:
                raise ContractError("Uncertain/outside-clip timing must not invent coordinates")
        if (
            row["onset"]["status"] == row["impact"]["status"] == "bounded"
            and row["onset"]["bounds_frame_ids"][0] > row["impact"]["bounds_frame_ids"][1]
        ):
            raise ContractError("Impact cannot definitely precede onset")
        if row.get("skeleton_quality") not in {
            "mostly_matches",
            "clear_wrong",
            "occluded_or_cropped",
            "uncertain",
            "missing",
        }:
            raise ContractError("Explicit overall skeleton review is required")
    return {
        "valid": True,
        "clips": len(rows),
        "annotation_origin": "human_self_reported",
        "note": "Self-reported human review; not independent multi-reviewer temporal ground truth",
    }


def build():
    if OUT.exists() or QA.exists():
        raise ContractError("Preserve completed outputs/QA; build refuses an existing directory")
    # Source contracts are snapshotted BEFORE any mutable progress documentation.
    previous = read(PARENT)["artifact_sha256"]
    for name, sha in previous.items():
        if digest(local_path(name)) != sha:
            raise ContractError(f"Previous finalized artifact changed: {name}")
    QA.mkdir(parents=True)
    OUT.mkdir(parents=True)
    (QA / "prior-contracts").mkdir()
    for name in MUTABLE:
        p = QA / "prior-contracts" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("xb") as stream:
            stream.write((ROOT / name).read_bytes())
    save(
        QA / "previous-stage.json",
        {"source": str(PARENT.relative_to(ROOT)), "sha256": digest(PARENT), "files": previous},
    )
    sources = {}

    def pin(p):
        sources[str(p.relative_to(ROOT))] = digest(p)

    pin(ROOT / "docs/ur-validation-alert-quality-20261013-plan.md")
    for p in (
        RUN / "dataset-index.json",
        RUN / "resolved-config.yaml",
        RUN / "threshold.json",
        RUN / "autoencoder.pt",
        RUN / "scaler.npz",
        HUMAN,
        LEGACY,
        ANNOTATIONS / "fall-intervals.json",
        ANNOTATIONS / "manifest.json",
    ):
        pin(p)
    for folder in (WIDE, DEMO):
        for name in ("sequences.csv", "metrics.json"):
            pin(folder / name)
    for name in ("windows.csv", "events.csv"):
        pin(WIDE / name)
    index, threshold = read(RUN / "dataset-index.json"), read(RUN / "threshold.json")["threshold"]
    cfg = yaml.safe_load((RUN / "resolved-config.yaml").read_text("utf8"))
    if (
        cfg["skeleton"]["window_size"] != 30
        or cfg["skeleton"]["coordinate_normalization"] != "per_frame_isotropic"
        or cfg["calibration"]["consecutive_windows"] != 3
    ):
        raise ContractError("Wrong reference version")
    source_rows = val_sources(index)
    ids = {r["sequence_id"] for r in source_rows}
    seq = csv_read(WIDE / "sequences.csv")
    demo = csv_read(DEMO / "sequences.csv")
    windows = parse_windows(csv_read(WIDE / "windows.csv"))
    events = csv_read(WIDE / "events.csv")
    for rows in (seq, demo):
        if len(rows) != 23 or {r["sequence_id"] for r in rows} != ids:
            raise ContractError("Sequence CSV is not exactly the original Validation split")
        for r in rows:
            r["label"] = int(r["label"])
            r["prediction"] = int(r["prediction"]) if r["prediction"] else None
    if not all(w["sequence_id"] in ids for w in windows + events):
        raise ContractError("Saved windows/events include a non-Validation sequence")
    metric, demo_metric = metrics(seq), metrics(demo)
    for folder, recomputed in ((WIDE, metric), (DEMO, demo_metric)):
        formal = read(folder / "metrics.json")
        for key in formal:
            if key in recomputed and formal[key] != recomputed[key]:
                raise ContractError(f"Saved metric mismatch: {folder.name}/{key}")
    ai_phase = {
        r["sequence_id"]: r
        for r in read(ANNOTATIONS / "fall-intervals.json")["sequences"]
        if r["split"] == "validation"
    }
    annotation_summary = {
        r["sequence_id"]: r
        for r in read(ANNOTATIONS / "manifest.json")["summary"]
        if r["split"] == "validation"
    }
    legacy = read(LEGACY)["coarse_intervals"]
    human = read(HUMAN)
    human_frames = {r["source_frame_id"]: r for r in human["frames"]}
    by_sid, demo_sid = {r["sequence_id"]: r for r in seq}, {r["sequence_id"]: r for r in demo}
    totals, all_frames, all_samples, enriched_windows, enriched_events = [], [], [], [], []
    observations_for_review = {}
    phase_counts = Counter()
    for source in source_rows:
        sid = source["sequence_id"]
        row = by_sid[sid]
        if row["label"] != source["label"] or demo_sid[sid]["label"] != source["label"]:
            raise ContractError("Label differs from original split manifest")
        pose_path, pose_meta = scoped_cache(index, source, "pose")
        feature_path, feature_meta = scoped_cache(index, source, "feature")
        for p in (
            pose_path,
            pose_path.with_suffix(".json"),
            feature_path,
            feature_path.with_suffix(".json"),
        ):
            pin(p)
        observations, _ = load_pose_cache(
            pose_path,
            {"sequence_id": sid, "split": "validation", "recipe_hash": index["recipe_hash"]},
        )
        fr, samples = pose_records(observations, cfg)
        all_frames.extend(fr)
        all_samples.extend(samples.values())
        ws = [w for w in windows if w["sequence_id"] == sid]
        scheduled = {w["sample_index"] for w in ws}
        valid = {w["sample_index"] for w in ws if w["raw_score"] is not None}
        if (
            len(scheduled) != int(row["n_scheduled_endpoints"])
            or len(valid) != int(row["n_valid_endpoints"])
            or len(valid) / len(scheduled) != number(row["window_coverage"])
        ):
            raise ContractError("Unique-endpoint denominator mismatch")
        records = feature_meta["records"]
        if len(records) != len(ws) or len({(w["sample_index"], w["track_id"]) for w in ws}) != len(
            ws
        ):
            raise ContractError("Metadata/window count or uniqueness mismatch")
        saved = {(r["sample_index"], r["track_id"]): r for r in records}
        for w in ws:
            rec = saved[(w["sample_index"], w["track_id"])]
            if (rec["row_index"] is not None) != (w["raw_score"] is not None):
                raise ContractError("Legal/unknown window differs from feature metadata")
            for key in ("start_ms", "end_ms", "available_at_ms", "window_id"):
                if rec[key] != w[key]:
                    raise ContractError("Saved timestamps/window identity differ")
            sample = samples[w["sample_index"]]
            good = w["track_id"] in sample["good_tracks"]
            enriched_windows.append(
                {
                    **w,
                    "source_frame_id_at_tick": sample["source_frame_id"],
                    "current_same_track_confidence_good": good,
                    "alert_with_current_good_pose": w["state"] == "suspected_fall" and good,
                    "human_fall19_phase": human_frames.get(sample["source_frame_id"], {}).get(
                        "phase"
                    )
                    if sid == "fall-19"
                    else None,
                }
            )
        score, ev = audit_events(ws, threshold, feature_meta["end_ms"])
        if (
            score != number(row["sequence_score"])
            or (None if score is None else int(score > threshold)) != row["prediction"]
        ):
            raise ContractError("Saved sequence prediction/score mismatch")
        old_ev = [e for e in events if e["sequence_id"] == sid]
        converted = [
            {
                **e,
                "track_id": int(e["track_id"]),
                **{
                    k: number(e[k])
                    for k in ("confirmed_start_ms", "last_active_ms", "peak_score", "end_ms")
                },
            }
            for e in old_ev
        ]
        if ev != converted or (ev[0]["confirmed_start_ms"] if ev else None) != number(
            row["first_alarm_ms"]
        ):
            raise ContractError("Saved event/first alarm mismatch")
        phase = ai_phase[sid]
        possible, core = phase["possible_descent_ms"], phase["core_descent_ms"]
        alerts = [
            w
            for w in enriched_windows
            if w["sequence_id"] == sid and w["state"] == "suspected_fall"
        ]
        good_alerts = [w for w in alerts if w["current_same_track_confidence_good"]]
        possible_good = [w for w in good_alerts if in_phase(w["available_at_ms"], possible)]
        starts = [e["confirmed_start_ms"] for e in ev]
        first = starts[0] if starts else None
        relation = (
            "not_applicable_ADL"
            if not row["label"]
            else "no_event"
            if first is None
            else "before_AI_possible"
            if first < possible[0]
            else "after_AI_possible"
            if first > possible[1]
            else "inside_AI_core"
            if in_phase(first, core)
            else "inside_AI_uncertainty_margin"
        )
        phase_counts[relation] += 1
        for e in ev:
            active_ws = [
                w
                for w in alerts
                if w["track_id"] == e["track_id"]
                and e["confirmed_start_ms"] <= w["available_at_ms"] <= e["last_active_ms"]
            ]
            enriched_events.append(
                {
                    **e,
                    "active_endpoint_count": len(active_ws),
                    "current_good_active_endpoint_count": sum(
                        w["current_same_track_confidence_good"] for w in active_ws
                    ),
                    "first_source_frame_id_at_tick": samples[active_ws[0]["sample_index"]][
                        "source_frame_id"
                    ],
                    "AI_possible_phase_hit": in_phase(e["confirmed_start_ms"], possible),
                    "AI_core_phase_hit": in_phase(e["confirmed_start_ms"], core),
                    "human_exact_timing_confirmed": False,
                }
            )
        any_pose = {r["source_frame_id"] for r in fr if r["has_pose"]}
        any_good = {r["source_frame_id"] for r in fr if r["confidence_good"]}
        outcome = (
            "unknown"
            if row["prediction"] is None
            else "TP"
            if row["label"] == row["prediction"] == 1
            else "TN"
            if row["label"] == row["prediction"] == 0
            else "FN"
            if row["label"]
            else "FP"
        )
        legacy_phase = legacy.get(sid, {}).get("fall_phase_ms")
        legacy_good = (
            [w for w in good_alerts if in_phase(w["available_at_ms"], legacy_phase)]
            if legacy_phase
            else None
        )
        totals.append(
            {
                "sequence_id": sid,
                "split": "validation",
                "label": row["label"],
                "wide_prediction": row["prediction"],
                "wide_outcome": outcome,
                "wide_sequence_K3_max": score,
                "wide_threshold": threshold,
                "wide_raw_max": max(
                    (w["raw_score"] for w in ws if w["raw_score"] is not None), default=None
                ),
                "demo_prediction": demo_sid[sid]["prediction"],
                "demo_first_alarm_ms": number(demo_sid[sid]["first_alarm_ms"]),
                "scheduled_unique_endpoints": len(scheduled),
                "valid_unique_endpoints": len(valid),
                "window_coverage": len(valid) / len(scheduled),
                "window_track_records": len(ws),
                "unknown_window_records": sum(w["state"] == "unknown" for w in ws),
                "warmup_window_records": sum(w["state"] == "warming_up" for w in ws),
                "window_invalid_reasons": dict(
                    Counter(w["reason"] for w in ws if w["state"] == "unknown")
                ),
                "source_frames": len(observations),
                "frames_any_pose": len(any_pose),
                "frames_no_pose": len(observations) - len(any_pose),
                "frames_any_confidence_good": len(any_good),
                "frames_pose_but_not_good": len(any_pose - any_good),
                "confident_joint_records": sum(r["confident_joints"] for r in fr),
                "confident_out_of_bounds_joint_records": sum(
                    r["confident_out_of_bounds_joints"] for r in fr
                ),
                "n_events": len(ev),
                "first_alarm_ms": first,
                "first_alarm_source_frame_id_at_tick": samples[alerts[0]["sample_index"]][
                    "source_frame_id"
                ]
                if alerts
                else None,
                "last_active_ms": max((e["last_active_ms"] for e in ev), default=None),
                "event_end_ms": max((e["end_ms"] for e in ev), default=None),
                "active_window_endpoints": len(alerts),
                "current_good_active_endpoints": len(good_alerts),
                "AI_possible_phase_ms": possible,
                "AI_core_phase_ms": core,
                "AI_phase_human_reviewed": phase["human_reviewed"],
                "first_alarm_AI_relation": relation,
                "AI_possible_event_starts": sum(in_phase(t, possible) for t in starts),
                "AI_core_event_starts": sum(in_phase(t, core) for t in starts),
                "AI_possible_good_active_endpoints": len(possible_good),
                "AI_possible_legal_unique_endpoints": len(
                    {
                        w["sample_index"]
                        for w in ws
                        if w["raw_score"] is not None and in_phase(w["available_at_ms"], possible)
                    }
                ),
                "legacy6_phase_ms": legacy_phase,
                "legacy6_phase_event_starts": sum(in_phase(t, legacy_phase) for t in starts)
                if legacy_phase
                else None,
                "legacy6_good_active_endpoints": len(legacy_good)
                if legacy_good is not None
                else None,
                "human_exact_onset_impact": "not_annotated" if row["label"] else "not_applicable",
                "human_pose_phase_review": "existing_self_reported_fall19"
                if sid == "fall-19"
                else "not_reviewed",
                "AI_sampled_visibility_region_counts": annotation_summary[sid][
                    "visual_region_counts"
                ],
            }
        )
        if sid in SELECTED:
            observations_for_review[sid] = (observations, pose_meta)
    metric.update(
        {
            "scheduled_unique_endpoints": sum(r["scheduled_unique_endpoints"] for r in totals),
            "valid_unique_endpoints": sum(r["valid_unique_endpoints"] for r in totals),
            "window_track_records": len(windows),
            "events": len(enriched_events),
            "AI_phase_relations": dict(phase_counts),
            "AI_possible_event_hit_videos": sum(r["AI_possible_event_starts"] > 0 for r in totals),
            "AI_core_event_hit_videos": sum(r["AI_core_event_starts"] > 0 for r in totals),
            "AI_possible_current_good_alert_videos": sum(
                r["AI_possible_good_active_endpoints"] > 0 for r in totals
            ),
            "legacy6_phase_hit_videos": sum(
                (r["legacy6_phase_event_starts"] or 0) > 0 for r in totals
            ),
            "legacy6_current_good_alert_videos": sum(
                (r["legacy6_good_active_endpoints"] or 0) > 0 for r in totals
            ),
            "human_exact_timing_videos": 0,
            "new_training": 0,
            "new_inference": 0,
            "new_calibration": 0,
            "test_cache_or_image_reads": 0,
            "new_test_evaluations": 0,
        }
    )
    metric["window_coverage"] = (
        metric["valid_unique_endpoints"] / metric["scheduled_unique_endpoints"]
    )
    if metric["window_coverage"] != read(WIDE / "metrics.json")["window_coverage"]:
        raise ContractError("Global coverage mismatch")
    csv_save(OUT / "validation-23.csv", totals)
    csv_save(OUT / "source-frame-pose-quality.csv", all_frames)
    csv_save(OUT / "causal-sampled-pose-quality.csv", all_samples)
    csv_save(OUT / "windows-with-quality.csv", enriched_windows)
    csv_save(OUT / "events-with-quality.csv", enriched_events)
    save(OUT / "metrics-rechecked.json", {"wide": metric, "demo_existing_validation": demo_metric})
    human_phase = {}
    for phase in ("upright", "transition", "ground"):
        selected = [
            w
            for w in enriched_windows
            if w["sequence_id"] == "fall-19" and w["human_fall19_phase"] == phase
        ]
        human_phase[phase] = {
            "scheduled_unique_endpoints": len({w["sample_index"] for w in selected}),
            "valid_unique_endpoints": len(
                {w["sample_index"] for w in selected if w["raw_score"] is not None}
            ),
            "event_active_endpoints": sum(w["state"] == "suspected_fall" for w in selected),
        }
    save(
        OUT / "fall19-existing-human-phase.json",
        {
            "source_sha256": digest(HUMAN),
            "phase_semantics": "whole_posture_phases_not_exact_onset_or_impact",
            "causal_endpoint_counts": human_phase,
        },
    )
    bundle = {
        "schema_version": 1,
        "id": TAG,
        "split": "validation",
        "annotation_origin": "pending_human_self_reported",
        "clips": [],
    }
    for sid in SELECTED:
        observations, meta = observations_for_review[sid]
        original_root = ROOT / "data/raw/urfall" / sid
        identities = {
            f["path"].replace("\\", "/"): f["sha256"] for f in meta["source_identity"]["files"]
        }
        # Short complete clips for 18/30; last pre-descent context through sequence end for 09.
        first_id = 103 if sid == "fall-09" else 1
        clip = {"sequence_id": sid, "split": "validation", "frames": []}
        for frame, poses in observations:
            if frame.source_frame_id < first_id:
                continue
            relative = f"{sid}-cam0-rgb/{sid}-cam0-rgb-{frame.source_frame_id:03d}.png"
            image = local_path(relative, original_root)
            sha = digest(image)
            if identities.get(relative) != sha:
                raise ContractError("Original Validation image differs from cached source identity")
            pin(image)
            clip["frames"].append(
                {
                    "source_frame_id": frame.source_frame_id,
                    "timestamp_ms": frame.timestamp_ms,
                    "width": frame.width,
                    "height": frame.height,
                    "source_sha256": sha,
                    "image": "data:image/png;base64,"
                    + base64.b64encode(image.read_bytes()).decode("ascii"),
                    "poses": [
                        {"track_id": p.track_id, "keypoints": p.keypoints.tolist()} for p in poses
                    ],
                }
            )
        bundle["clips"].append(clip)
    bundle["bundle_sha256"] = hashlib.sha256(
        json.dumps(bundle, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf8")
    ).hexdigest()
    save(OUT / "review-bundle.json", bundle)
    payload = json.dumps(bundle, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    template = ROOT / "scripts/templates/validation-alert-review.html"
    pin(template)
    with (OUT / "review.html").open("x", encoding="utf8") as stream:
        stream.write(template.read_text("utf8").replace("__BUNDLE__", payload))
    for p in (Path(__file__), ROOT / "tests/test_validation_alert_quality.py"):
        pin(p)
    save(
        QA / "source-manifest.json",
        {
            "id": TAG,
            "created_at": datetime.now(UTC).isoformat(),
            "baseline_recipe": index["recipe_hash"],
            "split_hash": index["split_manifest"]["split_hash"],
            "allowed_sequences": sorted(ids),
            "files": sources,
            "no_feature_arrays_loaded": True,
            "no_models_loaded": True,
            "raw_image_sequences": list(SELECTED),
            "test_access": "metadata_in_original_index_only_no_test_artifact_access",
        },
    )
    save(
        OUT / "review-selection.json",
        {
            "selected": list(SELECTED),
            "post_hoc_not_blind": True,
            "tasks_per_clip": "one_onset_interval_one_impact_interval_one_overall_pose_quality",
            "human_status": "pending",
            "human_exact_times_prefilled": False,
            "clips": [
                {
                    "sequence_id": c["sequence_id"],
                    "source_frame_ids": [
                        c["frames"][0]["source_frame_id"],
                        c["frames"][-1]["source_frame_id"],
                    ],
                    "timestamp_ms": [
                        c["frames"][0]["timestamp_ms"],
                        c["frames"][-1]["timestamp_ms"],
                    ],
                }
                for c in bundle["clips"]
            ],
        },
    )
    print(
        json.dumps(
            {"completed": True, "metrics": metric, "review": str(OUT / "review.html")},
            ensure_ascii=False,
        )
    )


def verify():
    manifest = read(QA / "source-manifest.json")
    for name, sha in manifest["files"].items():
        if digest(local_path(name)) != sha:
            raise ContractError(f"Pinned source changed: {name}")
    for name, sha in read(QA / "previous-stage.json")["files"].items():
        p = QA / "prior-contracts" / name if name in MUTABLE else local_path(name)
        if digest(p) != sha:
            raise ContractError(f"Previous artifact changed: {name}")
    summary = read(OUT / "metrics-rechecked.json")
    original = csv_read(WIDE / "sequences.csv")
    for r in original:
        r["prediction"] = int(r["prediction"]) if r["prediction"] else None
    for k, v in metrics(original).items():
        if summary["wide"][k] != v:
            raise ContractError(f"Reported metric changed: {k}")
    if (OUT / "artifact-manifest.json").exists():
        for name, sha in read(OUT / "artifact-manifest.json")["files"].items():
            if digest(local_path(name, OUT)) != sha:
                raise ContractError(f"New output changed: {name}")
    print(
        json.dumps(
            {
                "verified": True,
                "source_hashes": len(manifest["files"]),
                "previous_finalized_hashes": len(read(QA / "previous-stage.json")["files"]),
                "metrics": summary["wide"],
                "no_inference": True,
            },
            ensure_ascii=False,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("build", "verify", "serve", "validate-review"))
    parser.add_argument("--port", type=int, default=8830)
    parser.add_argument("--annotation", type=Path)
    args = parser.parse_args()
    if args.stage == "build":
        build()
    elif args.stage == "verify":
        verify()
    elif args.stage == "validate-review":
        if args.annotation is None:
            parser.error("--annotation is required")
        print(
            json.dumps(
                validate_review(read(args.annotation), read(OUT / "review-bundle.json")),
                ensure_ascii=False,
            )
        )
    else:
        from functools import partial
        from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

        verify()
        handler = partial(SimpleHTTPRequestHandler, directory=str(OUT))
        with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
            print(f"http://127.0.0.1:{args.port}/review.html", flush=True)
            server.serve_forever()


if __name__ == "__main__":
    main()
