"""Descriptive human-label / frozen-window comparisons; no fitting or calibration."""

import math
from bisect import bisect_right
from collections import Counter
from itertools import groupby, pairwise

from ..io_utils import ContractError


def runs(frames, key):
    result = []
    for value, group in groupby(frames, key=lambda row: row[key]):
        rows = list(group)
        result.append(
            {
                "label": value,
                "first_frame": rows[0]["source_frame_id"],
                "last_frame": rows[-1]["source_frame_id"],
                "frames": len(rows),
                "first_ms": rows[0]["timestamp_ms"],
                "last_ms": rows[-1]["timestamp_ms"],
            }
        )
    return result


def diagnose(annotation, data):
    """Describe labels already validated against the source manifest.

    Window labels refer to the last actual source observation at/before its
    endpoint, not a future RGB frame or every observation inside that window.
    """
    if (
        annotation.get("split") != "validation"
        or annotation.get("annotation_origin") != "human_self_reported"
    ):
        raise ContractError("Only validated self-reported Validation diagnosis")
    source = data["frames"]
    if not source or len(source) != len(annotation["frames"]):
        raise ContractError("Aligned full original frames required")
    for actual, label in zip(source, annotation["frames"]):
        if any(actual[k] != label[k] for k in ["source_frame_id", "timestamp_ms", "source_sha256"]):
            raise ContractError("Human labels must match frozen source identities")
    times = [r["timestamp_ms"] for r in source]
    if any(not math.isfinite(t) for t in times) or any(b <= a for a, b in pairwise(times)):
        raise ContractError("Strictly increasing finite source times required")
    frame_rows, warnings = [], []
    for actual, label in zip(source, annotation["frames"]):
        has_pose = bool(actual["poses"])
        good = any(p["current_good_pose"] for p in actual["poses"])
        if not has_pose and label["pose_check"] == "matches_person":
            warnings.append(
                {
                    "code": "matches_person_without_cached_pose",
                    "source_frame_id": actual["source_frame_id"],
                }
            )
        if has_pose and label["pose_check"] == "missing":
            warnings.append(
                {
                    "code": "missing_label_with_cached_pose",
                    "source_frame_id": actual["source_frame_id"],
                }
            )
        frame_rows.append(
            {
                **label,
                "cached_pose_present": has_pose,
                "cached_current_good": good,
                "human_wrong_but_confidence_good": good and label["pose_check"] == "wrong_or_drift",
            }
        )
    label_runs = {
        key: runs(annotation["frames"], key) for key in ["phase", "visibility", "pose_check"]
    }
    transition = [r for r in label_runs["phase"] if r["label"] == "transition"]
    ground = [r for r in label_runs["phase"] if r["label"] == "ground"]
    if (
        len(transition) == 1
        and len(ground) == 1
        and annotation["onset_bounds_frame_ids"]
        == [transition[0]["first_frame"], transition[0]["last_frame"]]
        and annotation["impact_bounds_frame_ids"]
        == [ground[0]["first_frame"], ground[0]["last_frame"]]
    ):
        warnings.append(
            {"code": "boundary_fields_equal_whole_phase_runs", "requires_human_clarification": True}
        )
    window_rows, summaries = [], {}
    for name, rows in data["windows"].items():
        groups = {}
        for row in rows:
            idx = bisect_right(times, row["available_at_ms"]) - 1
            if idx < 0:
                raise ContractError("No actual past source for endpoint")
            actual, label = source[idx], annotation["frames"][idx]
            matching = [p for p in actual["poses"] if p["track_id"] == row["track_id"]]
            good = any(p["current_good_pose"] for p in matching)
            threshold = data["models"][name]["threshold"]
            r = {
                "method": name,
                **row,
                "last_actual_source_frame_id": actual["source_frame_id"],
                "last_actual_source_ms": actual["timestamp_ms"],
                "human_phase_at_last_source": label["phase"],
                "human_visibility_at_last_source": label["visibility"],
                "human_pose_check_at_last_source": label["pose_check"],
                "current_track_pose_present": bool(matching),
                "current_track_good": good,
                "threshold": threshold,
                "above_raw_threshold": row["raw_score"] is not None
                and row["raw_score"] > threshold,
                "above_k3_threshold": row["event_score"] is not None
                and row["event_score"] > threshold,
            }
            window_rows.append(r)
            groups.setdefault(label["phase"], []).append(r)
        result = {}
        for phase, group in groups.items():
            valid = [r for r in group if r["quality"] == "valid"]
            matches = [r for r in valid if r["human_pose_check_at_last_source"] == "matches_person"]

            def maximum(values, key):
                finite = [r[key] for r in values if r[key] is not None]
                return max(finite) if finite else None

            result[phase] = {
                "scheduled": len(group),
                "valid": len(valid),
                "coverage": len(valid) / len(group),
                "quality_counts": dict(Counter(r["quality"] for r in group)),
                "legal_last_source_pose_labels": dict(
                    Counter(r["human_pose_check_at_last_source"] for r in valid)
                ),
                "legal_last_source_confidence_good": sum(r["current_track_good"] for r in valid),
                "max_raw": maximum(valid, "raw_score"),
                "max_k3": maximum(valid, "event_score"),
                "human_matches_last_source_max_raw": maximum(matches, "raw_score"),
                "human_matches_last_source_max_k3": maximum(matches, "event_score"),
                "raw_crossings": sum(r["above_raw_threshold"] for r in group),
                "k3_crossings": sum(r["above_k3_threshold"] for r in group),
                "threshold": data["models"][name]["threshold"],
            }
        summaries[name] = result
    return {
        "origin": "human_self_reported",
        "reviewed_frames": len(frame_rows),
        "label_runs": label_runs,
        "semantic_warnings": warnings,
        "wrong_but_confidence_good_frames": [
            r["source_frame_id"] for r in frame_rows if r["human_wrong_but_confidence_good"]
        ],
        "methods": summaries,
        "endpoint_label_scope": "Last actual source observation only, not whole-window pose accuracy",
        "precise_onset_latency_computed": False,
        "formal_time_accuracy_computed": False,
        "used_for_training": False,
        "used_for_calibration": False,
        "frame_rows": frame_rows,
        "window_rows": window_rows,
    }
