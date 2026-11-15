"""Paired observed-pose diagnostics. Original human labels do not label a new pose."""

from bisect import bisect_right
from copy import deepcopy
from itertools import pairwise

from ..io_utils import ContractError
from ..skeleton.preprocessing import good_frame


def resolution_config(base, cache_dir, manifest_dir, weights_manifest):
    result = deepcopy(base)
    result["pose"].update(input_height=512, input_width=384, weights_manifest=weights_manifest)
    result["data"].update(cache_dir=cache_dir, manifest_dir=manifest_dir)
    assert_resolution_only(base, result)
    return result


def assert_resolution_only(base, candidate):
    expected = deepcopy(base)
    expected["pose"].update(
        input_height=512,
        input_width=384,
        weights_manifest=candidate["pose"]["weights_manifest"],
    )
    expected["data"].update(
        cache_dir=candidate["data"]["cache_dir"], manifest_dir=candidate["data"]["manifest_dir"]
    )
    if (base["pose"]["input_height"], base["pose"]["input_width"]) != (
        256,
        192,
    ) or candidate != expected:
        raise ContractError("Only fixed 512x384 Pose crop and isolated output paths may change")


def scoped_source(index, sid):
    source = next(
        (s for s in index["split_manifest"]["sequences"] if s["sequence_id"] == sid), None
    )
    if source is None or source["split"] not in ("train", "validation"):
        raise ContractError("This experiment permits only Train/Validation source access")
    if source["split"] == "train" and source["label"] != 0:
        raise ContractError("Original Train must be ADL only")
    return source


def paired_pose_rows(old, new, quality, original_labels=None):
    if len(old) != len(new):
        raise ContractError("Compare every original observation without dropping missing frames")
    labels = {r["source_frame_id"]: r for r in original_labels or []}
    rows = []
    for (a, ap), (b, bp) in zip(old, new):
        if any(
            getattr(a, k) != getattr(b, k)
            for k in [
                "sequence_id",
                "source_frame_id",
                "timestamp_ms",
                "width",
                "height",
                "source_missing",
            ]
        ):
            raise ContractError("Paired observations must share exact original source/time/size")
        label = labels.get(a.source_frame_id)
        if label is not None and label["timestamp_ms"] != a.timestamp_ms:
            raise ContractError("Original human labels must match source time")
        rows.append(
            {
                "sequence_id": a.sequence_id,
                "source_frame_id": a.source_frame_id,
                "timestamp_ms": a.timestamp_ms,
                "old_pose_count": len(ap),
                "new_pose_count": len(bp),
                "old_good": any(good_frame(p.keypoints, quality) for p in ap),
                "new_good": any(good_frame(p.keypoints, quality) for p in bp),
                "old_track_ids": [p.track_id for p in ap],
                "new_track_ids": [p.track_id for p in bp],
                "human_phase": label["phase"] if label else None,
                "human_original_pose_check": label["pose_check"] if label else None,
                "human_new_pose_check": None,
            }
        )
    return rows


def human_phase_summary(windows, events, observations, labels, quality):
    """Phase membership uses the last actual past source; never precise event latency."""
    times = [f.timestamp_ms for f, _ in observations]
    if not times or any(b <= a for a, b in pairwise(times)):
        raise ContractError("Increasing source times required")
    label_map = {r["source_frame_id"]: r for r in labels}
    groups = {}
    for w in windows:
        i = bisect_right(times, w["available_at_ms"]) - 1
        if i < 0:
            raise ContractError("No actual past source for phase window")
        frame, poses = observations[i]
        label = label_map[frame.source_frame_id]
        if label["timestamp_ms"] != frame.timestamp_ms:
            raise ContractError("Phase label source/time mismatch")
        good = any(p.track_id == w["track_id"] and good_frame(p.keypoints, quality) for p in poses)
        groups.setdefault(label["phase"], []).append({**w, "current_good": good})
    result = {}
    for phase, rows in groups.items():
        endpoints = {w["sample_index"] for w in rows}
        valid = {w["sample_index"] for w in rows if w["raw_score"] is not None}
        starts = []
        for event in events:
            i = bisect_right(times, event["confirmed_start_ms"]) - 1
            if i >= 0 and label_map[observations[i][0].source_frame_id]["phase"] == phase:
                starts.append(event["confirmed_start_ms"])
        raw = [w["raw_score"] for w in rows if w["raw_score"] is not None]
        k3 = [w["event_score"] for w in rows if w["event_score"] is not None]
        result[phase] = {
            "scheduled_endpoints": len(endpoints),
            "valid_endpoints": len(valid),
            "coverage": len(valid) / len(endpoints),
            "max_raw": max(raw) if raw else None,
            "max_k3": max(k3) if k3 else None,
            "event_starts": starts,
            "good_alarm_endpoints": sum(
                w["state"] == "suspected_fall" and w["current_good"] for w in rows
            ),
        }
    return result
