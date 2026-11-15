"""Replay saved real window scores through the unmodified production display gates."""

import math
from collections import deque

import numpy as np

from ..detection.events import EventState
from ..io_utils import ContractError
from ..recipes import recipe_spec
from ..skeleton.preprocessing import good_frame
from .action_status import ACTION_LABELS, SCORE_COLUMNS
from .sample_display import EventHistory, unavailable


class CachedDisplayReplay:
    """The state/gate order of video.demo, with score lookup replacing model execution."""

    def __init__(self, config, threshold):
        self.config = config
        self.spec = recipe_spec(config)
        self.threshold = threshold
        self.events = EventState(threshold, 3, config["skeleton"]["stride"])
        self.history = EventHistory()
        self.statuses = {}
        self.max_age = (
            config["demo"]["score_hold_stride_multiplier"]
            * config["skeleton"]["stride"]
            * 1000
            / config["skeleton"]["target_fps"]
        )

    def step(self, sample, saved_windows):
        for w in saved_windows:
            if w["sequence_id"] != sample.sequence_id or w["sample_index"] != sample.sample_index:
                raise ContractError("Saved window does not belong to the current sample")
            if w["available_at_ms"] > sample.timestamp_ms + 1e-7:
                raise ContractError("A future score cannot be displayed")
            status = self.events.update(
                w["sequence_id"],
                w["track_id"],
                w["sample_index"],
                w["available_at_ms"],
                w["raw_score"],
                w["reason"] if w["raw_score"] is None else None,
            )
            for key in ("raw_score", "event_score", "state", "reason"):
                if status[key] != w[key]:
                    raise ContractError(f"Saved {key} differs from K3 replay")
            self.statuses[w["track_id"]] = {**status, "available_at_ms": w["available_at_ms"]}
            self.history.observe(w["track_id"], w["available_at_ms"], status)
        raw = {key: dict(value) for key, value in self.statuses.items()}
        present = {p.track_id for p in sample.poses}
        for track in list(self.statuses):
            if track not in present:
                self.statuses[track] = {
                    "state": "unknown",
                    "raw_score": None,
                    "event_score": None,
                    "available_at_ms": sample.timestamp_ms,
                    "reason": "track_gap",
                }
        visible = {}
        for pose in sample.poses:
            current = self.statuses.get(
                pose.track_id,
                {
                    "state": "warming_up",
                    "raw_score": None,
                    "event_score": None,
                    "available_at_ms": sample.timestamp_ms,
                    "reason": "collecting_window",
                },
            )
            age = sample.timestamp_ms - current["available_at_ms"]
            current = {**current, "score_age_ms": age}
            good = good_frame(
                pose.keypoints,
                self.config["quality"],
                self.spec["joints"],
                self.spec.get("derived_joints", ()),
            )
            if not good or age > self.max_age:
                current = {
                    **current,
                    "state": "unknown",
                    "raw_score": None,
                    "event_score": None,
                    "reason": "low_confidence" if not good else "stale_score",
                }
                self.statuses[pose.track_id] = current
            visible[pose.track_id] = current
        return raw, visible


def parse_cached_actions(rows, samples, observations, config):
    """Verify and reuse original per-observation classification; never call a classifier."""
    histories, snapshots = {}, {}
    max_age = config["pose"]["track_max_age"]
    max_gap = max_age * 1000 / config["skeleton"]["target_fps"]
    for frame, poses in observations:
        for track, h in list(histories.items()):
            if frame.sample_index - h[-1][0] > max_age or frame.timestamp_ms - h[-1][2] > max_gap:
                del histories[track]
        snapshot = {}
        for p in poses:
            h = histories.setdefault(p.track_id, deque(maxlen=30))
            h.append((frame.sample_index, frame.source_frame_id, frame.timestamp_ms))
            snapshot[p.track_id] = tuple(h)
        snapshots[frame.source_frame_id] = snapshot
    grouped = {}
    for row in rows:
        index = int(row["sample_index"])
        if index not in samples or float(row["timestamp_ms"]) != samples[index].timestamp_ms:
            raise ContractError("Cached action sample timestamp mismatch")
        sample = samples[index]
        fid = int(row["source_frame_id"]) if row["source_frame_id"] else None
        track = int(row["track_id"]) if row["track_id"] else None
        if fid != sample.source_frame_id:
            raise ContractError("Cached action source frame mismatch")
        poses = {p.track_id: p for p in sample.poses}
        target = grouped.setdefault(index, {})
        if track is None:
            if poses:
                raise ContractError("Cached action omitted a present track")
            continue
        if track in target or track not in poses:
            raise ContractError("Duplicate or non-present cached action track")
        observation = int(row["observation_frame_id"]) if row["observation_frame_id"] else None
        h = snapshots.get(fid, {}).get(track, ())
        if (
            observation != fid
            or not h
            or row["history_frame_ids"] != " ".join(str(v[1]) for v in h)
        ):
            raise ContractError("Cached action history is not the original actual observations")
        updated = float(row["available_at_ms"])
        if updated != h[-1][2] or updated > sample.timestamp_ms + 1e-7:
            raise ContractError("Cached action is from a future/different observation")
        if float(row["history_start_ms"]) != h[0][2] or float(row["history_end_ms"]) != h[-1][2]:
            raise ContractError("Cached action history bounds mismatch")
        pose_good = bool(good_frame(poses[track].keypoints, config["quality"]))
        if row["pose_good"] != str(pose_good):
            raise ContractError("Cached action quality differs from original pose")
        state = {
            **unavailable(),
            "state": row["state"],
            "label": row["label"],
            "available_at_ms": updated,
            "score_age_ms": float(row["score_age_ms"]),
            "source_frame_id": fid,
            "history_frame_ids": tuple(v[1] for v in h),
            "history_start_ms": h[0][2],
            "history_end_ms": h[-1][2],
            "pose_good": pose_good,
            "reason": row["reason"] or None,
        }
        if not math.isclose(state["score_age_ms"], sample.timestamp_ms - updated, abs_tol=1e-7):
            raise ContractError("Cached action age mismatch")
        if row["state"] == "valid":
            values = tuple(float(row[k]) for k in SCORE_COLUMNS)
            if len(h) != 30 or not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
                raise ContractError("Invalid original cached action scores/history")
            choice = int(np.argmax(values))
            if (
                int(row["class_index"]) != choice
                or row["label"] != ACTION_LABELS[choice]
                or float(row["head_score"]) != values[choice]
            ):
                raise ContractError("Cached action argmax/label/score mismatch")
            state.update(class_scores=values, class_index=choice, score=values[choice])
        elif row["state"] == "warming_up":
            if len(h) >= 30 or row["label"] != "PENDING" or any(row[k] for k in SCORE_COLUMNS):
                raise ContractError("Cached action warmup cannot invent scores")
        elif row["state"] != "unavailable":
            raise ContractError("Unknown cached action state")
        target[track] = state
    if set(grouped) != set(samples):
        raise ContractError("Cached action log omitted a scheduled sample")
    for index, sample in samples.items():
        if set(grouped[index]) != {p.track_id for p in sample.poses}:
            raise ContractError("Cached actions differ from sampled tracks")
    return grouped
