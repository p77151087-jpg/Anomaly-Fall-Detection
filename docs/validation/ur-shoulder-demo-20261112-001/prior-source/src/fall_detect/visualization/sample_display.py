"""Sample-style action display. Never supplies AE features, scores, or event decisions."""

from collections import deque

import cv2
import numpy as np

from ..contracts import RECIPE_ID
from ..io_utils import ContractError
from ..skeleton.preprocessing import good_frame
from .action_status import ACTION_LABELS

HEADER = 116
GREEN = (132, 223, 101)
RED = (103, 101, 255)
AMBER = (96, 207, 245)
GREY = (182, 193, 202)


def supports_sample_display(config):
    return (
        config["skeleton"]["recipe_id"] == RECIPE_ID
        and config["pose"]["backend"] == "human_alphapose"
        and config["features"]["backend"] == "human_tsstg"
    )


def unavailable(reason="tracking_lost"):
    return {
        "state": "unavailable",
        "label": "",
        "class_index": None,
        "score": None,
        "class_scores": None,
        "available_at_ms": None,
        "score_age_ms": None,
        "reason": reason,
        "source_frame_id": None,
        "history_frame_ids": (),
        "history_start_ms": None,
        "history_end_ms": None,
        "pose_good": False,
    }


class SampleActions:
    """30 actual confirmed-track observations, not AE windows or resampled duplicates.

    Histories survive short gaps as in the sample. Missing observations are never filled.
    Retire history after track_max_age source frames OR one nominal tracker lifetime.
    Warmup starts at the first confirmed observation exposed by our pose backend.
    """

    def __init__(self, config, extractor):
        if not supports_sample_display(config):
            raise ContractError("Sample action display requires genuine Human pose/ST-GCN")
        self.extractor = extractor
        self.quality = dict(config["quality"])
        self.max_age = config["pose"]["track_max_age"]
        self.max_gap_ms = self.max_age * 1000 / config["skeleton"]["target_fps"]
        self.histories = {}
        self.current = {}
        self.sequence_id = None
        self.last_timestamp = -1.0

    def push(self, frame, poses):
        if self.sequence_id is None:
            self.sequence_id = frame.sequence_id
        if frame.sequence_id != self.sequence_id or frame.timestamp_ms <= self.last_timestamp:
            raise ContractError("Sample actions require one increasing source timeline")
        self.last_timestamp = frame.timestamp_ms
        for key, history in list(self.histories.items()):
            if (
                frame.sample_index - history[-1][0] > self.max_age
                or frame.timestamp_ms - history[-1][2] > self.max_gap_ms
            ):
                del self.histories[key]
        current = {}
        if frame.source_missing and poses:
            raise ContractError("Missing source cannot contain observed actions")
        for pose in poses:
            if (
                pose.recipe_id != RECIPE_ID
                or pose.layout_id != "human13"
                or pose.sequence_id != frame.sequence_id
                or pose.source_frame_id != frame.source_frame_id
                or pose.timestamp_ms != frame.timestamp_ms
                or pose.frame_size_wh != (frame.width, frame.height)
            ):
                raise ContractError("Action observation/source contract mismatch")
            if pose.track_id in current:
                raise ContractError("Duplicate observed track")
            points = np.asarray(pose.keypoints)
            if points.shape != (13, 3) or not np.isfinite(points).all():
                raise ContractError("Actions require finite observed Human13 points")
            history = self.histories.setdefault(pose.track_id, deque(maxlen=30))
            if history and history[-1][4] != pose.frame_size_wh:
                history.clear()
            history.append(
                (
                    frame.sample_index,
                    frame.source_frame_id,
                    frame.timestamp_ms,
                    points.copy(),
                    pose.frame_size_wh,
                )
            )
            state = {
                **unavailable("collecting_observations"),
                "state": "warming_up",
                "label": "PENDING",
                "available_at_ms": frame.timestamp_ms,
                "score_age_ms": 0.0,
                "source_frame_id": frame.source_frame_id,
                "history_frame_ids": tuple(h[1] for h in history),
                "history_start_ms": history[0][2],
                "history_end_ms": history[-1][2],
                "pose_good": bool(good_frame(points, self.quality)),
            }
            if len(history) == 30:
                try:
                    scores = self.extractor.classify_observations(
                        np.stack([h[3] for h in history]), pose.frame_size_wh
                    )[0]
                except ContractError as exc:
                    if not str(exc).startswith("degenerate_pose"):
                        raise
                    state.update(state="unavailable", label="", reason="degenerate_pose")
                else:
                    if (
                        scores.shape != (7,)
                        or not np.isfinite(scores).all()
                        or ((scores < 0) | (scores > 1)).any()
                    ):
                        raise ContractError("Invalid independent action scores")
                    choice = int(scores.argmax())
                    state.update(
                        state="valid",
                        label=ACTION_LABELS[choice],
                        class_index=choice,
                        score=float(scores[choice]),
                        class_scores=tuple(float(v) for v in scores),
                        reason=None if state["pose_good"] else "low_confidence",
                    )
            current[pose.track_id] = state
        # A new mapping leaves the previous source snapshot intact for earlier grid ticks.
        self.current = current
        return current


def visible_actions(snapshot, poses, timestamp_ms):
    result = {}
    for pose in poses:
        state = snapshot.get(pose.track_id, unavailable())
        updated = state["available_at_ms"]
        if (
            updated is None
            or updated > timestamp_ms + 1e-7
            or state["source_frame_id"] != pose.source_frame_id
        ):
            state = unavailable("no_current_observation")
        else:
            state = {**state, "score_age_ms": max(0.0, timestamp_ms - updated)}
        result[pose.track_id] = state
    return result


class EventHistory:
    """Display history of already confirmed AE windows, without changing EventState."""

    def __init__(self):
        self.active = {}
        self.latest = None
        self.count = 0

    def observe(self, track_id, timestamp_ms, status):
        active = status["state"] == "suspected_fall"
        if active and not self.active.get(track_id, False):
            self.count += 1
            self.latest = {
                "track_id": track_id,
                "confirmed_start_ms": timestamp_ms,
                "event_number": self.count,
            }
        self.active[track_id] = active


def action_color(state):
    if state["label"] == "FALLING":
        return RED
    if state["label"] in ("LYING", "PENDING"):
        return AMBER
    return GREEN if state["state"] == "valid" else GREY


def action_hint(state):
    if state["state"] == "warming_up":
        return f"Collecting motion {len(state['history_frame_ids'])}/30"
    if state["state"] != "valid":
        return (
            "Tracking lost"
            if state["reason"] in ("tracking_lost", "no_current_observation")
            else "Pose unavailable"
        )
    freshness = f"Updated {state['available_at_ms'] / 1000:.2f}s"
    return freshness if state["pose_good"] else f"Pose uncertain | {freshness}"


def ae_hint(state):
    if state["state"] == "unknown":
        return "Current: unavailable (pose / tracking)"
    if state["state"] == "warming_up":
        return "Current: collecting motion"
    return "Current: alert" if state["state"] == "suspected_fall" else "Current: normal"


def draw_sample(image, poses, actions, ae_states, timestamp_ms, history, quality):
    """Motion label + explicit AE event history; no invented current action on loss."""
    from .video import EDGES

    height, width = image.shape[:2]
    canvas = np.full((height + HEADER, width, 3), (26, 21, 17), dtype=np.uint8)
    canvas[HEADER:] = image
    body = canvas[HEADER:]
    split = width // 2
    cv2.line(canvas, (split, 8), (split, HEADER - 8), (65, 65, 65), 1)

    def text(value, x, y, color=GREY, scale=0.44, max_width=None, thickness=1):
        max_width = split - 24 if max_width is None else max_width
        while cv2.getTextSize(value, 0, scale, thickness)[0][0] > max_width and scale > 0.18:
            scale -= 0.02
        cv2.putText(canvas, value, (x, y), 0, scale, color, thickness, cv2.LINE_AA)

    priority = {"FALLING": 3, "LYING": 2}
    track_id = (
        max(
            actions,
            key=lambda k: (
                priority.get(actions[k]["label"], 0),
                actions[k]["state"] == "valid",
                -k,
            ),
        )
        if actions
        else None
    )
    action = actions.get(track_id, unavailable())
    ae = ae_states.get(track_id, {"state": "unknown"})
    text("ACTION" + (f" / ID {track_id}" if len(poses) > 1 else ""), 14, 19)
    text(action["label"], 14, 52, action_color(action), 0.9, thickness=2)
    text(action_hint(action), 14, 76)
    text(f"{timestamp_ms / 1000:.2f} s", 14, 99)
    text("AE EVENT HISTORY", split + 14, 19)
    latest = history.latest
    text(
        "ALERT RECORDED" if latest else "NO EVENT YET",
        split + 14,
        49,
        RED if latest else GREY,
        0.65,
        thickness=2,
    )
    if latest:
        text(
            f"ID {latest['track_id']} | event at {latest['confirmed_start_ms'] / 1000:.2f}s",
            split + 14,
            72,
            AMBER,
        )
    text(ae_hint(ae), split + 14, 97, GREY, 0.40)
    for pose in poses:
        action = actions.get(pose.track_id, unavailable())
        if not action["label"]:
            continue
        color = action_color(action)
        points = pose.keypoints
        # Keep visual geometry conservative; classification itself follows sample-style argmax.
        if good_frame(points, quality):
            inside = (
                np.isfinite(points).all(axis=1)
                & (points[:, 0] >= 0)
                & (points[:, 0] < width)
                & (points[:, 1] >= 0)
                & (points[:, 1] < height)
                & (points[:, 2] >= quality["min_joint_confidence"])
            )
            for a, b in EDGES:
                if inside[a] and inside[b]:
                    p, q = tuple(points[a, :2].astype(int)), tuple(points[b, :2].astype(int))
                    cv2.line(body, p, q, (25, 20, 15), 4, cv2.LINE_AA)
                    cv2.line(body, p, q, color, 2, cv2.LINE_AA)
            for (x, y, _), valid in zip(points, inside):
                if valid:
                    cv2.circle(body, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)
        x1, y1, x2, y2 = np.asarray(pose.bbox_xyxy, dtype=int)
        x1, x2 = np.clip([x1, x2], 0, width - 1)
        y1, y2 = np.clip([y1, y2], 0, height - 1)
        if action["pose_good"]:
            cv2.rectangle(body, (x1, y1), (x2, y2), color, 1)
        label = action["label"] + (f" / ID {pose.track_id}" if len(poses) > 1 else "")
        scale = 0.6
        size, baseline = cv2.getTextSize(label, 0, scale, 2)
        left = max(0, min(width - size[0] - 12, x1))
        top = max(size[1] + 7, min(height - baseline - 4, y1 - 5))
        cv2.rectangle(
            body,
            (left, top - size[1] - 7),
            (left + size[0] + 12, top + baseline + 3),
            (26, 21, 17),
            -1,
        )
        cv2.putText(body, label, (left + 6, top), 0, scale, color, 2, cv2.LINE_AA)
    return canvas
