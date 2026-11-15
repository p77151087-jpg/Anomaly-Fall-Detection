"""Display-only Human action classifications, independent of AE events and metrics."""

import cv2
import numpy as np

from ..io_utils import ContractError
from ..recipes import HUMAN_RECIPE
from ..skeleton.preprocessing import good_frame

# Exact class order from the pinned Human ActionsEstLoader.TSSTG checkpoint contract.
SOURCE_CLASSES = (
    "Standing",
    "Walking",
    "Sitting",
    "Lying Down",
    "Stand up",
    "Sit down",
    "Fall Down",
)
ACTION_LABELS = (
    "STANDING",
    "WALKING",
    "SITTING",
    "LYING",
    "STANDING UP",
    "SITTING DOWN",
    "FALLING",
)
SCORE_COLUMNS = ("standing", "walking", "sitting", "lying", "stand_up", "sit_down", "fall_down")
ACTION_HEADER = 92


def supports_actions(config):
    skeleton = config["skeleton"]
    return (
        skeleton["recipe_id"] == HUMAN_RECIPE
        and skeleton["window_size"] == 30
        and skeleton.get("coordinate_normalization", "per_frame") == "per_frame"
    )


def unavailable(state="unknown", reason="no_person", timestamp_ms=None):
    return {
        "state": state,
        "label": "PENDING" if state == "warming_up" else "UNKNOWN",
        "class_index": None,
        "score": None,
        "class_scores": None,
        "available_at_ms": timestamp_ms,
        "score_age_ms": None,
        "reason": reason,
    }


class ActionDisplayState:
    def __init__(self, config):
        if not supports_actions(config):
            raise ContractError("Seven actions require the original30 Human per-frame recipe")
        self.quality = dict(config["quality"])
        self.max_age = (
            config["demo"]["score_hold_stride_multiplier"]
            * config["skeleton"]["stride"]
            * 1000
            / config["skeleton"]["target_fps"]
        )
        self.states = {}

    def update(self, track_id, scores, timestamp_ms, reason=None):
        if scores is None:
            self.states[track_id] = unavailable(
                reason=reason or "invalid_window", timestamp_ms=timestamp_ms
            )
            return
        values = np.asarray(scores)
        if (
            values.shape != (7,)
            or not np.isfinite(values).all()
            or ((values < 0) | (values > 1)).any()
        ):
            raise ContractError("Action head must return seven finite sigmoid scores")
        chosen = int(values.argmax())
        self.states[track_id] = {
            "state": "valid",
            "label": ACTION_LABELS[chosen],
            "class_index": chosen,
            "score": float(values[chosen]),
            "class_scores": tuple(float(v) for v in values),
            "available_at_ms": timestamp_ms,
            "score_age_ms": 0.0,
            "reason": None,
        }

    def visible(self, poses, timestamp_ms):
        present = {p.track_id for p in poses}
        for track_id in set(self.states) - present:
            self.states[track_id] = unavailable(reason="track_gap", timestamp_ms=timestamp_ms)
        result = {}
        for pose in poses:
            current = self.states.get(
                pose.track_id, unavailable("warming_up", "collecting_window", timestamp_ms)
            )
            updated = current["available_at_ms"]
            age = timestamp_ms - updated if updated is not None else None
            if not good_frame(pose.keypoints, self.quality, 13):
                current = unavailable(reason="low_confidence", timestamp_ms=timestamp_ms)
                self.states[pose.track_id] = current
            elif current["state"] == "valid" and (age is None or age < 0 or age > self.max_age):
                current = unavailable(reason="stale_action", timestamp_ms=timestamp_ms)
                self.states[pose.track_id] = current
            result[pose.track_id] = {
                **current,
                "score_age_ms": age if current["state"] == "valid" else None,
            }
        return result


def action_hint(status):
    if status["state"] == "warming_up":
        return "Collecting motion"
    if status["state"] != "valid":
        return "No valid result"
    return f"Updated {status['available_at_ms'] / 1000:.2f} s"


def action_color(status):
    if status["state"] != "valid":
        return (75, 203, 248)
    if status["class_index"] == 6:
        return (110, 110, 255)
    if status["class_index"] == 3:
        return (80, 192, 250)
    return (166, 220, 108)


def draw_action_header(
    output, action, ae_label, ae_color, ae_hint, timestamp_ms, track_id=None, multiple=False
):
    """Both columns refer to the same person; each box also carries both labels."""
    width = output.shape[1]
    output[:ACTION_HEADER] = (26, 21, 17)
    middle = width // 2
    cv2.line(output, (middle, 10), (middle, ACTION_HEADER - 10), (65, 65, 65), 1)

    def text(value, x, y, size, color, thickness=1, available=None):
        available = available or middle - 30
        measured = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, size, thickness)[0][0]
        size *= min(1.0, available / max(1, measured))
        cv2.putText(
            output, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA
        )

    scale = max(0.55, min(1.3, width / 640))
    text("ACTION", 16, 18, 0.38 * scale, (195, 195, 195))
    text("ANOMALY (AE)", middle + 16, 18, 0.38 * scale, (195, 195, 195), available=middle - 100)
    text(action["label"], 16, 50, 0.88 * scale, action_color(action), 2)
    text(ae_label, middle + 16, 50, 0.88 * scale, ae_color, 2)
    prefix = f"T{track_id} | " if multiple else ""
    text(prefix + action_hint(action), 16, 75, 0.38 * scale, (190, 197, 201))
    text(prefix + ae_hint, middle + 16, 75, 0.38 * scale, (190, 197, 201))
    clock = f"{timestamp_ms / 1000:.2f} s"
    clock_width = cv2.getTextSize(clock, cv2.FONT_HERSHEY_SIMPLEX, 0.38 * scale, 1)[0][0]
    text(
        clock,
        width - clock_width - 12,
        18,
        0.38 * scale,
        (190, 197, 201),
        available=clock_width + 1,
    )
