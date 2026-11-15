"""Compact status overlay; display changes never update inference state."""

import cv2
import numpy as np

from ..recipes import RECIPES
from ..skeleton.preprocessing import OPENPOSE18_EDGES, good_frame
from .action_status import ACTION_HEADER, action_color, draw_action_header, unavailable
from .presentation import COLORS

LABELS = {
    "warming_up": "PENDING",
    "normal": "NORMAL",
    "suspected_fall": "FALL ALERT",
    "unknown": "UNKNOWN",
}
HEADER = 64


def simple_size(width, height, actions=False):
    return width, height + (ACTION_HEADER if actions else HEADER)


def status_hint(status):
    """Keep unavailable results distinct from collecting a new motion window."""
    if status["state"] == "unknown":
        return "No valid result"
    if status["state"] == "warming_up":
        return "Collecting motion"
    updated = status.get("available_at_ms")
    if updated is None or status.get("raw_score") is None:
        return "No valid result"
    return f"Updated {updated / 1000:.2f} s"


def draw_simple(
    image, poses, statuses, timestamp_ms, threshold, source_kind="real", quality=None, actions=None
):
    from .video import EDGES

    height, width = image.shape[:2]
    header = ACTION_HEADER if actions is not None else HEADER
    output = np.full((height + header, width, 3), (26, 21, 17), dtype=np.uint8)
    output[header:] = image
    body = output[header:]
    live = [(p.track_id, statuses.get(p.track_id, {"state": "warming_up"})) for p in poses]
    priority = {"suspected_fall": 3, "unknown": 2, "warming_up": 1, "normal": 0}
    headline_id, headline = (
        max(live, key=lambda s: priority[s[1]["state"]]) if live else (None, {"state": "unknown"})
    )
    color = tuple(reversed(COLORS[headline["state"]]))
    scale = min(1.3, max(0.5, width / 640))
    cv2.rectangle(output, (0, 0), (5, HEADER - 1), color, -1)
    cv2.putText(
        output,
        LABELS[headline["state"]],
        (16, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95 * scale,
        color,
        2,
        cv2.LINE_AA,
    )
    hint = status_hint(headline)
    if source_kind != "real":
        hint = f"SIMULATED INPUT | {hint}"
    cv2.putText(
        output,
        hint,
        (17, 53),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42 * scale,
        (190, 197, 201),
        1,
        cv2.LINE_AA,
    )
    clock = f"{timestamp_ms / 1000:.2f} s"
    clock_size = cv2.getTextSize(clock, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, 1)[0]
    cv2.putText(
        output,
        clock,
        (max(2, width - clock_size[0] - 16), 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55 * scale,
        (220, 226, 230),
        1,
        cv2.LINE_AA,
    )
    if actions is not None:
        draw_action_header(
            output,
            actions.get(headline_id, unavailable()),
            LABELS[headline["state"]],
            color,
            hint,
            timestamp_ms,
            track_id=headline_id,
            multiple=len(poses) > 1,
        )
    for pose in poses:
        spec = RECIPES[pose.recipe_id]
        display_quality = (
            quality
            if quality is not None
            else {
                "min_joint_confidence": 0.2,
                "min_valid_joints": 12 if pose.layout_id == "openpose18" else 9,
            }
        )
        if not good_frame(
            pose.keypoints, display_quality, spec["joints"], spec.get("derived_joints", ())
        ):
            # Keep the headline's UNKNOWN status; do not draw a failed pose as a person.
            continue
        status = statuses.get(pose.track_id, {"state": "warming_up"})
        color = tuple(reversed(COLORS[status["state"]]))
        if actions is not None:
            action = actions.get(pose.track_id, unavailable())
            color = action_color(action)
        points = pose.keypoints
        joint_min = display_quality["min_joint_confidence"]
        for a, b in OPENPOSE18_EDGES if pose.layout_id == "openpose18" else EDGES:
            if (
                min(points[a, 2], points[b, 2]) >= joint_min
                and np.isfinite(points[[a, b], :2]).all()
            ):
                start, end = tuple(points[a, :2].astype(int)), tuple(points[b, :2].astype(int))
                cv2.line(body, start, end, (25, 20, 15), 4, cv2.LINE_AA)
                cv2.line(body, start, end, color, 2, cv2.LINE_AA)
        for x, y, confidence in points:
            if confidence >= joint_min and np.isfinite([x, y]).all():
                cv2.circle(body, (int(x), int(y)), 4, (25, 20, 15), -1, cv2.LINE_AA)
                cv2.circle(body, (int(x), int(y)), 2, color, -1, cv2.LINE_AA)
        x1, y1, x2, y2 = np.asarray(pose.bbox_xyxy, dtype=int)
        cv2.rectangle(body, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        label = LABELS[status["state"]]
        if actions is not None:
            label = f"{action['label']} / AE {label}"
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62 * scale, 2)
        left = max(0, min(width - label_size[0] - 16, x1))
        top = max(
            0, min(height - label_size[1] - baseline - 12, y1 - label_size[1] - baseline - 12)
        )
        cv2.rectangle(
            body,
            (left, top),
            (left + label_size[0] + 16, top + label_size[1] + baseline + 12),
            (26, 21, 17),
            -1,
        )
        cv2.putText(
            body,
            label,
            (left + 8, top + label_size[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62 * scale,
            color,
            2,
            cv2.LINE_AA,
        )
    return output
