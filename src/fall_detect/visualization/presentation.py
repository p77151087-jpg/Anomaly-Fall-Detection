"""Chinese presentation renderer. It never updates inference or event state."""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..io_utils import ContractError
from ..skeleton.preprocessing import OPENPOSE18_EDGES

LABELS = {
    "normal": "正常活動",
    "suspected_fall": "疑似跌倒！",
    "warming_up": "分析中",
    "unknown": "暫時無法判定",
}
COLORS = {
    "normal": (91, 227, 163),
    "suspected_fall": (255, 101, 113),
    "warming_up": (162, 185, 212),
    "unknown": (255, 205, 95),
}
REASONS = {
    "no_person": "未偵測到人物",
    "track_gap": "人物追蹤中斷",
    "low_confidence": "骨架可信度不足",
    "stale_score": "分數已逾有效時間",
    "collecting_window": "收集動作片段",
    "insufficient_consecutive_windows": "等待連續有效窗口",
    "too_many_missing": "骨架缺漏過多",
    "low_valid_ratio": "有效骨架比例不足",
}
HEADER, PANEL = 88, 320


def resolve_font(path=None):
    choices = [Path(path)] if path else [Path("C:/Windows/Fonts/msjh.ttc")]
    for candidate in choices:
        if candidate.is_file():
            return candidate
    raise ContractError("Chinese font missing; use --font PATH or --overlay debug")


@lru_cache(maxsize=24)
def font(path, size):
    return ImageFont.truetype(str(path), size=size)


def presentation_size(width, height):
    return width + PANEL, max(height, 480) + HEADER


def display_values(status):
    """Never display stale or unknown numeric values, even on malformed UI input."""
    if status["state"] == "unknown":
        return "—", "—", "分數暫停更新"
    raw = status.get("raw_score")
    event = status.get("event_score")
    updated = status.get("available_at_ms")
    return (
        "—" if raw is None else f"{raw:.4f}",
        "—" if event is None else f"{event:.4f}",
        "尚無有效分數" if raw is None or updated is None else f"分數更新於 {updated / 1000:.2f} 秒",
    )


def draw_presentation(
    image, poses, statuses, timestamp_ms, threshold, source_kind="real", font_path=None
):
    path = resolve_font(font_path)
    height, width = image.shape[:2]
    working = image.copy()
    # Coordinates remain in source space; add the title and information panel afterwards.
    from .video import EDGES

    for pose in poses:
        status = statuses.get(pose.track_id, {"state": "warming_up"})
        rgb = COLORS[status["state"]]
        color = tuple(reversed(rgb))
        points = pose.keypoints
        for a, b in OPENPOSE18_EDGES if pose.layout_id == "openpose18" else EDGES:
            if min(points[a, 2], points[b, 2]) >= 0.2 and np.isfinite(points[[a, b], :2]).all():
                cv2.line(
                    working,
                    tuple(points[a, :2].astype(int)),
                    tuple(points[b, :2].astype(int)),
                    (15, 20, 30),
                    6,
                    cv2.LINE_AA,
                )
                cv2.line(
                    working,
                    tuple(points[a, :2].astype(int)),
                    tuple(points[b, :2].astype(int)),
                    color,
                    3,
                    cv2.LINE_AA,
                )
        for x, y, confidence in points:
            if confidence >= 0.2 and np.isfinite([x, y]).all():
                cv2.circle(working, (int(x), int(y)), 5, (15, 20, 30), -1, cv2.LINE_AA)
                cv2.circle(working, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)
        x1, y1, x2, y2 = np.asarray(pose.bbox_xyxy, dtype=int)
        cv2.rectangle(working, (x1, y1), (x2, y2), color, 2)

    canvas = Image.new("RGB", presentation_size(width, height), (14, 23, 38))
    canvas.paste(Image.fromarray(cv2.cvtColor(working, cv2.COLOR_BGR2RGB)), (0, HEADER))
    draw = ImageDraw.Draw(canvas)
    live = [(p.track_id, statuses.get(p.track_id, {"state": "warming_up"})) for p in poses]
    if not live:
        live = [(None, {"state": "unknown", "reason": "no_person"})]
    priority = {"suspected_fall": 3, "unknown": 2, "warming_up": 1, "normal": 0}
    headline = max(live, key=lambda row: priority[row[1]["state"]])[1]
    color = COLORS[headline["state"]]
    draw.rectangle((0, 0, 8, HEADER - 1), fill=color)
    draw.text((24, 12), LABELS[headline["state"]], font=font(path, 34), fill=color)
    draw.text((26, 56), "ST-GCN 特徵  ·  正常資料 AE", font=font(path, 17), fill=(190, 204, 222))
    x = width + 20
    draw.text((x, 19), f"原影片  {timestamp_ms / 1000:06.2f} 秒", font=font(path, 22), fill="white")
    draw.text(
        (x, 53),
        "真實推論" if source_kind == "real" else "測試資料",
        font=font(path, 16),
        fill=(160, 180, 205),
    )
    for i, (track_id, status) in enumerate(live[:2]):
        y = HEADER + 18 + i * 188
        col = COLORS[status["state"]]
        draw.text(
            (x, y),
            f"人物 {track_id if track_id is not None else '—'}  ·  {LABELS[status['state']]}",
            font=font(path, 21),
            fill=col,
        )
        raw, event, freshness = display_values(status)
        draw.text((x, y + 39), f"AE 分數     {raw}", font=font(path, 23), fill="white")
        draw.text((x, y + 76), f"連續三窗   {event}", font=font(path, 23), fill="white")
        draw.text((x, y + 116), freshness, font=font(path, 18), fill=(185, 202, 222))
        reason = status.get("reason")
        if reason:
            draw.text(
                (x, y + 145), REASONS.get(reason, "骨架或時間窗不足"), font=font(path, 18), fill=col
            )
    if len(live) > 2:
        draw.text(
            (x, HEADER + 390),
            f"另有 {len(live) - 2} 位人物；詳見紀錄",
            font=font(path, 16),
            fill="white",
        )
    bottom = canvas.height
    draw.line((x, bottom - 91, canvas.width - 20, bottom - 91), fill=(57, 77, 99), width=1)
    draw.text((x, bottom - 77), f"警示門檻   {threshold:.4f}", font=font(path, 22), fill="white")
    draw.text(
        (x, bottom - 41), "AE 重建誤差，不是跌倒機率", font=font(path, 17), fill=(185, 202, 222)
    )
    for pose in poses:
        status = statuses.get(pose.track_id, {"state": "warming_up"})
        label = f"{pose.track_id}  {LABELS[status['state']]}"
        box = draw.textbbox((0, 0), label, font=font(path, 22))
        label_width = box[2] + 18
        x1 = max(0, min(width - label_width, int(pose.bbox_xyxy[0])))
        y1 = max(HEADER + 2, min(canvas.height - 34, int(pose.bbox_xyxy[1]) + HEADER - 35))
        draw.rectangle((x1, y1, x1 + label_width, y1 + 33), fill=(14, 23, 38))
        draw.text((x1 + 8, y1 + 1), label, font=font(path, 22), fill=COLORS[status["state"]])
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
