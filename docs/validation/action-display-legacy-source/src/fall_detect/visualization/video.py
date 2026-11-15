import time
from pathlib import Path

import cv2
import numpy as np

from ..io_utils import ContractError, write_csv, write_json
from ..recipes import recipe_spec
from ..skeleton.preprocessing import OPENPOSE18_EDGES

EDGES = [
    (0, 1),
    (0, 2),
    (1, 2),
    (1, 3),
    (3, 5),
    (2, 4),
    (4, 6),
    (1, 7),
    (2, 8),
    (7, 8),
    (7, 9),
    (9, 11),
    (8, 10),
    (10, 12),
]
COLORS = {
    "normal": (80, 210, 80),
    "suspected_fall": (30, 80, 255),
    "warming_up": (220, 200, 60),
    "unknown": (170, 170, 170),
}


def draw_overlay(image, poses, statuses, timestamp_ms, threshold, source_kind="real"):
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 66), (20, 20, 20), -1)
    cv2.putText(
        output,
        f"{source_kind}  t={timestamp_ms / 1000:.2f}s  threshold={threshold:.6g}",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    if not poses:
        cv2.putText(
            output,
            "unknown: no_person",
            (10, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLORS["unknown"],
            1,
            cv2.LINE_AA,
        )
    for pose in poses:
        status = statuses.get(
            pose.track_id, {"state": "warming_up", "raw_score": None, "reason": "warming_up"}
        )
        color = COLORS[status["state"]]
        keypoints = pose.keypoints
        for a, b in OPENPOSE18_EDGES if pose.layout_id == "openpose18" else EDGES:
            if (
                min(keypoints[a, 2], keypoints[b, 2]) >= 0.2
                and np.isfinite(keypoints[[a, b], :2]).all()
            ):
                cv2.line(
                    output,
                    tuple(keypoints[a, :2].astype(int)),
                    tuple(keypoints[b, :2].astype(int)),
                    color,
                    2,
                )
        for x, y, confidence in keypoints:
            if confidence >= 0.2 and np.isfinite([x, y]).all():
                cv2.circle(output, (int(x), int(y)), 3, color, -1)
        x1, y1, x2, y2 = np.asarray(pose.bbox_xyxy, dtype=int)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        score = "-" if status.get("raw_score") is None else f"{status['raw_score']:.5g}"
        text = f"ID {pose.track_id} {status['state']} score={score} age={status.get('score_age_ms', 0):.0f}ms"
        cv2.putText(
            output,
            text,
            (max(2, x1), max(85, min(output.shape[0] - 28, y1))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
        if status.get("reason"):
            cv2.putText(
                output,
                status["reason"],
                (max(2, x1), max(100, min(output.shape[0] - 10, y1 + 16))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )
    return output


class VideoOutput:
    def __init__(self, path, fps, size):
        self.path = Path(path)
        if self.path.exists():
            raise ContractError(f"Video output already exists: {path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.codec = (
            "mp4v"
            if self.path.suffix.lower() == ".mp4"
            else "MJPG"
            if self.path.suffix.lower() == ".avi"
            else None
        )
        if self.codec is None:
            raise ContractError("Video output must be .mp4 or .avi")
        self.writer = cv2.VideoWriter(
            str(self.path), cv2.VideoWriter_fourcc(*self.codec), fps, size
        )
        if not self.writer.isOpened():
            raise ContractError(f"VideoWriter codec {self.codec} is unavailable; try .avi")
        self.size, self.fps, self.count = size, fps, 0

    def write(self, frame):
        if (frame.shape[1], frame.shape[0]) != self.size:
            raise ContractError("Output frame dimensions changed")
        self.writer.write(frame)
        self.count += 1

    def close(self):
        self.writer.release()
        if not self.path.exists() or self.path.stat().st_size == 0:
            raise ContractError("Empty output video")
        cap = cv2.VideoCapture(str(self.path))
        count = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if (frame.shape[1], frame.shape[0]) != self.size:
                    raise ContractError("Decoded video dimensions differ")
                count += 1
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
        finally:
            cap.release()
        if count != self.count or count == 0 or abs(actual_fps - self.fps) > 0.1:
            raise ContractError("Output video frame count/FPS verification failed")
        return {
            "codec": self.codec,
            "frames": count,
            "fps": actual_fps,
            "size_wh": list(self.size),
            "duration_ms": count / self.fps * 1000,
            "decoded_verified": True,
        }


def demo(run, source, output, device=None, no_display=True, overlay_mode="debug", font_path=None):
    from ..assets import require_weights
    from ..backends import create_backends
    from ..config import recipe_fingerprint
    from ..data.readers import read_source
    from ..detection.events import EventState
    from ..skeleton.preprocessing import good_frame
    from ..skeleton.windows import CausalResampler, WindowBuilder
    from ..training.experiment import Scorer

    if overlay_mode not in ("debug", "presentation", "simple"):
        raise ContractError("Unknown overlay mode")
    if overlay_mode == "presentation":
        from .presentation import draw_presentation, presentation_size, resolve_font

        font_path = resolve_font(font_path)
    elif overlay_mode == "simple":
        from .simple import draw_simple, simple_size
    scorer = Scorer(run, device)
    if scorer.index["source_kind"] != "real":
        raise ContractError(
            "demo requires a real, calibrated run; synthetic fixture models cannot classify real video"
        )
    config = scorer.config
    resources = require_weights(config)
    recipe_hash = recipe_fingerprint(config, {k: v["sha256"] for k, v in resources.items()})
    if recipe_hash != scorer.index["recipe_hash"]:
        raise ContractError("Current pose/ST-GCN recipe differs from calibrated run")
    backend, extractor = create_backends(config, resources, scorer.device)
    joint_count = recipe_spec(config)["joints"]
    builder, resampler = (
        WindowBuilder(config),
        CausalResampler(
            config["skeleton"]["target_fps"],
            config["skeleton"]["resampling_hold_policy"],
            config["skeleton"]["timestamp_tolerance_ms"],
        ),
    )
    events = EventState(scorer.threshold["threshold"], 3, config["skeleton"]["stride"])
    writer, previous_frame, statuses, log = None, None, {}, []
    max_age = (
        config["demo"]["score_hold_stride_multiplier"]
        * config["skeleton"]["stride"]
        * 1000
        / config["skeleton"]["target_fps"]
    )
    sync = None
    if Path(source).is_dir():
        candidates = list(Path(source).glob("*-data.csv"))
        sync = candidates[0] if len(candidates) == 1 else None
    started = time.perf_counter()
    last_time = 0.0
    try:
        for frame in read_source(
            source,
            sync_path=sync,
            fallback_fps=config["data"]["fallback_fps"],
            allow_missing_timestamp_tail=config["data"]["allow_missing_timestamp_tail"],
        ):
            poses = backend.predict(frame)
            if writer is None:
                writer = VideoOutput(
                    output,
                    config["skeleton"]["target_fps"],
                    presentation_size(frame.width, frame.height)
                    if overlay_mode == "presentation"
                    else simple_size(frame.width, frame.height)
                    if overlay_mode == "simple"
                    else (frame.width, frame.height),
                )
            for sample in resampler.push(frame, poses):
                last_time = sample.timestamp_ms
                image = (
                    frame.image_bgr
                    if sample.source_frame_id == frame.source_frame_id
                    else previous_frame.image_bgr
                )
                if image is None:
                    image = np.zeros((frame.height, frame.width, 3), dtype=np.uint8)
                for window in builder.push(sample):
                    score = (
                        float(
                            scorer.score(
                                extractor.extract_features(window.keypoints, window.frame_size_wh)
                            )[0]
                        )
                        if window.quality == "valid"
                        else None
                    )
                    status = events.update(
                        window.sequence_id,
                        window.track_id,
                        window.sample_index,
                        window.end_ms,
                        score,
                        window.quality if score is None else None,
                    )
                    statuses[window.track_id] = {**status, "available_at_ms": window.end_ms}
                present = {pose.track_id for pose in sample.poses}
                for track_id in list(statuses):
                    if track_id not in present:
                        statuses[track_id] = {
                            "state": "unknown",
                            "raw_score": None,
                            "event_score": None,
                            "available_at_ms": sample.timestamp_ms,
                            "reason": "track_gap",
                        }
                visible = {}
                for pose in sample.poses:
                    current = statuses.get(
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
                    if (
                        not good_frame(
                            pose.keypoints,
                            config["quality"],
                            joint_count,
                            recipe_spec(config).get("derived_joints", ()),
                        )
                        or age > max_age
                    ):
                        current = {
                            **current,
                            "state": "unknown",
                            "raw_score": None,
                            "event_score": None,
                            "reason": "low_confidence"
                            if not good_frame(
                                pose.keypoints,
                                config["quality"],
                                joint_count,
                                recipe_spec(config).get("derived_joints", ()),
                            )
                            else "stale_score",
                        }
                        statuses[pose.track_id] = current
                    visible[pose.track_id] = current
                if overlay_mode == "presentation":
                    rendered = draw_presentation(
                        image,
                        sample.poses,
                        visible,
                        sample.timestamp_ms,
                        scorer.threshold["threshold"],
                        font_path=font_path,
                    )
                elif overlay_mode == "simple":
                    rendered = draw_simple(
                        image,
                        sample.poses,
                        visible,
                        sample.timestamp_ms,
                        scorer.threshold["threshold"],
                        quality=config["quality"],
                    )
                else:
                    rendered = draw_overlay(
                        image,
                        sample.poses,
                        visible,
                        sample.timestamp_ms,
                        scorer.threshold["threshold"],
                    )
                writer.write(rendered)
                for track_id, status in visible.items() or [
                    (
                        None,
                        {
                            "state": "unknown",
                            "reason": "no_person",
                            "raw_score": None,
                            "event_score": None,
                        },
                    )
                ]:
                    log.append(
                        {
                            "sample_index": sample.sample_index,
                            "timestamp_ms": sample.timestamp_ms,
                            "source_frame_id": sample.source_frame_id,
                            "track_id": track_id,
                            **status,
                        }
                    )
            previous_frame = frame
            # EOF follows the actual source end, including the fraction after the last grid tick.
            # Cached score_sequence uses the same raw-source end timestamp.
            last_time = frame.timestamp_ms
        if writer is None:
            raise ContractError("Input contains no frames")
        info = writer.close()
        writer = None
    finally:
        if writer is not None:
            writer.writer.release()
    output = Path(output)
    write_csv(
        output.with_suffix(".csv"),
        log,
        [
            "sample_index",
            "timestamp_ms",
            "source_frame_id",
            "track_id",
            "state",
            "reason",
            "raw_score",
            "event_score",
            "available_at_ms",
            "score_age_ms",
        ],
    )
    write_csv(
        output.with_name(output.stem + "-events.csv"),
        events.finish(last_time),
        [
            "sequence_id",
            "track_id",
            "confirmed_start_ms",
            "last_active_ms",
            "end_ms",
            "peak_score",
            "end_reason",
        ],
    )
    write_json(
        output.with_suffix(".json"),
        {
            **info,
            "source_kind": "real",
            "rendering": {
                "overlay_mode": overlay_mode,
                "font_path": str(font_path) if font_path else None,
                "hide_failed_pose_geometry": overlay_mode == "simple",
            },
            "run": str(run),
            "source": str(source),
            "recipe_hash": recipe_hash,
            "threshold": scorer.threshold,
            "config": config,
            "processing_fps": info["frames"] / (time.perf_counter() - started),
            "video_grid_fps": config["skeleton"]["target_fps"],
        },
    )
    return output
