from collections import deque
from dataclasses import replace

import numpy as np

from ..contracts import Sample, Window
from ..io_utils import ContractError
from ..recipes import recipe_spec
from .preprocessing import good_frame, human_preprocess, openpose_preprocess


class CausalResampler:
    """Emit a fixed grid using only the most recent observation at or before each tick.

    Legacy observations expire after one target interval. The source_interval policy
    permits a lower-FPS video observation until its nominal next source frame is due.
    FPS comes from reader metadata, never future timestamps or future poses.
    push never uses a later frame for an earlier tick; finish never invents future duration.
    """

    def __init__(self, target_fps, hold_policy="target_interval", timestamp_tolerance_ms=0.0):
        self.period = 1000.0 / target_fps
        if hold_policy not in {"target_interval", "source_interval", "bounded_jitter"}:
            raise ContractError("Unknown resampling hold policy")
        self.hold_policy = hold_policy
        if (
            type(timestamp_tolerance_ms) not in (int, float)
            or not np.isfinite(timestamp_tolerance_ms)
            or not 0 <= timestamp_tolerance_ms <= 1.0
        ):
            raise ContractError("Timestamp tolerance must be between 0 and 1 ms")
        self.timestamp_tolerance_ms = timestamp_tolerance_ms
        self.previous = None
        self.origin = None
        self.next_index = 0
        self.last_source_id = None

    def _sample(self, tick, observation):
        frame, poses = observation
        max_age = self.period
        if self.hold_policy == "bounded_jitter":
            # Explicit alternative: at 30 Hz, tolerate capture jitter up to 50 ms,
            # the same nominal boundary used by the pose tracker. No future frames.
            max_age = 1.5 * self.period
        if (
            self.hold_policy in {"source_interval", "bounded_jitter"}
            and frame.source_fps is not None
        ):
            if (
                isinstance(frame.source_fps, bool)
                or not isinstance(frame.source_fps, (int, float))
                or not np.isfinite(frame.source_fps)
                or frame.source_fps <= 0
            ):
                raise ContractError("Source FPS must be finite and positive")
            max_age = max(max_age, 1000.0 / frame.source_fps)
        # Integer-millisecond source timestamps can quantize a nominal frame interval.
        # Only expiry gains tolerance: never shift the grid or consume a future pose.
        missing = (
            frame.source_missing
            or tick - frame.timestamp_ms > max_age + self.timestamp_tolerance_ms + 1e-7
        )
        repeated = self.last_source_id == frame.source_frame_id
        self.last_source_id = frame.source_frame_id
        result = Sample(
            frame.sequence_id,
            self.next_index,
            tick,
            None if missing else frame.source_frame_id,
            [] if missing else [replace(p, sample_index=self.next_index) for p in poses],
            (frame.width, frame.height),
            repeated=repeated,
            missing=missing,
        )
        self.next_index += 1
        return result

    def push(self, frame, poses):
        if self.origin is None:
            self.origin = frame.timestamp_ms
        if self.previous:
            old = self.previous[0]
            if frame.sequence_id != old.sequence_id or frame.timestamp_ms <= old.timestamp_ms:
                raise ContractError(
                    "Resampler requires one sequence with strictly increasing timestamps"
                )
        result = []
        tick = self.origin + self.next_index * self.period
        while tick < frame.timestamp_ms - 1e-7:
            if self.previous is None:
                break
            result.append(self._sample(tick, self.previous))
            tick = self.origin + self.next_index * self.period
        self.previous = (frame, poses)
        if abs(tick - frame.timestamp_ms) < 1e-7:
            result.append(self._sample(tick, self.previous))
        return result


class WindowBuilder:
    """Every sequence shares endpoints 29+j*stride, regardless of detected tracks."""

    def __init__(self, config):
        self.config = config
        self.spec = recipe_spec(config)
        self.size = config["skeleton"]["window_size"]
        self.stride = config["skeleton"]["stride"]
        self.samples = deque(maxlen=self.size)
        self.sequence_id = None
        self.last_index = -1

    def push(self, sample):
        if self.sequence_id is None:
            self.sequence_id = sample.sequence_id
        if sample.sequence_id != self.sequence_id or sample.sample_index != self.last_index + 1:
            raise ContractError("WindowBuilder requires consecutive samples from one sequence")
        self.last_index = sample.sample_index
        self.samples.append(sample)
        if (
            sample.sample_index < self.size - 1
            or (sample.sample_index - self.size + 1) % self.stride
        ):
            return []
        track_ids = sorted({p.track_id for item in self.samples for p in item.poses})
        result = []
        for track_id in track_ids or [None]:
            selected = [
                next((p for p in item.poses if p.track_id == track_id), None)
                for item in self.samples
            ]
            points = None
            ratio = 0.0
            if track_id is None:
                quality = "no_person"
            elif any(item.missing for item in self.samples) or any(p is None for p in selected):
                quality = "track_gap"
            elif any(
                p.layout_id != self.spec["layout"]
                or p.recipe_id != self.config["skeleton"]["recipe_id"]
                for p in selected
            ):
                raise ContractError("Pose layout/recipe differs from configured recipe")
            else:
                points = np.stack([p.keypoints for p in selected]).astype(np.float32)
                ratio = (
                    sum(
                        good_frame(
                            p,
                            self.config["quality"],
                            self.spec["joints"],
                            self.spec.get("derived_joints", ()),
                        )
                        for p in points
                    )
                    / self.size
                )
                quality = (
                    "valid"
                    if ratio >= self.config["quality"]["min_good_frame_ratio"]
                    else "low_confidence"
                )
                if any(p.frame_size_wh != sample.frame_size_wh for p in selected):
                    quality = "frame_size_changed"
                if quality == "valid":
                    try:
                        if self.spec["layout"] == "human13":
                            human_preprocess(
                                points,
                                sample.frame_size_wh,
                                self.config["skeleton"]["span_tolerance"],
                            )
                        else:
                            openpose_preprocess(points, sample.frame_size_wh)
                    except ContractError:
                        quality = "degenerate_pose"
            result.append(
                Window(
                    sample.sequence_id,
                    track_id,
                    sample.sample_index,
                    self.samples[0].timestamp_ms,
                    sample.timestamp_ms,
                    points if quality == "valid" else None,
                    sample.frame_size_wh,
                    quality,
                    ratio,
                )
            )
        return result
