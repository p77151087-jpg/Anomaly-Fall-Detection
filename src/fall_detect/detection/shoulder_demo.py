"""Explicit, frozen shoulder candidate for streaming demos."""

from collections import Counter, deque
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..data.cache import save_pose_cache
from ..io_utils import ContractError, read_json, sha256, write_csv, write_npz
from ..skeleton.preprocessing import good_frame
from .events import EventState


def coherent_shoulder_descent(previous, current, box, quality):
    if not good_frame(previous, quality, 13) or not good_frame(current, quality, 13):
        return None, "anchor_confidence"
    minimum = quality["min_joint_confidence"]
    if (previous[1:3, 2] < minimum).any() or (current[1:3, 2] < minimum).any():
        return None, "shoulder_confidence"
    height = float(box[3] - box[1])
    if not np.isfinite(height) or height <= 0:
        return None, "invalid_reference_height"
    delta = float((current[1:3, 1].astype(np.float64) - previous[1:3, 1]).min())
    return max(0.0, delta / height), None


class ShoulderDemo:
    """Two separate K3 histories, followed by a per-track logical union."""

    @classmethod
    def from_candidate(cls, directory, scorer):
        directory = Path(directory)
        manifest = read_json(directory / "artifact-manifest.json")["files"]
        for name in ("lock.json", "normal-calibration.json"):
            if sha256(directory / name) != manifest[name]:
                raise ContractError(f"Frozen shoulder candidate changed: {name}")
        lock = read_json(directory / "lock.json")
        calibration = read_json(directory / "normal-calibration.json")
        recipe = lock["recipe"]
        if (
            lock["baseline"] != scorer.run.name
            or recipe["lag_ticks"] != 6
            or recipe["fps"] != 30
            or recipe["k"] != 3
            or recipe["fusion"] != "per-track OR after two independent K3 histories"
            or recipe["wide_recipe_hash"] != scorer.index["recipe_hash"]
            or recipe["split_hash"] != scorer.index["split_manifest"]["split_hash"]
            or recipe["wide_threshold"] != scorer.threshold["threshold"]
            or scorer.residual_weights is not None
        ):
            raise ContractError("Shoulder candidate differs from the frozen wide run")
        run_key = f"runs/{scorer.run.name}/resolved-config.yaml"
        if sha256(scorer.run / "resolved-config.yaml") != lock["source_sha256"][run_key]:
            raise ContractError("Shoulder candidate quality/timing configuration changed")
        known = {r["sequence_id"]: r for r in scorer.index["split_manifest"]["sequences"]}
        normal = calibration["normal_video_k3"]
        expected = {
            sid
            for sid, row in known.items()
            if row["label"] == 0 and row["split"] in {"train", "validation"}
        }
        if (
            set(normal) != expected
            or Counter(row["split"] for row in normal.values()) != {"train": 24, "validation": 8}
            or any(row["split"] != known[sid]["split"] for sid, row in normal.items())
            or calibration["comparison"] != "gt"
            or calibration["threshold"] != max(r["max_k3"] for r in normal.values())
            or not np.isfinite(calibration["threshold"])
            or calibration["threshold"] <= 0
        ):
            raise ContractError("Shoulder normal calibration contract differs")
        provenance = {
            "candidate": str(directory),
            "candidate_lock_sha256": manifest["lock.json"],
            "calibration_sha256": manifest["normal-calibration.json"],
            "comparison_recipe_sha256": lock["comparison_recipe_sha256"],
            "wide_threshold": recipe["wide_threshold"],
            "shoulder_threshold": calibration["threshold"],
            "decision_threshold": 1.0,
            "score_semantics": "raw_score=max available normalized raw values; event_score=max of two independently formed normalized K3 values",
            "fusion": recipe["fusion"],
        }
        return cls(scorer.config, recipe["wide_threshold"], calibration["threshold"], provenance)

    def __init__(self, config, wide_threshold, shoulder_threshold, provenance=None):
        self.config = config
        self.wide_threshold = float(wide_threshold)
        self.shoulder_threshold = float(shoulder_threshold)
        self.provenance = provenance or {}
        self.samples = deque(maxlen=7)
        self.shoulder = EventState(shoulder_threshold, 3, config["skeleton"]["stride"])
        self.combined = EventState(1.0, 1, config["skeleton"]["stride"])
        self.window_rows, self.features, self.observations = [], [], []

    @staticmethod
    def audit_paths(output):
        output = Path(output)
        return [
            output.with_name(output.stem + "-windows.csv"),
            output.with_name(output.stem + "-wide-events.csv"),
            output.with_name(output.stem + "-poses.npz"),
            output.with_name(output.stem + "-poses.json"),
            output.with_name(output.stem + "-features.npz"),
        ]

    def observe_source(self, frame, poses):
        self.observations.append((replace(frame, image_bgr=None), list(poses)))

    def observe_sample(self, sample):
        if self.samples and (
            sample.sequence_id != self.samples[-1].sequence_id
            or sample.sample_index != self.samples[-1].sample_index + 1
        ):
            raise ContractError("Shoulder samples must be one consecutive sequence")
        self.samples.append(sample)

    def update(self, window, features, wide):
        current_sample = self.samples[-1]
        if (current_sample.sequence_id, current_sample.sample_index) != (
            window.sequence_id,
            window.sample_index,
        ):
            raise ContractError("Shoulder window does not belong to the current sample")
        raw, reason = None, window.quality if features is None else "anchor_track_gap"
        reference_frame = None
        if features is not None and len(self.samples) == 7:
            past = self.samples[0]
            reference_frame = past.source_frame_id
            a = next((p for p in past.poses if p.track_id == window.track_id), None)
            b = next((p for p in current_sample.poses if p.track_id == window.track_id), None)
            if (
                a is not None
                and b is not None
                and reference_frame is not None
                and current_sample.source_frame_id is not None
            ):
                if current_sample.source_frame_id <= reference_frame:
                    reason = "no_distinct_causal_frame"
                else:
                    raw, reason = coherent_shoulder_descent(
                        a.keypoints, b.keypoints, a.bbox_xyxy, self.config["quality"]
                    )
        cue = self.shoulder.update(
            window.sequence_id,
            window.track_id,
            window.sample_index,
            window.end_ms,
            raw,
            reason,
        )
        ratios = [
            status["event_score"] / threshold
            for status, threshold in ((wide, self.wide_threshold), (cue, self.shoulder_threshold))
            if status["event_score"] is not None
        ]
        ratio = max(ratios, default=None)
        closed_before = len(self.combined.events)
        combined = self.combined.update(
            window.sequence_id,
            window.track_id,
            window.sample_index,
            window.end_ms,
            ratio,
            wide["reason"] or cue["reason"],
        )
        if (
            len(self.combined.events) > closed_before
            and self.combined.events[-1]["end_reason"] != "time_gap"
        ):
            self.combined.events[-1]["end_reason"] = wide["reason"] or cue["reason"] or "low_score"
        active = wide["state"] == "suspected_fall" or cue["state"] == "suspected_fall"
        if (combined["state"] == "suspected_fall") != active:
            raise ContractError("Independent K3 fusion changed the strict comparison")
        instantaneous = [
            status["raw_score"] / threshold
            for status, threshold in ((wide, self.wide_threshold), (cue, self.shoulder_threshold))
            if status["raw_score"] is not None
        ]
        status = {
            "raw_score": max(instantaneous, default=None),
            "event_score": ratio,
            "state": "suspected_fall" if active else wide["state"],
            "reason": None if active else wide["reason"],
        }
        row_index = None
        if features is not None:
            values = np.asarray(features, dtype=np.float32)
            if values.shape != (1, 512) or not np.isfinite(values).all():
                raise ContractError("Fresh demo features must be finite [1,512]")
            row_index = len(self.features)
            self.features.append(values[0].copy())
        self.window_rows.append(
            {
                "sequence_id": window.sequence_id,
                "track_id": window.track_id,
                "sample_index": window.sample_index,
                "available_at_ms": window.end_ms,
                "start_ms": window.start_ms,
                "source_frame_id": current_sample.source_frame_id,
                "reference_source_frame_id": reference_frame,
                "row_index": row_index,
                "quality": window.quality,
                "good_frame_ratio": window.good_frame_ratio,
                "wide_raw": wide["raw_score"],
                "wide_k3": wide["event_score"],
                "wide_state": wide["state"],
                "wide_reason": wide["reason"],
                "shoulder_raw": raw,
                "shoulder_k3": cue["event_score"],
                "shoulder_state": cue["state"],
                "shoulder_reason": cue["reason"],
                "combined_k3_ratio": ratio,
                "combined_state": status["state"],
            }
        )
        return status

    def finish(self, end_ms):
        return self.combined.finish(end_ms)

    def save_audit(self, output, metadata, wide_events):
        output = Path(output)
        write_csv(output.with_name(output.stem + "-windows.csv"), self.window_rows)
        write_csv(
            output.with_name(output.stem + "-wide-events.csv"),
            wide_events,
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
        array = np.stack(self.features) if self.features else np.empty((0, 512), dtype=np.float32)
        write_npz(output.with_name(output.stem + "-features.npz"), features=array)
        save_pose_cache(output.with_name(output.stem + "-poses.npz"), self.observations, metadata)
