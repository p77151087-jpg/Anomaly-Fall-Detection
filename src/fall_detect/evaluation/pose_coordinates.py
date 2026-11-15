"""Narrow, non-deployable coordinate intervention. Never fit or calibrate here."""

from collections import deque
from copy import deepcopy
from dataclasses import asdict

import numpy as np

from ..io_utils import ContractError, fingerprint
from ..skeleton.windows import CausalResampler, WindowBuilder

DIAGNOSTIC = "ur-pose-coordinate-diagnostic-20261013-001"
BASE_RECIPE = "713ee59a0853f20641cb7f664df2920359baa0734807acfa5cf86fb87a4e2f5b"
FRAMES = tuple(range(44, 52))
JOINT = 3  # human13 anatomical left elbow; viewer's right


def validate_annotation(annotation, reference, require_human=False):
    """Coordinates must refer to the locked original observations, not another Pose."""
    immutable = (
        "schema_version",
        "diagnostic_id",
        "sequence_id",
        "split",
        "source_hash",
        "base_recipe_hash",
        "pose_array_sha256",
        "pose_metadata_sha256",
    )
    for key in immutable:
        if annotation.get(key) != reference.get(key):
            raise ContractError(f"Coordinate identity mismatch: {key}")
    if annotation["split"] != "validation" or annotation["sequence_id"] != "fall-19":
        raise ContractError("Only locked Validation fall-19 may be intervened on")
    if annotation.get("author_kind") not in {"ai_provisional", "human_self_reported"}:
        raise ContractError("Unknown coordinate author")
    human = annotation.get("author_kind") == "human_self_reported"
    if (
        type(annotation.get("human_confirmed")) is not bool
        or annotation["human_confirmed"] != human
    ):
        raise ContractError("Human attestation must be an explicit boolean")
    if require_human and not human:
        raise ContractError("AI positions are not human corrections")
    if human and (
        not isinstance(annotation.get("reviewer"), str) or not annotation["reviewer"].strip()
    ):
        raise ContractError("Human coordinates require a reviewer")
    if human and annotation.get("reviewed_frame_ids") != list(FRAMES):
        raise ContractError("Human reviewer must explicitly review all eight frames")
    rows = annotation.get("points", [])
    if len(rows) != len(FRAMES):
        raise ContractError("Exactly eight locked frame decisions are required")
    expected = {r["source_frame_id"]: r for r in reference["points"]}
    seen = set()
    for row in rows:
        fid = row.get("source_frame_id")
        if type(fid) is not int or fid not in expected or fid in seen:
            raise ContractError("Duplicate or unlocked frame")
        seen.add(fid)
        ref = expected[fid]
        if type(row.get("track_id")) is not int or type(row.get("joint_index")) is not int:
            raise ContractError("Track and joint IDs must be integers")
        for key in (
            "track_id",
            "joint_index",
            "joint_name",
            "timestamp_ms",
            "image_sha256",
            "frame_size_wh",
            "original_xy",
            "original_confidence",
        ):
            if row.get(key) != ref[key]:
                raise ContractError(f"Locked point field changed: {fid}/{key}")
        if row.get("visibility") not in {"visible", "not_visible_or_uncertain"}:
            raise ContractError("Explicit visibility decision required")
        xy = row.get("xy")
        if row["visibility"] != "visible":
            if xy is not None:
                raise ContractError("Uncertain or hidden joints must not be reconstructed")
            continue
        if not isinstance(xy, list) or len(xy) != 2 or any(type(v) not in (int, float) for v in xy):
            raise ContractError("A visible coordinate needs two numeric values")
        w, h = row["frame_size_wh"]
        if not np.isfinite(xy).all() or not (0 <= xy[0] < w and 0 <= xy[1] < h):
            raise ContractError("Coordinates must be finite and inside the original image")
    return annotation


def apply_coordinates(observations, annotation, reference, require_human=False):
    validate_annotation(annotation, reference, require_human)
    result = deepcopy(observations)
    rows = {r["source_frame_id"]: r for r in annotation["points"]}
    applied = []
    for frame, poses in result:
        if frame.sequence_id != "fall-19":
            raise ContractError("Foreign sequence in coordinate intervention")
        row = rows.get(frame.source_frame_id)
        if row is None:
            continue
        selected = [p for p in poses if p.track_id == row["track_id"]]
        if len(selected) != 1 or frame.timestamp_ms != row["timestamp_ms"]:
            raise ContractError("Locked pose or timestamp not found")
        pose = selected[0]
        if pose.layout_id != "human13" or tuple(pose.frame_size_wh) != tuple(row["frame_size_wh"]):
            raise ContractError("Locked pose layout/size mismatch")
        np.testing.assert_array_equal(pose.keypoints[JOINT, :2], np.float32(row["original_xy"]))
        if float(pose.keypoints[JOINT, 2]) != row["original_confidence"]:
            raise ContractError("Original confidence mismatch")
        if row["visibility"] == "visible":
            pose.keypoints[JOINT, :2] = row["xy"]
            applied.append(frame.source_frame_id)
    if {f.source_frame_id for f, _ in result}.intersection(rows) != set(FRAMES):
        raise ContractError("Missing locked frame")
    return result, applied


def intervention_recipe(annotation):
    return fingerprint(
        {
            "kind": "frozen_model_coordinate_sensitivity_v1",
            "base_recipe_hash": BASE_RECIPE,
            "annotation": annotation,
        }
    )


def assert_same_records(original, derived):
    if [asdict(r) for r in original.records] != [asdict(r) for r in derived.records]:
        raise ContractError("Coordinate change altered quality, tracking or window schedule")
    for key in (
        "n_samples",
        "stride",
        "window_size",
        "quality_counts",
        "end_ms",
        "repeated_samples",
        "missing_samples",
    ):
        if original.metadata[key] != derived.metadata[key]:
            raise ContractError(f"Coordinate change altered timing/quality: {key}")


def window_sources(observations, config):
    """Audit exact past-source membership of each same-track window (no interpolation)."""
    c = config["skeleton"]
    resampler = CausalResampler(
        c["target_fps"], c["resampling_hold_policy"], c["timestamp_tolerance_ms"]
    )
    builder = WindowBuilder(config)
    history = deque(maxlen=c["window_size"])
    times = {f.source_frame_id: f.timestamp_ms for f, _ in observations}
    rows = []
    for frame, poses in observations:
        for sample in resampler.push(frame, poses):
            if (
                sample.source_frame_id is not None
                and times[sample.source_frame_id] > sample.timestamp_ms + 1e-7
            ):
                raise ContractError("Future source used in a window")
            history.append(sample)
            for window in builder.push(sample):
                ids = [
                    s.source_frame_id
                    for s in history
                    if any(p.track_id == window.track_id for p in s.poses)
                ]
                rows.append(
                    {
                        "sample_index": window.sample_index,
                        "track_id": window.track_id,
                        "source_frame_ids": ids,
                        "quality": window.quality,
                    }
                )
    return rows
