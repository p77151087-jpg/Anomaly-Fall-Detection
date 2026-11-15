from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..contracts import RECIPE_ID, FeatureCache, FeatureRecord, FrameRecord, PoseRecord
from ..io_utils import ContractError, read_json, sha256, write_json, write_npz
from ..recipes import RECIPES


def save_pose_cache(path, observations, metadata):
    path = Path(path)
    keypoints, boxes, poses, frames = [], [], [], []
    for frame, frame_poses in observations:
        indices = []
        for pose in frame_poses:
            indices.append(len(poses))
            row = asdict(pose)
            keypoints.append(row.pop("keypoints"))
            boxes.append(row.pop("bbox_xyxy"))
            poses.append(row)
        frames.append(
            {
                "sequence_id": frame.sequence_id,
                "source_frame_id": frame.source_frame_id,
                "sample_index": frame.sample_index,
                "timestamp_ms": frame.timestamp_ms,
                "width": frame.width,
                "height": frame.height,
                "timestamp_estimated": frame.timestamp_estimated,
                "source_missing": frame.source_missing,
                "source_fps": frame.source_fps,
                "source_fps_estimated": frame.source_fps_estimated,
                "pose_indices": indices,
            }
        )
    recipe_id = metadata.get("recipe_id", RECIPE_ID)
    if recipe_id not in RECIPES:
        raise ContractError("Unknown pose cache recipe")
    spec = RECIPES[recipe_id]
    if any(p["layout_id"] != spec["layout"] or p["recipe_id"] != recipe_id for p in poses):
        raise ContractError("Pose cache mixes layouts/recipes")
    if any(p.shape != (spec["joints"], 3) for p in keypoints):
        raise ContractError("Pose keypoint shape mismatch")
    write_npz(
        path,
        keypoints=np.asarray(keypoints, dtype=np.float32).reshape(-1, spec["joints"], 3),
        boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
    )
    write_json(
        path.with_suffix(".json"),
        {
            **metadata,
            "recipe_id": recipe_id,
            "layout_id": spec["layout"],
            "schema_version": 1,
            "kind": "pose_cache",
            "frames": frames,
            "poses": poses,
            "array_sha256": sha256(path),
        },
    )


def load_pose_cache(path, expected):
    path = Path(path)
    meta = read_json(path.with_suffix(".json"))
    check_metadata(meta, expected, "pose_cache")
    if meta["array_sha256"] != sha256(path):
        raise ContractError("Pose cache checksum mismatch")
    with np.load(path, allow_pickle=False) as arrays:
        keypoints, boxes = arrays["keypoints"], arrays["boxes"]
    recipe_id = meta.get("recipe_id", RECIPE_ID)
    if recipe_id not in RECIPES:
        raise ContractError("Unknown pose cache recipe")
    spec = RECIPES[recipe_id]
    if keypoints.shape != (len(meta["poses"]), spec["joints"], 3) or boxes.shape != (
        len(meta["poses"]),
        4,
    ):
        raise ContractError("Pose cache array/metadata count mismatch")
    if not np.isfinite(keypoints).all() or not np.isfinite(boxes).all():
        raise ContractError("Pose cache contains non-finite values")
    observations, used = [], []
    previous = -1.0
    for frame_meta in meta["frames"]:
        record = dict(frame_meta)
        indices = record.pop("pose_indices")
        if record["timestamp_ms"] <= previous:
            raise ContractError("Pose cache timestamps are not increasing")
        previous = record["timestamp_ms"]
        frame = FrameRecord(**record, image_bgr=None)
        frame_poses = []
        for index in indices:
            if not 0 <= index < len(keypoints):
                raise ContractError("Invalid pose row index")
            fields = dict(meta["poses"][index])
            if fields["layout_id"] != spec["layout"] or fields["recipe_id"] != recipe_id:
                raise ContractError("Pose cache record layout/recipe mismatch")
            fields["frame_size_wh"] = tuple(fields["frame_size_wh"])
            if (
                fields["sequence_id"] != frame.sequence_id
                or fields["source_frame_id"] != frame.source_frame_id
            ):
                raise ContractError("Pose/frame association mismatch")
            frame_poses.append(
                PoseRecord(**fields, keypoints=keypoints[index], bbox_xyxy=boxes[index])
            )
            used.append(index)
        observations.append((frame, frame_poses))
    if sorted(used) != list(range(len(keypoints))):
        raise ContractError("Duplicate or unreferenced pose rows")
    return observations, meta


def check_metadata(meta, expected, kind):
    if meta.get("schema_version") != 1 or meta.get("kind") != kind:
        raise ContractError("Cache schema/kind mismatch; recompute with prepare")
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ContractError(f"Cache {key} mismatch; recompute with prepare")


def validate_feature_cache(cache):
    features, records, meta = cache.features, cache.records, cache.metadata
    dim = meta["feature_dim"]
    if meta.get("recipe_id") not in RECIPES or dim != RECIPES[meta["recipe_id"]]["dim"]:
        raise ContractError("Feature dimension/recipe mismatch")
    if features.ndim != 2 or features.shape[1] != dim or not np.isfinite(features).all():
        raise ContractError("Invalid feature shape or non-finite feature values")
    indices, identities = [], set()
    for row in records:
        if (
            row.schema_version != 1
            or row.recipe_id != meta["recipe_id"]
            or row.feature_dim != dim
            or row.sequence_id != meta["sequence_id"]
            or row.split != meta["split"]
        ):
            raise ContractError("Feature record metadata mismatch")
        if row.available_at_ms < row.end_ms or row.end_ms < row.start_ms:
            raise ContractError("Noncausal feature timestamps")
        identity = (row.sample_index, row.track_id)
        if identity in identities:
            raise ContractError("Duplicate feature endpoint/track")
        identities.add(identity)
        if row.quality == "valid":
            if row.track_id is None or row.row_index is None:
                raise ContractError("Valid feature needs a real track and row")
            indices.append(row.row_index)
        elif row.row_index is not None:
            raise ContractError("Invalid window cannot have a formal feature row")
    if sorted(indices) != list(range(len(features))):
        raise ContractError("Feature row indices must match arrays exactly once")
    size, stride = meta["window_size"], meta["stride"]
    expected_endpoints = set(range(size - 1, meta["n_samples"], stride))
    if {r.sample_index for r in records} != expected_endpoints:
        raise ContractError("Cache omitted or invented scheduled endpoints")
    return cache


def save_feature_cache(path, cache):
    validate_feature_cache(cache)
    path = Path(path)
    write_npz(path, features=cache.features)
    write_json(
        path.with_suffix(".json"),
        {
            **cache.metadata,
            "schema_version": 1,
            "kind": "feature_cache",
            "array_sha256": sha256(path),
            "records": [asdict(r) for r in cache.records],
        },
    )


def load_feature_cache(path, expected=None):
    path = Path(path)
    meta = read_json(path.with_suffix(".json"))
    check_metadata(meta, expected or {}, "feature_cache")
    if meta["array_sha256"] != sha256(path):
        raise ContractError("Feature cache checksum mismatch")
    with np.load(path, allow_pickle=False) as arrays:
        features = arrays["features"]
    records = [FeatureRecord(**row) for row in meta.pop("records")]
    return validate_feature_cache(FeatureCache(features, records, meta))
