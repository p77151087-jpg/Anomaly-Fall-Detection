"""Reuse observed poses for resampling or feature-window scheduling changes."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from ..config import project_path, recipe_fingerprint
from ..io_utils import ContractError, sha256
from .cache import load_pose_cache, save_pose_cache
from .readers import read_source


def validate_pose_reuse(index, config, manifest, weight_hashes):
    if index.get("source_kind") != "real" or index.get("split_manifest") != manifest:
        raise ContractError("Pose reuse requires the same complete real split manifest")
    previous = index.get("preparation_config")
    if previous is None:
        raise ContractError("Pose reuse requires saved preparation configuration")
    if {k: v["sha256"] for k, v in index["weights"].items()} != weight_hashes:
        raise ContractError("Pose reuse weight hashes differ")
    comparison = deepcopy(config)
    comparison["skeleton"]["resampling_hold_policy"] = previous["skeleton"].get(
        "resampling_hold_policy", "target_interval"
    )
    comparison["skeleton"]["timestamp_tolerance_ms"] = previous["skeleton"].get(
        "timestamp_tolerance_ms", 0.0
    )
    comparison["skeleton"]["stride"] = previous["skeleton"]["stride"]
    if recipe_fingerprint(comparison, weight_hashes) != index["recipe_hash"]:
        raise ContractError(
            "Pose reuse permits only resampling hold-policy/tolerance or window-stride changes"
        )


def reuse_pose_observations(index, sid, metadata, directory, destination, config):
    row = index["sequences"].get(sid, {})
    if row.get("status") != "prepared":
        raise ContractError(f"{sid}: source pose cache is not prepared")
    path = project_path(row["pose_path"])
    if Path(destination).resolve() == path.resolve():
        raise ContractError("Pose reuse must not overwrite its source cache")
    expected = {**metadata, "recipe_hash": index["recipe_hash"]}
    expected.setdefault("annotation_intervals_ms", None)
    observations, old_meta = load_pose_cache(path, expected)
    if not observations:
        raise ContractError("Cannot reuse an empty observation sequence")
    # Source identity was rehashed by prepare before this call. Decode its first frame
    # to obtain container FPS, not an estimate made from future cached timestamps.
    if directory.is_file():
        reader = read_source(
            directory, sequence_id=sid, fallback_fps=config["data"]["fallback_fps"], limit=1
        )
        try:
            first = next(reader)
        finally:
            reader.close()
        old_first = observations[0][0]
        if (
            first.source_frame_id != old_first.source_frame_id
            or abs(first.timestamp_ms - old_first.timestamp_ms) > 1e-7
            or (first.width, first.height) != (old_first.width, old_first.height)
        ):
            raise ContractError("Source header/first frame disagrees with cached observations")
        observations = [
            (
                replace(
                    frame,
                    source_fps=first.source_fps,
                    source_fps_estimated=first.source_fps_estimated,
                ),
                poses,
            )
            for frame, poses in observations
        ]
    provenance = {
        "pose_array_sha256": sha256(path),
        "pose_metadata_sha256": sha256(path.with_suffix(".json")),
        "original_recipe_hash": index["recipe_hash"],
        "source_fps": observations[0][0].source_fps,
        "source_fps_estimated": observations[0][0].source_fps_estimated,
        "operation": "unchanged observed keypoints and track IDs; resampling/window scheduling only",
    }
    save_pose_cache(
        destination,
        observations,
        {**old_meta, **metadata, "reused_pose_only": provenance},
    )
    if sha256(destination) != provenance["pose_array_sha256"]:
        raise ContractError("Pose arrays changed during observation reuse")
    return observations
