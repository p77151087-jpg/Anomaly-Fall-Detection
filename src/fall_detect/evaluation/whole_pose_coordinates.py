"""Validate human review data and apply only visible corrections, without fitting."""

import hashlib
import json
from collections import Counter
from copy import deepcopy
from datetime import datetime

import numpy as np

from ..io_utils import ContractError, fingerprint
from ..skeleton.preprocessing import HUMAN13_NAMES
from .pose_coordinates import BASE_RECIPE

ID = "ur-fall19-whole-coordinate-20261013-001"
PARENT = "ur-fall19-joint-localization-20261013-001"
REVIEW_FRAMES = (44, 47, 49)
STATUSES = {
    "matches_original",
    "mislocated_visible",
    "mislocated_location_uncertain",
    "not_visible_or_uncertain",
    "unreviewed",
}


def validate_review(annotation, bundle):
    if bundle.get("sequence_id") != "fall-19" or bundle.get("split") != "validation":
        raise ContractError("Only original Validation fall-19 bundle")
    content = {k: v for k, v in bundle.items() if k != "bundle_sha256"}
    payload = json.dumps(content, ensure_ascii=False, sort_keys=True, allow_nan=False)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if bundle.get("bundle_sha256") != digest:
        raise ContractError("Bundle digest mismatch")
    if tuple(bundle["names"]) != tuple(HUMAN13_NAMES):
        raise ContractError("Wrong joint mapping")
    if annotation.get("schema_version") != 1 or type(annotation["schema_version"]) is not int:
        raise ContractError("Review schema must be integer 1")
    for key in (
        "diagnostic_id",
        "sequence_id",
        "split",
        "pose_array_sha256",
        "pose_metadata_sha256",
        "source_manifest_sha256",
    ):
        if annotation.get(key) != bundle.get(key):
            raise ContractError(f"Review identity mismatch: {key}")
    if bundle["diagnostic_id"] != PARENT or annotation.get("parent_bundle_sha256") != digest:
        raise ContractError("Wrong parent review")
    if (
        annotation.get("author_kind") != "human_self_reported"
        or annotation.get("human_confirmed") is not True
    ):
        raise ContractError("Explicit human self-report required")
    if not isinstance(annotation.get("reviewer"), str) or not annotation["reviewer"].strip():
        raise ContractError("Reviewer required")
    try:
        reviewed = datetime.fromisoformat(annotation["reviewed_at"])
        if reviewed.tzinfo is None:
            raise ValueError("timezone required")
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ContractError("Review date with timezone required") from exc
    expected = {f["source_frame_id"]: f for f in bundle["frames"]}
    frames = annotation.get("frames")
    if not isinstance(frames, list) or len(frames) != 3 or set(expected) != set(REVIEW_FRAMES):
        raise ContractError("Exactly three locked frames")
    seen, counts, corrections = set(), Counter(), []
    for row in frames:
        fid = row.get("source_frame_id")
        if type(fid) is not int or fid not in expected or fid in seen:
            raise ContractError("Duplicate or unlocked frame")
        seen.add(fid)
        ref = expected[fid]
        if type(row.get("track_id")) is not int or row["track_id"] != 1:
            raise ContractError("Track 1 required")
        for key in ("track_id", "timestamp_ms", "image_sha256", "frame_size_wh"):
            if row.get(key) != ref[key]:
                raise ContractError(f"Changed frame identity: {fid}/{key}")
        points = row.get("points")
        if not isinstance(points, list) or len(points) != 13:
            raise ContractError("All 13 joint decisions required")
        joints = set()
        for point in points:
            j = point.get("joint_index")
            if type(j) is not int or not 0 <= j < 13 or j in joints:
                raise ContractError("Duplicate or unlocked joint")
            joints.add(j)
            original = ref["keypoints"][j]
            if (
                point.get("joint_name") != HUMAN13_NAMES[j]
                or point.get("original_xy") != original[:2]
                or point.get("original_confidence") != original[2]
            ):
                raise ContractError("Original joint identity changed")
            status, xy = point.get("status"), point.get("xy")
            if status not in STATUSES or (j >= 7 and status == "unreviewed"):
                raise ContractError("Explicit required joint review missing")
            counts[status] += 1
            if status in {"matches_original", "mislocated_visible"}:
                if (
                    not isinstance(xy, list)
                    or len(xy) != 2
                    or any(type(v) not in (int, float) for v in xy)
                    or not np.isfinite(xy).all()
                ):
                    raise ContractError("Finite visible XY required")
                width, height = row["frame_size_wh"]
                if not (0 <= xy[0] < width and 0 <= xy[1] < height):
                    raise ContractError("Visible coordinate outside original image")
                if status == "matches_original" and xy != original[:2]:
                    raise ContractError("Matching joint must preserve original XY")
                if status == "mislocated_visible":
                    corrections.append(
                        {
                            "source_frame_id": fid,
                            "track_id": 1,
                            "joint_index": j,
                            "joint_name": HUMAN13_NAMES[j],
                            "original_xy": original[:2],
                            "xy": xy,
                            "displacement_px": float(np.linalg.norm(np.array(xy) - original[:2])),
                        }
                    )
            elif xy is not None:
                raise ContractError("Uncertain or unreviewed joints must have null XY")
    return {"counts": dict(counts), "corrections": corrections, "reviewed_frames": sorted(seen)}


def apply_review(observations, annotation, bundle):
    audit = validate_review(annotation, bundle)
    result = deepcopy(observations)
    reviews = {f["source_frame_id"]: f for f in annotation["frames"]}
    seen = set()
    for frame, poses in result:
        if frame.sequence_id != "fall-19":
            raise ContractError("Foreign sequence; no Test intervention")
        review = reviews.get(frame.source_frame_id)
        if review is None:
            continue
        if frame.source_frame_id in seen:
            raise ContractError("Duplicate observed frame")
        seen.add(frame.source_frame_id)
        selected = [p for p in poses if p.track_id == 1]
        if len(selected) != 1 or frame.timestamp_ms != review["timestamp_ms"]:
            raise ContractError("Original same-track observation unavailable")
        pose = selected[0]
        if pose.layout_id != "human13" or list(pose.frame_size_wh) != review["frame_size_wh"]:
            raise ContractError("Original layout/size mismatch")
        for point in review["points"]:
            j = point["joint_index"]
            if (
                pose.keypoints[j, :2].astype(float).tolist() != point["original_xy"]
                or float(pose.keypoints[j, 2]) != point["original_confidence"]
            ):
                raise ContractError("Observed XY/confidence mismatch")
            if point["status"] == "mislocated_visible":
                pose.keypoints[j, :2] = point["xy"]
    if seen != set(REVIEW_FRAMES):
        raise ContractError("Missing reviewed frame; never interpolate")
    return result, audit


def recipe_hash(annotation, raw_sha256):
    return fingerprint(
        {
            "kind": "frozen_whole_human_visible_xy_control_v1",
            "id": ID,
            "base_recipe_hash": BASE_RECIPE,
            "raw_annotation_sha256": raw_sha256,
            "annotation": annotation,
            "uncertain_policy": "retain_original",
            "fit": False,
            "calibrate": False,
            "combine_old_elbow": False,
        }
    )
