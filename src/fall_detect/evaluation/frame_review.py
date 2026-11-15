"""Causal diagnostic alignment and strictly separate self-reported human labels."""

import math

from ..io_utils import ContractError

PHASES = {"upright", "transition", "ground", "uncertain"}
VISIBILITY = {"full", "partial", "out_of_frame", "uncertain"}
POSE_CHECKS = {"matches_person", "wrong_or_drift", "missing", "uncertain"}


def latest_windows(rows, timestamp_ms, freshness_ms=2000 / 30):
    """Latest scheduled result per track at/before RGB time; expose age, never backfill."""
    if not math.isfinite(timestamp_ms):
        raise ContractError("Finite source time required")
    selected = {}
    for row in sorted(rows, key=lambda r: (r["available_at_ms"], r["sample_index"])):
        if row["available_at_ms"] <= timestamp_ms:
            selected[row["track_id"]] = row
    return [
        {
            **r,
            "age_ms": timestamp_ms - r["available_at_ms"],
            "fresh": timestamp_ms - r["available_at_ms"] <= freshness_ms,
        }
        for r in selected.values()
    ]


def annotation_template(manifest, manifest_sha256):
    return {
        "schema_version": 1,
        "sequence_id": manifest["sequence_id"],
        "split": manifest["split"],
        "manifest_sha256": manifest_sha256,
        "annotation_origin": "human_self_reported",
        "reviewer": "",
        "human_attested": False,
        "onset_bounds_frame_ids": None,
        "impact_bounds_frame_ids": None,
        "notes": "",
        "frames": [
            {
                "source_frame_id": r["source_frame_id"],
                "timestamp_ms": r["timestamp_ms"],
                "source_sha256": r["source_sha256"],
                "phase": "unreviewed",
                "visibility": "unreviewed",
                "pose_check": "unreviewed",
                "reviewed": False,
            }
            for r in manifest["frames"]
        ],
    }


def validate_annotation(annotation, manifest, manifest_sha256):
    """Validate an attestation; cannot independently establish a person's identity or GT."""
    if (
        manifest["split"] != "validation"
        or annotation.get("schema_version") != 1
        or annotation.get("sequence_id") != manifest["sequence_id"]
        or annotation.get("split") != "validation"
        or annotation.get("manifest_sha256") != manifest_sha256
        or annotation.get("annotation_origin") != "human_self_reported"
        or annotation.get("human_attested") is not True
        or not isinstance(annotation.get("reviewer"), str)
        or not annotation["reviewer"].strip()
        or annotation["reviewer"].strip().lower() in {"ai", "codex", "assistant", "chatgpt"}
    ):
        raise ContractError("Requires explicit human self-attestation on this Validation source")
    supplied = annotation.get("frames", [])
    if not isinstance(supplied, list) or len(supplied) != len(manifest["frames"]):
        raise ContractError("All original frames must be reviewed, including uncertainty")
    for actual, source in zip(supplied, manifest["frames"]):
        if not isinstance(actual, dict) or (
            any(
                actual.get(k) != source[k]
                for k in ["source_frame_id", "timestamp_ms", "source_sha256"]
            )
            or actual.get("reviewed") is not True
            or actual.get("phase") not in PHASES
            or actual.get("visibility") not in VISIBILITY
            or actual.get("pose_check") not in POSE_CHECKS
        ):
            raise ContractError("Unreviewed/misaligned frame or invalid label")
    ids = {r["source_frame_id"] for r in manifest["frames"]}
    bounds = []
    for name in ["onset_bounds_frame_ids", "impact_bounds_frame_ids"]:
        pair = annotation.get(name)
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(type(x) is not int or x not in ids for x in pair)
            or pair[0] > pair[1]
        ):
            raise ContractError("Human time bounds must be ordered original frame IDs")
        bounds.append(pair)
    if bounds[0][1] > bounds[1][0]:
        raise ContractError("Onset uncertainty must precede impact uncertainty")
    return {
        "validated": True,
        "origin": "human_self_reported",
        "reviewer": annotation["reviewer"].strip(),
        "reviewed_frames": len(supplied),
        "identity_independently_verified": False,
        "used_for_training": False,
        "used_for_calibration": False,
        "scope": "Validation diagnosis only; not complete independent ground truth",
    }
