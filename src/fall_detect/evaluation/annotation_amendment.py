"""Apply an explicit human clarification without inventing precise event times."""

from copy import deepcopy

from ..io_utils import ContractError
from .frame_review import validate_annotation


def amend_phase_annotation(original, manifest, manifest_sha256, original_sha256, data, amendment):
    """Validate the original export, then derive a separate phase-only schema v2.

    This bounded amendment accepts the user's two corrections for fall-19.
    The old schema v1 validator is used only on the unchanged original. Its
    ordered bounds check cannot establish the semantic meaning of those bounds.
    """
    validate_annotation(original, manifest, manifest_sha256)
    if (
        original.get("sequence_id") != "fall-19"
        or amendment.get("schema_version") != 1
        or amendment.get("origin") != "human_text_clarification"
        or amendment.get("human_attested") is not True
        or amendment.get("original_sha256") != original_sha256
        or amendment.get("range_semantics") != "whole_posture_phases"
        or amendment.get("pose_corrections")
        != [
            {"source_frame_id": 1, "before": "matches_person", "after": "missing"},
            {"source_frame_id": 2, "before": "matches_person", "after": "missing"},
        ]
    ):
        raise ContractError("Requires this source's explicit human phase/pose clarification")
    frames = original["frames"]
    ranges = {}
    for phase, key in [
        ("transition", "onset_bounds_frame_ids"),
        ("ground", "impact_bounds_frame_ids"),
    ]:
        ids = [r["source_frame_id"] for r in frames if r["phase"] == phase]
        if not ids or ids != list(range(ids[0], ids[-1] + 1)) or original[key] != [ids[0], ids[-1]]:
            raise ContractError("Clarified ranges must match whole contiguous posture phases")
        ranges[phase] = [ids[0], ids[-1]]
    if len(data["frames"]) != len(frames) or any(
        any(a[k] != b[k] for k in ["source_frame_id", "timestamp_ms", "source_sha256"])
        for a, b in zip(frames, data["frames"])
    ):
        raise ContractError("Frozen observations must match the original export")
    result = deepcopy(original)
    for correction in amendment["pose_corrections"]:
        idx = next(
            (
                i
                for i, r in enumerate(frames)
                if r["source_frame_id"] == correction["source_frame_id"]
            ),
            None,
        )
        if (
            idx is None
            or frames[idx]["pose_check"] != correction["before"]
            or data["frames"][idx]["poses"]
        ):
            raise ContractError("Only explicitly corrected absent cached poses may change")
        result["frames"][idx]["pose_check"] = correction["after"]
    result.update(
        schema_version=2,
        original_schema_version=original["schema_version"],
        derived_from_sha256=original_sha256,
        amendment_origin="human_text_clarification",
        temporal_semantics="whole_posture_phases; precise onset and impact not supplied",
        phase_ranges_frame_ids=ranges,
        onset_bounds_frame_ids=None,
        impact_bounds_frame_ids=None,
        precise_time_ground_truth=False,
        used_for_training=False,
        used_for_calibration=False,
    )
    return result
