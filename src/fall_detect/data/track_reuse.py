"""Conservative cache reuse for an explicit floating-point time tolerance change."""

from copy import deepcopy

from ..config import recipe_fingerprint
from ..io_utils import ContractError
from ..pose.lightweight import tracking_gap_exceeded


def validate_tracking_reuse(index, config):
    old = index.get("preparation_config")
    if not old or config["pose"]["backend"] != "lightweight_openpose":
        raise ContractError("Tracking reuse requires a saved real OpenPose configuration")
    hashes = {role: entry["sha256"] for role, entry in index["weights"].items()}
    if recipe_fingerprint(old, hashes) != index["recipe_hash"]:
        raise ContractError("Reuse index recipe does not match its saved configuration")
    equivalent = deepcopy(config)
    equivalent["pose"]["track_gap_tolerance_ms"] = old["pose"].get("track_gap_tolerance_ms", 0.0)
    if recipe_fingerprint(equivalent, hashes) != index["recipe_hash"]:
        raise ContractError("Reuse may change only tracking time tolerance")


def identical_tracking_resets(observations, old, new):
    previous = None
    for frame, _ in observations:
        # Both backends unconditionally reset on an explicitly missing frame.
        if not frame.source_missing and tracking_gap_exceeded(
            previous, frame.timestamp_ms, old
        ) != tracking_gap_exceeded(previous, frame.timestamp_ms, new):
            return False
        previous = frame.timestamp_ms
    return True
