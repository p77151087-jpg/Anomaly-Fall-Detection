"""Interpret human onset/impact uncertainty without inventing an exact fall time."""

import math

from ..io_utils import ContractError


def checked_interval(value):
    if value is None:
        return None
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in value)
        or value[0] > value[1]
    ):
        raise ContractError("Expected two ordered finite nonnegative time bounds")
    return [float(v) for v in value]


def temporal_relation(time_ms, onset_ms, impact_ms):
    onset = checked_interval(onset_ms)
    impact = checked_interval(impact_ms)
    if onset is not None and impact is not None and onset[0] > impact[1]:
        raise ContractError("Impact definitely precedes onset")
    if time_ms is not None and (
        type(time_ms) not in (int, float) or not math.isfinite(time_ms) or time_ms < 0
    ):
        raise ContractError("Alarm timestamp must be finite and nonnegative")
    result = {
        "relation": "no_alarm" if time_ms is None else "unresolved_time_bounds",
        "possible_fall_envelope_ms": [onset[0], impact[1]]
        if onset is not None and impact is not None
        else None,
        "guaranteed_fall_interval_ms": [onset[1], impact[0]]
        if onset is not None and impact is not None and onset[1] <= impact[0]
        else None,
        "inside_possible_fall_envelope": None,
        "inside_guaranteed_fall_interval": None,
        "definitely_before_impact": None,
        "definitely_after_impact": None,
        "inside_impact_uncertainty": None,
        "delay_from_onset_bounds_ms": None,
        "delay_from_impact_bounds_ms": None,
    }
    if time_ms is None:
        return result
    time = float(time_ms)
    if onset is not None:
        result["delay_from_onset_bounds_ms"] = [time - onset[1], time - onset[0]]
    if impact is not None:
        result.update(
            definitely_before_impact=time < impact[0],
            definitely_after_impact=time > impact[1],
            inside_impact_uncertainty=impact[0] <= time <= impact[1],
            delay_from_impact_bounds_ms=[time - impact[1], time - impact[0]],
        )
    if onset is None or impact is None:
        return result
    result["inside_possible_fall_envelope"] = onset[0] <= time <= impact[1]
    result["inside_guaranteed_fall_interval"] = (
        onset[1] <= time <= impact[0] if onset[1] <= impact[0] else False
    )
    if time < onset[0]:
        relation = "before_onset"
    elif time > impact[1]:
        relation = "after_impact"
    elif onset[0] <= time <= onset[1] and impact[0] <= time <= impact[1]:
        relation = "overlapping_onset_impact_uncertainty"
    elif impact[0] <= time <= impact[1]:
        relation = "impact_uncertainty"
    elif time < onset[1]:
        relation = "onset_uncertainty"
    else:
        relation = "guaranteed_fall_interval"
    result["relation"] = relation
    return result


def frame_time_bounds(mark, frames):
    if not isinstance(mark, dict) or mark.get("status") not in {
        "bounded",
        "uncertain",
        "outside_clip",
    }:
        raise ContractError("Invalid human timing status")
    if mark["status"] != "bounded":
        if mark.get("bounds_frame_ids") is not None:
            raise ContractError("Unavailable timing must have null bounds")
        return None
    bounds = mark.get("bounds_frame_ids")
    by_id = {f["source_frame_id"]: f for f in frames}
    if len(by_id) != len(frames):
        raise ContractError("Duplicate source frame IDs")
    if (
        not isinstance(bounds, list)
        or len(bounds) != 2
        or any(type(n) is not int or n not in by_id for n in bounds)
        or bounds[0] > bounds[1]
    ):
        raise ContractError("Human bounds must reference ordered observed frame IDs")
    return checked_interval([by_id[n]["timestamp_ms"] for n in bounds])
