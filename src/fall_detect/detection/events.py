import math
from collections import deque

from ..io_utils import ContractError


def above_threshold(score, threshold):
    if not math.isfinite(float(score)) or not math.isfinite(float(threshold)):
        raise ContractError("Scores and threshold must be finite float64 values")
    return float(score) > float(threshold)


class EventState:
    def __init__(self, threshold, consecutive_windows=3, stride=5):
        self.threshold = float(threshold)
        self.k, self.stride = consecutive_windows, stride
        self.tracks = {}
        self.events = []

    def _close(self, state, reason, end_ms):
        active = state["active"]
        if active is not None:
            self.events.append({**active, "end_ms": float(end_ms), "end_reason": reason})
        state["active"] = None

    def update(self, sequence_id, track_id, sample_index, available_at_ms, raw_score, reason=None):
        key = (sequence_id, track_id)
        state = self.tracks.setdefault(
            key, {"history": deque(maxlen=self.k), "last_index": None, "active": None}
        )
        if state["last_index"] is not None and sample_index <= state["last_index"]:
            raise ContractError("Event windows must be strictly increasing per track")
        adjacent = state["last_index"] is None or sample_index == state["last_index"] + self.stride
        if not adjacent:
            self._close(state, "time_gap", available_at_ms)
            state["history"].clear()
        state["last_index"] = sample_index
        if raw_score is None:
            self._close(state, reason or "unknown", available_at_ms)
            state["history"].clear()
            return {
                "raw_score": None,
                "event_score": None,
                "state": "unknown",
                "reason": reason or "unknown",
            }
        raw_score = float(raw_score)
        above_threshold(raw_score, self.threshold)
        state["history"].append(raw_score)
        event_score = min(state["history"]) if len(state["history"]) == self.k else None
        active = event_score is not None and above_threshold(event_score, self.threshold)
        if active:
            if state["active"] is None:
                state["active"] = {
                    "sequence_id": sequence_id,
                    "track_id": track_id,
                    "confirmed_start_ms": float(available_at_ms),
                    "last_active_ms": float(available_at_ms),
                    "peak_score": raw_score,
                }
            else:
                state["active"]["last_active_ms"] = float(available_at_ms)
                state["active"]["peak_score"] = max(state["active"]["peak_score"], raw_score)
        else:
            self._close(state, "low_score", available_at_ms)
        return {
            "raw_score": raw_score,
            "event_score": event_score,
            "state": "suspected_fall"
            if active
            else "normal"
            if event_score is not None
            else "warming_up",
            "reason": None if event_score is not None else "insufficient_consecutive_windows",
        }

    def finish(self, end_ms, reason="sequence_end"):
        for state in self.tracks.values():
            self._close(state, reason, end_ms)
        self.tracks.clear()
        return self.events


def score_sequence(cache, scores, threshold, k=3):
    if len(scores) != len(cache.features):
        raise ContractError("Score count differs from feature rows")
    state = EventState(threshold, k, cache.metadata["stride"])
    windows, valid_endpoints = [], set()
    for record in sorted(cache.records, key=lambda r: (r.sample_index, r.track_id or -1)):
        raw_score = float(scores[record.row_index]) if record.row_index is not None else None
        output = state.update(
            record.sequence_id,
            record.track_id,
            record.sample_index,
            record.available_at_ms,
            raw_score,
            record.quality if raw_score is None else None,
        )
        if raw_score is not None:
            valid_endpoints.add(record.sample_index)
        windows.append(
            {
                "sequence_id": record.sequence_id,
                "track_id": record.track_id,
                "sample_index": record.sample_index,
                "window_id": record.window_id,
                "start_ms": record.start_ms,
                "end_ms": record.end_ms,
                "available_at_ms": record.available_at_ms,
                **output,
            }
        )
    event_scores = [w["event_score"] for w in windows if w["event_score"] is not None]
    sequence_score = max(event_scores) if event_scores else None
    total = len({r.sample_index for r in cache.records})
    events = state.finish(cache.metadata.get("end_ms", 0))
    result = {
        "sequence_id": cache.metadata["sequence_id"],
        "sequence_score": sequence_score,
        "prediction": int(above_threshold(sequence_score, threshold))
        if sequence_score is not None
        else None,
        "unknown_reason": None if sequence_score is not None else "no_legal_event_score",
        "n_scheduled_endpoints": total,
        "n_valid_endpoints": len(valid_endpoints),
        "window_coverage": len(valid_endpoints) / total if total else None,
        "first_alarm_ms": events[0]["confirmed_start_ms"] if events else None,
    }
    return result, windows, events
