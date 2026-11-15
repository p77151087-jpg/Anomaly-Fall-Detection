import numpy as np
import pytest

from fall_detect.detection.events import EventState, above_threshold
from fall_detect.evaluation.metrics import sequence_metrics
from fall_detect.io_utils import ContractError, read_json, write_json
from fall_detect.training.experiment import choose_threshold


def feed(scores, threshold=1.0, tracks=None, indices=None):
    state = EventState(threshold)
    rows = [
        state.update(
            "synthetic",
            tracks[i] if tracks else 1,
            indices[i] if indices else 29 + i * 5,
            1000 + i * 167,
            score,
            "track_gap",
        )
        for i, score in enumerate(scores)
    ]
    return rows, state.finish(3000)


def test_three_high_confirm_and_low_ends_alarm():
    rows, events = feed([2.0, 3.0, 4.0, 0.5, 3.0, 4.0, 5.0])
    assert [r["state"] for r in rows] == [
        "warming_up",
        "warming_up",
        "suspected_fall",
        "normal",
        "normal",
        "normal",
        "suspected_fall",
    ]
    assert len(events) == 2
    assert events[0]["confirmed_start_ms"] == 1334
    assert events[0]["end_reason"] == "low_score"


def test_low_windows_are_evaluable_and_unknown_resets():
    rows, events = feed([0.1, 0.2, 0.3, None, 2, 2, 2])
    assert rows[2]["event_score"] == 0.1 and rows[2]["state"] == "normal"
    assert rows[3]["state"] == "unknown" and rows[4]["state"] == "warming_up"
    assert rows[6]["state"] == "suspected_fall"
    assert len(events) == 1


def test_tracks_and_gap_cannot_pool_three_high_windows():
    rows, events = feed([2, 2, 2], tracks=[1, 2, 1])
    assert all(r["event_score"] is None for r in rows) and events == []
    rows, events = feed([2, 2, 2], indices=[29, 34, 44])
    assert all(r["event_score"] is None for r in rows) and events == []


def test_calibration_streaming_float64_nextafter_equivalence(tmp_path):
    value = float(np.float32(0.25))
    threshold = float(np.nextafter(value, -np.inf))
    write_json(tmp_path / "threshold.json", {"value": threshold})
    assert not above_threshold(value, value)
    assert above_threshold(value, read_json(tmp_path / "threshold.json")["value"])
    assert feed([value] * 3, value)[0][-1]["state"] == "normal"
    assert feed([value] * 3, threshold)[0][-1]["state"] == "suspected_fall"
    rng = np.random.default_rng(42)
    for _ in range(10):
        scores = rng.random(20).tolist()
        scores[7] = None
        event_scores = feed(scores, 0)[0]
        sequence_score = max(r["event_score"] for r in event_scores if r["event_score"] is not None)
        for threshold in (sequence_score, float(np.nextafter(sequence_score, -np.inf)), 0.5):
            assert any(
                r["state"] == "suspected_fall" for r in feed(scores, threshold)[0]
            ) == above_threshold(sequence_score, threshold)


def test_calibration_candidates_unknown_and_tie_break():
    rows = [
        {"sequence_id": "normal", "label": 0, "sequence_score": 1.0},
        {"sequence_id": "fall", "label": 1, "sequence_score": 3.0},
        {"sequence_id": "unknown", "label": 1, "sequence_score": None},
    ]
    selected, candidates = choose_threshold(rows)
    assert selected["threshold"] == 1.0
    assert selected["metrics"]["n_unknown"] == 1
    assert len(candidates) == 3 and candidates[0]["threshold"] < 1.0
    assert candidates[-1]["threshold"] == 3.0
    with pytest.raises(ContractError):
        choose_threshold(rows[:1])


def test_hand_calculated_metrics_unknown_never_tn():
    pairs = [(1, 1), (1, 1), (1, 0), (1, None), (0, 0), (0, 1), (0, None)]
    rows = [
        {
            "sequence_id": str(i),
            "label": y,
            "prediction": p,
            "n_scheduled_endpoints": 10,
            "n_valid_endpoints": 5 if p is not None else 0,
        }
        for i, (y, p) in enumerate(pairs)
    ]
    metrics = sequence_metrics(rows)
    assert [metrics[k] for k in ("TP", "FP", "FN", "TN")] == [2, 1, 1, 1]
    assert metrics["sequence_fall_f1"] == pytest.approx(2 / 3)
    assert metrics["sequence_accuracy"] == 3 / 5
    assert metrics["operational_accuracy"] == 3 / 7
    assert metrics["end_to_end_fall_recall"] == 2 / 4
    assert metrics["outcomes_with_unknown"] == [[1, 1, 1], [1, 2, 1]]
    assert metrics["window_coverage"] == 25 / 70
    assert sequence_metrics([])["sequence_accuracy"] is None
    assert sequence_metrics([{"sequence_id": "u", "label": 0, "prediction": None}])["TN"] == 0
