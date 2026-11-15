import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def control(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("fall19_shoulder_descent")


@pytest.fixture
def anchors():
    previous = np.ones((13, 3), dtype=np.float32)
    return previous, previous.copy(), np.array([0, 0, 80, 200], dtype=np.float32)


QUALITY = {"min_joint_confidence": 0.2, "min_valid_joints": 9}


def test_descent_requires_both_shoulders_and_uses_reference_height(control, anchors):
    previous, current, box = anchors
    current[1:3, 1] += [60, 40]
    assert control.coherent_descent(previous, current, box, QUALITY) == (0.2, None, 40, 200)
    current[2, 1] = previous[2, 1] - 20
    assert control.coherent_descent(previous, current, box, QUALITY) == (0, None, -20, 200)


def test_shoulder_confidence_has_an_explicit_missing_state(control, anchors):
    previous, current, box = anchors
    current[2, 2] = 0.19
    assert control.coherent_descent(previous, current, box, QUALITY)[:2] == (
        None,
        "shoulder_confidence",
    )


def test_degenerate_reference_is_unavailable(control, anchors):
    previous, current, box = anchors
    box[3] = box[1]
    assert control.coherent_descent(previous, current, box, QUALITY)[:2] == (
        None,
        "invalid_reference_height",
    )


def stream(scores, indices=None):
    indices = indices if indices is not None else range(29, 29 + len(scores))
    return [
        {
            "sequence_id": "synthetic-adl",
            "track_id": 1,
            "sample_index": i,
            "available_at_ms": i * (1000 / 30),
            "raw_score": score,
            "missing_reason": "anchor_confidence" if score is None else None,
            "original_legal": True,
        }
        for i, score in zip(indices, scores, strict=True)
    ]


def test_two_branches_cannot_pool_alternating_high_scores(control):
    wide, _ = control.score_branch(stream([2, 0, 2]), 1, 2000)
    shoulder, _ = control.score_branch(stream([0, 2, 0]), 1, 2000)
    merged = control.fuse_rows(wide, shoulder, 1, 1)
    assert merged[-1]["combined_state"] == "normal"
    assert control.combined_events(merged, 2000) == []


def test_wide_event_survives_an_unavailable_shoulder_branch(control):
    wide, _ = control.score_branch(stream([2, 2, 2, 2]), 1, 2000)
    shoulder, _ = control.score_branch(stream([None] * 4), 1, 2000)
    merged = control.fuse_rows(wide, shoulder, 1, 1)
    (event,) = control.combined_events(merged, 2000)
    assert event["wide_active_endpoints"] == 2
    assert event["shoulder_active_endpoints"] == 0
    assert event["end_reason"] == "sequence_end"


def test_gap_and_missing_reset_shoulder_and_equality_stays_normal(control):
    samples = [29, 30, 32, 33, 34, 35, 36, 37, 38]
    rows, events = control.score_branch(stream([2, 2, 1, 1, 1, None, 2, 2, 2], samples), 1, 2000)
    assert rows[2]["event_score"] is None
    assert rows[4]["event_score"] == 1 and rows[4]["state"] == "normal"
    assert rows[7]["event_score"] is None
    (event,) = events
    assert event["confirmed_start_ms"] == rows[8]["available_at_ms"]
