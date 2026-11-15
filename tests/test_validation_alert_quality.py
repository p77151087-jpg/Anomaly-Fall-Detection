import importlib.util
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fall_detect.io_utils import ContractError

MODULE = Path(__file__).resolve().parents[1] / "scripts/validation_alert_quality.py"
SPEC = importlib.util.spec_from_file_location("validation_alert_quality", MODULE)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def window(index, raw, score=None, state="warming_up", track=1, reason=None):
    return {
        "sequence_id": "fall-18",
        "track_id": track,
        "sample_index": index,
        "available_at_ms": index * 30.0,
        "start_ms": 0.0,
        "end_ms": index * 30.0,
        "raw_score": raw,
        "event_score": score,
        "state": state,
        "reason": reason,
    }


def annotation():
    b = {
        "bundle_sha256": "fixed",
        "clips": [
            {"sequence_id": sid, "frames": [{"source_frame_id": n} for n in range(1, 10)]}
            for sid in m.SELECTED
        ],
    }
    a = {
        "schema_version": 1,
        "bundle_sha256": "fixed",
        "annotation_origin": "human_self_reported",
        "human_attested": True,
        "reviewer": "tester",
        "clips": [
            {
                "sequence_id": sid,
                "reviewed": True,
                "onset": {"status": "bounded", "bounds_frame_ids": [2, 3]},
                "impact": {"status": "uncertain", "bounds_frame_ids": None},
                "skeleton_quality": "uncertain",
            }
            for sid in m.SELECTED
        ],
    }
    return a, b


def test_unique_validation_denominator_and_test_ignored():
    rows = [
        {"sequence_id": f"adl-{n:02}", "split": "validation", "label": 0} for n in range(1, 9)
    ] + [{"sequence_id": f"fall-{n:02}", "split": "validation", "label": 1} for n in range(1, 16)]
    index = {
        "split_manifest": {
            "sequences": rows + [{"split": "test", "sequence_id": "DO_NOT_ACCESS", "label": 1}]
        }
    }
    assert len(m.val_sources(index)) == 23
    with pytest.raises(ContractError, match="Only Validation"):
        m.scoped_cache({}, {"split": "test"}, "pose")
    index["split_manifest"]["sequences"].append(rows[0])
    with pytest.raises(ContractError):
        m.val_sources(index)


def test_cache_metadata_test_rejected_before_array_read(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "ROOT", tmp_path)
    root = tmp_path / "data/cache/ur-human-isotropic-001"
    root.mkdir(parents=True)
    m.save(root / "bad.json", {"split": "test"})
    index = {
        "sequences": {
            "fall-18": {
                "status": "prepared",
                "pose_path": str(root / "bad.npz"),
                "source_hash": "a",
            }
        },
        "recipe_hash": "b",
        "split_manifest": {"split_hash": "c"},
    }
    with pytest.raises(ContractError, match="Cache"):
        m.scoped_cache(index, {"split": "validation", "sequence_id": "fall-18"}, "pose")


def test_strict_threshold_equality_and_warmup():
    ws = [window(1, 1), window(2, 1), window(3, 1, 1, "normal")]
    score, events = m.audit_events(ws, 1, 120)
    assert score == 1 and events == []


def test_unknown_and_gap_reset_history_and_event():
    ws = [
        window(1, 2),
        window(2, 2),
        window(3, 2, 2, "suspected_fall"),
        window(4, None, None, "unknown", reason="track_gap"),
        window(5, 2),
        window(6, 2),
        window(8, 2),
        window(9, 2),
        window(10, 2, 2, "suspected_fall"),
    ]
    _, events = m.audit_events(ws, 1, 330)
    assert len(events) == 2
    assert events[0]["end_reason"] == "track_gap" and events[0]["confirmed_start_ms"] == 90
    assert events[1]["confirmed_start_ms"] == 300 and events[1]["end_ms"] == 330


def test_tracks_do_not_combine_and_duplicate_rejected():
    ws = [window(1, 2, track=1), window(1, 2, track=2), window(2, 2, track=1)]
    assert m.audit_events(ws, 1, 90) == (None, [])
    with pytest.raises(ContractError, match="Duplicate"):
        m.audit_events(ws + [deepcopy(ws[-1])], 1, 90)


def test_unknown_metric_denominators():
    result = m.metrics(
        [
            {"label": 1, "prediction": 1},
            {"label": 1, "prediction": None},
            {"label": 0, "prediction": 0},
        ]
    )
    assert result["sequence_accuracy"] == 1 and result["operational_accuracy"] == 2 / 3
    assert result["sequence_fall_recall"] == 1 and result["end_to_end_fall_recall"] == 0.5
    assert result["overall_f1_unknown_as_miss"] == 2 / 3
    assert result["outcomes_with_unknown"] == [[1, 0, 0], [0, 1, 1]]


def test_no_event_and_no_gt_are_not_correct_time():
    assert not m.in_phase(None, [1, 2]) and not m.in_phase(1, None)
    assert m.in_phase(2, [1, 2])


def test_causal_grid_uses_past_source_not_future_frame():
    sampler = m.CausalResampler(30, "bounded_jitter", 0)
    samples = []
    timestamps = {1: 0, 2: 33, 3: 66, 4: 101}
    for source_id, time in timestamps.items():
        frame = SimpleNamespace(
            sequence_id="fall-18",
            timestamp_ms=time,
            source_frame_id=source_id,
            source_missing=False,
            source_fps=None,
            width=640,
            height=480,
        )
        samples.extend(sampler.push(frame, []))
    assert [s.source_frame_id for s in samples] == [1, 2, 3, 3]
    assert all(timestamps[s.source_frame_id] <= s.timestamp_ms for s in samples)


def test_confidence_good_does_not_assert_geometric_correctness():
    points = np.ones((13, 3))
    points[:, :2] = -100
    assert m.good_frame(points, {"min_joint_confidence": 0.2, "min_valid_joints": 9})


def test_exclusive_files_and_directory_refusal(tmp_path, monkeypatch):
    f = tmp_path / "x.json"
    m.save(f, {"x": 1})
    before = f.read_bytes()
    with pytest.raises(FileExistsError):
        m.save(f, {"x": 2})
    assert f.read_bytes() == before
    monkeypatch.setattr(m, "OUT", tmp_path)
    monkeypatch.setattr(m, "QA", tmp_path / "qa")
    with pytest.raises(ContractError, match="Preserve"):
        m.build()


def test_source_path_cannot_escape(tmp_path):
    with pytest.raises(ContractError, match="escapes"):
        m.local_path("../elsewhere.json", tmp_path)


def test_review_accepts_uncertain_and_rejects_invented_time():
    a, b = annotation()
    assert m.validate_review(a, b)["valid"]
    a["clips"][0]["impact"]["bounds_frame_ids"] = [5, 6]
    with pytest.raises(ContractError, match="invent"):
        m.validate_review(a, b)


@pytest.mark.parametrize("bad_bounds", [[4, 2], [0, 3], [2.0, 3], [True, 3], [2, None]])
def test_review_rejects_invalid_bounds(bad_bounds):
    a, b = annotation()
    a["clips"][0]["onset"]["bounds_frame_ids"] = bad_bounds
    with pytest.raises(ContractError, match="ordered"):
        m.validate_review(a, b)


def test_review_is_explicit_self_report_and_complete():
    a, b = annotation()
    for field, value in [
        ("human_attested", False),
        ("reviewer", ""),
        ("bundle_sha256", "other"),
        ("annotation_origin", "AI"),
    ]:
        modified = deepcopy(a)
        modified[field] = value
        with pytest.raises(ContractError):
            m.validate_review(modified, b)
    a["clips"][0]["reviewed"] = False
    with pytest.raises(ContractError, match="reviewed"):
        m.validate_review(a, b)
