import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

from fall_detect.evaluation.human_timing import frame_time_bounds, temporal_relation
from fall_detect.io_utils import ContractError


@pytest.mark.parametrize(
    "time,relation,possible,guaranteed",
    [
        (99, "before_onset", False, False),
        (100, "onset_uncertainty", True, False),
        (149, "onset_uncertainty", True, False),
        (150, "guaranteed_fall_interval", True, True),
        (199, "guaranteed_fall_interval", True, True),
        (200, "impact_uncertainty", True, True),
        (225, "impact_uncertainty", True, False),
        (250, "impact_uncertainty", True, False),
        (251, "after_impact", False, False),
    ],
)
def test_boundary_semantics(time, relation, possible, guaranteed):
    r = temporal_relation(time, [100, 150], [200, 250])
    assert r["relation"] == relation
    assert r["inside_possible_fall_envelope"] is possible
    assert r["inside_guaranteed_fall_interval"] is guaranteed
    assert r["definitely_before_impact"] is (time < 200)


def test_sub_millisecond_boundary_not_rounded_to_source_frame():
    r = temporal_relation(1866.6666666666667, [1167, 1534], [1635, 1868])
    assert r["relation"] == "impact_uncertainty"
    assert r["delay_from_impact_bounds_ms"][0] == pytest.approx(-1.3333333333333)
    assert not r["definitely_before_impact"] and not r["definitely_after_impact"]


def test_uncertainty_preserved_not_midpoint():
    r = temporal_relation(5900, [4304, 4571], [4705, 5973])
    assert r["delay_from_onset_bounds_ms"] == [1329, 1596]
    assert r["delay_from_impact_bounds_ms"] == [-73, 1195]
    assert r["guaranteed_fall_interval_ms"] == [4571, 4705]


def test_missing_bounds_and_no_alarm_are_unknown():
    for onset, impact in [(None, [200, 250]), ([100, 150], None), (None, None)]:
        r = temporal_relation(225, onset, impact)
        assert r["inside_possible_fall_envelope"] is None
        assert r["inside_guaranteed_fall_interval"] is None
    r = temporal_relation(None, [100, 150], [200, 250])
    assert r["relation"] == "no_alarm" and r["inside_possible_fall_envelope"] is None


def test_overlap_does_not_invent_guaranteed_descent():
    r = temporal_relation(190, [100, 220], [180, 250])
    assert r["guaranteed_fall_interval_ms"] is None
    assert r["inside_guaranteed_fall_interval"] is False
    assert r["relation"] == "overlapping_onset_impact_uncertainty"


@pytest.mark.parametrize(
    "time,onset,impact",
    [
        (True, [100, 150], [200, 250]),
        (float("nan"), [100, 150], [200, 250]),
        (100, [150, 100], [200, 250]),
        (100, [100, float("inf")], [200, 250]),
        (100, [300, 350], [200, 250]),
        (100, [-1, 150], [200, 250]),
    ],
)
def test_invalid_numeric_or_causal_bounds(time, onset, impact):
    with pytest.raises(ContractError):
        temporal_relation(time, onset, impact)


def test_source_frame_ids_use_recorded_time_not_30hz_index():
    frames = [
        {"source_frame_id": 50, "timestamp_ms": 1635},
        {"source_frame_id": 57, "timestamp_ms": 1868},
    ]
    assert frame_time_bounds({"status": "bounded", "bounds_frame_ids": [50, 57]}, frames) == [
        1635,
        1868,
    ]
    with pytest.raises(ContractError):
        frame_time_bounds({"status": "bounded", "bounds_frame_ids": [49, 57]}, frames)
    with pytest.raises(ContractError):
        frame_time_bounds({"status": "uncertain", "bounds_frame_ids": [50, 57]}, frames)


def test_build_refuses_existing_output_before_reading_upload(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "validation_human_timing", scripts / "validation_human_timing.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "OUT", tmp_path)
    monkeypatch.setattr(module, "QA", tmp_path / "qa")
    with pytest.raises(ContractError, match="Preserve"):
        module.build(tmp_path / "nonexistent-annotation.json")


@pytest.mark.parametrize("separator", ["/", "\\"])
def test_archived_docs_normalize_separators_and_still_detect_tampering(
    tmp_path, monkeypatch, separator
):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "timing_preservation", scripts / "validation_human_timing.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "QA", tmp_path / "qa")
    old = tmp_path / "qa/prior-contracts/docs/current-progress.md"
    current = tmp_path / "docs/current-progress.md"
    old.parent.mkdir(parents=True)
    current.parent.mkdir(parents=True)
    old.write_bytes(b"original")
    current.write_bytes(b"updated and deliberately different")
    files = {f"docs{separator}current-progress.md": hashlib.sha256(b"original").hexdigest()}
    module.check_hashes(files, archived=True)
    old.write_bytes(b"tampered")
    with pytest.raises(ContractError, match="Preserved"):
        module.check_hashes(files, archived=True)
