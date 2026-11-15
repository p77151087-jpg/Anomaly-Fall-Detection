from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import sample

from fall_detect.cli import parser
from fall_detect.contracts import Window
from fall_detect.detection.events import EventState
from fall_detect.detection.shoulder_demo import ShoulderDemo, coherent_shoulder_descent
from fall_detect.io_utils import ContractError


def test_cli_shoulder_is_explicit():
    base = ["demo", "--run", "x", "--input", "y", "--output", "z"]
    assert parser().parse_args(base).shoulder_candidate is None
    assert (
        parser().parse_args(base + ["--shoulder-candidate", "frozen"]).shoulder_candidate
        == "frozen"
    )


def test_exact_shoulder_formula_and_confidence(config, skeleton):
    before = skeleton[0].copy()
    after = before.copy()
    after[1, 1] += 30
    after[2, 1] += 20
    box = np.array([20, 30, 300, 230], np.float32)
    assert coherent_shoulder_descent(before, after, box, config["quality"]) == (0.1, None)
    after[2, 2] = 0.1
    assert coherent_shoulder_descent(before, after, box, config["quality"])[0] is None
    after[:, 2] = 0.1
    assert (
        coherent_shoulder_descent(before, after, box, config["quality"])[1] == "anchor_confidence"
    )


def stream(config, skeleton, monkeypatch, cue_values, wide_values, missing_at=None):
    config = deepcopy(config)
    config["skeleton"]["stride"] = 1
    branch = ShoulderDemo(config, 1.0, 1.0)
    wide = EventState(1.0, 3, 1)
    iterator = iter(cue_values)
    monkeypatch.setattr(
        "fall_detect.detection.shoulder_demo.coherent_shoulder_descent",
        lambda *args: (next(iterator), None),
    )
    result = []
    for i in range(6 + len(wide_values)):
        s = sample(i, skeleton[i % 30])
        branch.observe_sample(s)
        if i < 6:
            continue
        legal = i != missing_at
        w = Window(
            s.sequence_id,
            1,
            i,
            0.0,
            s.timestamp_ms,
            np.tile(skeleton[0], (30, 1, 1)),
            (640, 480),
            "valid" if legal else "low_confidence",
            1.0 if legal else 0.0,
        )
        a = wide.update(
            s.sequence_id, 1, i, s.timestamp_ms, wide_values[i - 6] if legal else None, w.quality
        )
        f = np.ones((1, 512), np.float32) if legal else None
        result.append(branch.update(w, f, a))
    return branch, result


def test_independent_k3_prevents_alternating_branch_pooling(config, skeleton, monkeypatch):
    branch, rows = stream(config, skeleton, monkeypatch, [0, 2, 0, 2, 0, 2], [2, 0, 2, 0, 2, 0])
    assert all(r["state"] != "suspected_fall" for r in rows)
    assert branch.finish(500) == []


def test_shoulder_only_k3_and_missing_reset(config, skeleton, monkeypatch):
    branch, rows = stream(config, skeleton, monkeypatch, [2, 2, 2, 2, 2, 2], [0] * 7, missing_at=9)
    assert rows[2]["state"] == "suspected_fall"
    assert rows[3]["state"] == "unknown"
    assert rows[4]["state"] != "suspected_fall"
    assert rows[5]["state"] != "suspected_fall"
    assert rows[6]["state"] == "suspected_fall"
    assert branch.window_rows[3]["shoulder_raw"] is None
    assert len(branch.finish(500)) == 2


def test_strict_threshold_and_wide_preservation(config, skeleton, monkeypatch):
    _, rows = stream(config, skeleton, monkeypatch, [1, 1, 1, 0], [2, 2, 2, 2])
    assert rows[2]["state"] == rows[3]["state"] == "suspected_fall"
    assert rows[2]["event_score"] == 2


def test_grid_gap_is_rejected(config, skeleton):
    branch = ShoulderDemo(config, 1, 1)
    branch.observe_sample(sample(0, skeleton[0]))
    with pytest.raises(ContractError, match="consecutive sequence"):
        branch.observe_sample(sample(2, skeleton[0]))


def test_missing_or_held_reference_is_not_zero_filled(config, skeleton, monkeypatch):
    config = deepcopy(config)
    config["skeleton"]["stride"] = 1
    branch = ShoulderDemo(config, 1, 1)
    for i in range(7):
        s = sample(i, skeleton[0])
        s.source_frame_id = 1
        branch.observe_sample(s)
    w = Window("synthetic", 1, 6, 0, 200, skeleton, (640, 480), "valid", 1)
    branch.update(
        w,
        np.ones((1, 512), np.float32),
        {"raw_score": 0, "event_score": None, "state": "warming_up", "reason": None},
    )
    assert branch.window_rows[0]["shoulder_reason"] == "no_distinct_causal_frame"
    assert branch.window_rows[0]["shoulder_raw"] is None


def candidate_fixture(config, monkeypatch):
    known = [
        {"sequence_id": f"adl-{i}", "label": 0, "split": "train" if i < 24 else "validation"}
        for i in range(32)
    ]
    normal = {r["sequence_id"]: {"split": r["split"], "max_k3": 0.5} for r in known}
    scorer = SimpleNamespace(
        run=Path("wide"),
        config=config,
        residual_weights=None,
        index={
            "recipe_hash": "recipe",
            "split_manifest": {"split_hash": "split", "sequences": known},
        },
        threshold={"threshold": 1.0},
    )
    lock = {
        "baseline": "wide",
        "recipe": {
            "lag_ticks": 6,
            "fps": 30,
            "k": 3,
            "fusion": "per-track OR after two independent K3 histories",
            "wide_recipe_hash": "recipe",
            "split_hash": "split",
            "wide_threshold": 1.0,
        },
        "source_sha256": {"runs/wide/resolved-config.yaml": "config"},
        "comparison_recipe_sha256": "comparison",
    }
    docs = {
        "lock.json": lock,
        "normal-calibration.json": {
            "normal_video_k3": normal,
            "comparison": "gt",
            "threshold": 0.5,
        },
        "artifact-manifest.json": {
            "files": {"lock.json": "lock", "normal-calibration.json": "cal"}
        },
    }
    hashes = {
        "lock.json": "lock",
        "normal-calibration.json": "cal",
        "resolved-config.yaml": "config",
    }
    monkeypatch.setattr("fall_detect.detection.shoulder_demo.read_json", lambda p: docs[p.name])
    monkeypatch.setattr("fall_detect.detection.shoulder_demo.sha256", lambda p: hashes[p.name])
    return scorer, docs, hashes


def test_frozen_candidate_binding(config, monkeypatch):
    scorer, _, _ = candidate_fixture(config, monkeypatch)
    branch = ShoulderDemo.from_candidate(Path("candidate"), scorer)
    assert branch.shoulder_threshold == 0.5
    assert branch.wide_threshold == 1.0


@pytest.mark.parametrize("changed", ["hash", "run", "lag", "quality", "test", "threshold"])
def test_candidate_rejects_changed_contract(config, monkeypatch, changed):
    scorer, docs, hashes = candidate_fixture(config, monkeypatch)
    if changed == "hash":
        hashes["lock.json"] = "changed"
    elif changed == "run":
        scorer.run = Path("different-run")
    elif changed == "lag":
        docs["lock.json"]["recipe"]["lag_ticks"] = 5
    elif changed == "quality":
        hashes["resolved-config.yaml"] = "changed-quality"
    elif changed == "test":
        docs["normal-calibration.json"]["normal_video_k3"]["adl-0"]["split"] = "test"
    else:
        docs["normal-calibration.json"]["threshold"] = 0.49
    with pytest.raises(ContractError):
        ShoulderDemo.from_candidate(Path("candidate"), scorer)


@pytest.mark.parametrize("sidecar", range(5))
def test_optional_audit_files_refuse_overwrite_before_weights(
    config, monkeypatch, tmp_path, sidecar
):
    from fall_detect.training import experiment
    from fall_detect.visualization.video import demo

    scorer = SimpleNamespace(
        config=config, threshold={"threshold": 1}, index={"source_kind": "real"}
    )
    monkeypatch.setattr(experiment, "Scorer", lambda *args: scorer)
    branch = ShoulderDemo(config, 1, 1)
    monkeypatch.setattr(ShoulderDemo, "from_candidate", lambda *args: branch)
    output = tmp_path / "new-demo.mp4"
    path = branch.audit_paths(output)[sidecar]
    path.write_bytes(b"preserve me")
    with pytest.raises(ContractError, match="already exists"):
        demo("run", "input", output, shoulder_candidate="candidate")
    assert path.read_bytes() == b"preserve me"
