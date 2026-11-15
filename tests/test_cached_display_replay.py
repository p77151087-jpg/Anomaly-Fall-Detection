import importlib.util
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import observation

from fall_detect.contracts import Sample
from fall_detect.io_utils import ContractError
from fall_detect.visualization.action_status import SCORE_COLUMNS
from fall_detect.visualization.cached_replay import CachedDisplayReplay, parse_cached_actions
from fall_detect.visualization.sample_display import EventHistory, draw_sample


@pytest.fixture(name="config")
def display_config(config):
    config = deepcopy(config)
    config["skeleton"]["stride"] = 1
    config["skeleton"]["target_fps"] = 30
    return config


def sample(i, points):
    f, poses = observation(i, points)
    return Sample(f.sequence_id, i, f.timestamp_ms, f.source_frame_id, poses, (640, 480))


def window(s, raw, event=None, state="warming_up", reason="insufficient_consecutive_windows"):
    return {
        "sequence_id": s.sequence_id,
        "track_id": 1,
        "sample_index": s.sample_index,
        "available_at_ms": s.timestamp_ms,
        "raw_score": raw,
        "event_score": event,
        "state": state,
        "reason": reason,
    }


def test_event_history_precedes_bad_pose_mask_without_changing_csv_event(config, skeleton):
    engine = CachedDisplayReplay(config, 0.5)
    bad = skeleton[0].copy()
    bad[:, 2] = 0.05
    for i in [29, 30]:
        s = sample(i, skeleton[0])
        engine.step(s, [window(s, 0.7)])
    s = sample(31, bad)
    raw, visible = engine.step(s, [window(s, 0.7, 0.7, "suspected_fall", None)])
    assert raw[1]["state"] == "suspected_fall"
    assert visible[1]["state"] == "unknown" and visible[1]["reason"] == "low_confidence"
    assert visible[1]["raw_score"] is None and visible[1]["event_score"] is None
    assert (
        engine.history.count == 1 and engine.history.latest["confirmed_start_ms"] == s.timestamp_ms
    )
    assert engine.events.finish(s.timestamp_ms)[0]["confirmed_start_ms"] == s.timestamp_ms


def test_score_staleness_and_track_loss_are_not_normal(config, skeleton):
    engine = CachedDisplayReplay(config, 0.5)
    s = sample(29, skeleton[0])
    engine.step(s, [window(s, 0.4)])
    later = replace(sample(30, skeleton[0]), timestamp_ms=s.timestamp_ms + engine.max_age + 1)
    _, visible = engine.step(later, [])
    assert visible[1]["state"] == "unknown" and visible[1]["reason"] == "stale_score"
    empty = replace(later, sample_index=31, poses=[])
    _, visible = engine.step(empty, [])
    assert visible == {} and engine.statuses[1]["reason"] == "track_gap"


def test_strict_equality_never_creates_event(config, skeleton):
    engine = CachedDisplayReplay(config, 0.5)
    for i in [29, 30]:
        s = sample(i, skeleton[0])
        engine.step(s, [window(s, 0.5)])
    s = sample(31, skeleton[0])
    engine.step(s, [window(s, 0.5, 0.5, "normal", None)])
    assert engine.history.count == 0 and engine.events.finish(s.timestamp_ms) == []


def test_unknown_resets_k3(config, skeleton):
    engine = CachedDisplayReplay(config, 0.5)
    for i in [29, 30]:
        s = sample(i, skeleton[0])
        engine.step(s, [window(s, 0.7)])
    s = sample(31, skeleton[0])
    engine.step(s, [window(s, None, None, "unknown", "track_gap")])
    s = sample(32, skeleton[0])
    _, visible = engine.step(s, [window(s, 0.7)])
    assert visible[1]["state"] == "warming_up" and engine.history.count == 0


@pytest.mark.parametrize("change", ["future", "sequence", "state"])
def test_saved_window_identity_and_future_and_state_rejected(config, skeleton, change):
    engine = CachedDisplayReplay(config, 0.5)
    s = sample(29, skeleton[0])
    w = window(s, 0.7)
    if change == "future":
        w["available_at_ms"] += 1
    elif change == "sequence":
        w["sequence_id"] = "other"
    else:
        w["state"] = "suspected_fall"
    with pytest.raises(ContractError):
        engine.step(s, [w])


def action_fixture(skeleton, bad=False):
    points = skeleton[0].copy()
    if bad:
        points[:, 2] = 0.05
    observations = [observation(i, points) for i in range(30)]
    s = sample(29, points)
    values = [0.1] * 6 + [0.9]
    row = {
        "sample_index": "29",
        "timestamp_ms": str(s.timestamp_ms),
        "source_frame_id": "30",
        "track_id": "1",
        "observation_frame_id": "30",
        "history_frame_ids": " ".join(map(str, range(1, 31))),
        "available_at_ms": str(s.timestamp_ms),
        "score_age_ms": "0",
        "history_start_ms": "0",
        "history_end_ms": str(s.timestamp_ms),
        "pose_good": str(not bad),
        "state": "valid",
        "label": "FALLING",
        "class_index": "6",
        "head_score": ".9",
        "reason": "low_confidence" if bad else "",
        **{k: str(v) for k, v in zip(SCORE_COLUMNS, values, strict=True)},
    }
    return row, s, observations


def test_cached_classifier_keeps_label_at_bad_pose_without_making_ae_event(config, skeleton):
    row, s, observations = action_fixture(skeleton, bad=True)
    result = parse_cached_actions([row], {29: s}, observations, config)
    assert result[29][1]["label"] == "FALLING" and not result[29][1]["pose_good"]
    engine = CachedDisplayReplay(config, 0.5)
    assert engine.history.count == 0


@pytest.mark.parametrize("change", ["history", "future", "track", "argmax", "quality"])
def test_cached_action_integrity_rejected(config, skeleton, change):
    row, s, observations = action_fixture(skeleton)
    if change == "history":
        row["history_frame_ids"] += " 31"
    elif change == "future":
        row["available_at_ms"] = str(s.timestamp_ms + 1)
    elif change == "track":
        row["track_id"] = "2"
    elif change == "argmax":
        row["class_index"] = "0"
    else:
        row["pose_good"] = "False"
    with pytest.raises(ContractError):
        parse_cached_actions([row], {29: s}, observations, config)


def test_native_renderer_hides_failed_geometry_but_retains_class_and_history(config, skeleton):
    row, s, observations = action_fixture(skeleton, bad=True)
    actions = parse_cached_actions([row], {29: s}, observations, config)[29]
    history = EventHistory()
    history.observe(1, s.timestamp_ms, {"state": "suspected_fall"})
    visible = {1: {"state": "unknown"}}
    pixels = np.full((480, 640, 3), 60, np.uint8)
    before = deepcopy(actions)
    first = draw_sample(
        pixels, s.poses, actions, visible, s.timestamp_ms, history, config["quality"]
    )
    altered = s.poses[0].keypoints.copy()
    altered[:, :2] += 50
    second = draw_sample(
        pixels,
        [replace(s.poses[0], keypoints=altered)],
        actions,
        visible,
        s.timestamp_ms,
        history,
        config["quality"],
    )
    np.testing.assert_array_equal(first, second)
    assert actions == before and history.count == 1
    # The history title really has red pixels; low pose quality is still UNKNOWN.
    red = first[30:60, 334:635]
    assert ((red[:, :, 2] > 190) & (red[:, :, 0] < 160)).sum() > 30


def script_module(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "validation_display_replay", scripts / "validation_display_replay.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_output_and_encoder_refuse_overwrite_before_any_model_read(tmp_path, monkeypatch):
    m = script_module(monkeypatch)
    monkeypatch.setattr(m, "OUT", tmp_path)
    with pytest.raises(ContractError, match="Preserve"):
        m.build()
    target = tmp_path / "exists.mp4"
    target.write_bytes(b"original")
    with pytest.raises(ContractError, match="Preserve"):
        m.H264Output(target, (32, 24))
    assert target.read_bytes() == b"original"


def test_h264_real_codec_synthetic_decode(tmp_path, monkeypatch):
    m = script_module(monkeypatch)
    monkeypatch.setattr(m, "OUT", tmp_path)
    writer = m.H264Output(tmp_path / "synthetic.mp4", (32, 24))
    for value in [20, 90, 180]:
        writer.write(np.full((24, 32, 3), value, np.uint8))
    info = writer.close()
    assert info["frames"] == 3
    cap = cv2.VideoCapture(str(tmp_path / "synthetic.mp4"))
    decoded = []
    while True:
        ok, image = cap.read()
        if not ok:
            break
        decoded.append(image)
    assert cap.get(cv2.CAP_PROP_FPS) == 30
    cap.release()
    assert len(decoded) == 3 and all(image.shape == (24, 32, 3) for image in decoded)
