from copy import deepcopy
from dataclasses import replace

import cv2
import numpy as np
import pytest
import torch
from conftest import observation

from fall_detect.config import load_config, project_path
from fall_detect.detection.events import EventState
from fall_detect.io_utils import ContractError, read_json
from fall_detect.models.stgcn import HumanFeatureExtractor
from fall_detect.skeleton.preprocessing import human_preprocess
from fall_detect.visualization.sample_display import (
    HEADER,
    EventHistory,
    SampleActions,
    action_hint,
    draw_sample,
    supports_sample_display,
    visible_actions,
)


class HeadFixture:
    def __init__(self):
        self.inputs = []

    def classify_observations(self, points, size):
        self.inputs.append((points.copy(), size))
        return np.array([[0.1] * 6 + [0.9]])


def test_actual_observations_not_ae_quality_or_interpolated_frames(config, skeleton):
    head = HeadFixture()
    actions = SampleActions(config, head)
    bad = skeleton.copy()
    bad[:, :, 2] = 0.05
    for i in range(29):
        state = actions.push(*observation(i, bad[i]))[1]
        assert state["label"] == "PENDING"
    assert not head.inputs
    # A short gap adds neither an observation nor a made-up action.
    assert actions.push(*observation(29, None)) == {}
    state = actions.push(*observation(30, bad[29]))[1]
    assert state["label"] == "FALLING" and state["state"] == "valid"
    assert not state["pose_good"] and "Pose uncertain" in action_hint(state)
    assert state["history_frame_ids"] == (*range(1, 30), 31)
    np.testing.assert_array_equal(head.inputs[0][0], bad)
    assert head.inputs[0][1] == (640, 480)
    assert visible_actions(actions.current, [], 1000) == {}


@pytest.mark.parametrize("index,timestamp", [(61, 1100), (31, 2500)])
def test_long_gap_retires_history_by_frame_count_or_time(config, skeleton, index, timestamp):
    actions = SampleActions(config, HeadFixture())
    for i in range(30):
        actions.push(*observation(i, skeleton[i]))
    state = actions.push(*observation(index, skeleton[0], timestamp=timestamp))[1]
    assert state["label"] == "PENDING" and len(state["history_frame_ids"]) == 1


def test_tracks_and_source_snapshots_are_separate_and_causal(config, skeleton):
    actions = SampleActions(config, HeadFixture())
    for i in range(30):
        frame, first = observation(i, skeleton[i])
        _, second = observation(i, skeleton[i], track=2)
        actions.push(frame, first + second if i >= 10 else first)
    previous = actions.current
    assert previous[1]["label"] == "FALLING"
    assert previous[2]["label"] == "PENDING"
    later, later_poses = observation(30, skeleton[0])
    current = actions.push(later, later_poses)
    assert 2 in previous and 2 not in current
    assert visible_actions(previous, first, frame.timestamp_ms)[1]["label"] == "FALLING"
    assert visible_actions(current, first, frame.timestamp_ms)[1]["label"] == ""
    assert visible_actions(current, later_poses, later.timestamp_ms - 1)[1]["label"] == ""
    assert visible_actions(current, later_poses, later.timestamp_ms)[1]["label"] == "FALLING"


@pytest.mark.parametrize("failure", ["duplicate", "source", "nan", "missing", "timeline"])
def test_invalid_observations_fail_closed(config, skeleton, failure):
    actions = SampleActions(config, HeadFixture())
    frame, poses = observation(0, skeleton[0])
    if failure == "duplicate":
        poses *= 2
    elif failure == "source":
        poses = [replace(poses[0], source_frame_id=999)]
    elif failure == "nan":
        poses[0].keypoints[0, 0] = np.nan
    elif failure == "missing":
        frame = replace(frame, source_missing=True)
    else:
        actions.push(frame, poses)
    with pytest.raises(ContractError):
        actions.push(frame, poses)


def test_degenerate_pose_is_not_given_a_fake_action(config, skeleton):
    class Degenerate(HeadFixture):
        def classify_observations(self, points, size):
            raise ContractError("degenerate_pose: zero span")

    actions = SampleActions(config, Degenerate())
    for i in range(30):
        state = actions.push(*observation(i, skeleton[i]))[1]
    assert state["state"] == "unavailable" and state["label"] == ""
    assert state["class_scores"] is None and action_hint(state) == "Pose unavailable"


def test_event_history_matches_actual_k3_transitions_and_survives_unknown():
    events = EventState(0.5, 3, 1)
    history = EventHistory()
    values = [0.8, 0.8, 0.8, 0.8, None, 0.9, 0.9, 0.9]
    for i, score in enumerate(values):
        status = events.update("synthetic", 1, i, i * 1000 / 30, score)
        before = deepcopy(status)
        history.observe(1, i * 1000 / 30, status)
        assert status == before
        if i == 4:
            assert history.count == 1 and history.latest["confirmed_start_ms"] == 2 * 1000 / 30
    actual = events.finish(8 * 1000 / 30)
    assert history.count == len(actual) == 2
    assert history.latest["confirmed_start_ms"] == actual[-1]["confirmed_start_ms"]


def test_render_keeps_low_quality_action_and_separate_ae_history(config, skeleton, monkeypatch):
    actions = SampleActions(config, HeadFixture())
    for i in range(30):
        frame, poses = observation(i, skeleton[i])
        poses[0].keypoints[:, 2] = 0.01
        states = actions.push(frame, poses)
    history = EventHistory()
    history.observe(1, 500, {"state": "suspected_fall"})
    ae = {1: {"state": "unknown"}}
    before = deepcopy((states, ae))
    image = np.zeros((480, 640, 3), np.uint8)
    texts = []
    original = cv2.putText

    def capture(image, text, *args, **kwargs):
        texts.append(text)
        return original(image, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", capture)
    result = draw_sample(image, poses, states, ae, 1000, history, config["quality"])
    assert result.shape == (480 + HEADER, 640, 3)
    assert "FALLING" in texts and "AE EVENT HISTORY" in texts and "ALERT RECORDED" in texts
    assert any("Pose uncertain" in s for s in texts)
    assert "Current: unavailable (pose / tracking)" in texts
    assert not any("UNKNOWN" in s or "%" in s for s in texts)
    assert (states, ae) == before and not image.any()
    texts.clear()
    draw_sample(image, [], {}, {}, 1100, history, config["quality"])
    assert "Tracking lost" in texts and "ALERT RECORDED" in texts
    assert not any(s in ("FALLING", "LYING", "STANDING") for s in texts)


@pytest.mark.parametrize("size,normalization", [(30, "per_frame_isotropic"), (20, "per_window")])
def test_genuine_independent_forward_uses_original_recipe_and_preserves_ae(
    skeleton, size, normalization
):
    manifest = project_path("weights/sample-human-manifest.json")
    if not manifest.exists():
        pytest.skip("Genuine Human checkpoint manifest unavailable")
    resources = {r["role"]: r for r in read_json(manifest)["resources"]}
    if any(not project_path(r["local_path"]).exists() for r in resources.values()):
        pytest.skip("Genuine Human resources incomplete")
    config = load_config(project_path("configs/ur-sample-human-s1-001.yaml"))
    config["skeleton"].update(window_size=size, coordinate_normalization=normalization)
    assert supports_sample_display(config)
    extractor = HumanFeatureExtractor(config, resources, torch.device("cpu"))
    saved = {k: v.clone() for k, v in extractor.model.state_dict().items()}
    original = skeleton.copy()
    ae_before = extractor.extract_features(skeleton[:size], (640, 480))
    actual = extractor.classify_observations(skeleton, (640, 480))
    points, motion = human_preprocess(
        skeleton, (640, 480), window_size=30, normalization="per_frame"
    )
    with torch.inference_mode():
        expected = extractor.model((torch.from_numpy(points), torch.from_numpy(motion))).numpy()
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        ae_before, extractor.extract_features(skeleton[:size], (640, 480))
    )
    np.testing.assert_array_equal(skeleton, original)
    assert all(torch.equal(v, saved[k]) for k, v in extractor.model.state_dict().items())
    assert all(not p.requires_grad for p in extractor.model.parameters())
    with pytest.raises(ContractError):
        extractor.classify_observations(skeleton[:20], (640, 480))


def test_non_human_recipe_rejected(config):
    config["features"]["backend"] = "reference"
    assert not supports_sample_display(config)
    with pytest.raises(ContractError):
        SampleActions(config, HeadFixture())
