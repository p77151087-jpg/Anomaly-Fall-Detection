from copy import deepcopy

import cv2
import numpy as np
import pytest
import torch
from conftest import observation

from fall_detect.cli import parser
from fall_detect.config import load_config, project_path
from fall_detect.io_utils import ContractError, read_json
from fall_detect.models.stgcn import HumanFeatureExtractor
from fall_detect.recipes import YOLO_RECIPE
from fall_detect.skeleton.preprocessing import human_preprocess
from fall_detect.visualization.action_status import (
    ACTION_HEADER,
    ACTION_LABELS,
    SOURCE_CLASSES,
    ActionDisplayState,
    supports_actions,
)
from fall_detect.visualization.simple import draw_simple


def test_genuine_action_head_matches_full_pretrained_forward(skeleton):
    manifest = project_path("weights/sample-human-manifest.json")
    if not manifest.exists():
        pytest.skip("Genuine Human checkpoint manifest unavailable")
    resources = {r["role"]: r for r in read_json(manifest)["resources"]}
    if any(not project_path(r["local_path"]).exists() for r in resources.values()):
        pytest.skip("Genuine Human resources incomplete")
    config = load_config(project_path("configs/ur-sample-human-s1-001.yaml"))
    extractor = HumanFeatureExtractor(config, resources, torch.device("cpu"))
    before = skeleton.copy()
    features = extractor.extract_features(skeleton, (640, 480))
    saved_features = features.copy()
    actual = extractor.action_scores(features)
    points, motion = human_preprocess(skeleton, (640, 480))
    with torch.inference_mode():
        expected = extractor.model((torch.from_numpy(points), torch.from_numpy(motion))).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
    np.testing.assert_array_equal(features, saved_features)
    np.testing.assert_array_equal(skeleton, before)
    assert actual.shape == (1, 7)
    assert extractor.load_report["strict"] and extractor.forward_verified
    assert all(not p.requires_grad for p in extractor.model.parameters())
    # Check the pinned upstream order rather than trusting our display names alone.
    import ast

    tree = ast.parse(
        project_path("third_party/human/ActionsEstLoader.py").read_text(encoding="utf8")
    )
    names = [
        ast.literal_eval(n.value)
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == "class_names" for t in n.targets)
    ]
    assert names == [list(SOURCE_CLASSES)]
    for malformed in (np.zeros((1, 7)), np.full((1, 512), np.nan), np.zeros((0, 512))):
        with pytest.raises(ContractError):
            extractor.action_scores(malformed)
    extractor.normalization = "per_window"
    with pytest.raises(ContractError, match="original30"):
        extractor.action_scores(features)


@pytest.mark.parametrize("index,label", list(enumerate(ACTION_LABELS)))
def test_seven_class_order_is_preserved_without_ae_input(config, skeleton, index, label):
    tracker = ActionDisplayState(config)
    scores = np.full(7, 0.1)
    scores[index] = 0.6  # Scores deliberately do not sum to one.
    tracker.update(1, scores, 1000)
    _, poses = observation(30, skeleton[0])
    state = tracker.visible(poses, 1000)[1]
    assert state["label"] == label and state["class_index"] == index
    assert state["score"] == 0.6 and state["state"] == "valid"
    scores[index] = 0.0
    assert tracker.states[1]["class_scores"][index] == 0.6


def test_quality_gap_staleness_and_invalid_window_clear_action(config, skeleton):
    tracker = ActionDisplayState(config)
    _, good = observation(30, skeleton[0])
    assert tracker.visible(good, 0)[1]["label"] == "PENDING"
    scores = [0.1] * 6 + [0.9]
    tracker.update(1, scores, 1000)
    bad = deepcopy(good)
    bad[0].keypoints[:, 2] = 0
    state = tracker.visible(bad, 1000)[1]
    assert state["label"] == "UNKNOWN" and state["class_scores"] is None
    assert tracker.visible(good, 1010)[1]["label"] == "UNKNOWN"
    tracker.update(1, scores, 1020)
    assert tracker.visible([], 1021) == {}
    assert tracker.visible(good, 1022)[1]["reason"] == "track_gap"
    tracker.update(1, scores, 2000)
    assert tracker.visible(good, 2000 + tracker.max_age + 1)[1]["reason"] == "stale_action"
    tracker.update(1, scores, 3000)
    tracker.update(1, None, 3033, "degenerate")
    assert tracker.visible(good, 3033)[1]["reason"] == "degenerate"
    assert tracker.visible(good, 3033)[1]["score"] is None
    with pytest.raises(ContractError):
        tracker.update(1, [np.nan] * 7, 3040)


def test_dual_overlay_keeps_conflicting_results_and_inputs(config, skeleton, monkeypatch):
    image = np.zeros((480, 640, 3), np.uint8)
    _, poses = observation(30, skeleton[0])
    tracker = ActionDisplayState(config)
    tracker.update(1, [0.1] * 6 + [0.9], 1000)
    actions = tracker.visible(poses, 1000)
    ae = {1: {"state": "normal", "raw_score": 0.1, "available_at_ms": 1000}}
    before = deepcopy((actions, ae))
    points = poses[0].keypoints.copy()
    texts = []
    original = cv2.putText

    def capture(img, text, *args, **kwargs):
        texts.append(text)
        return original(img, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", capture)
    result = draw_simple(image, poses, ae, 1000, 0.5, quality=config["quality"], actions=actions)
    assert result.shape == (480 + ACTION_HEADER, 640, 3)
    assert "FALLING" in texts and "ANOMALY (AE)" in texts and "NORMAL" in texts
    assert "FALLING / AE NORMAL" in texts and not any("%" in s for s in texts)
    assert (actions, ae) == before
    np.testing.assert_array_equal(points, poses[0].keypoints)
    assert not image.any()


def test_action_labels_are_separate_per_track(config, skeleton):
    tracker = ActionDisplayState(config)
    _, one = observation(30, skeleton[0], track=1)
    _, two = observation(30, skeleton[0], track=2)
    tracker.update(1, [0.9] + [0.1] * 6, 1000)
    tracker.update(2, [0.1] * 6 + [0.9], 1000)
    visible = tracker.visible(one + two, 1000)
    assert visible[1]["label"] == "STANDING" and visible[2]["label"] == "FALLING"
    tracker.visible(one, 1010)
    visible = tracker.visible(one + two, 1020)
    assert visible[1]["label"] == "STANDING" and visible[2]["label"] == "UNKNOWN"


@pytest.mark.parametrize(
    "field,value",
    [("window_size", 20), ("coordinate_normalization", "per_window"), ("recipe_id", YOLO_RECIPE)],
)
def test_incompatible_recipe_cannot_reuse_seven_class_head(config, field, value):
    config["skeleton"][field] = value
    assert not supports_actions(config)
    with pytest.raises(ContractError, match="original30"):
        ActionDisplayState(config)


def test_cli_chooses_auto_by_default():
    args = parser().parse_args(
        ["demo", "--run", "runs/example", "--input", "clip", "--output", "new.mp4"]
    )
    assert args.overlay == "auto"
