import csv

import cv2
import numpy as np
import pytest
from conftest import observation

from fall_detect.contracts import PoseRecord
from fall_detect.data.cache import load_pose_cache, save_pose_cache
from fall_detect.detection.events import EventState, score_sequence
from fall_detect.io_utils import read_json
from fall_detect.pipeline import features_from_observations
from fall_detect.skeleton.windows import CausalResampler, WindowBuilder
from fall_detect.visualization.video import demo


class SyntheticExtractor:
    feature_dim = 512

    def extract_features(self, points, size):
        # Only a plumbing fixture; never instantiated by a production CLI path.
        return np.full((1, 512), points[:, :, 0].mean() / 100, dtype=np.float32)


def test_batch_and_streaming_share_windows_scores_events(config, skeleton, tmp_path):
    observations = [observation(i, None if i == 46 else skeleton[i % 30]) for i in range(100)]
    meta = {"sequence_id": "synthetic", "split": "test", "source_kind": "synthetic"}
    pose_path = tmp_path / "synthetic-poses.npz"
    save_pose_cache(pose_path, observations, meta)
    restored, _ = load_pose_cache(pose_path, meta)
    extractor = SyntheticExtractor()
    batch = features_from_observations(restored, extractor, config, meta)
    scores = batch.features.mean(axis=1).astype(np.float64)
    _, batch_windows, batch_events = score_sequence(batch, scores, 1.0)
    state = EventState(1.0)
    builder, resampler = WindowBuilder(config), CausalResampler(30)
    streaming = []
    for frame, poses in observations:
        for sample in resampler.push(frame, poses):
            for window in builder.push(sample):
                score = (
                    float(extractor.extract_features(window.keypoints, window.frame_size_wh).mean())
                    if window.quality == "valid"
                    else None
                )
                streaming.append(
                    state.update(
                        window.sequence_id,
                        window.track_id,
                        window.sample_index,
                        window.end_ms,
                        score,
                        window.quality if score is None else None,
                    )
                )
    assert len(streaming) == len(batch_windows)
    for batch_row, streaming_row in zip(batch_windows, streaming):
        assert all(batch_row[key] == value for key, value in streaming_row.items())
    assert state.finish(observations[-1][0].timestamp_ms) == batch_events


def test_short_clip_has_no_endpoints_and_null_coverage(config, skeleton):
    cache = features_from_observations(
        [observation(i, skeleton[i]) for i in range(20)],
        SyntheticExtractor(),
        config,
        {"sequence_id": "synthetic", "split": "test"},
    )
    result, _, _ = score_sequence(cache, [], 1.0)
    assert result["n_scheduled_endpoints"] == 0
    assert result["window_coverage"] is None and result["prediction"] is None


@pytest.mark.parametrize("tail_delay", [0.0, 7.0])
@pytest.mark.parametrize("overlay_mode", ["debug", "actions", "auto"])
def test_demo_pipeline_with_explicit_synthetic_dependencies(
    tmp_path, config, skeleton, monkeypatch, tail_delay, overlay_mode
):
    from fall_detect import assets
    from fall_detect import config as config_module
    from fall_detect.models import stgcn
    from fall_detect.pose import human
    from fall_detect.training import experiment

    source = tmp_path / "synthetic-source"
    source.mkdir()
    for i in range(85):
        ok, png = cv2.imencode(".png", np.full((480, 640, 3), 40 + i % 20, np.uint8))
        assert ok
        png.tofile(source / f"image-{i + 1}.png")
    (source / "fixture-data.csv").write_text(
        "".join(f"{i + 1},{i * 1000 / 30 + (tail_delay if i == 84 else 0)}\n" for i in range(85)),
        encoding="utf-8",
    )
    finished_at = []
    original_finish = EventState.finish

    def capture_finish(self, timestamp):
        finished_at.append(timestamp)
        return original_finish(self, timestamp)

    monkeypatch.setattr(EventState, "finish", capture_finish)

    class FakeScorer:
        def __init__(self, *args, **kwargs):
            self.config = config
            self.device = "cpu"
            # The actual Scorer rejects synthetic runs on real videos. This fake tests rendering only.
            self.index = {"source_kind": "real", "recipe_hash": "synthetic-test-double"}
            self.threshold = {"threshold": 1.0}

        def score(self, features):
            return features.mean(axis=1)

    class FakePose:
        def __init__(self, *args):
            pass

        def predict(self, frame):
            if frame.sample_index == 46:
                return []
            points = skeleton[frame.sample_index % 30]
            return [
                PoseRecord(
                    frame.sequence_id,
                    frame.source_frame_id,
                    frame.sample_index,
                    frame.timestamp_ms,
                    1,
                    np.array([80, 30, 300, 300]),
                    points,
                    (640, 480),
                    0.9,
                )
            ]

    class FakeExtractor(SyntheticExtractor):
        def __init__(self, *args):
            pass

        def action_scores(self, features):
            return np.array([[0.1] * 6 + [0.9]], dtype=np.float32)

        def classify_observations(self, points, size):
            assert points.shape == (30, 13, 3)
            return np.array([[0.1] * 6 + [0.9]], dtype=np.float32)

    monkeypatch.setattr(experiment, "Scorer", FakeScorer)
    monkeypatch.setattr(assets, "require_weights", lambda c: {})
    monkeypatch.setattr(config_module, "recipe_fingerprint", lambda c, w: "synthetic-test-double")
    monkeypatch.setattr(human, "HumanPoseBackend", FakePose)
    monkeypatch.setattr(stgcn, "HumanFeatureExtractor", FakeExtractor)
    output = tmp_path / "synthetic-render.mp4"
    demo("synthetic-fake-run", source, output, overlay_mode=overlay_mode)
    metadata = read_json(output.with_suffix(".json"))
    assert metadata["decoded_verified"] and metadata["frames"] == 85
    with output.with_suffix(".csv").open(encoding="utf-8-sig", newline="") as handle:
        logs = list(csv.DictReader(handle))
    assert len(logs) == 85
    assert logs[0]["state"] == "warming_up"
    assert logs[39]["state"] == "suspected_fall"
    assert logs[46]["state"] == logs[47]["state"] == "unknown"
    assert logs[46]["raw_score"] == ""
    assert 2.8 < metadata["duration_ms"] / 1000 < 2.9
    assert finished_at[-1] == pytest.approx(84 * 1000 / 30 + tail_delay)
    if overlay_mode == "actions":
        with (tmp_path / "synthetic-render-actions.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            actions = list(csv.DictReader(handle))
        assert len(actions) == len(logs)
        assert actions[0]["label"] == "PENDING"
        assert actions[39]["label"] == "FALLING"
        assert actions[46]["label"] == actions[47]["label"] == "UNKNOWN"
        assert actions[46]["head_score"] == actions[47]["head_score"] == ""
    if overlay_mode == "auto":
        assert metadata["rendering"]["overlay_mode"] == "sample"
        with (tmp_path / "synthetic-render-actions.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            actions = list(csv.DictReader(handle))
        assert actions[29]["label"] == actions[39]["label"] == "FALLING"
        assert actions[46]["label"] == ""
        assert actions[47]["label"] == "FALLING"  # Short gap keeps actual history.
        with (tmp_path / "synthetic-render-display.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            display = list(csv.DictReader(handle))
        assert display[46]["history_event_count"] == "1"
        assert display[46]["history_event_start_ms"] == display[39]["history_event_start_ms"]


def test_missing_timeline_does_not_shrink_coverage_denominator():
    from fall_detect.evaluation.metrics import sequence_metrics

    rows = [
        {
            "sequence_id": "a",
            "label": 0,
            "prediction": 0,
            "n_scheduled_endpoints": 10,
            "n_valid_endpoints": 10,
        },
        {
            "sequence_id": "b",
            "label": 1,
            "prediction": None,
            "n_scheduled_endpoints": None,
            "n_valid_endpoints": 0,
        },
    ]
    metrics = sequence_metrics(rows)
    assert metrics["window_coverage"] is None and metrics["n_missing_timelines"] == 1
    assert metrics["sequence_coverage"] == 0.5
