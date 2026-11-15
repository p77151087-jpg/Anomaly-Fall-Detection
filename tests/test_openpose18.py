"""Alternate-recipe risks. Fixtures test contracts, never fall-detection accuracy."""

import ast
import math
from copy import deepcopy
from dataclasses import replace

import cv2
import numpy as np
import pytest
import torch

from fall_detect.assets import ensure_weight_manifest, require_weights
from fall_detect.config import load_config, project_path, recipe_fingerprint, validate
from fall_detect.contracts import FrameRecord, PoseRecord
from fall_detect.data.cache import (
    load_feature_cache,
    load_pose_cache,
    save_feature_cache,
    save_pose_cache,
)
from fall_detect.io_utils import ContractError
from fall_detect.models.openpose_stgcn import OpenPoseFeatureExtractor
from fall_detect.models.weights import select_device
from fall_detect.pipeline import features_from_observations
from fall_detect.pose.lightweight import LightweightPoseBackend, prepare_image
from fall_detect.recipes import OPENPOSE_RECIPE
from fall_detect.skeleton.preprocessing import good_frame, openpose_preprocess
from fall_detect.skeleton.windows import CausalResampler, WindowBuilder


@pytest.fixture
def alternate():
    return load_config(project_path("configs/openpose18.yaml"), "cpu")


@pytest.fixture
def points18():
    t, k = np.arange(30)[:, None], np.arange(18)[None, :]
    return np.stack(
        (100 + k * 5 + t * 0.3, 70 + k * 15 + t * 0.2, np.full((30, 18), 0.8)), axis=-1
    ).astype(np.float32)


def observations(points, n=85, gap=None):
    for i in range(n):
        frame = FrameRecord("fixture18", i + 1, i, i * 1000 / 30, None, 640, 480)
        pose = PoseRecord(
            "fixture18",
            i + 1,
            i,
            frame.timestamp_ms,
            1,
            np.array([90, 60, 200, 400]),
            points[i % 30].copy(),
            (640, 480),
            0.8,
            layout_id="openpose18",
            recipe_id=OPENPOSE_RECIPE,
        )
        yield frame, [] if i == gap else [pose]


def test_openpose_normalization_mask_and_nonsquare(points18):
    points18[:, 17] = [999, 777, 0]
    original = points18.copy()
    actual = openpose_preprocess(points18, (640, 480))
    # Independent Kinetics feeder coordinate contract, full person M=1.
    expected = original.transpose(2, 0, 1).copy()
    expected[0] /= 640
    expected[1] /= 480
    expected[:2] -= 0.5
    expected[0][expected[2] == 0] = 0
    expected[1][expected[2] == 0] = 0
    np.testing.assert_array_equal(actual, expected[None, :, :, :, None])
    np.testing.assert_array_equal(points18, original)
    with pytest.raises(ContractError):
        openpose_preprocess(points18[:, :13], (640, 480))
    broken = points18.copy()
    broken[0, 0, 0] = np.nan
    with pytest.raises(ContractError):
        openpose_preprocess(broken, (640, 480))


def test_pretrained_image_transform_parity():
    path = project_path("third_party/lightweight_openpose/val.py")
    if not path.exists():
        pytest.skip("Pinned pose source unavailable")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"normalize", "pad_width"}
    ]
    scope = {"np": np, "cv2": cv2, "math": math}
    # Execute only two function definitions from the pinned source for numeric parity.
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), scope)  # noqa: S102
    image = np.random.default_rng(42).integers(0, 255, (333, 511, 3), dtype=np.uint8)
    actual, scale, pad = prepare_image(image, 256)
    resized = cv2.resize(image, (0, 0), fx=256 / 333, fy=256 / 333, interpolation=cv2.INTER_LINEAR)
    expected = scope["normalize"](
        resized, np.array([128, 128, 128], np.float32), np.float32(1 / 256)
    )
    expected, old_pad = scope["pad_width"](
        expected, 8, (0, 0, 0), [256, max(expected.shape[1], 256)]
    )
    np.testing.assert_array_equal(actual.numpy()[0].transpose(1, 2, 0), expected)
    assert pad == old_pad and scale == 256 / 333


def test_recipe_cannot_mix_weights_or_dimensions(alternate, tmp_path):
    changed = deepcopy(alternate)
    changed["features"]["expected_dim"] = 512
    with pytest.raises(ContractError):
        validate(changed)
    changed = deepcopy(alternate)
    changed["pose"]["weights_manifest"] = str(tmp_path / "weights.json")
    ensure_weight_manifest(changed)
    human = load_config()
    human["pose"]["weights_manifest"] = changed["pose"]["weights_manifest"]
    with pytest.raises(ContractError, match="recipe/roles"):
        ensure_weight_manifest(human)
    assert recipe_fingerprint(human, {}) != recipe_fingerprint(alternate, {})
    changed["skeleton"]["recipe_id"] = "unregistered"
    with pytest.raises(ContractError):
        validate(changed)


def test_openpose_windows_gaps_layout_and_quality(alternate, points18):
    assert good_frame(points18[0], alternate["quality"], 18)
    missing = points18[0].copy()
    missing[:7, 2] = 0
    assert not good_frame(missing, alternate["quality"], 18)
    resampler, builder = CausalResampler(30), WindowBuilder(alternate)
    windows = []
    for frame, poses in observations(points18, gap=32):
        for sample in resampler.push(frame, poses):
            windows.extend(builder.push(sample))
    assert windows[0].quality == "valid"
    assert windows[1].quality == "track_gap"
    assert windows[-1].quality == "valid"
    assert all(w.keypoints is None for w in windows if w.quality != "valid")
    resampler, builder = CausalResampler(30), WindowBuilder(alternate)
    with pytest.raises(ContractError, match="layout/recipe"):
        for frame, poses in observations(points18, n=30):
            for sample in resampler.push(frame, [replace(poses[0], layout_id="human13")]):
                builder.push(sample)


def test_pose_cache_roundtrip_and_empty_shape(alternate, points18, tmp_path):
    metadata = {"recipe_id": OPENPOSE_RECIPE, "source_hash": "fixture"}
    path = tmp_path / "poses.npz"
    obs = list(observations(points18))
    save_pose_cache(path, obs, metadata)
    restored, _ = load_pose_cache(path, metadata)
    np.testing.assert_array_equal(restored[0][1][0].keypoints, points18[0])
    assert restored[0][1][0].layout_id == "openpose18"
    with pytest.raises(ContractError):
        load_pose_cache(path, {"recipe_id": "human_tsstg_coco_cut_v1"})
    save_pose_cache(path, [(frame, []) for frame, _ in obs], metadata)
    with np.load(path, allow_pickle=False) as arrays:
        assert arrays["keypoints"].shape == (0, 18, 3)


def test_256_feature_cache_and_streaming_share_endpoints(alternate, points18, tmp_path):
    class FixtureExtractor:
        feature_dim = 256

        def extract_features(self, points, size):
            return np.full((1, 256), points.mean(), np.float32)

    meta = {"sequence_id": "fixture18", "split": "train", "source_kind": "synthetic"}
    cache = features_from_observations(observations(points18), FixtureExtractor(), alternate, meta)
    path = tmp_path / "features.npz"
    save_feature_cache(path, cache)
    loaded = load_feature_cache(path)
    np.testing.assert_array_equal(loaded.features, cache.features)
    assert all(
        row.recipe_id == OPENPOSE_RECIPE and row.feature_dim == 256 for row in loaded.records
    )
    builder, resampler = WindowBuilder(alternate), CausalResampler(30)
    streaming = []
    for frame, poses in observations(points18):
        for sample in resampler.push(frame, poses):
            for window in builder.push(sample):
                streaming.append((window.sample_index, window.track_id, window.quality))
    assert streaming == [(r.sample_index, r.track_id, r.quality) for r in cache.records]
    cache.metadata["recipe_id"] = "human_tsstg_coco_cut_v1"
    with pytest.raises(ContractError):
        save_feature_cache(path, cache)


def test_parallel_download_worker_reuses_complete_sequence(tmp_path):
    import zipfile

    from fall_detect.assets import fetch_sequence
    from fall_detect.io_utils import read_json, sha256

    archive = tmp_path / "archives" / "adl-01-cam0-rgb.zip"
    archive.parent.mkdir()
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("adl-01-cam0-rgb/frame-001.png", b"fixture archive, not a real image")
    sync = tmp_path / "adl-01" / "adl-01-data.csv"
    sync.parent.mkdir()
    sync.write_text("1,0.0\n", encoding="utf-8")
    links = {
        archive.name: "https://example.invalid/unused.zip",
        sync.name: "https://example.invalid/unused.csv",
    }
    assert fetch_sequence(tmp_path, "adl-01", links)["status"] == "downloaded"
    marker = read_json(sync.parent / "source.json")
    assert marker["archive_sha256"] == sha256(archive)
    assert marker["sync_sha256"] == sha256(sync)
    # Existing complete files require no network; a second call preserves provenance.
    assert fetch_sequence(tmp_path, "adl-01", links)["status"] == "downloaded"
    assert read_json(sync.parent / "source.json") == marker


def test_download_worker_failure_is_reported_per_sequence(tmp_path, monkeypatch):
    import requests

    from fall_detect.assets import fetch_sequence

    def fail(*args, **kwargs):
        raise requests.ConnectionError("fixture network failure")

    monkeypatch.setattr(requests.Session, "get", fail)
    result = fetch_sequence(
        tmp_path, "adl-01", {"adl-01-cam0-rgb.zip": "https://example.invalid/failure"}
    )
    assert result == {"sequence_id": "adl-01", "error": "fixture network failure"}
    assert not (tmp_path / "adl-01/source.json").exists()


def test_missing_timestamp_tail_is_opt_in_and_causal(tmp_path, alternate):
    from fall_detect.data.readers import read_images

    for index in range(1, 5):
        cv2.imencode(".png", np.full((20, 30, 3), index, np.uint8))[1].tofile(
            tmp_path / f"frame-{index:03}.png"
        )
    sync = tmp_path / "times.csv"
    sync.write_text("1,0\n2,40\n", encoding="utf-8")
    with pytest.raises(ContractError, match="synchronization"):
        list(read_images(tmp_path, sync_path=sync))
    frames = list(read_images(tmp_path, sync_path=sync, allow_missing_timestamp_tail=True))
    assert [f.timestamp_estimated for f in frames] == [False, False, True, True]
    np.testing.assert_allclose(
        [f.timestamp_ms for f in frames], [0, 40, 40 + 1000 / 30, 40 + 2000 / 30]
    )
    assert all(not f.source_missing for f in frames)
    # Interior timestamp gaps remain errors; no use of future timestamps for filling.
    sync.write_text("1,0\n3,80\n4,120\n", encoding="utf-8")
    with pytest.raises(ContractError, match="synchronization"):
        list(read_images(tmp_path, sync_path=sync, allow_missing_timestamp_tail=True))
    other = deepcopy(alternate)
    other["data"]["allow_missing_timestamp_tail"] = False
    assert recipe_fingerprint(other, {}) != recipe_fingerprint(alternate, {})


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
@pytest.mark.parametrize("weighted", [False, True])
def test_ae_scoring_batch_and_streaming_are_identical(device_name, weighted):
    from fall_detect.detection.events import above_threshold
    from fall_detect.models.autoencoder import Autoencoder
    from fall_detect.training.experiment import Scorer
    from fall_detect.training.scaler import FeatureScaler

    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    values = np.random.default_rng(12).normal(size=(17, 256)).astype(np.float32)
    scorer = object.__new__(Scorer)
    scorer.device = select_device(device_name)
    scorer.scaler = FeatureScaler.fit(values)
    torch.manual_seed(42)
    scorer.model = Autoencoder(256, 128, 32).to(scorer.device).eval()
    # This unit fixture intentionally bypasses __init__; include its new scoring field.
    scorer.residual_weights = np.linspace(0.5, 1.5, 256, dtype=np.float32) if weighted else None
    batch = scorer.score(values)
    streaming = np.array([scorer.score(row[None])[0] for row in values])
    np.testing.assert_array_equal(batch, streaming)
    for expected, actual in zip(batch, streaming):
        assert not above_threshold(actual, float(expected))


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_real_checkpoints_frozen_heads_and_pose_tracking(alternate, points18, device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    manifest = ensure_weight_manifest(alternate)
    if any(not project_path(e["local_path"]).exists() for e in manifest["resources"]):
        pytest.skip("Real alternate checkpoints unavailable")
    resources = require_weights(alternate)
    device = select_device(device_name)
    extractor = OpenPoseFeatureExtractor(alternate, resources, device)
    before = {k: v.clone() for k, v in extractor.model.named_buffers()}
    features = extractor.extract_features(points18, (640, 480))
    assert features.shape == (1, 256) and extractor.forward_verified
    assert extractor.load_report["strict"] and not any(
        p.requires_grad for p in extractor.model.parameters()
    )
    for key, value in extractor.model.named_buffers():
        torch.testing.assert_close(value, before[key])
    pose = LightweightPoseBackend(alternate, resources, device)
    assert pose.load_report["strict"] and not any(p.requires_grad for p in pose.model.parameters())
    # Actual image forward (when the bundled training smoke sequence is available).
    from fall_detect.data.readers import read_source

    source = project_path("data/raw/urfall/adl-05")
    if source.exists():
        frame = next(read_source(source, sequence_id="adl-05", limit=1))
        predicted = pose.predict(frame)
        assert pose.forward_verified and predicted
        assert all(p.keypoints.shape == (18, 3) and p.layout_id == "openpose18" for p in predicted)
        first_ids = [p.track_id for p in predicted]
        following = pose.predict(replace(frame, timestamp_ms=1000 / 30, source_frame_id=2))
        assert [p.track_id for p in following] == first_ids
        pose.reset()
        assert [p.track_id for p in pose.predict(frame)] == first_ids
        # CPU/GPU feature inference on a real observed skeleton window, when available.
        index_path = project_path(
            "outputs/real_feature_smoke/openpose18-cuda-v1/dataset-index.json"
        )
        if index_path.exists():
            from fall_detect.io_utils import read_json

            index = read_json(index_path)
            cached, _ = load_pose_cache(project_path(index["sequences"]["adl-05"]["pose_path"]), {})
            builder, resampler = WindowBuilder(alternate), CausalResampler(30)
            for f, poses in cached:
                for sample in resampler.push(f, poses):
                    for window in builder.push(sample):
                        if window.quality == "valid":
                            assert np.isfinite(
                                extractor.extract_features(window.keypoints, window.frame_size_wh)
                            ).all()
                            return
            pytest.fail("No real skeleton window in smoke cache")
