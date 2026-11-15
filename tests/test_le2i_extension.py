import csv
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from fall_detect.config import load_config, project_path, recipe_fingerprint, validate
from fall_detect.data.le2i import (
    annotated_intervals,
    annotation_groups,
    build_extended_manifest,
    download_resumable,
    find_videos,
)
from fall_detect.data.splits import validate_split
from fall_detect.io_utils import ContractError, fingerprint, read_json
from fall_detect.pipeline import features_from_observations
from fall_detect.pose.lightweight import restore_rotation, suppress_duplicate_poses


def csv_file(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def le2i_fixture(tmp_path, monkeypatch):
    import fall_detect.config as config_module

    monkeypatch.setattr(config_module, "ROOT", tmp_path)
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    labels, videos = [], {}
    for index in range(6):
        name = f"Home_01/video_{index + 1}"
        path = tmp_path / "data/raw/le2i/videos/Home_01" / f"video ({index + 1}).avi"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture-{index}".encode())
        videos[name] = path
        labels.append(
            {
                "path": name,
                "label": 0 if index % 2 == 0 else 1,
                "start": 0,
                "end": 4,
                "subject": index // 2,
                "cam": 1,
                "dataset": "le2i",
            }
        )
    csv_file(annotations / "labels.csv", list(labels[0]), labels)
    for split, ids in zip(("train", "validation", "test"), ((1, 2), (3, 4), (5, 6))):
        csv_file(
            annotations / f"{split}.csv", ["path"], [{"path": f"Home_01/video_{i}"} for i in ids]
        )
    return annotations, videos


def rehash(manifest):
    manifest["split_hash"] = fingerprint({k: v for k, v in manifest.items() if k != "split_hash"})
    return manifest


def test_whole_video_fall_exclusion_and_original_split(le2i_fixture):
    annotations, videos = le2i_fixture
    manifest = build_extended_manifest(annotations, videos)
    assert len(manifest["sequences"]) == 75
    assert len(manifest["excluded"]) == 1
    assert all(r["label"] == 0 for r in manifest["sequences"] if r["split"] == "train")
    assert find_videos(next(iter(videos.values())).parents[1]) == videos
    modified = deepcopy(manifest)
    modified["sequences"][0]["split"] = "test"
    with pytest.raises(ContractError, match="frozen UR Fall"):
        validate_split(rehash(modified))
    modified = deepcopy(manifest)
    ext = [r for r in modified["sequences"] if r["dataset"] == "le2i"]
    ext[-1]["subject"] = ext[0]["subject"]
    with pytest.raises(ContractError, match="subject crosses"):
        validate_split(rehash(modified))
    modified = deepcopy(manifest)
    row = next(r for r in modified["sequences"] if r["dataset"] == "le2i" and r["label"] == 1)
    row.update(label=0, contains_fall=False, split="train")
    with pytest.raises(ContractError):
        validate_split(rehash(modified))


def test_annotation_original_video_leakage_rejected(le2i_fixture):
    annotations, _ = le2i_fixture
    with (annotations / "test.csv").open("a", encoding="utf-8") as handle:
        handle.write("Home_01/video_1\n")
    with pytest.raises(ContractError, match="crosses splits"):
        annotation_groups(annotations)


def test_identical_video_bytes_rejected(le2i_fixture):
    annotations, videos = le2i_fixture
    videos["Home_01/video_5"].write_bytes(videos["Home_01/video_1"].read_bytes())
    with pytest.raises(ContractError, match="Duplicate video bytes"):
        build_extended_manifest(annotations, videos)


def test_within_split_duplicate_groups_excluded_wholly(le2i_fixture):
    annotations, videos = le2i_fixture
    videos["Home_01/video_4"].write_bytes(videos["Home_01/video_3"].read_bytes())
    manifest = build_extended_manifest(annotations, videos)
    assert len(manifest["sequences"]) == 73
    excluded = [r for r in manifest["excluded"] if r["reason"] == "duplicate_source_bytes"]
    assert {r["original_video"] for r in excluded} == {"Home_01/video_3", "Home_01/video_4"}


def test_unlabelled_gaps_never_enter_training(config, skeleton):
    from conftest import observation

    class Extractor:
        feature_dim = 512

        def extract_features(self, points, size):
            return np.ones((1, 512), np.float32)

    intervals = annotated_intervals(
        [{"start": 0, "end": 0.4}, {"start": 0.4, "end": 1}, {"start": 2, "end": 4}]
    )
    assert intervals == [[0, 1000], [2000, 4000]]
    observations = [observation(i, skeleton[i % 30]) for i in range(121)]
    cache = features_from_observations(
        observations,
        Extractor(),
        config,
        {"split": "train", "sequence_id": "synthetic", "annotation_intervals_ms": intervals},
    )
    assert any(r.quality == "unlabelled_interval" for r in cache.records)
    for r in cache.records:
        if r.row_index is not None:
            assert r.end_ms <= 1000 or r.start_ms >= 2000


@pytest.mark.parametrize("angle", [-90, 0, 90])
def test_rotation_inverse_non_square_missing_joints(angle):
    original = np.array([[10, 20, 0.9], [200, 100, 0.8], [0, 0, 0]], np.float32)
    rotated = original.copy()
    if angle == 90:
        rotated[:2, 0], rotated[:2, 1] = original[:2, 1], 639 - original[:2, 0]
    elif angle == -90:
        rotated[:2, 0], rotated[:2, 1] = 479 - original[:2, 1], original[:2, 0]
    np.testing.assert_array_equal(restore_rotation(rotated, angle, 640, 480), original)


def test_rotation_deduplicates_without_merging_people():
    points = np.array([[20 + i * 3, 30 + i * 7, 0.8] for i in range(18)], np.float32)
    shifted = points.copy()
    shifted[:, 0] += 300
    duplicate = points.copy()
    duplicate[:3, 2] = 0
    selected = suppress_duplicate_poses([(points, 10), (duplicate, 9), (shifted, 10)])
    assert len(selected) == 2
    np.testing.assert_array_equal(selected[0][0], points)


def test_historical_recipe_hash_and_new_rotation_isolation():
    run = project_path("runs/openpose18-baseline-001")
    if not run.exists():
        pytest.skip("Local historical run not present")
    config = load_config(run / "resolved-config.yaml")
    index = read_json(run / "dataset-index.json")
    hashes = {role: entry["sha256"] for role, entry in index["weights"].items()}
    assert recipe_fingerprint(config, hashes) == index["recipe_hash"]
    config["pose"]["rotation_angles"] = [0, -90, 90]
    assert recipe_fingerprint(validate(config), hashes) != index["recipe_hash"]
    config["pose"]["rotation_angles"] = [0, {}]
    with pytest.raises(ContractError):
        validate(config)


def test_resume_range_mismatch_preserves_partial(tmp_path):
    class Response:
        status_code = 206

        def __init__(self):
            self.headers = {"Content-Range": "bytes 0-9/10"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

    class Http:
        def get(self, *args, **kwargs):
            return Response()

    target = tmp_path / "source.zip"
    partial = Path(str(target) + ".part")
    partial.write_bytes(b"12345")
    with pytest.raises(ContractError, match="range mismatch"):
        download_resumable(Http(), "https://example.invalid/data", target, 10)
    assert partial.read_bytes() == b"12345" and not target.exists()


def test_reuse_keeps_source_immutable_and_rejects_semantic_changes(tmp_path, config, skeleton):
    from conftest import observation

    from fall_detect.data.cache import load_feature_cache, save_feature_cache, save_pose_cache
    from fall_detect.io_utils import sha256
    from fall_detect.pipeline import reuse_prepared_sequence

    class Extractor:
        feature_dim = 512

        def extract_features(self, points, size):
            return np.ones((1, 512), np.float32)

    metadata = {
        "split": "train",
        "sequence_id": "synthetic",
        "split_hash": "old-split",
        "recipe_hash": "fixture-recipe",
        "source_hash": "fixture-source",
        "source_kind": "synthetic",
        "recipe_id": config["skeleton"]["recipe_id"],
        "limit_frames": None,
    }
    observations = [observation(i, skeleton[i % 30]) for i in range(45)]
    cache = features_from_observations(observations, Extractor(), config, metadata)
    old_feature, old_pose = tmp_path / "old-feature.npz", tmp_path / "old-pose.npz"
    save_feature_cache(old_feature, cache)
    save_pose_cache(old_pose, observations, metadata)
    files = [old_feature, old_pose, old_feature.with_suffix(".json"), old_pose.with_suffix(".json")]
    original_hashes = [sha256(p) for p in files]
    index = {
        "recipe_hash": metadata["recipe_hash"],
        "split_manifest": {"split_hash": "old-split"},
        "sequences": {
            "synthetic": {
                "status": "prepared",
                "feature_path": str(old_feature),
                "pose_path": str(old_pose),
            }
        },
    }
    new_meta = {**metadata, "split_hash": "new-split"}
    destination, pose = tmp_path / "new-feature.npz", tmp_path / "new-pose.npz"
    reuse_prepared_sequence(index, "synthetic", new_meta, destination, pose)
    reused = load_feature_cache(destination, new_meta)
    np.testing.assert_array_equal(reused.features, cache.features)
    assert [sha256(p) for p in files] == original_hashes
    for key, value in [
        ("split", "test"),
        ("recipe_hash", "different"),
        ("source_hash", "changed"),
        ("annotation_intervals_ms", [[0, 500]]),
    ]:
        with pytest.raises(ContractError, match="mismatch"):
            reuse_prepared_sequence(index, "synthetic", {**new_meta, key: value}, destination, pose)


def test_video_premature_end_rejected_but_explicit_smoke_limit_allowed(monkeypatch):
    import cv2

    from fall_detect.data.readers import read_video

    class Capture:
        def __init__(self, *args):
            self.count = 0

        def isOpened(self):
            return True

        def get(self, prop):
            return {
                cv2.CAP_PROP_FPS: 25,
                cv2.CAP_PROP_FRAME_COUNT: 10,
                cv2.CAP_PROP_POS_MSEC: (self.count - 1) * 40,
            }[prop]

        def read(self):
            self.count += 1
            return (True, np.zeros((10, 10, 3), np.uint8)) if self.count <= 5 else (False, None)

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", Capture)
    assert len(list(read_video("fixture.avi", limit=5))) == 5
    with pytest.raises(ContractError, match="before declared frame count"):
        list(read_video("fixture.avi"))


@pytest.mark.parametrize("good_primary,expected_calls", [(True, 1), (False, 3)])
def test_selective_rotation_retains_observed_pose_and_avoids_unneeded_calls(
    good_primary, expected_calls
):
    from types import SimpleNamespace

    from fall_detect.contracts import FrameRecord
    from fall_detect.pose.lightweight import LightweightPoseBackend

    class Pose:
        last_id = -1

        def __init__(self, points, score):
            self.bbox, self.id, self.confidence = (0, 0, 50, 50), 0, score

    backend = LightweightPoseBackend.__new__(LightweightPoseBackend)
    backend.config = load_config(project_path("configs/openpose18-rotation.yaml"))
    backend.pose_module = SimpleNamespace(Pose=Pose, track_poses=lambda *args, **kwargs: None)
    backend.reset()
    points = np.array([[10 + i, 20 + i, 0.9] for i in range(18)], np.float32)
    calls = []

    def decode(image):
        calls.append(image.shape)
        return [(points.copy(), 12)] if good_primary or len(calls) == 2 else []

    backend._decode = decode
    frame = FrameRecord("fixture", 0, 0, 0, np.zeros((480, 640, 3), np.uint8), 640, 480)
    poses = backend._predict(frame)
    assert len(calls) == expected_calls and len(poses) == 1
    expected = points if good_primary else restore_rotation(points, -90, 640, 480)
    np.testing.assert_array_equal(poses[0].keypoints, expected)
