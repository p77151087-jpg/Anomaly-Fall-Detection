import zipfile

import cv2
import numpy as np
import pytest

from fall_detect.assets import safe_extract
from fall_detect.config import load_config
from fall_detect.data.readers import (
    image_inventory,
    read_images,
    read_timestamps,
    validate_raw_label,
)
from fall_detect.data.splits import build_split, ensure_split, smoke_ids, validate_split
from fall_detect.io_utils import ContractError, fingerprint, read_json, write_json


def test_numeric_sort_and_frame_id_timestamp_alignment(tmp_path):
    for number in [10, 2, 1]:
        ok, png = cv2.imencode(".png", np.zeros((32, 64, 3), np.uint8))
        assert ok
        png.tofile(tmp_path / f"frame-{number}.png")
    sync = tmp_path / "sync.csv"
    sync.write_text("1,0\n2,40\n10,500\n", encoding="utf-8")
    paths, report = image_inventory(tmp_path)
    assert [p.stem for p in paths] == ["frame-1", "frame-2", "frame-10"]
    assert report["missing_frame_ids"] == list(range(3, 10))
    frames = list(read_images(tmp_path, sync_path=sync))
    assert [
        (f.source_frame_id, f.sample_index, f.timestamp_ms) for f in frames if not f.source_missing
    ] == [
        (1, 0, 0),
        (2, 1, 40),
        (10, 9, 500),
    ]
    assert len(frames) == 10 and sum(f.source_missing for f in frames) == 7
    assert (frames[0].width, frames[0].height) == (64, 32)
    assert not frames[0].timestamp_estimated
    sync.write_text("1,0\n2,40\n", encoding="utf-8")
    with pytest.raises(ContractError, match="no matching"):
        list(read_images(tmp_path, sync_path=sync))


@pytest.mark.parametrize("text", ["1,0\n1,33", "1,50\n2,40", "2,0\n1,33", "1,0\n2,nan", "1.5,0"])
def test_bad_timestamps_rejected(tmp_path, text):
    path = tmp_path / "times.csv"
    path.write_text(text)
    with pytest.raises(ContractError):
        read_timestamps(path)


def test_zip_traversal_rejected_before_extraction(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("safe.txt", "safe")
        z.writestr("../escape.txt", "bad")
    with pytest.raises(ContractError, match="escapes"):
        safe_extract(archive, tmp_path / "dest")
    assert not (tmp_path / "dest/safe.txt").exists()
    assert not (tmp_path / "escape.txt").exists()


def test_fixed_split_isolation_and_persistence(tmp_path):
    manifest = ensure_split(tmp_path)
    assert manifest == ensure_split(tmp_path) == build_split()
    assert len(manifest["sequences"]) == 70
    assert len({s["sequence_id"] for s in manifest["sequences"]}) == 70
    assert sum(s["split"] == "train" for s in manifest["sequences"]) == 24
    assert all(s["label"] == 0 for s in manifest["sequences"] if s["split"] == "train")
    assert smoke_ids(manifest) == ["adl-05", "fall-02"]
    manifest["sequences"][0]["label"] = 1
    manifest["split_hash"] = fingerprint({k: v for k, v in manifest.items() if k != "split_hash"})
    with pytest.raises(ContractError):
        validate_split(manifest)


def test_config_typo_and_invalid_values(tmp_path):
    path = tmp_path / "bad.yaml"
    for content in [
        "skeleton:\n  strdie: 5",
        "skeleton:\n  stride: 0",
        "quality:\n  min_good_frame_ratio: 1.2",
        "features:\n  expected_dim: 256",
        "runtime:\n  num_workers: -1",
        "skeleton:\n  stride: 5.5",
    ]:
        path.write_text(content)
        with pytest.raises(ContractError):
            load_config(path)


def test_transition_label_is_preserved():
    assert validate_raw_label(0) == {"raw_label": 0, "posture": "transition"}
    with pytest.raises(ContractError):
        validate_raw_label(2)


def test_strict_json_and_float64_roundtrip(tmp_path):
    score = float(np.float32(0.3))
    threshold = float(np.nextafter(score, -np.inf))
    write_json(tmp_path / "threshold.json", {"threshold": threshold})
    assert read_json(tmp_path / "threshold.json")["threshold"] == threshold < score
    with pytest.raises(ValueError):
        write_json(tmp_path / "bad.json", {"value": np.inf})
