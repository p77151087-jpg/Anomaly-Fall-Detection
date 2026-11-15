"""Synthetic boundary evidence only; no real annotations used as fixtures."""

import hashlib
import importlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from fall_detect.contracts import FrameRecord, PoseRecord
from fall_detect.evaluation.whole_pose_coordinates import (
    PARENT,
    apply_review,
    recipe_hash,
    validate_review,
)
from fall_detect.io_utils import ContractError
from fall_detect.skeleton.preprocessing import HUMAN13_NAMES


@pytest.fixture
def inputs():
    points = np.array([[200 + j * 10, 100 + j * 12, 0.8] for j in range(13)], dtype=np.float32)
    observations, frames, reviews = [], [], []
    for fid in range(1, 55):
        t = (fid - 1) * 1000 / 30
        f = FrameRecord("fall-19", fid, fid - 1, t, None, 640, 480)
        p = PoseRecord(
            "fall-19",
            fid,
            fid - 1,
            t,
            1,
            np.array([100, 50, 500, 450]),
            points.copy(),
            (640, 480),
            0.9,
        )
        observations.append((f, [p]))
        if fid in (44, 47, 49):
            frame = {
                "source_frame_id": fid,
                "timestamp_ms": t,
                "track_id": 1,
                "image_sha256": "image",
                "frame_size_wh": [640, 480],
            }
            frames.append({**frame, "keypoints": points.astype(float).tolist()})
            rows = [
                {
                    "joint_index": j,
                    "joint_name": HUMAN13_NAMES[j],
                    "status": "matches_original",
                    "xy": points[j, :2].astype(float).tolist(),
                    "original_xy": points[j, :2].astype(float).tolist(),
                    "original_confidence": float(points[j, 2]),
                }
                for j in range(13)
            ]
            if fid == 47:
                rows[11].update(status="mislocated_visible", xy=[305, 220])
                rows[12].update(status="not_visible_or_uncertain", xy=None)
            reviews.append({**frame, "points": rows})
    bundle = {
        "diagnostic_id": PARENT,
        "sequence_id": "fall-19",
        "split": "validation",
        "pose_array_sha256": "pose",
        "pose_metadata_sha256": "meta",
        "source_manifest_sha256": "manifest",
        "names": list(HUMAN13_NAMES),
        "frames": frames,
    }
    digest = hashlib.sha256(
        json.dumps(bundle, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    bundle["bundle_sha256"] = digest
    a = {
        k: bundle[k]
        for k in (
            "diagnostic_id",
            "sequence_id",
            "split",
            "pose_array_sha256",
            "pose_metadata_sha256",
            "source_manifest_sha256",
        )
    }
    a.update(
        schema_version=1,
        author_kind="human_self_reported",
        human_confirmed=True,
        reviewer="Synthetic reviewer",
        reviewed_at="2026-10-13T08:00:00Z",
        parent_bundle_sha256=digest,
        frames=reviews,
    )
    return observations, a, bundle


def test_only_confirmed_xy_changed_and_no_interpolation(inputs):
    obs, a, bundle = inputs
    derived, audit = apply_review(obs, a, bundle)
    assert len(audit["corrections"]) == 1 and audit["counts"]["not_visible_or_uncertain"] == 1
    for (f, ps), (g, qs) in zip(obs, derived):
        assert f == g
        p, q = ps[0], qs[0]
        expected = p.keypoints.copy()
        if f.source_frame_id == 47:
            expected[11, :2] = [305, 220]
        np.testing.assert_array_equal(expected, q.keypoints)
        np.testing.assert_array_equal(p.keypoints[:, 2], q.keypoints[:, 2])
        np.testing.assert_array_equal(p.bbox_xyxy, q.bbox_xyxy)
        assert p.quality == q.quality and p.track_id == q.track_id
        assert p.keypoints is not q.keypoints
    assert obs[46][1][0].keypoints[11, 0] == 310


@pytest.mark.parametrize(
    "key,value",
    [
        ("split", "test"),
        ("sequence_id", "fall-20"),
        ("parent_bundle_sha256", "other"),
        ("pose_array_sha256", "other"),
        ("source_manifest_sha256", "other"),
        ("human_confirmed", "true"),
        ("human_confirmed", False),
        ("author_kind", "ai_provisional"),
        ("schema_version", True),
        ("reviewer", ""),
        ("reviewed_at", "2026-10-13"),
    ],
)
def test_identity_human_and_date(inputs, key, value):
    _, a, b = inputs
    a[key] = value
    with pytest.raises(ContractError):
        validate_review(a, b)


@pytest.mark.parametrize(
    "change",
    [
        "duplicate_frame",
        "duplicate_joint",
        "missing_joint",
        "name",
        "original_xy",
        "confidence",
        "time",
        "track",
        "image",
        "matching_xy",
        "required_pending",
        "uncertain_xy",
        "bool_joint",
    ],
)
def test_locked_frame_and_joint_fields(inputs, change):
    _, a, b = inputs
    f, p = a["frames"][1], a["frames"][1]["points"][11]
    if change == "duplicate_frame":
        a["frames"][2] = deepcopy(f)
    if change == "duplicate_joint":
        f["points"][10] = deepcopy(p)
    if change == "missing_joint":
        f["points"].pop()
    if change == "name":
        p["joint_name"] = "right_ankle"
    if change == "original_xy":
        p["original_xy"][0] += 1
    if change == "confidence":
        p["original_confidence"] = 0.9
    if change == "time":
        f["timestamp_ms"] += 1
    if change == "track":
        f["track_id"] = 2
    if change == "image":
        f["image_sha256"] = "wrong"
    if change == "matching_xy":
        f["points"][0]["xy"][0] += 1
    if change == "required_pending":
        p.update(status="unreviewed", xy=None)
    if change == "uncertain_xy":
        f["points"][12]["xy"] = [200, 200]
    if change == "bool_joint":
        p["joint_index"] = True
    with pytest.raises(ContractError):
        validate_review(a, b)


@pytest.mark.parametrize(
    "xy", [[float("nan"), 3], [float("inf"), 3], [-1, 3], [640, 3], [3, 480], [True, 3], [3], None]
)
def test_invalid_visible_coordinates(inputs, xy):
    _, a, b = inputs
    a["frames"][1]["points"][11]["xy"] = xy
    with pytest.raises(ContractError):
        validate_review(a, b)


def test_bundle_digest_and_mapping(inputs):
    _, a, b = inputs
    b["frames"][0]["keypoints"][0][0] += 1
    with pytest.raises(ContractError):
        validate_review(a, b)


@pytest.mark.parametrize("change", ["missing", "track", "foreign"])
def test_do_not_fill_missing_or_other_track(inputs, change):
    obs, a, b = inputs
    if change == "missing":
        obs[46] = (obs[46][0], [])
    if change == "track":
        obs[46][1][0].track_id = 2
    if change == "foreign":
        obs[0][0].sequence_id = "test-video"
    with pytest.raises(ContractError):
        apply_review(obs, a, b)


def test_recipe_separates_annotation_bytes_and_joint_values(inputs):
    _, a, _ = inputs
    first = recipe_hash(a, "first")
    assert first != recipe_hash(a, "second")
    a["frames"][1]["points"][11]["xy"][0] += 1
    assert first != recipe_hash(a, "first")


@pytest.fixture
def driver(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("whole_pose_coordinate_diagnostic")


def test_test_metadata_rejected_before_any_array_load(driver, monkeypatch):
    monkeypatch.setattr(driver, "read_json", lambda _: {"split": "test"})

    def forbidden(*args, **kwargs):
        pytest.fail("Array read before split check")

    monkeypatch.setattr(driver, "load_feature_cache", forbidden)
    monkeypatch.setattr(driver, "load_pose_cache", forbidden)
    index = {"sequences": {"fall-19": {"feature_path": "fake.npz", "pose_path": "fake-pose.npz"}}}
    with pytest.raises(ContractError):
        driver.original_inputs("fall-19", index)
    with pytest.raises(ContractError):
        driver.original_inputs("fall-20", {})


def test_existing_stage_refuses_before_source_read(driver, monkeypatch, tmp_path):
    monkeypatch.setattr(driver, "OUT", tmp_path)
    monkeypatch.setattr(
        driver, "sha256", lambda _: pytest.fail("Should refuse before source access")
    )
    with pytest.raises(ContractError):
        driver.prepare(tmp_path / "unused.json")


def test_exclusive_file_does_not_overwrite(driver, tmp_path):
    p = tmp_path / "file.json"
    driver.exclusive_bytes(p, b"original")
    with pytest.raises(FileExistsError):
        driver.exclusive_bytes(p, b"replacement")
    assert p.read_bytes() == b"original"
