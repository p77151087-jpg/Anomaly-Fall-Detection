from copy import deepcopy

import pytest

from fall_detect.config import ROOT, load_config
from fall_detect.data.caucafall import (
    ACTIVITIES,
    ARCHIVE_BYTES,
    ARCHIVE_ETAG,
    PREFIX,
    ArchiveRanges,
    build_manifest,
    video_identity,
)
from fall_detect.data.splits import validate_split
from fall_detect.io_utils import ContractError, fingerprint, read_json


@pytest.fixture
def manifest():
    base = read_json(ROOT / "data/manifests/le2i-extension/split-v1.json")
    inventory = [
        {
            "sequence_id": f"caucafall-S{s:02d}-{a}",
            "subject": s,
            "activity": a,
            "label": int(a.startswith("Fall")),
            "archive_member": f"{PREFIX}Subject.{s}/{folder}/{a}S{s}.avi",
            "source_path": f"data/raw/caucafall/original/Subject.{s}/{a}S{s}.avi",
            "source_sha256": fingerprint([s, a]),
        }
        for s in range(1, 11)
        for a, folder in ACTIVITIES.items()
    ]
    return build_manifest(base, inventory)


def resign(manifest):
    manifest["split_hash"] = fingerprint({k: v for k, v in manifest.items() if k != "split_hash"})
    return manifest


def test_subject_group_split_normal_only_and_frozen_base(manifest):
    assert len(manifest["sequences"]) == 231
    assert manifest["sequences"][:161] == manifest["base_manifest"]["sequences"]
    cauca = [r for r in manifest["sequences"] if r.get("dataset") == "caucafall"]
    for split, total, normals in (("train", 30, 30), ("validation", 20, 10), ("test", 20, 10)):
        rows = [r for r in cauca if r["split"] == split]
        assert len(rows) == total and sum(r["label"] == 0 for r in rows) == normals
    for subject in range(1, 11):
        assert len({r["split"] for r in cauca if r["subject"] == subject}) == 1
    assert len(manifest["excluded"]) == 30
    assert all(r["label"] == 1 and r["split"] == "train" for r in manifest["excluded"])
    assert (
        load_config(ROOT / "configs/openpose18-caucafall.yaml")["data"]["protocol_id"]
        == manifest["protocol_id"]
    )


@pytest.mark.parametrize(
    "corruption",
    ["subject_leak", "train_fall", "missing", "duplicate", "old_split", "path", "activity"],
)
def test_reject_dataset_leaks_and_corruption(manifest, corruption):
    altered = deepcopy(manifest)
    rows = [r for r in altered["sequences"] if r.get("dataset") == "caucafall"]
    if corruption == "subject_leak":
        rows[0]["split"] = "test" if rows[0]["split"] != "test" else "validation"
    elif corruption == "train_fall":
        next(r for r in rows if r["split"] == "train")["label"] = 1
    elif corruption == "missing":
        altered["sequences"].pop()
    elif corruption == "duplicate":
        rows[1]["source_sha256"] = rows[0]["source_sha256"]
    elif corruption == "old_split":
        altered["sequences"][0]["split"] = "test"
    elif corruption == "path":
        rows[0]["source_path"] = "data/raw/caucafall/../../secret.avi"
    else:
        rows[0]["activity"] = "Walk"
    with pytest.raises(ContractError):
        validate_split(resign(altered))


@pytest.mark.parametrize(
    "member",
    [f"{PREFIX}Subject.1/Walk/WalkS2.avi", f"{PREFIX}Subject.1/Walk/UnknownS1.avi", "../evil.avi"],
)
def test_original_video_identity_strict(member):
    with pytest.raises(ContractError):
        video_identity(member)


@pytest.mark.parametrize(
    "status,range_header,etag",
    [
        (200, "bytes 0-2/8326349365", ARCHIVE_ETAG),
        (206, "bytes 1-3/8326349365", ARCHIVE_ETAG),
        (206, "bytes 0-2/8326349365", '"changed"'),
    ],
)
def test_range_download_rejects_unbounded_or_changed_response(status, range_header, etag):
    class Response:
        status_code = status

        def __init__(self):
            self.headers = {"Content-Range": range_header, "ETag": etag}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        @property
        def content(self):
            pytest.fail("Must reject before reading an unexpected archive body")

    class Http:
        def get(self, *args, **kwargs):
            return Response()

        def close(self):
            pass

    remote = ArchiveRanges.__new__(ArchiveRanges)
    remote.position, remote.url, remote.http = 0, "https://example.test/archive", Http()
    with pytest.raises(ContractError):
        remote.read(3)
    with pytest.raises(ContractError):
        remote.read(ARCHIVE_BYTES)
    remote.close()
