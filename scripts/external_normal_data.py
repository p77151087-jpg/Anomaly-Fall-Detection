"""Acquire a fixed, provenance-bound normal-only RGB subset; never access Test."""

import argparse
import hashlib
import re
import struct
import zipfile
import zlib
from pathlib import Path

import cv2
import requests

from fall_detect.io_utils import ContractError, read_json, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/external-normal-20261012"
LOCK = ROOT / "docs/validation/external-normal-20261012-data-lock.json"
PROFILE = ROOT / "docs/validation/external-normal-20261012-data.json"


def save_new(path, obj):
    if path.exists():
        raise ContractError(f"Preserve existing evidence: {path}")
    write_json(path, obj)


def verify_frozen_source(path, expected):
    path = Path(path)
    if sha256(path) == expected:
        return
    repair = read_json(ROOT / "docs/validation/external-normal-20261012-source-repair.json")
    entry = repair["files"].get(path.name, {})
    if (
        entry.get("from_sha256") != expected
        or entry.get("to_sha256") != sha256(path)
        or sha256(ROOT / entry["archived_source"]) != expected
    ):
        raise ContractError(
            "Frozen implementation changed without an exact source archive/repair binding"
        )


def safe_member(name):
    p = Path(name)
    if p.is_absolute() or name.startswith("/") or ".." in p.parts or ":" in name or "\\" in name:
        raise ContractError("Unsafe archive member")
    return p


def select_members(rows):
    """First eight subjects with all three activities in the public inventory."""
    pattern = re.compile(r"VideoDataset/ADL/SBJ_(0[1-9])_LOC(\d+)/ACT(13|18|24)_F_1/[^/]+\.mp4")
    matches = [r for r in rows if pattern.fullmatch(r["name"])]
    selected = []
    for subject in (1, 2, 3, 4, 6, 7, 8, 9):
        for activity in (13, 18, 24):
            group = [
                r
                for r in matches
                if pattern.fullmatch(r["name"]).group(1) == f"{subject:02d}"
                and int(pattern.fullmatch(r["name"]).group(3)) == activity
            ]
            if not group:
                raise ContractError(f"Missing preregistered subject/activity: {subject}/{activity}")
            selected.append(min(group, key=lambda r: r["name"]))
    return selected


def freeze():
    inv = read_json(RAW / "video-inventory-full.json")
    selected = select_members(inv["members"])
    save_new(
        LOCK,
        {
            "schema_version": 1,
            "script_sha256": sha256(__file__),
            "source": "https://figshare.com/articles/dataset/28596332/2",
            "uma_source": "https://figshare.com/articles/dataset/4214283/8",
            "license": "CC BY 4.0",
            "source_label": "ADL directory / official videos_ADL archive",
            "selection": "subjects 01,02,03,04,06,07,08,09 (05 lacks ACT18/24 in inventory), ACT13/18/24, F view, trial 1, lexicographically first location; all 8 UMA demonstrations",
            "selection_uses_ur_validation_scores": False,
            "selection_uses_test": False,
            "new_sources_assignment": "external_train_only; not a new independent evaluation split",
            "members": selected,
            "file": inv["file"],
            "inventory_sha256": sha256(RAW / "video-inventory-full.json"),
            "official_metadata_sha256": {
                str(a): sha256(RAW / f"figshare-{a}-metadata.json") for a in (28596332, 4214283)
            },
            "uma_archive_sha256": sha256(RAW / "videos_ADL.zip"),
            "whole_video_archive_downloaded": False,
            "whole_video_archive_published_md5_not_verified": True,
            "download_integrity": "individual ZIP central-directory sizes + CRC32, then local SHA256; TLS verified",
            "model_validation_not_started": True,
        },
    )
    print({"selected_video_dataset": len(selected), "uma": 8}, flush=True)


def remote_member(session, file, row):
    safe_member(row["name"])
    # Include the filename/extra header before the compressed payload.
    start = row["offset"]
    response = session.get(
        file["download_url"],
        headers={
            "Range": f"bytes={start}-{start + 30 + len(row['name'].encode('utf-8')) + 65535 + row['compressed'] - 1}"
        },
        timeout=(15, 60),
    )
    response.raise_for_status()
    if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(
        f"bytes {start}-"
    ):
        raise ContractError("Server did not honor the requested source range")
    b = response.content
    h = struct.unpack_from("<4s5H3I2H", b)
    if h[0] != b"PK\x03\x04" or h[3] != row["method"] or h[2] & 1:
        raise ContractError("ZIP header/method/encryption contract")
    name = b[30 : 30 + h[9]].decode("utf-8" if h[2] & 2048 else "cp437")
    if name != row["name"]:
        raise ContractError("ZIP member identity mismatch")
    offset = 30 + h[9] + h[10]
    payload = b[offset : offset + row["compressed"]]
    data = zlib.decompress(payload, -15) if row["method"] == 8 else payload
    if len(data) != row["size"] or (zlib.crc32(data) & 0xFFFFFFFF) != row["crc"]:
        raise ContractError("ZIP member size/CRC mismatch")
    return data


def acquire():
    if PROFILE.exists():
        raise ContractError("Acquisition already complete; use status")
    lock = read_json(LOCK)
    verify_frozen_source(__file__, lock["script_sha256"])
    rows = []
    with requests.Session() as session:
        for r in lock["members"]:
            dest = RAW / safe_member(r["name"])
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                dest.write_bytes(remote_member(session, lock["file"], r))
            data = dest.read_bytes()
            if len(data) != r["size"] or (zlib.crc32(data) & 0xFFFFFFFF) != r["crc"]:
                raise ContractError("Existing downloaded member failed integrity")
            match = re.search(r"SBJ_(\d+)_LOC(\d+)/ACT(\d+)_F_1/", r["name"])
            sid = f"ursul-S{int(match[1]):02d}-A{match[3]}"
            rows.append(
                {
                    "sequence_id": sid,
                    "dataset": "ursul_video_v2",
                    "subject": int(match[1]),
                    "activity_code": int(match[3]),
                    "location": int(match[2]),
                    "split": "external_train",
                    "label": 0,
                    "archive_member": r["name"],
                    "source_path": dest.relative_to(ROOT).as_posix(),
                    "source_sha256": sha256(dest),
                    "source_label": "ADL",
                    "label_confidence": "publisher label, sampled visual verification pending",
                }
            )
            print(f"Acquired {sid}: {len(data)} bytes", flush=True)
    with zipfile.ZipFile(RAW / "videos_ADL.zip") as z:
        expected_md5 = read_json(RAW / "figshare-4214283-metadata.json")["files"][0]["computed_md5"]
        if hashlib.md5((RAW / "videos_ADL.zip").read_bytes()).hexdigest() != expected_md5:
            raise ContractError("UMAFall published archive MD5 mismatch")
        for i, name in enumerate(sorted(z.namelist()), 1):
            if not name.endswith(".mp4"):
                raise ContractError("Unexpected UMA ADL archive content")
            dest = RAW / "uma" / safe_member(name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            data = z.read(name)
            if not dest.exists():
                dest.write_bytes(data)
            if dest.read_bytes() != data:
                raise ContractError("Preserved UMA video changed")
            rows.append(
                {
                    "sequence_id": f"uma-adl-{i:02d}",
                    "dataset": "umafall_v8",
                    "subject": None,
                    "split": "external_train",
                    "label": 0,
                    "archive_member": name,
                    "source_path": dest.relative_to(ROOT).as_posix(),
                    "source_sha256": sha256(dest),
                    "source_label": "official videos_ADL",
                    "label_confidence": "publisher ADL demonstration",
                }
            )
    hashes = [r["source_sha256"] for r in rows]
    if len(hashes) != len(set(hashes)):
        raise ContractError("Duplicate acquired RGB bytes")
    for r in rows:
        cap = cv2.VideoCapture(str(ROOT / r["source_path"]))
        if not cap.isOpened():
            raise ContractError("Cannot decode acquired RGB")
        r.update(
            fps=cap.get(cv2.CAP_PROP_FPS),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            declared_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        cap.release()
    save_new(
        PROFILE,
        {
            "schema_version": 1,
            "source_kind": "real",
            "lock_sha256": sha256(LOCK),
            "sequences": rows,
            "test_reads": 0,
            "not_independent_evaluation": True,
        },
    )
    print(
        {"acquired": len(rows), "declared_frames": sum(r["declared_frames"] for r in rows)},
        flush=True,
    )


def status():
    lock = read_json(LOCK)
    verify_frozen_source(__file__, lock["script_sha256"])
    data = read_json(PROFILE)
    for row in data["sequences"]:
        if (
            row["label"] != 0
            or row["split"] != "external_train"
            or sha256(ROOT / row["source_path"]) != row["source_sha256"]
        ):
            raise ContractError("External Train source changed")
    print({"acquired_normal_rgb": len(data["sequences"]), "test_reads": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "acquire", "status"))
    globals()[parser.parse_args().stage]()
