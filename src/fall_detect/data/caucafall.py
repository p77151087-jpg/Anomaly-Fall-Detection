"""Official CAUCAFall v4 videos, with subject-disjoint normal-only expansion."""

import io
import re
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..config import ROOT, project_path
from ..io_utils import ContractError, atomic_bytes, fingerprint, read_json, sha256, write_json

PAGE = "https://data.mendeley.com/datasets/7w7fccy7ky/4"
ARCHIVE_URL = "https://data.mendeley.com/public-api/zip/7w7fccy7ky/download/4"
ARCHIVE_BYTES = 8_326_349_365
ARCHIVE_PUBLISHED_SHA256 = "f8cab337dfae0d0ff210cb94e12cec320cb6be867667d6e29c8b27ccc7ea76ee"
ARCHIVE_ETAG = '"3508e08a1a37df3b9de1849ceadc07eb-398"'
BASE_SPLIT_HASH = "a6454c4389e4a8ef7b161920088dd290489dd9e361b24b1afab56507bf11e4b8"
PREFIX = "Dataset CAUCAFall/CAUCAFall/"
ACTIVITIES = {
    "FallBackwards": "Fall backwards",
    "FallForward": "Fall forward",
    "FallLeft": "Fall left",
    "FallRight": "Fall right",
    "FallSitting": "Fall sitting",
    "Hop": "Hop",
    "Kneel": "Kneel",
    "Pickupobject": "Pick up object",
    "SitDown": "Sit down",
    "Walk": "Walk",
}


def subject_assignments():
    subjects = np.random.default_rng(42).permutation(np.arange(1, 11)).tolist()
    return {
        subject: split
        for split, group in (
            ("train", subjects[:6]),
            ("validation", subjects[6:8]),
            ("test", subjects[8:]),
        )
        for subject in group
    }


def video_identity(member):
    match = re.fullmatch(
        re.escape(PREFIX) + r"Subject\.(\d+)/([^/]+)/([A-Za-z]+)S(\d+)\.avi", member
    )
    if not match:
        raise ContractError(f"Unexpected CAUCAFall original video path: {member}")
    subject, folder, activity, suffix_subject = match.groups()
    subject = int(subject)
    if (
        subject not in range(1, 11)
        or subject != int(suffix_subject)
        or ACTIVITIES.get(activity) != folder
    ):
        raise ContractError("CAUCAFall subject/activity identity mismatch")
    return subject, activity, int(activity.startswith("Fall"))


class ArchiveRanges(io.RawIOBase):
    """Read ZIP members over HTTPS ranges; never silently download the 8.33 GB archive."""

    def __init__(self):
        self.position = 0
        self.http = requests.Session()
        retry = Retry(total=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504))
        self.http.mount("https://", HTTPAdapter(max_retries=retry))
        self.refresh_url()

    def refresh_url(self):
        with self.http.get(ARCHIVE_URL, stream=True, timeout=(20, 30)) as response:
            response.raise_for_status()
            if (
                int(response.headers.get("Content-Length", 0)) != ARCHIVE_BYTES
                or response.headers.get("ETag") != ARCHIVE_ETAG
            ):
                raise ContractError("Official CAUCAFall archive identity changed")
            self.url = response.url  # Temporary signed public URL; never persist credentials.

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = (
            offset
            if whence == 0
            else self.position + offset
            if whence == 1
            else ARCHIVE_BYTES + offset
        )
        if position < 0 or position > ARCHIVE_BYTES or whence not in (0, 1, 2):
            raise ContractError("Invalid archive range")
        self.position = position
        return position

    def read(self, size=-1):
        size = (
            ARCHIVE_BYTES - self.position if size < 0 else min(size, ARCHIVE_BYTES - self.position)
        )
        if not size:
            return b""
        if size > 16 * 1024 * 1024:
            raise ContractError("Unexpectedly large ZIP member/range")
        expected_range = f"bytes {self.position}-{self.position + size - 1}/{ARCHIVE_BYTES}"
        for attempt in range(2):
            with self.http.get(
                self.url,
                headers={
                    "Range": f"bytes={self.position}-{self.position + size - 1}",
                    "If-Match": ARCHIVE_ETAG,
                },
                stream=True,
                timeout=(20, 60),
            ) as response:
                if response.status_code == 403 and attempt == 0:
                    self.refresh_url()
                    continue
                response.raise_for_status()
                if (
                    response.status_code != 206
                    or response.headers.get("Content-Range") != expected_range
                    or response.headers.get("ETag") != ARCHIVE_ETAG
                ):
                    raise ContractError("Server did not honor the pinned archive range")
                payload = response.content
                if len(payload) != size:
                    raise ContractError("Incomplete archive range")
                self.position += size
                return payload
        raise ContractError("Cannot read public archive")

    def close(self):
        self.http.close()
        super().close()


def fetch_videos(destination):
    """Preserve exact original AVI bytes; ZIP CRC verified and actual SHA256 retained."""
    destination = Path(destination)
    with ArchiveRanges() as remote, zipfile.ZipFile(remote) as archive:
        videos = [info for info in archive.infolist() if info.filename.lower().endswith(".avi")]
    identities = [video_identity(info.filename)[:2] for info in videos]
    expected = {(subject, activity) for subject in range(1, 11) for activity in ACTIVITIES}
    if len(videos) != 100 or set(identities) != expected:
        raise ContractError("Official archive must contain exactly 10 subjects x 10 activities")

    def fetch_group(group):
        rows = []
        with ArchiveRanges() as remote, zipfile.ZipFile(remote) as archive:
            for info in group:
                subject, activity, label = video_identity(info.filename)
                relative = f"Subject.{subject}/{activity}S{subject}.avi"
                path = destination / relative
                if path.exists():
                    content = path.read_bytes()
                    if len(content) != info.file_size or zlib.crc32(content) != info.CRC:
                        raise ContractError(f"Existing original video changed: {relative}")
                else:
                    atomic_bytes(path, archive.read(info))
                rows.append(
                    {
                        "sequence_id": f"caucafall-S{subject:02d}-{activity}",
                        "subject": subject,
                        "activity": activity,
                        "label": label,
                        "archive_member": info.filename,
                        "crc32": info.CRC,
                        "source_path": path.resolve().relative_to(ROOT).as_posix(),
                        "source_sha256": sha256(path),
                        "size_bytes": info.file_size,
                    }
                )
                print(f"Verified original CAUCAFall: {relative}", flush=True)
        return rows

    with ThreadPoolExecutor(max_workers=4) as pool:
        groups = list(pool.map(fetch_group, [videos[i::4] for i in range(4)]))
    rows = sorted([row for group in groups for row in group], key=lambda row: row["sequence_id"])
    if len({row["source_sha256"] for row in rows}) != 100:
        raise ContractError("Duplicate original CAUCAFall video content")
    return rows


def build_manifest(base, inventory):
    from .splits import CAUCA_PROTOCOL, validate_split

    validate_split(base)
    if base["split_hash"] != BASE_SPLIT_HASH:
        raise ContractError("CAUCAFall extension requires the frozen 161-sequence baseline")
    assignments = subject_assignments()
    entries, excluded = list(base["sequences"]), []
    for row in inventory:
        split = assignments[row["subject"]]
        entry = {**row, "split": split, "dataset": "caucafall", "camera": 0, "modality": "rgb"}
        if split == "train" and row["label"]:
            excluded.append({**entry, "reason": "whole_fall_video_excluded_from_train"})
        else:
            entries.append(entry)
    manifest = {
        "schema_version": 1,
        "protocol_id": CAUCA_PROTOCOL,
        "seed": 42,
        "base_manifest": base,
        "base_split_hash": base["split_hash"],
        "subject_independent": False,
        "caucafall_subject_independent": True,
        "caucafall_subject_assignments": {str(k): v for k, v in assignments.items()},
        "label_rule": "Official whole activity video: Fall* is Fall; five other activities are ADL. No clips from Fall videos in Train.",
        "excluded": excluded,
        "sequences": entries,
    }
    manifest["split_hash"] = fingerprint(manifest)
    return validate_split(manifest)


def fetch_caucafall(config):
    if config["data"]["dataset"] != "urfall_le2i_caucafall":
        raise ContractError("Use the separate CAUCAFall extension config")
    base = read_json(project_path("data/manifests/le2i-extension/split-v1.json"))
    inventory = fetch_videos(project_path("data/raw/caucafall/original"))
    manifest = build_manifest(base, inventory)
    directory = project_path(config["data"]["manifest_dir"])
    target = directory / "split-v1.json"
    if target.exists() and read_json(target) != manifest:
        raise ContractError("Refusing to overwrite a different CAUCAFall split")
    report = {
        "source_page": PAGE,
        "doi": "10.17632/7w7fccy7ky.4",
        "license": "CC BY 4.0",
        "authors": ["Jose Camilo Eraso", "Elena Munoz", "Mariela Munoz", "Jesus Pinto"],
        "archive_url": ARCHIVE_URL,
        "archive_bytes": ARCHIVE_BYTES,
        "archive_published_sha256": ARCHIVE_PUBLISHED_SHA256,
        "archive_full_sha256_verified": False,
        "verification": "HTTPS byte ranges from official v4 archive; pinned size/ETag; each original AVI ZIP CRC verified; actual per-video SHA256 saved. Entire archive not downloaded or hashed.",
        "video_bytes": sum(row["size_bytes"] for row in inventory),
        "videos": inventory,
        "n_downloaded": len(inventory),
        "n_selected": 70,
        "n_excluded_train_fall": 30,
        "errors": [],
    }
    write_json(directory / "caucafall-sources.json", report)
    write_json(target, manifest)
    return report
