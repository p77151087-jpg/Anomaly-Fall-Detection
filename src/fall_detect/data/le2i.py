"""Le2i original videos and pinned OmniFall annotations; no generated examples."""

import re
import time
from pathlib import Path

from ..io_utils import ContractError

LE2I_PAGE = (
    "https://search-data.ubfc.fr/imvia/FR-13002091000019-2024-04-09_Fall-Detection-Dataset.html"
)
LE2I_URL = "https://search-data.ubfc.fr/imvia/dl_data.php?file=101"
ARCHIVE_BYTES = 9_608_701_260
ANNOTATION_REVISION = "83572a37b9e3081df8c06a56874b1d1f2a19386c"
ANNOTATION_REPO = "https://huggingface.co/datasets/simplexsigil2/omnifall"
ANNOTATION_HASHES = {
    "labels.csv": "bb144196f7d384d2b5153022d73ea349132f428229986ce8183902a25e213132",
    "train.csv": "b03d4d0aeb84dbe9e4302ae46a7c28840bba567731bff632b9d439852e8f3ffe",
    "validation.csv": "91cfcfedd9c1d2a7216048d8dd6ff908c861fd24fdee4621d8222bb6594f8660",
    "test.csv": "b7f5ca62ba341dac8899c27b62ab7171dd05762a7e7151150bdde60a64332e04",
}


def annotated_intervals(segments):
    """Union of known intervals; gaps remain unknown even in an ADL-labelled video."""
    merged = []
    for segment in sorted(segments, key=lambda row: row["start"]):
        start, end = segment["start"] * 1000, segment["end"] * 1000
        if merged and start <= merged[-1][1] + 1e-7:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def annotation_groups(directory):
    import csv
    import math
    from collections import defaultdict

    directory = Path(directory)
    groups = defaultdict(list)
    for row in csv.DictReader((directory / "labels.csv").open(encoding="utf-8")):
        name = row["path"]
        if not re.fullmatch(r"[A-Za-z0-9_]+/video_\d+", name) or row["dataset"] != "le2i":
            raise ContractError("Unexpected annotation source path/dataset")
        value = {
            "label": int(row["label"]),
            "start": float(row["start"]),
            "end": float(row["end"]),
            "subject": int(row["subject"]),
            "camera": int(row["cam"]),
        }
        if (
            value["label"] not in range(10)
            or not all(math.isfinite(value[k]) for k in ("start", "end"))
            or not 0 <= value["start"] < value["end"]
        ):
            raise ContractError("Invalid Le2i interval or label")
        groups[name].append(value)
    assignments, subjects = {}, {}
    for split in ("train", "validation", "test"):
        for row in csv.DictReader((directory / f"{split}.csv").open(encoding="utf-8")):
            name = row["path"]
            if name not in groups or name in assignments:
                raise ContractError("Missing labels or original video crosses splits")
            identities = {r["subject"] for r in groups[name]}
            cameras = {r["camera"] for r in groups[name]}
            if len(identities) != 1 or len(cameras) != 1:
                raise ContractError("Inconsistent Le2i subject/camera identity")
            subject = next(iter(identities))
            if subject < 0 or subject in subjects and subjects[subject] != split:
                raise ContractError("Le2i subject crosses splits")
            subjects[subject] = split
            assignments[name] = split
    if set(groups) != set(assignments):
        raise ContractError("Every original Le2i video must have exactly one split")
    return dict(groups), assignments


def find_videos(directory):
    """Map original AVI names without transcoding or altering the source timing."""
    result = {}
    for path in Path(directory).rglob("*.avi"):
        match = re.fullmatch(r"video[ _]*(?:\((\d+)\)|(\d+))", path.stem, re.IGNORECASE)
        scenes = [
            part.replace(" ", "_")
            for part in path.parts
            if re.fullmatch(r"Coffee_room_0[12]|Home_0[12]|Office|Lecture[ _]room", part)
        ]
        if not match or not scenes:
            continue
        name = f"{scenes[-1]}/video_{int(match[1] or match[2])}"
        if name in result:
            raise ContractError(f"Duplicate original video: {name}")
        result[name] = path
    return result


def build_extended_manifest(annotation_dir, videos):
    from ..config import ROOT
    from ..io_utils import fingerprint, sha256
    from .splits import EXTENDED_PROTOCOL, build_split, validate_split

    groups, assignments = annotation_groups(annotation_dir)
    if set(groups) - set(videos):
        raise ContractError(f"Missing original videos: {sorted(set(groups) - set(videos))[:5]}")
    original = build_split()
    entries = [
        {**r, "dataset": "urfall", "source_path": f"data/raw/urfall/{r['sequence_id']}"}
        for r in original["sequences"]
    ]
    from collections import defaultdict

    content_groups = defaultdict(list)
    digests = {name: sha256(videos[name]) for name in groups}
    for name, digest in digests.items():
        content_groups[digest].append(name)
    duplicates = {digest: names for digest, names in content_groups.items() if len(names) > 1}
    if any(len({assignments[name] for name in names}) > 1 for names in duplicates.values()):
        raise ContractError("Duplicate video bytes cross original splits")
    excluded = []
    for name in sorted(groups):
        segments = groups[name]
        label = int(any(r["label"] in (1, 2) for r in segments))
        split = assignments[name]
        if split == "train" and label:
            excluded.append(
                {"original_video": name, "reason": "contains_fall_or_fallen", "source_split": split}
            )
            continue
        path = videos[name]
        digest = digests[name]
        if digest in duplicates:
            excluded.append(
                {
                    "original_video": name,
                    "reason": "duplicate_source_bytes",
                    "source_split": split,
                    "source_sha256": digest,
                    "duplicate_group": duplicates[digest],
                }
            )
            continue
        entries.append(
            {
                "sequence_id": "le2i-" + name.replace("/", "-"),
                "split": split,
                "label": label,
                "dataset": "le2i",
                "modality": "rgb",
                "camera": segments[0]["camera"],
                "subject": segments[0]["subject"],
                "original_video": name,
                "source_path": path.resolve().relative_to(ROOT).as_posix(),
                "source_sha256": digest,
                "contains_fall": bool(label),
                "activity_labels": sorted({r["label"] for r in segments}),
                "annotation_end_seconds": max(r["end"] for r in segments),
                "annotation_intervals_ms": annotated_intervals(segments),
            }
        )
    result = {
        "schema_version": 1,
        "protocol_id": EXTENDED_PROTOCOL,
        "seed": 42,
        "base_split_hash": original["split_hash"],
        "subject_independent": False,
        "le2i_subject_independent": True,
        "annotation_revision": ANNOTATION_REVISION,
        "annotation_hashes": {
            p.name: sha256(p) for p in sorted(Path(annotation_dir).glob("*.csv"))
        },
        "label_rule": "whole video: any fall(1)/fallen(2) -> Fall; otherwise ADL; exclude all Fall source videos from train",
        "excluded": excluded,
        "duplicate_policy": "reject cross-split duplicates; exclude all members of within-split duplicate groups",
        "sequences": entries,
    }
    result["split_hash"] = fingerprint(result)
    return validate_split(result)


def fetch_le2i(config):
    import zipfile

    from ..assets import download, safe_extract, session
    from ..config import project_path
    from ..io_utils import read_json, sha256, write_json

    if config["data"]["dataset"] != "urfall_le2i":
        raise ContractError("Use the separate urfall_le2i extension config")
    root = project_path("data/raw/le2i")
    annotation_dir = root / "annotations"
    resources = []
    with session() as http:
        for remote, name in [
            ("labels/le2i.csv", "labels.csv"),
            ("splits/cs/le2i/train.csv", "train.csv"),
            ("splits/cs/le2i/val.csv", "validation.csv"),
            ("splits/cs/le2i/test.csv", "test.csv"),
        ]:
            url = f"{ANNOTATION_REPO}/resolve/{ANNOTATION_REVISION}/{remote}"
            path = annotation_dir / name
            download(http, url, path)
            if sha256(path) != ANNOTATION_HASHES[name]:
                raise ContractError(f"Pinned annotation hash mismatch: {name}")
            resources.append({"url": url, "sha256": sha256(path), "bytes": path.stat().st_size})
    annotation_groups(annotation_dir)
    archive = root / "FallDataset.zip"
    download_archive(archive)
    marker = root / "extracted.json"
    archive_hash = sha256(archive)
    if marker.exists() and read_json(marker)["archive_sha256"] != archive_hash:
        raise ContractError("Archive changed since its CRC-verified extraction")
    if not marker.exists():
        safe_extract(archive, root / "unpacked")
        for nested in sorted((root / "unpacked").glob("*.zip")):
            print(f"Extracting {nested.name}", flush=True)
            safe_extract(nested, root / "videos" / nested.stem)
        # Opening every nested stream through ZipFile verifies its CRC while extracting.
        with zipfile.ZipFile(archive) as handle:
            if not handle.namelist():
                raise ContractError("Empty Le2i archive")
        write_json(marker, {"archive_sha256": archive_hash})
    manifest = build_extended_manifest(annotation_dir, find_videos(root / "videos"))
    destination = project_path(config["data"]["manifest_dir"])
    manifest_path = destination / "split-v1.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ContractError("Existing extension split differs; use a new manifest directory")
    write_json(manifest_path, manifest)
    report = {
        "source_kind": "real",
        "errors": [],
        "source_page": LE2I_PAGE,
        "video_url": LE2I_URL,
        "archive_sha256": archive_hash,
        "archive_bytes": ARCHIVE_BYTES,
        "video_license": "CC BY-NC-SA 3.0",
        "annotation_repo": ANNOTATION_REPO,
        "annotation_revision": ANNOTATION_REVISION,
        "annotation_resources": resources,
        "n_planned": len(manifest["sequences"]),
        "n_excluded_train_fall": sum(
            r["reason"] == "contains_fall_or_fallen" for r in manifest["excluded"]
        ),
        "n_excluded_duplicates": sum(
            r["reason"] == "duplicate_source_bytes" for r in manifest["excluded"]
        ),
        "manifest": str(manifest_path),
    }
    write_json(destination / "le2i-sources.json", report)
    return report


def download_archive(destination):
    """Six bounded range requests; completed batches remain resumable on disk."""
    from concurrent.futures import ThreadPoolExecutor

    from ..assets import session

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != ARCHIVE_BYTES:
            raise ContractError("Le2i archive size mismatch")
        return
    partial = destination.with_suffix(".zip.part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > ARCHIVE_BYTES:
        raise ContractError("Partial archive too large")

    def fetch(bounds):
        start, end = bounds
        chunk_path = destination.parent / f"range-{start}-{end}.part"
        for attempt in range(3):
            try:
                with (
                    session() as http,
                    http.get(
                        LE2I_URL,
                        headers={"Range": f"bytes={start}-{end}"},
                        stream=True,
                        timeout=(30, 120),
                    ) as response,
                ):
                    response.raise_for_status()
                    if (
                        response.status_code != 206
                        or response.headers.get("Content-Range")
                        != f"bytes {start}-{end}/{ARCHIVE_BYTES}"
                    ):
                        raise ContractError("Server did not confirm exact archive range")
                    with chunk_path.open("wb") as handle:
                        for chunk in response.iter_content(1024 * 1024):
                            handle.write(chunk)
                if chunk_path.stat().st_size != end - start + 1:
                    raise OSError("Incomplete archive range")
                return chunk_path
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(2)

    import shutil

    size = 64 * 1024 * 1024
    with ThreadPoolExecutor(max_workers=6) as pool:
        while offset < ARCHIVE_BYTES:
            spans = [
                (start, min(start + size, ARCHIVE_BYTES) - 1)
                for start in range(offset, min(offset + size * 6, ARCHIVE_BYTES), size)
            ]
            chunks = list(pool.map(fetch, spans))
            with partial.open("ab") as handle:
                for chunk in chunks:
                    with chunk.open("rb") as source:
                        shutil.copyfileobj(source, handle)
                    offset += chunk.stat().st_size
                    chunk.unlink()
            print(f"Le2i downloaded {offset / 1e9:.2f}/{ARCHIVE_BYTES / 1e9:.2f} GB", flush=True)
    partial.replace(destination)


def download_resumable(http, url, destination, expected_bytes):
    """Resume only a server-confirmed byte range; publish after complete length."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != expected_bytes:
            raise ContractError("Existing Le2i archive has an unexpected size")
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected_bytes:
            raise ContractError("Partial archive exceeds expected size")
        if offset == expected_bytes:
            partial.replace(destination)
            return
        try:
            with http.get(
                url, headers={"Range": f"bytes={offset}-"}, stream=True, timeout=(30, 120)
            ) as response:
                response.raise_for_status()
                if response.status_code == 206:
                    match = re.fullmatch(
                        r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
                    )
                    if not match or int(match[1]) != offset or int(match[3]) != expected_bytes:
                        raise ContractError("Download range mismatch")
                elif response.status_code != 200 or offset:
                    raise ContractError("Server did not confirm resume offset")
                if "text/html" in response.headers.get("Content-Type", ""):
                    raise ContractError("HTML returned instead of archive")
                last_log = offset
                with partial.open("ab" if offset else "wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        handle.write(chunk)
                        offset += len(chunk)
                        if offset > expected_bytes:
                            raise ContractError("Download larger than expected")
                        if offset - last_log >= 256 * 1024 * 1024:
                            print(
                                f"Le2i downloaded {offset / 1e9:.2f}/{expected_bytes / 1e9:.2f} GB",
                                flush=True,
                            )
                            last_log = offset
            if partial.stat().st_size != expected_bytes:
                raise OSError("Incomplete Le2i download; resuming")
            partial.replace(destination)
            return
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2)
