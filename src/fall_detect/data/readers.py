"""Bounded readers. Source IDs and synchronization timestamps never use row position."""

import csv
import re
from pathlib import Path

import cv2
import numpy as np

from ..contracts import FrameRecord
from ..io_utils import ContractError, fingerprint, sha256


def numeric_frame_id(path):
    match = re.search(r"(\d+)$", Path(path).stem)
    if not match:
        raise ContractError(f"Cannot determine source frame ID: {path}")
    return int(match[1])


def image_inventory(directory):
    paths = sorted(
        (p for p in Path(directory).rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}),
        key=numeric_frame_id,
    )
    if not paths:
        raise ContractError(f"No input images: {directory}")
    ids = [numeric_frame_id(p) for p in paths]
    if len(ids) != len(set(ids)):
        raise ContractError(f"Duplicate frame IDs in {directory}")
    missing = sorted(set(range(ids[0], ids[-1] + 1)) - set(ids))
    return paths, {
        "n_images": len(ids),
        "first_frame_id": ids[0],
        "last_frame_id": ids[-1],
        "missing_frame_ids": missing,
        "numeric_sort": True,
    }


def read_timestamps(path):
    result = {}
    previous_id, previous_time = -1, -1.0
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), 1):
            if not row:
                continue
            try:
                frame_float, timestamp = float(row[0]), float(row[1])
                frame_id = int(frame_float)
            except (ValueError, IndexError) as exc:
                raise ContractError(f"Invalid timestamp row {line_number} in {path}") from exc
            if (
                frame_id != frame_float
                or frame_id in result
                or frame_id <= previous_id
                or timestamp < 0
                or not np.isfinite(timestamp)
                or timestamp <= previous_time
            ):
                raise ContractError(
                    f"Duplicate/nonmonotonic frame or timestamp at {path}:{line_number}"
                )
            result[frame_id] = timestamp
            previous_id, previous_time = frame_id, timestamp
    if not result:
        raise ContractError(f"Empty synchronization data: {path}")
    return result


def source_identity(source, sync_path=None):
    source = Path(source)
    paths = image_inventory(source)[0] if source.is_dir() else [source]
    files = [
        {"path": str(p.relative_to(source)) if source.is_dir() else p.name, "sha256": sha256(p)}
        for p in paths
    ]
    if sync_path and Path(sync_path).exists():
        files.append({"path": Path(sync_path).name, "sha256": sha256(sync_path)})
    return {"files": files, "source_hash": fingerprint(files)}


def read_images(
    directory,
    sequence_id=None,
    sync_path=None,
    fallback_fps=30,
    limit=None,
    allow_missing_timestamp_tail=False,
):
    paths, _ = image_inventory(directory)
    sid = sequence_id or Path(directory).name
    times = read_timestamps(sync_path) if sync_path and Path(sync_path).exists() else None
    if fallback_fps <= 0:
        raise ContractError("fallback_fps must be positive")
    by_id = {numeric_frame_id(path): path for path in paths}
    first = min(min(by_id), min(times)) if times else min(by_id)
    last = max(max(by_id), max(times)) if times else max(by_id)
    first_image = cv2.imdecode(np.fromfile(paths[0], dtype=np.uint8), cv2.IMREAD_COLOR)
    if first_image is None:
        raise ContractError(f"Undecodable image: {paths[0]}")
    height, width = first_image.shape[:2]
    size = (width, height)
    previous_time = -1.0
    for index, frame_id in enumerate(range(first, last + 1)):
        if limit is not None and index >= limit:
            break
        path = by_id.get(frame_id)
        if (
            times is not None
            and frame_id not in times
            and path is not None
            and not (allow_missing_timestamp_tail and frame_id > max(times))
        ):
            raise ContractError(f"RGB frame {frame_id} has no matching synchronization frame ID")
        image = None
        if path is not None:
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ContractError(f"Undecodable image: {path}")
            if size != (image.shape[1], image.shape[0]):
                raise ContractError("Changing frame dimensions within sequence")
        estimated = times is None or frame_id not in times
        timestamp = (
            times[frame_id]
            if not estimated
            else (
                (frame_id - first) * 1000 / fallback_fps
                if times is None
                else max(0, previous_time + 1000 / fallback_fps)
            )
        )
        if timestamp <= previous_time:
            raise ContractError(
                "Missing-frame fallback conflicts with source time; supply explicit timestamps"
            )
        previous_time = timestamp
        yield FrameRecord(
            sid,
            frame_id,
            index,
            timestamp,
            image,
            width,
            height,
            timestamp_estimated=estimated,
            source_missing=path is None,
        )


def read_video(path, sequence_id=None, fallback_fps=30, limit=None):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ContractError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_estimated = not np.isfinite(fps) or fps <= 0
    fps = fallback_fps if fps_estimated else fps
    declared = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    declared = int(declared) if np.isfinite(declared) and declared > 0 else None
    previous, index = -1.0, 0
    try:
        while limit is None or index < limit:
            ok, image = cap.read()
            if not ok:
                if declared is not None and index < declared:
                    raise ContractError(
                        f"Video ended before declared frame count: {index}/{declared}: {path}"
                    )
                break
            source_time = cap.get(cv2.CAP_PROP_POS_MSEC)
            estimated = not np.isfinite(source_time) or source_time < 0 or source_time <= previous
            timestamp = index * 1000 / fps if estimated else float(source_time)
            if timestamp <= previous:
                raise ContractError("Video timestamp regression; cannot preserve causal timing")
            height, width = image.shape[:2]
            yield FrameRecord(
                sequence_id or Path(path).stem,
                index,
                index,
                timestamp,
                image,
                width,
                height,
                timestamp_estimated=estimated,
                source_fps=float(fps),
                source_fps_estimated=fps_estimated,
            )
            previous, index = timestamp, index + 1
    finally:
        cap.release()
    if index == 0:
        raise ContractError(f"No decoded frames: {path}")


def read_source(source, **kwargs):
    if Path(source).is_dir():
        yield from read_images(source, **kwargs)
    else:
        kwargs.pop("sync_path", None)
        kwargs.pop("allow_missing_timestamp_tail", None)
        yield from read_video(source, **kwargs)


def validate_raw_label(value):
    """Preserve official posture labels; none of these is an event-onset annotation."""
    if value not in (-1, 0, 1):
        raise ContractError("Official posture label must be -1, 0 or 1")
    return {"raw_label": value, "posture": {-1: "not_lying", 0: "transition", 1: "lying"}[value]}
