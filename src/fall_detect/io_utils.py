"""Strict JSON, content fingerprints and atomic artifact writers."""

import csv
import hashlib
import io
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np


class ContractError(ValueError):
    """An input violates the data or experiment contract."""


def canonical_json(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )


def fingerprint(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".part", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        # Windows readers/antivirus can briefly deny replacement of an open file.
        # Keep the old complete file visible and retry only these sharing/access errors.
        for attempt in range(9):
            try:
                os.replace(temporary, path)
                break
            except OSError as exc:
                if getattr(exc, "winerror", None) not in (5, 32, 33) or attempt == 8:
                    raise
                time.sleep(min(0.01 * 2**attempt, 0.25))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    atomic_bytes(
        path,
        (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"),
    )


def read_json(path):
    def reject(value):
        raise ContractError(f"Non-finite JSON value: {value}")

    return json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=reject)


def write_npz(path, **arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    atomic_bytes(path, buffer.getvalue())


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else [])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, buffer.getvalue().encode("utf-8-sig"))


def contained_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ContractError(f"Path escapes destination: {relative}")
    return path
