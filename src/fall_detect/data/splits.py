from pathlib import Path

import numpy as np

from ..io_utils import ContractError, fingerprint, read_json, write_json

PROTOCOL = "urfall-cam0-rgb-sequence-v1"
EXTENDED_PROTOCOL = "urfall-plus-le2i-cs-sequence-v1"
CAUCA_PROTOCOL = "urfall-le2i-caucafall-subject-sequence-v1"


def build_split():
    rng = np.random.default_rng(42)
    adl = rng.permutation([f"adl-{i:02d}" for i in range(1, 41)]).tolist()
    fall = rng.permutation([f"fall-{i:02d}" for i in range(1, 31)]).tolist()
    groups = {"train": adl[:24], "validation": adl[24:32] + fall[:15], "test": adl[32:] + fall[15:]}
    entries = [
        {
            "sequence_id": sid,
            "split": split,
            "label": int(sid.startswith("fall-")),
            "camera": 0,
            "modality": "rgb",
        }
        for split, ids in groups.items()
        for sid in ids
    ]
    result = {
        "schema_version": 1,
        "protocol_id": PROTOCOL,
        "seed": 42,
        "numpy_version": np.__version__,
        "subject_independent": False,
        "sequences": entries,
    }
    result["split_hash"] = fingerprint(result)
    return result


def validate_split(manifest):
    payload = {k: v for k, v in manifest.items() if k != "split_hash"}
    if fingerprint(payload) != manifest.get("split_hash"):
        raise ContractError("Split manifest hash mismatch")
    entries = manifest["sequences"]
    ids = [s["sequence_id"] for s in entries]
    if manifest.get("protocol_id") == PROTOCOL:
        expected = {f"adl-{i:02d}" for i in range(1, 41)} | {f"fall-{i:02d}" for i in range(1, 31)}
        if len(ids) != 70 or set(ids) != expected:
            raise ContractError("v1 split requires all 70 original sequence IDs")
        expected_counts = {
            ("train", 0): 24,
            ("train", 1): 0,
            ("validation", 0): 8,
            ("validation", 1): 15,
            ("test", 0): 8,
            ("test", 1): 15,
        }
        for (split, label), count in expected_counts.items():
            if sum(s["split"] == split and s["label"] == label for s in entries) != count:
                raise ContractError("Split counts violate v1")
        for row in entries:
            if row["label"] != int(row["sequence_id"].startswith("fall-")):
                raise ContractError("Sequence label must follow source class")
    if len(ids) != len(set(ids)) or any(s["split"] == "train" and s["label"] != 0 for s in entries):
        raise ContractError("Split overlap or non-normal train data")
    if any(
        s["split"] not in {"train", "validation", "test"} or s["label"] not in (0, 1)
        for s in entries
    ):
        raise ContractError("Invalid sequence split or label")
    if manifest.get("protocol_id") == EXTENDED_PROTOCOL:
        validate_extended_split(manifest)
    if manifest.get("protocol_id") == CAUCA_PROTOCOL:
        validate_caucafall_split(manifest)
    return manifest


def validate_extended_split(manifest):
    """Retain UR Fall assignments and disjoint Le2i original-video/subject groups."""
    baseline = build_split()
    original = {r["sequence_id"]: r for r in baseline["sequences"]}
    urfall = [r for r in manifest["sequences"] if r.get("dataset") == "urfall"]
    if len(urfall) != 70 or {r["sequence_id"] for r in urfall} != set(original):
        raise ContractError("Extended split must retain all 70 UR Fall sequences")
    for row in urfall:
        expected = original[row["sequence_id"]]
        if any(row[k] != expected[k] for k in ("split", "label", "camera", "modality")):
            raise ContractError("Extended split changed a frozen UR Fall assignment")
        if row.get("source_path") != f"data/raw/urfall/{row['sequence_id']}":
            raise ContractError("Unexpected UR Fall source path")
    seen_paths, subjects, seen_hashes = {}, {}, set()
    le2i = [r for r in manifest["sequences"] if r.get("dataset") == "le2i"]
    if not le2i or len(le2i) + len(urfall) != len(manifest["sequences"]):
        raise ContractError("Extended split needs Le2i, and accepts no unknown dataset")
    for row in le2i:
        from ..io_utils import contained_path

        source = row.get("source_path", "")
        if not source.startswith("data/raw/le2i/") or not row["sequence_id"].startswith("le2i-"):
            raise ContractError("Unexpected Le2i path or sequence namespace")
        contained_path(Path("data/raw/le2i"), source.removeprefix("data/raw/le2i/"))
        if source in seen_paths:
            raise ContractError("Repeated original Le2i video")
        seen_paths[source] = row["split"]
        import re

        digest = row.get("source_sha256", "")
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or digest in seen_hashes:
            raise ContractError("Missing or repeated Le2i source hash")
        seen_hashes.add(digest)
        subject = row.get("subject")
        if type(subject) is not int or subject < 0:
            raise ContractError("Le2i subject identity is required")
        if subject in subjects and subjects[subject] != row["split"]:
            raise ContractError("Le2i subject crosses splits")
        subjects[subject] = row["split"]
        labels = row.get("activity_labels", [])
        if not labels or any(type(x) is not int or x not in range(10) for x in labels):
            raise ContractError("Le2i activity labels required")
        if row.get("contains_fall") != bool(row["label"]) or bool(row["label"]) != any(
            x in (1, 2) for x in labels
        ):
            raise ContractError("Le2i normal-only label contract mismatch")
        intervals = row.get("annotation_intervals_ms")
        if not isinstance(intervals, list) or not intervals:
            raise ContractError("Le2i annotated intervals required")
        last = -1
        for interval in intervals:
            import math

            if len(interval) != 2 or any(
                not isinstance(v, (int, float)) or not math.isfinite(v) for v in interval
            ):
                raise ContractError("Invalid annotation interval")
            start, end = interval
            if start < 0 or start <= last or end <= start:
                raise ContractError("Annotation intervals must be disjoint and increasing")
            last = end


def validate_caucafall_split(manifest):
    from .caucafall import ACTIVITIES, BASE_SPLIT_HASH, subject_assignments, video_identity

    base = manifest.get("base_manifest", {})
    if base.get("protocol_id") != EXTENDED_PROTOCOL:
        raise ContractError("CAUCAFall requires the existing Le2i extension baseline")
    validate_split(base)
    if base["split_hash"] != BASE_SPLIT_HASH or manifest.get("base_split_hash") != BASE_SPLIT_HASH:
        raise ContractError("Frozen baseline split changed")
    retained = [r for r in manifest["sequences"] if r.get("dataset") != "caucafall"]
    if retained != base["sequences"]:
        raise ContractError("CAUCAFall extension changed existing sequence assignments")
    selected = [r for r in manifest["sequences"] if r.get("dataset") == "caucafall"]
    excluded = manifest.get("excluded", [])
    if len(selected) != 70 or len(excluded) != 30:
        raise ContractError("CAUCAFall requires 70 selected and 30 excluded Train Fall videos")
    assignments = subject_assignments()
    if manifest.get("caucafall_subject_assignments") != {str(k): v for k, v in assignments.items()}:
        raise ContractError("CAUCAFall subject split changed")
    identities, paths = set(), set()
    digests = {row["source_sha256"] for row in base["sequences"] if "source_sha256" in row}
    import re

    for row in selected + excluded:
        subject, activity, label = video_identity(row.get("archive_member", ""))
        expected_path = f"data/raw/caucafall/original/Subject.{subject}/{activity}S{subject}.avi"
        if (
            row.get("subject") != subject
            or row.get("activity") != activity
            or row.get("label") != label
            or row.get("split") != assignments[subject]
            or row.get("sequence_id") != f"caucafall-S{subject:02d}-{activity}"
            or row.get("source_path") != expected_path
            or row.get("dataset") != "caucafall"
            or row.get("camera") != 0
            or row.get("modality") != "rgb"
        ):
            raise ContractError("CAUCAFall source/label/subject assignment mismatch")
        digest = row.get("source_sha256", "")
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or digest in digests or expected_path in paths:
            raise ContractError("Missing or duplicate CAUCAFall video hash/path")
        digests.add(digest)
        paths.add(expected_path)
        identities.add((subject, activity))
        if row in excluded and (row["split"] != "train" or label != 1):
            raise ContractError("Only whole Train Fall videos may be excluded")
    if identities != {(s, a) for s in range(1, 11) for a in ACTIVITIES}:
        raise ContractError("Incomplete CAUCAFall activity inventory")


def dataset_split(config):
    from ..config import project_path

    directory = project_path(config["data"]["manifest_dir"])
    if config["data"]["dataset"] == "urfall":
        return ensure_split(directory)
    path = directory / "split-v1.json"
    if not path.exists():
        raise ContractError(
            "Extension manifest missing; run fetch-assets with the matching dataset/config"
        )
    manifest = validate_split(read_json(path))
    if manifest["protocol_id"] != config["data"]["protocol_id"]:
        raise ContractError("Expected the configured extension protocol")
    return manifest


def ensure_split(directory):
    path = Path(directory) / "split-v1.json"
    if path.exists():
        return validate_split(read_json(path))
    result = build_split()
    write_json(path, result)
    return result


def smoke_ids(manifest, limit=1):
    if limit < 1:
        raise ContractError("limit-per-class must be positive")
    train = sorted(s["sequence_id"] for s in manifest["sequences"] if s["split"] == "train")
    falls = sorted(
        s["sequence_id"]
        for s in manifest["sequences"]
        if s["split"] == "validation" and s["label"] == 1
    )
    return train[:limit] + falls[:limit]
