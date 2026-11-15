"""Strict original RGB -> Human pose -> frozen ST-GCN preparation for new Train ADL."""

import argparse
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import numpy as np
from external_normal_data import PROFILE, ROOT, save_new, verify_frozen_source
from PIL import Image, ImageDraw

from fall_detect.backends import create_backends
from fall_detect.config import load_config, recipe_fingerprint
from fall_detect.data.cache import load_feature_cache, save_feature_cache, save_pose_cache
from fall_detect.data.readers import read_source, source_identity
from fall_detect.io_utils import ContractError, read_json, sha256, write_json
from fall_detect.models.weights import select_device
from fall_detect.pipeline import features_from_observations

CACHE = ROOT / "data/cache/external-normal-human-isotropic-20261012-001"
EVIDENCE = ROOT / "docs/validation/external-normal-20261012-prepared.json"
BEFORE = ROOT / "docs/validation/external-normal-20261012-prepare-lock.json"


def metadata_for(row, base):
    if row["label"] != 0 or row["split"] != "external_train":
        raise ContractError("Only newly assigned external Train ADL is permitted")
    source = ROOT / row["source_path"]
    if sha256(source) != row["source_sha256"]:
        raise ContractError("External original RGB hash changed")
    return {
        "source_kind": "real",
        "sequence_id": row["sequence_id"],
        "split": "train",
        "external_assignment": "external_train",
        "label": 0,
        "recipe_hash": base["recipe_hash"],
        "recipe_id": "human_tsstg_coco_cut_v1",
        "split_hash": base["split_manifest"]["split_hash"],
        "source_hash": source_identity(source)["source_hash"],
        "raw_video_sha256": row["source_sha256"],
        "external_source": row,
        "assignment_manifest_sha256": sha256(PROFILE),
        "ur_split_hash_is_retained_evaluation_only": True,
    }


def freeze():
    data = read_json(PROFILE)
    base = read_json(ROOT / "runs/ur-human-isotropic-001/dataset-index.json")
    save_new(
        BEFORE,
        {
            "script_sha256": sha256(__file__),
            "data_profile_sha256": sha256(PROFILE),
            "source_index_sha256": sha256(ROOT / "runs/ur-human-isotropic-001/dataset-index.json"),
            "recipe_hash": base["recipe_hash"],
            "quality_unchanged": True,
            "fps_unchanged": True,
            "test_reads": 0,
            "fixed_exclusion": "zero usable feature windows only; no UR score-dependent exclusion",
            "weights": {
                v["local_path"]: sha256(ROOT / v["local_path"]) for v in base["weights"].values()
            },
            "planned_sources": [r["sequence_id"] for r in data["sequences"]],
        },
    )


def prepare():
    if EVIDENCE.exists():
        raise ContractError("Preparation complete; use status")
    lock = read_json(BEFORE)
    verify_frozen_source(__file__, lock["script_sha256"])
    if sha256(PROFILE) != lock["data_profile_sha256"]:
        raise ContractError("Frozen source/profile changed")
    base = read_json(ROOT / "runs/ur-human-isotropic-001/dataset-index.json")
    cfg = load_config(ROOT / "configs/ur-human-isotropic-001.yaml")
    for p, digest in lock["weights"].items():
        if sha256(ROOT / p) != digest:
            raise ContractError("Genuine weight checksum mismatch")
    if (
        recipe_fingerprint(cfg, {k: v["sha256"] for k, v in base["weights"].items()})
        != base["recipe_hash"]
    ):
        raise ContractError("Expected original genuine isotropic recipe")
    device = select_device(cfg["runtime"]["device"], cfg["runtime"]["torch_threads"])
    backend, extractor = create_backends(cfg, base["weights"], device)
    entries = []
    for row in read_json(PROFILE)["sequences"]:
        sid = row["sequence_id"]
        feature_path, pose_path = CACHE / sid / "features.npz", CACHE / sid / "poses.npz"
        meta = metadata_for(row, base)
        if feature_path.exists():
            cache = load_feature_cache(feature_path, meta)
            decoded = read_json(pose_path.with_suffix(".json"))["frames"]
            print(f"Resumed preserved features: {sid}", flush=True)
        else:
            if pose_path.exists():
                raise ContractError("Incomplete extraction retained; use a new preparation ID")
            backend.reset()
            observations = []
            for frame in read_source(ROOT / row["source_path"], sequence_id=sid):
                poses = backend.predict(frame)
                frame.image_bgr = None
                observations.append((frame, poses))
                if frame.sample_index % 90 == 0:
                    print(
                        f"{sid}: RGB frame {frame.source_frame_id}, tracks {len(poses)}", flush=True
                    )
            save_pose_cache(pose_path, observations, meta)
            cache = features_from_observations(observations, extractor, cfg, meta)
            save_feature_cache(feature_path, cache)
            decoded = read_json(pose_path.with_suffix(".json"))["frames"]
        entries.append(
            {
                **row,
                "feature_path": feature_path.relative_to(ROOT).as_posix(),
                "pose_path": pose_path.relative_to(ROOT).as_posix(),
                "feature_sha256": sha256(feature_path),
                "pose_sha256": sha256(pose_path),
                "source_hash": meta["source_hash"],
                "valid_features": len(cache.features),
                "decoded_frames": len(decoded),
                "n_samples": cache.metadata["n_samples"],
                "quality_counts": cache.metadata["quality_counts"],
                "eligible": len(cache.features) > 0,
            }
        )
        write_json(CACHE / "preparation-progress.json", {"completed": entries, "test_reads": 0})
    save_new(
        EVIDENCE,
        {
            "schema_version": 1,
            "evidence_tier": "real_end_to_end",
            "scope": "Original new normal RGB, strict real pose and genuine frozen ST-GCN; no AE scores yet",
            "preparation_lock_sha256": sha256(BEFORE),
            "data_profile_sha256": sha256(PROFILE),
            "recipe_hash": base["recipe_hash"],
            "entries": entries,
            "test_reads": 0,
            "new_test_evaluations": 0,
            "pose_strict_load": True,
            "feature_strict_load": True,
            "pose_forward": backend.forward_verified,
            "feature_forward": extractor.forward_verified,
        },
    )
    print(
        {
            "prepared": len(entries),
            "eligible": sum(r["eligible"] for r in entries),
            "feature_windows": sum(r["valid_features"] for r in entries),
        },
        flush=True,
    )


def sheets():
    dest = ROOT / "outputs/external-normal-review-20261012"
    if dest.exists():
        raise ContractError("Preserve external review sheets")
    dest.mkdir()
    rows = read_json(PROFILE)["sequences"]
    for block in range(0, len(rows), 4):
        canvas = Image.new("RGB", (1280, 4 * 230), "white")
        draw = ImageDraw.Draw(canvas)
        for i, row in enumerate(rows[block : block + 4]):
            cap = cv2.VideoCapture(str(ROOT / row["source_path"]))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for j, index in enumerate(np.linspace(0, n - 1, 5).round().astype(int)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(index))
                ok, img = cap.read()
                if not ok:
                    raise ContractError("Review frame decode failed")
                im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                im.thumbnail((254, 202))
                canvas.paste(im, (j * 256, i * 230 + 23))
                draw.text(
                    (j * 256 + 2, i * 230 + 3), f"{row['sequence_id']} f{index}", fill="black"
                )
            cap.release()
        canvas.save(dest / f"group-{block // 4:02d}.jpg", quality=95)
    print({"review_sheets": (len(rows) + 3) // 4})


def status():
    verify_frozen_source(__file__, read_json(BEFORE)["script_sha256"])
    data = read_json(EVIDENCE)
    for r in data["entries"]:
        for field, digest in (
            ("source_path", "source_sha256"),
            ("feature_path", "feature_sha256"),
            ("pose_path", "pose_sha256"),
        ):
            if sha256(ROOT / r[field]) != r[digest]:
                raise ContractError("External prepared source changed")
    print(
        {
            "prepared": len(data["entries"]),
            "eligible": sum(r["eligible"] for r in data["entries"]),
            "windows": sum(r["valid_features"] for r in data["entries"]),
            "test_reads": 0,
        }
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("freeze", "prepare", "sheets", "status"))
    globals()[p.parse_args().stage]()
