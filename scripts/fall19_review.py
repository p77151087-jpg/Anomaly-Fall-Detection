"""Build/read a local Validation-only frame-review package; no fitting or calibration."""

import argparse
import csv
import json
import shutil
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
from ur_motion_novelty import BASE, OLD, ROOT
from ur_motion_novelty import TAG as MOTION_TAG

from fall_detect.config import project_path
from fall_detect.data.cache import load_pose_cache
from fall_detect.data.readers import image_inventory, numeric_frame_id
from fall_detect.evaluation.frame_review import (
    annotation_template,
    latest_windows,
    validate_annotation,
)
from fall_detect.io_utils import (
    ContractError,
    atomic_bytes,
    read_json,
    sha256,
    write_csv,
    write_json,
)
from fall_detect.skeleton.preprocessing import good_frame
from fall_detect.skeleton.windows import CausalResampler, WindowBuilder
from fall_detect.training.experiment import Scorer, cache_for, load_index

TAG = "ur-fall19-review-20261012-001"
SID = "fall-19"
E = ROOT / "docs/validation" / TAG
O = ROOT / "outputs" / TAG
MUTABLE = [
    "AGENTS.md",
    "Fall_Detection_Codex_Implementation_Spec.md",
    "docs/current-progress.md",
    "docs/acceptance-results.md",
    "docs/course-alignment-plan.md",
]
SOURCES = [
    "scripts/fall19_review.py",
    "scripts/ur_motion_novelty.py",
    "scripts/fall19_review.html",
    "src/fall_detect/evaluation/frame_review.py",
    "src/fall_detect/data/cache.py",
    "src/fall_detect/data/readers.py",
    "src/fall_detect/skeleton/preprocessing.py",
    "src/fall_detect/skeleton/windows.py",
    "src/fall_detect/training/experiment.py",
    "src/fall_detect/io_utils.py",
]
CSV = {
    "wide30": ROOT / "outputs" / OLD / BASE / "window-results.csv",
    "isotropic20": ROOT
    / "outputs/ur-short20-isotropic-20261012/ur-short20-isotropic-20261012-001/window-results.csv",
    "motion_fusion": ROOT / "outputs" / MOTION_TAG / "window-results.csv",
}


def save(path, value):
    if path.exists():
        raise ContractError(f"Preserve completed artifact: {path}")
    write_json(path, value)


def windows_from_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        selected = [r for r in csv.DictReader(stream) if r["sequence_id"] == SID]
    for r in selected:
        for k in ["sample_index", "track_id", "window_id"]:
            r[k] = int(r[k]) if r[k] != "" else None
        for k in ["start_ms", "end_ms", "available_at_ms", "raw_score", "event_score"]:
            r[k] = float(r[k]) if r[k] != "" else None
        r["reason"] = r["reason"] or None
    return selected


def verify_sources():
    lock = read_json(E / "lock.json")
    for category in ["source_hashes", "input_hashes"]:
        for name, digest in lock[category].items():
            if sha256(ROOT / name) != digest:
                raise ContractError(f"Review source changed: {name}")
    return lock


def write_image(path, image):
    if path.exists():
        raise ContractError("Preserve review image")
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ContractError("Image encode failed")
    encoded.tofile(path)


def build():
    if E.exists() or O.exists():
        raise ContractError("Completed or attempted review package exists")
    E.mkdir(parents=True)
    old_before = read_json(ROOT / "docs/validation" / MOTION_TAG / "before.json")
    old_qa = read_json(ROOT / "docs/validation" / MOTION_TAG / "final-qa.json")
    assets = {**old_before["assets"], **old_qa["artifacts"]}
    for name in MUTABLE:
        assets.pop(name, None)
        target = E / "prior-contract" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    for name in [
        f"docs/validation/{MOTION_TAG}/final-qa.json",
        f"docs/validation/{MOTION_TAG}/before.json",
    ]:
        assets[name] = sha256(ROOT / name)
    save(
        E / "before.json",
        {
            "assets": assets,
            "contracts": {n: sha256(ROOT / n) for n in MUTABLE},
            "provenance": "Union of immutable preceding preservation and final QA, independently rehashed at verification",
            "test_bytes_hash_only": True,
        },
    )
    index_path = ROOT / "data/cache" / MOTION_TAG / "dataset-index.json"
    index = load_index(index_path)
    split = next(s for s in index["split_manifest"]["sequences"] if s["sequence_id"] == SID)
    if split["split"] != "validation" or index.get("preparation_scope") != ["train", "validation"]:
        raise ContractError("fall-19 must remain original scoped Validation")
    entry = index["sequences"][SID]
    posepath = project_path(entry["pose_path"])
    observations, posemeta = load_pose_cache(
        posepath,
        {
            "sequence_id": SID,
            "source_hash": entry["source_hash"],
            "recipe_hash": index["recipe_hash"],
        },
    )
    paths = {numeric_frame_id(p): p for p in image_inventory(ROOT / "data/raw/urfall" / SID)[0]}
    if set(paths) != {f.source_frame_id for f, _ in observations}:
        raise ContractError("Raw source/frame identities do not match full pose cache")
    # Check all original raw hashes from the preparation identity, never from Test.
    raw_root = ROOT / "data/raw/urfall" / SID
    inputs = {
        index_path.relative_to(ROOT).as_posix(): sha256(index_path),
        posepath.relative_to(ROOT).as_posix(): sha256(posepath),
        posepath.with_suffix(".json").relative_to(ROOT).as_posix(): sha256(
            posepath.with_suffix(".json")
        ),
    }
    for item in posemeta["source_identity"]["files"]:
        path = raw_root / item["path"]
        if sha256(path) != item["sha256"]:
            raise ContractError("Original RGB/sync changed")
        inputs[path.relative_to(ROOT).as_posix()] = item["sha256"]
    for path in CSV.values():
        inputs[path.relative_to(ROOT).as_posix()] = sha256(path)
    inputs["docs/ur-fall19-review-20261012-plan.md"] = sha256(
        ROOT / "docs/ur-fall19-review-20261012-plan.md"
    )
    save(
        E / "lock.json",
        {
            "sequence_id": SID,
            "split": "validation",
            "source_hashes": {n: sha256(ROOT / n) for n in SOURCES},
            "input_hashes": inputs,
            "no_fit": True,
            "no_calibration": True,
            "no_test": True,
            "no_public_upload": True,
        },
    )
    O.mkdir(parents=True)
    (O / "raw").mkdir()
    models, windows = {}, {}
    wide = Scorer(ROOT / "runs" / BASE)
    short = Scorer(ROOT / "runs/ur-short20-isotropic-20261012-001")
    motion_threshold = read_json(ROOT / "runs" / MOTION_TAG / "threshold.json")["threshold"]
    models["wide30"] = {"window_size": 30, "threshold": wide.threshold["threshold"], "run": BASE}
    models["isotropic20"] = {
        "window_size": 20,
        "threshold": short.threshold["threshold"],
        "run": "ur-short20-isotropic-20261012-001",
    }
    models["motion_fusion"] = {"window_size": 30, "threshold": motion_threshold, "run": MOTION_TAG}
    full_windows = {}
    for name, scorer, idx in [
        ("wide30", wide, index),
        ("isotropic20", short, short.index),
        ("motion_fusion", wide, index),
    ]:
        cache = cache_for(idx, SID)
        original = {(r.sample_index, r.track_id): r for r in cache.records}
        rows = windows_from_csv(CSV[name])
        if len(rows) != len(original):
            raise ContractError("Window rows do not match cache")
        quality = {}
        resampler = CausalResampler(30, scorer.config["skeleton"]["resampling_hold_policy"])
        builder = WindowBuilder(scorer.config)
        for frame, poses in observations:
            for sample in resampler.push(frame, poses):
                for w in builder.push(sample):
                    r = original[(w.sample_index, w.track_id)]
                    if w.quality != r.quality or w.start_ms != r.start_ms or w.end_ms != r.end_ms:
                        raise ContractError("Reconstructed timing/quality changed")
                    quality[(w.sample_index, w.track_id)] = {
                        "quality": w.quality,
                        "good_frame_ratio": w.good_frame_ratio,
                    }
        if name != "motion_fusion":
            raw_scores = scorer.score(cache.features)
        else:
            with np.load(ROOT / "runs" / MOTION_TAG / "scores.npz", allow_pickle=False) as saved:
                raw_scores = saved[SID]
        for row in rows:
            rec = original[(row["sample_index"], row["track_id"])]
            expected = None if rec.row_index is None else float(raw_scores[rec.row_index])
            if row["raw_score"] != expected:
                raise ContractError("Original score changed")
            row.update(quality[(rec.sample_index, rec.track_id)])
        windows[name] = rows
        full_windows[name] = [dict(r) for r in rows]
        write_csv(O / f"{name}-windows.csv", rows)
    frames = []
    for frame, poses in observations:
        path = paths[frame.source_frame_id]
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (frame.height, frame.width):
            raise ContractError("RGB decode/dimensions failed")
        raw_output = O / "raw" / f"{frame.source_frame_id:03d}.png"
        shutil.copyfile(path, raw_output)
        annotated = image.copy()
        pose_rows = []
        for pose in poses:
            reliable = pose.keypoints[:, 2] >= 0.2
            outside = (
                (pose.keypoints[:, 0] < 0)
                | (pose.keypoints[:, 0] >= frame.width)
                | (pose.keypoints[:, 1] < 0)
                | (pose.keypoints[:, 1] >= frame.height)
            )
            pose_rows.append(
                {
                    "track_id": pose.track_id,
                    "reported_quality": pose.quality,
                    "current_good_pose": bool(good_frame(pose.keypoints, wide.config["quality"])),
                    "reliable_joints": int(reliable.sum()),
                    "reliable_joints_outside": int((outside & reliable).sum()),
                    "min_confidence": float(pose.keypoints[:, 2].min()),
                    "keypoints": pose.keypoints.tolist(),
                }
            )
            for j, (x, y, confidence) in enumerate(pose.keypoints):
                cv2.circle(
                    annotated,
                    (round(x), round(y)),
                    4,
                    (65, 225, 65) if confidence >= 0.2 else (55, 70, 255),
                    -1,
                )
                cv2.putText(annotated, str(j), (round(x) + 4, round(y)), 0, 0.4, (255, 255, 255), 1)
            cv2.putText(
                annotated,
                f"track {pose.track_id}",
                (5, 20 + 20 * len(pose_rows)),
                0,
                0.6,
                (255, 230, 70),
                2,
            )
        pose_output = O / "pose" / f"{frame.source_frame_id:03d}.png"
        write_image(pose_output, annotated)
        frames.append(
            {
                "sequence_id": SID,
                "source_frame_id": frame.source_frame_id,
                "timestamp_ms": frame.timestamp_ms,
                "timestamp_estimated": frame.timestamp_estimated,
                "source_sha256": sha256(path),
                "raw_path": raw_output.relative_to(O).as_posix(),
                "pose_path": pose_output.relative_to(O).as_posix(),
                "poses": pose_rows,
                "models": {
                    name: latest_windows(rs, frame.timestamp_ms) for name, rs in windows.items()
                },
            }
        )
    manifest = {
        "schema_version": 1,
        "sequence_id": SID,
        "split": "validation",
        "source_hash": entry["source_hash"],
        "human_confirmed": False,
        "frames": [
            {k: r[k] for k in ["source_frame_id", "timestamp_ms", "source_sha256"]} for r in frames
        ],
    }
    save(O / "source-manifest.json", manifest)
    template = annotation_template(manifest, sha256(O / "source-manifest.json"))
    save(O / "annotation-template.json", template)
    save(
        O / "review-data.json",
        {
            "frames": frames,
            "models": models,
            "annotation_template": template,
            "windows": full_windows,
            "human_confirmed": False,
            "source": "Frozen original RGB/Pose and Validation scores; not new inference or GT",
        },
    )
    atomic_bytes(
        O / "review-data.js",
        (
            "window.REVIEW_DATA="
            + json.dumps(
                {"frames": frames, "models": models, "annotation_template": template},
                ensure_ascii=False,
                allow_nan=False,
            )
            + ";\n"
        ).encode("utf-8"),
    )
    shutil.copyfile(ROOT / "scripts/fall19_review.html", O / "index.html")
    flat = []
    for frame in frames:
        flat.append(
            {
                "source_frame_id": frame["source_frame_id"],
                "timestamp_ms": frame["timestamp_ms"],
                "pose_tracks": ";".join(str(p["track_id"]) for p in frame["poses"]),
                "current_good_pose": any(p["current_good_pose"] for p in frame["poses"]),
                "max_reliable_joints": max([p["reliable_joints"] for p in frame["poses"]] or [0]),
                **{
                    name: ";".join(
                        f"track{r['track_id']}:{r['quality']}" for r in frame["models"][name]
                    )
                    for name in models
                },
            }
        )
    write_csv(O / "frame-timeline.csv", flat)
    for selection, name in [
        (frames, "all"),
        ([r for r in frames if 1400 <= r["timestamp_ms"] <= 2600], "dense"),
    ]:
        for mode in ["raw", "pose"]:
            for offset in range(0, len(selection), 20):
                tiles = []
                for f in selection[offset : offset + 20]:
                    im = cv2.imdecode(
                        np.fromfile(O / f[f"{mode}_path"], dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    tile = cv2.copyMakeBorder(
                        cv2.resize(im, (320, 240)), 26, 0, 0, 0, cv2.BORDER_CONSTANT
                    )
                    cv2.putText(
                        tile,
                        f"f{f['source_frame_id']:03d} {f['timestamp_ms']:.0f}ms",
                        (5, 18),
                        0,
                        0.55,
                        (240, 240, 240),
                        1,
                    )
                    tiles.append(tile)
                while len(tiles) % 4:
                    tiles.append(np.zeros_like(tiles[0]))
                write_image(
                    O / "sheets" / f"{name}-{mode}-{offset // 20 + 1:02d}.jpg",
                    np.vstack([np.hstack(tiles[i : i + 4]) for i in range(0, len(tiles), 4)]),
                )
    save(
        E / "build.json",
        {
            "frames": len(frames),
            "windows": {n: len(r) for n, r in windows.items()},
            "valid_windows": {
                n: sum(r["quality"] == "valid" for r in rows) for n, rows in windows.items()
            },
            "human_confirmed": False,
            "training_epochs": 0,
            "test_feature_reads": 0,
            "test_evaluations": 0,
            "new_pose_forward": False,
            "new_stgcn_forward": False,
            "reloaded_ae_score_parity": True,
            "artifacts": {
                p.relative_to(ROOT).as_posix(): sha256(p) for p in O.rglob("*") if p.is_file()
            },
        },
    )
    print(
        {
            "frames": len(frames),
            "models": models,
            "valid_windows": read_json(E / "build.json")["valid_windows"],
            "review": str(O / "index.html"),
        },
        flush=True,
    )


def import_annotation(path):
    verify_sources()
    manifest = read_json(O / "source-manifest.json")
    annotation = read_json(path)
    verified = validate_annotation(annotation, manifest, sha256(O / "source-manifest.json"))
    folder = E / "human-annotations" / sha256(path)
    if folder.exists():
        raise ContractError("Preserve imported annotation")
    folder.mkdir(parents=True)
    shutil.copyfile(path, folder / "original-export.json")
    save(folder / "validation.json", verified)
    print(verified, flush=True)


def serve(port):
    verify_sources()
    handler = partial(SimpleHTTPRequestHandler, directory=str(O))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"Review URL: http://127.0.0.1:{server.server_port}/index.html", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["build", "status", "serve", "import"])
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--annotation")
    args = parser.parse_args()
    if args.stage == "build":
        build()
    elif args.stage == "status":
        print({"lock": verify_sources(), "build": read_json(E / "build.json")}, flush=True)
    elif args.stage == "serve":
        serve(args.port)
    elif args.annotation:
        import_annotation(project_path(args.annotation))
    else:
        parser.error("import requires --annotation")
