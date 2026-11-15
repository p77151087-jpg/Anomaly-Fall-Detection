"""Read-only Train/Validation pose audit; visual annotations are not official GT."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from fall_detect.config import project_path
from fall_detect.data.cache import load_pose_cache
from fall_detect.data.readers import image_inventory, numeric_frame_id
from fall_detect.io_utils import ContractError, read_json, sha256, write_json

ROOT = project_path(".")
OUT = ROOT / "outputs/human-keyframes"
INDEX = ROOT / "data/cache/ur-sample-human-s1-001/dataset-index.json"
IDS = ("fall-19", "fall-18", "fall-30")


def write_image(path, image):
    path = Path(path)
    if path.exists():
        raise ContractError(f"Preserve existing image: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ContractError("Image encoding failed")
    encoded.tofile(path)


def sheets():
    if OUT.exists():
        raise ContractError("Keyframe sheets already exist")
    index = read_json(INDEX)
    manifest = index["split_manifest"]
    rows = []
    for sid in IDS:
        split = next(s for s in manifest["sequences"] if s["sequence_id"] == sid)
        if split["split"] != "validation":
            raise ContractError("Diagnostic IDs must be Validation")
        observations, _ = load_pose_cache(project_path(index["sequences"][sid]["pose_path"]), {})
        paths = {numeric_frame_id(p): p for p in image_inventory(ROOT / "data/raw/urfall" / sid)[0]}
        selected = observations[::5]
        if selected[-1][0].source_frame_id != observations[-1][0].source_frame_id:
            selected.append(observations[-1])
        for mode in ("raw", "pose"):
            tiles = []
            for frame, poses in selected:
                raw = paths[frame.source_frame_id]
                image = cv2.imdecode(np.fromfile(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
                if mode == "pose":
                    for pose in poses:
                        for j, (x, y, conf) in enumerate(pose.keypoints):
                            color = (0, 255, 0) if conf >= 0.2 else (0, 0, 255)
                            cv2.circle(image, (round(x), round(y)), 4, color, -1)
                            cv2.putText(image, str(j), (round(x) + 4, round(y)), 0, 0.45, color, 1)
                tile = cv2.resize(image, (320, 240))
                tile = cv2.copyMakeBorder(tile, 24, 0, 0, 0, cv2.BORDER_CONSTANT)
                cv2.putText(
                    tile,
                    f"{sid} f{frame.source_frame_id} {frame.timestamp_ms:.0f}ms",
                    (5, 17),
                    0,
                    0.48,
                    (255, 255, 255),
                    1,
                )
                tiles.append(tile)
                if mode == "raw":
                    rows.append(
                        {
                            "sequence_id": sid,
                            "frame_id": frame.source_frame_id,
                            "timestamp_ms": frame.timestamp_ms,
                            "timestamp_estimated": frame.timestamp_estimated,
                            "source": str(raw.relative_to(ROOT)),
                            "sha256": sha256(raw),
                        }
                    )
            while len(tiles) % 4:
                tiles.append(np.zeros_like(tiles[0]))
            sheet = np.vstack([np.hstack(tiles[i : i + 4]) for i in range(0, len(tiles), 4)])
            write_image(OUT / f"{sid}-{mode}.jpg", sheet)
    write_json(
        OUT / "inventory.json",
        {
            "evidence": "real_feature_smoke",
            "official_ground_truth": False,
            "index_sha256": sha256(INDEX),
            "frames": rows,
        },
    )
    print(f"Saved {len(rows)} sampled frames in six contact sheets")


def diagnose():
    from fall_detect.skeleton.preprocessing import HUMAN13_NAMES, good_frame, human_preprocess

    target = ROOT / "docs/validation/human-keyframes-diagnosis.json"
    if target.exists():
        raise ContractError("Preserve diagnosis")
    annotations = read_json(ROOT / "docs/validation/human-keyframes-annotations.json")
    index = read_json(INDEX)
    summary, reviewed = [], []
    for sid in IDS:
        entry = index["sequences"][sid]
        observations, _ = load_pose_cache(project_path(entry["pose_path"]), {})
        outside, high_outside, good = 0, 0, []
        extents = []
        for f, poses in observations:
            for p in poses:
                xy, conf = p.keypoints[:, :2], p.keypoints[:, 2]
                out = (
                    (xy[:, 0] < 0) | (xy[:, 0] >= f.width) | (xy[:, 1] < 0) | (xy[:, 1] >= f.height)
                )
                outside += int(out.sum())
                high_outside += int((out & (conf >= 0.2)).sum())
                if good_frame(p.keypoints, {"min_joint_confidence": 0.2, "min_valid_joints": 9}):
                    good.append(f.timestamp_ms)
                    span = np.ptp(xy, axis=0)
                    pts, _ = human_preprocess(
                        np.repeat(p.keypoints[None], 30, 0), (f.width, f.height)
                    )
                    norm_span = np.ptp(pts[0, :2, 0, :13], axis=1)
                    extents.append(
                        {
                            "frame_id": f.source_frame_id,
                            "timestamp_ms": f.timestamp_ms,
                            "pixel_bbox_aspect": float(span[0] / span[1]),
                            "normalized_bbox_aspect": float(norm_span[0] / norm_span[1]),
                        }
                    )
        summary.append(
            {
                "sequence_id": sid,
                "observations": len(observations),
                "out_of_image_joint_observations": outside,
                "confident_out_of_image_joint_observations": high_outside,
                "last_good_pose_ms": max(good),
                "good_frame_extents": extents,
            }
        )
        paths = {numeric_frame_id(p): p for p in image_inventory(ROOT / "data/raw/urfall" / sid)[0]}
        for ann in annotations["frames"]:
            if ann["sequence_id"] != sid:
                continue
            f, poses = next((f, p) for f, p in observations if f.source_frame_id == ann["frame_id"])
            raw = paths[f.source_frame_id]
            image = cv2.imdecode(np.fromfile(raw, np.uint8), 1)
            comparisons = []
            for joint, region in ann["approximate_visible_regions"].items():
                x, y, radius = region
                cv2.circle(image, (x, y), radius, (255, 255, 0), 1)
                j = HUMAN13_NAMES.index(joint)
                if poses:
                    px, py, confidence = poses[0].keypoints[j]
                    distance = float(np.linalg.norm(np.array([px, py]) - [x, y]))
                    comparisons.append(
                        {
                            "joint": joint,
                            "pose_xy": [float(px), float(py)],
                            "confidence": float(confidence),
                            "distance_from_approximate_center_px": distance,
                            "inside_review_region": distance <= radius,
                        }
                    )
                    cv2.circle(image, (round(px), round(py)), 3, (0, 255, 0), -1)
                cv2.putText(
                    image,
                    joint,
                    (max(0, x - radius), max(12, y - radius)),
                    0,
                    0.35,
                    (255, 255, 0),
                    1,
                )
            out = OUT / f"{sid}-f{f.source_frame_id}-review.png"
            write_image(out, image)
            reviewed.append(
                {
                    **ann,
                    "timestamp_ms": f.timestamp_ms,
                    "source": str(raw.relative_to(ROOT)),
                    "source_sha256": sha256(raw),
                    "review_image": str(out.relative_to(ROOT)),
                    "comparisons": comparisons,
                }
            )
    write_json(
        target,
        {
            "evidence": "real_feature_smoke",
            "official_ground_truth": False,
            "annotations_sha256": sha256(ROOT / "docs/validation/human-keyframes-annotations.json"),
            "summary": summary,
            "reviewed": reviewed,
            "proven": "Legacy per-frame independent X/Y scaling maps every nondegenerate joint bounding box to aspect ratio 1, removing its original aspect. Low-confidence coordinates participate. Out-of-frame joint coordinates can retain high model confidence.",
            "hypothesis": "Preserving pixel aspect ratio may improve separation before pose quality disappears; not proved by geometry or this visual audit.",
            "limits": "Nine approximate visual reviews, not independent or official keypoint GT; overlays were also inspected. Visibility is uncertain under blur and occlusion; no hidden joint coordinates fabricated. No Test or Accuracy calculation in this audit.",
        },
    )
    print(
        {
            "reviewed_frames": len(reviewed),
            "summary": [{k: v for k, v in s.items() if k != "good_frame_extents"} for s in summary],
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["sheets", "diagnose"])
    globals()[parser.parse_args().stage]()
