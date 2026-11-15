"""Read-only Validation Pose diagnosis and a separately saved human review bundle."""

import argparse
import base64
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
ID = "ur-fall19-joint-localization-20261013-001"
OUT = ROOT / "outputs" / ID
QA = ROOT / "docs/validation" / ID
PRIOR = ROOT / "outputs/ur-fall19-review-20261012-001"
POSE = ROOT / (
    "data/cache/ur-human-isotropic-001/poses/"
    "fall-19-b845bd9db6378b523e05c3475d64ad09babf9e4a304aecbaae6ac4de62602b53.npz"
)
RAW = ROOT / "data/raw/urfall/fall-19/fall-19-cam0-rgb"
HUMAN = ROOT / "outputs/ur-fall19-human-amendment-20261013-001/derived-annotation.json"
PLAN = ROOT / "docs/ur-fall19-joint-localization-20261013-plan.md"
TEMPLATE = ROOT / "src/fall_detect/visualization/joint_localization_review.html"
FRAMES = tuple(range(44, 52))
NAMES = (
    "nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)
ZH_NAMES = (
    "鼻", "左肩", "右肩", "左肘", "右肘", "左腕", "右腕", "左髖", "右髖",
    "左膝", "右膝", "左踝", "右踝",
)
EDGES = ((1, 2), (1, 3), (3, 5), (2, 4), (4, 6), (1, 7), (2, 8),
         (7, 8), (7, 9), (9, 11), (8, 10), (10, 12))
MUTABLE = ("docs/current-progress.md", "docs/acceptance-results.md")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def load_original():
    """Only the exact fall-19 Validation cache and its eight original images."""
    meta, manifest, human = read(POSE.with_suffix(".json")), read(PRIOR / "source-manifest.json"), read(HUMAN)
    if any(d.get("sequence_id") != "fall-19" or d.get("split") != "validation"
           for d in (meta, manifest, human)):
        raise ValueError("Validation fall-19 only; no Test access")
    if meta["source_kind"] != "real" or meta["recipe_hash"] != (
        "713ee59a0853f20641cb7f664df2920359baa0734807acfa5cf86fb87a4e2f5b"
    ):
        raise ValueError("Wrong frozen original Pose recipe")
    if sha(POSE) != meta["array_sha256"] or manifest["source_hash"] != meta["source_hash"]:
        raise ValueError("Source/Pose checksum mismatch")
    if human["manifest_sha256"] != sha(PRIOR / "source-manifest.json"):
        raise ValueError("Human annotation manifest mismatch")
    with np.load(POSE, allow_pickle=False) as arrays:
        keypoints = arrays["keypoints"].copy()
    if keypoints.shape != (len(meta["poses"]), 13, 3) or not np.isfinite(keypoints).all():
        raise ValueError("Non-finite or unexpected human13 Pose")
    derived = {f["source_frame_id"]: f for f in read(PRIOR / "review-data.json")["frames"]}
    identities = {p["path"].replace("\\", "/"): p["sha256"]
                  for p in meta["source_identity"]["files"]}
    manifest_frames = {f["source_frame_id"]: f for f in manifest["frames"]}
    human_frames = {f["source_frame_id"]: f for f in human["frames"]}
    records = []
    for fid in FRAMES:
        frame = next(f for f in meta["frames"] if f["source_frame_id"] == fid)
        indices = frame["pose_indices"]
        if len(indices) != 1 or (frame["width"], frame["height"]) != (640, 480):
            raise ValueError("Unexpected frame/track count")
        pose = meta["poses"][indices[0]]
        points = keypoints[indices[0]]
        prior_pose = derived[fid]["poses"][0]
        if pose["track_id"] != 1 or pose["layout_id"] != "human13":
            raise ValueError("Unexpected track/layout")
        if not np.array_equal(points, np.asarray(prior_pose["keypoints"], dtype=points.dtype)):
            raise ValueError("Previously displayed skeleton differs from original cache")
        image_path = RAW / f"fall-19-cam0-rgb-{fid:03d}.png"
        digest = sha(image_path)
        expected_image = identities[f"fall-19-cam0-rgb/{image_path.name}"]
        if not (digest == expected_image == manifest_frames[fid]["source_sha256"]
                == human_frames[fid]["source_sha256"] == derived[fid]["source_sha256"]):
            raise ValueError("Image source mismatch")
        if not (frame["timestamp_ms"] == manifest_frames[fid]["timestamp_ms"]
                == human_frames[fid]["timestamp_ms"] == derived[fid]["timestamp_ms"]):
            raise ValueError("Timestamp mismatch")
        with Image.open(image_path) as im:
            if im.size != (640, 480):
                raise ValueError("Image dimensions mismatch")
        records.append({
            "source_frame_id": fid, "timestamp_ms": frame["timestamp_ms"],
            "track_id": 1, "image_path": relative(image_path), "image_sha256": digest,
            "frame_size_wh": [640, 480], "keypoints": points.astype(float).tolist(),
            "original_reported_quality": pose["quality"],
            "original_current_good_pose": prior_pose["current_good_pose"],
            "human_whole_pose_check": human_frames[fid]["pose_check"],
            "human_phase": human_frames[fid]["phase"],
        })
    return records


def measurements(records):
    joints, bones = [], []
    previous = None
    for record in records:
        points = np.asarray(record["keypoints"], dtype=float)
        if points.shape != (13, 3) or not np.isfinite(points).all():
            raise ValueError("Measurements require finite human13 points")
        if previous is not None:
            if (record["source_frame_id"] != previous["source_frame_id"] + 1
                    or record["track_id"] != previous["track_id"]
                    or record["timestamp_ms"] <= previous["timestamp_ms"]):
                raise ValueError("No bridging tracks/gaps or reversed time")
            delta = points[:, :2] - np.asarray(previous["keypoints"])[:, :2]
            median_delta = np.median(delta, axis=0)
            elapsed = record["timestamp_ms"] - previous["timestamp_ms"]
        for j, name in enumerate(NAMES):
            row = {
                "source_frame_id": record["source_frame_id"], "track_id": 1,
                "timestamp_ms": record["timestamp_ms"], "joint_index": j, "joint_name": name,
                "x": points[j, 0], "y": points[j, 1], "confidence": points[j, 2],
                "previous_frame_id": None if previous is None else previous["source_frame_id"],
                "elapsed_ms": None if previous is None else elapsed,
                "dx_px": None if previous is None else delta[j, 0],
                "dy_px": None if previous is None else delta[j, 1],
                "step_px": None if previous is None else float(np.linalg.norm(delta[j])),
                "median_motion_residual_px": None if previous is None else float(
                    np.linalg.norm(delta[j] - median_delta)),
                "outside_image": not (0 <= points[j, 0] < 640 and 0 <= points[j, 1] < 480),
                "human_joint_confirmation": "not_provided_by_whole_pose_label",
            }
            joints.append(row)
        for a, b in EDGES:
            bones.append({"source_frame_id": record["source_frame_id"], "joint_a": a,
                          "joint_b": b, "name": NAMES[a] + "--" + NAMES[b],
                          "length_px": float(np.linalg.norm(points[a, :2] - points[b, :2])),
                          "interpretation": "diagnostic_projection_not_correctness_ground_truth"})
        previous = record
    return joints, bones


def csv_write(path, rows):
    with Path(path).open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inventory():
    paths = {POSE, POSE.with_suffix(".json"), HUMAN, PLAN, Path(__file__).resolve(), TEMPLATE,
             ROOT / "AGENTS.md", ROOT / "Fall_Detection_Codex_Implementation_Spec.md",
             ROOT / "docs/course-alignment-plan.md",
             ROOT / "docs/ur-pose-coordinate-human-20261013.md",
             ROOT / "docs/ur-pose-coordinate-diagnostic-20261013-plan.md"}
    for name in MUTABLE:
        paths.add(ROOT / name)
    folders = (
        "outputs/ur-fall19-review-20261012-001",
        "outputs/ur-fall19-human-amendment-20261013-001",
        "outputs/ur-pose-coordinate-diagnostic-20261013-001",
        "data/cache/ur-pose-coordinate-diagnostic-20261013-001",
        "runs/ur-isotropic-wide-20261012-001",
        "runs/ur-sample-human-s1-001",
        "runs/ur-short20-isotropic-20261012-001",
    )
    for folder in folders:
        paths.update(p for p in (ROOT / folder).rglob("*") if p.is_file())
    paths.update(RAW / f"fall-19-cam0-rgb-{fid:03d}.png" for fid in FRAMES)
    return {relative(path): sha(path) for path in sorted(paths)}


def render(record):
    raw = Image.open(ROOT / record["image_path"]).convert("RGB")
    overlay = raw.copy()
    draw = ImageDraw.Draw(overlay)
    points = np.asarray(record["keypoints"])
    for a, b in EDGES:
        draw.line([tuple(points[a, :2]), tuple(points[b, :2])], fill=(210, 210, 210), width=1)
    for j, point in enumerate(points):
        x, y = point[:2]
        color = (255, 178, 48) if j >= 7 else (79, 222, 242)
        draw.ellipse((x-3, y-3, x+3, y+3), outline=color, width=2)
        draw.text((x+4, y-10), str(j), fill=color, stroke_width=1, stroke_fill="black")
    crop = (260, 240, 430, 480)
    panel = Image.new("RGB", (1040, 760), "#16202d")
    panel.paste(raw.crop(crop).resize((510, 720), Image.Resampling.NEAREST), (0, 32))
    panel.paste(overlay.crop(crop).resize((510, 720), Image.Resampling.NEAREST), (530, 32))
    draw = ImageDraw.Draw(panel)
    draw.text((12, 10), f"Frame {record['source_frame_id']} | {record['timestamp_ms']:.0f} ms | ORIGINAL RGB", fill="white")
    draw.text((542, 10), "ORIGINAL POSE | orange: hips/knees/ankles; cyan: upper", fill="white")
    return panel


def inspect():
    if OUT.exists() or QA.exists():
        raise ValueError("Diagnostic directory exists; refuse overwrite")
    records = load_original()
    before = inventory()
    OUT.mkdir(parents=True, exist_ok=False)
    QA.mkdir(parents=True, exist_ok=False)
    (OUT / "all-frames").mkdir()
    (QA / "prior-contracts").mkdir()
    for name in MUTABLE:
        dest = QA / "prior-contracts" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    write(QA / "old-assets-before.json", before)
    write(QA / "lock.json", {"diagnostic_id": ID, "plan_sha256": sha(PLAN),
                              "script_sha256": sha(Path(__file__)), "training": "not_run",
                              "calibration": "not_run", "test": "not_accessed",
                              "selection_limit": 3, "coordinate_intervention": "not_run"})
    write(OUT / "original-frames.json", records)
    joints, bones = measurements(records)
    csv_write(OUT / "joint-measurements.csv", joints)
    csv_write(OUT / "bone-measurements.csv", bones)
    sheet = Image.new("RGB", (2080, 760), "#16202d")
    for i, record in enumerate(records):
        panel = render(record)
        panel.save(OUT / "all-frames" / f"frame-{record['source_frame_id']:03d}.png")
        sheet.paste(panel.resize((520, 380)), ((i % 4)*520, (i // 4)*380))
    sheet.save(OUT / "all-eight-contact.png")
    print(json.dumps({"diagnostic_id": ID, "frames": len(records), "joints": len(joints),
                      "bones": len(bones), "old_files": len(before), "test": "not_accessed"}))


def package(assessment_path):
    if (OUT / "review").exists() or (OUT / "ai-assessment.json").exists():
        raise ValueError("Review bundle exists; refuse overwrite")
    original = read(OUT / "original-frames.json")
    if original != load_original():
        raise ValueError("Original source changed since inspection")
    assessment = read(assessment_path)
    selected = assessment["selected_frame_ids"]
    if (assessment.get("human_confirmed") is not False or assessment.get("author_kind") != "ai_provisional"
            or not 1 <= len(selected) <= 3 or len(set(selected)) != len(selected)
            or any(fid not in FRAMES for fid in selected)):
        raise ValueError("Invalid independent AI selection; maximum three original frames")
    if set(assessment["frame_observations"]) != {str(fid) for fid in FRAMES}:
        raise ValueError("AI observation required for each original frame")
    write(OUT / "ai-assessment.json", assessment)
    data = {
        "diagnostic_id": ID, "sequence_id": "fall-19", "split": "validation",
        "pose_array_sha256": sha(POSE), "pose_metadata_sha256": sha(POSE.with_suffix(".json")),
        "source_manifest_sha256": sha(PRIOR / "source-manifest.json"),
        "names": list(NAMES), "zh_names": list(ZH_NAMES), "edges": EDGES,
        "required_joint_indices": list(range(7, 13)), "frames": [],
    }
    review = OUT / "review"
    review.mkdir(exist_ok=False)
    for fid in selected:
        record = next(r for r in original if r["source_frame_id"] == fid)
        copied = dict(record)
        copied["image"] = "data:image/png;base64," + base64.b64encode(
            (ROOT / record["image_path"]).read_bytes()).decode("ascii")
        # No scores or AI-suggested coordinates enter the human review page.
        data["frames"].append(copied)
        shutil.copyfile(OUT / "all-frames" / f"frame-{fid:03d}.png", review / f"frame-{fid:03d}.png")
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, allow_nan=False)
    data["bundle_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    write(review / "review-data.json", data)
    html = TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", json.dumps(data, ensure_ascii=False))
    with (review / "index.html").open("x", encoding="utf-8") as stream:
        stream.write(html)
    write(OUT / "summary.json", {
        "diagnostic_id": ID, "frames_inspected": list(FRAMES), "joints_measured": 104,
        "selected_frame_ids": selected, "required_joint_indices": list(range(7, 13)),
        "required_human_decisions": len(selected)*6, "human_review": "pending",
        "human_joint_ground_truth": "not_available", "coordinates_replaced": 0,
        "new_feature_inference": "not_run", "training": "not_run", "calibration": "not_run",
        "test": "not_accessed_not_evaluated", "formal_validation_changed": False,
        "accuracy_improvement": "not_evaluated_no_claim", "demo_changed": False,
        "bundle_sha256": data["bundle_sha256"],
    })
    print(json.dumps({"selected": selected, "required_human_decisions": len(selected)*6,
                      "human_review": "pending", "review": str(review / "index.html")}))


def verify():
    before = read(QA / "old-assets-before.json")
    for name, digest in before.items():
        path = ROOT / name
        if name in MUTABLE and sha(path) != digest:
            path = QA / "prior-contracts" / name
        if sha(path) != digest:
            raise ValueError(f"Old artifact modified: {name}")
    records = read(OUT / "original-frames.json")
    if records != load_original():
        raise ValueError("Original source/observation mismatch")
    joints, bones = measurements(records)
    for path, expected in ((OUT / "joint-measurements.csv", joints), (OUT / "bone-measurements.csv", bones)):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != len(expected):
            raise ValueError("CSV denominator mismatch")
        for observed, target in zip(rows, expected):
            for key, value in target.items():
                wanted = "" if value is None else str(value)
                if observed[key] != wanted:
                    raise ValueError(f"CSV field mismatch: {key}")
    summary = read(OUT / "summary.json")
    data = read(OUT / "review/review-data.json")
    digest = data.pop("bundle_sha256")
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, allow_nan=False)
    if hashlib.sha256(payload.encode("utf-8")).hexdigest() != digest or summary["bundle_sha256"] != digest:
        raise ValueError("Review bundle checksum mismatch")
    if [f["source_frame_id"] for f in data["frames"]] != summary["selected_frame_ids"]:
        raise ValueError("Selection mismatch")
    for frame in data["frames"]:
        if hashlib.sha256(base64.b64decode(frame["image"].split(",", 1)[1])).hexdigest() != frame["image_sha256"]:
            raise ValueError("Embedded original image changed")
        original = next(r for r in records if r["source_frame_id"] == frame["source_frame_id"])
        if {k: frame[k] for k in original} != original:
            raise ValueError("Review observation changed")
    html = (OUT / "review/index.html").read_text(encoding="utf-8")
    if "__DATA__" in html or 'const DATA=' not in html:
        raise ValueError("Review page incomplete")
    result = {"verified": True, "old_artifacts_or_archived_contracts": len(before),
              "frames": 8, "joint_rows": len(joints), "bone_rows": len(bones),
              "selected_frame_ids": summary["selected_frame_ids"], "images_exact": True,
              "test": "not_accessed", "feature_smoke": "not_run", "end_to_end": "not_run"}
    print(json.dumps(result, ensure_ascii=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "package", "verify", "serve"))
    parser.add_argument("--assessment", type=Path)
    parser.add_argument("--port", type=int, default=8820)
    args = parser.parse_args()
    if args.action == "inspect":
        inspect()
    elif args.action == "package":
        if args.assessment is None:
            parser.error("package requires --assessment")
        package(args.assessment)
    elif args.action == "verify":
        verify()
    else:
        from functools import partial
        from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
        handler = partial(SimpleHTTPRequestHandler, directory=str(OUT / "review"))
        print(f"http://127.0.0.1:{args.port}/index.html", flush=True)
        ThreadingHTTPServer(("127.0.0.1", args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
