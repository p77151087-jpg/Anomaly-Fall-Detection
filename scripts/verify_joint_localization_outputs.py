"""Independent math/CSV checks using only the exact original Validation Pose."""

import base64
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ID = "ur-fall19-joint-localization-20261013-001"
OUT = ROOT / "outputs" / ID
QA = ROOT / "docs/validation" / ID


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    output = QA / "independent-verification.json"
    if output.exists():
        raise ValueError("Verification output already exists; use localize verify for read-only recheck")
    pose = ROOT / (
        "data/cache/ur-human-isotropic-001/poses/"
        "fall-19-b845bd9db6378b523e05c3475d64ad09babf9e4a304aecbaae6ac4de62602b53.npz"
    )
    meta = read(pose.with_suffix(".json"))
    assert meta["sequence_id"] == "fall-19" and meta["split"] == "validation"
    assert sha(pose) == meta["array_sha256"]
    with np.load(pose, allow_pickle=False) as arrays:
        points = arrays["keypoints"]
    mapping = {}
    for frame in meta["frames"]:
        fid = frame["source_frame_id"]
        if 44 <= fid <= 51:
            indices = frame["pose_indices"]
            assert len(indices) == 1
            assert meta["poses"][indices[0]]["track_id"] == 1
            mapping[fid] = (frame, points[indices[0]])
    with (OUT / "joint-measurements.csv").open(encoding="utf-8-sig", newline="") as stream:
        joints = list(csv.DictReader(stream))
    assert len(joints) == 104
    for row in joints:
        fid, j = int(row["source_frame_id"]), int(row["joint_index"])
        frame, array = mapping[fid]
        assert int(row["track_id"]) == 1 and float(row["timestamp_ms"]) == frame["timestamp_ms"]
        assert all(float(row[k]) == float(array[j, i]) for i, k in enumerate(("x", "y", "confidence")))
        if fid == 44:
            assert row["step_px"] == row["median_motion_residual_px"] == ""
        else:
            past = mapping[fid - 1][1]
            dx = float(array[j, 0]) - float(past[j, 0])
            dy = float(array[j, 1]) - float(past[j, 1])
            median_x = statistics.median(float(array[k, 0]) - float(past[k, 0]) for k in range(13))
            median_y = statistics.median(float(array[k, 1]) - float(past[k, 1]) for k in range(13))
            assert math.isclose(float(row["step_px"]), math.hypot(dx, dy), abs_tol=1e-9)
            assert math.isclose(float(row["median_motion_residual_px"]), math.hypot(dx-median_x, dy-median_y), abs_tol=1e-9)
    with (OUT / "bone-measurements.csv").open(encoding="utf-8-sig", newline="") as stream:
        bones = list(csv.DictReader(stream))
    assert len(bones) == 96
    for row in bones:
        array = mapping[int(row["source_frame_id"])][1]
        a, b = int(row["joint_a"]), int(row["joint_b"])
        value = math.hypot(float(array[a, 0]) - float(array[b, 0]), float(array[a, 1]) - float(array[b, 1]))
        assert math.isclose(float(row["length_px"]), value, abs_tol=1e-9)
    data = read(OUT / "review/review-data.json")
    assert [f["source_frame_id"] for f in data["frames"]] == [44, 47, 49]
    assert data["required_joint_indices"] == [7, 8, 9, 10, 11, 12]
    for frame in data["frames"]:
        assert frame["keypoints"] == mapping[frame["source_frame_id"]][1].astype(float).tolist()
        image = ROOT / frame["image_path"]
        assert base64.b64decode(frame["image"].split(",", 1)[1]) == image.read_bytes()
        assert sha(image) == frame["image_sha256"]
    before = read(QA / "old-assets-before.json")
    for name, digest in before.items():
        candidate = ROOT / name
        if name in ("docs/current-progress.md", "docs/acceptance-results.md") and sha(candidate) != digest:
            candidate = QA / "prior-contracts" / name
        assert sha(candidate) == digest, name
    protected = {p: sha(p) for folder in (OUT, QA) for p in folder.rglob("*") if p.is_file()}
    refusals = []
    for args in (["inspect"], ["package", "--assessment", str(QA / "ai-visual-selection.json")]):
        completed = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "scripts/localize_fall19_joints.py"), *args], cwd=ROOT, text=True, capture_output=True, encoding="utf-8", check=False)
        assert completed.returncode != 0 and "refuse overwrite" in completed.stderr
        refusals.append({"command": args, "returncode": completed.returncode, "stderr": completed.stderr})
    assert all(sha(p) == digest for p, digest in protected.items())
    result = {
        "independent_check": True, "passed": True, "joint_rows": 104, "bone_rows": 96,
        "selected_frames": [44, 47, 49], "original_pose_exact": True, "embedded_images_exact": True,
        "old_artifacts_or_archived_contracts": len(before), "refusal_checks": refusals,
        "protected_outputs_during_refusals": len(protected), "test_access": "none",
        "feature_inference": "not_run", "training": "not_run", "real_end_to_end": "not_run",
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "refusal_checks"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
