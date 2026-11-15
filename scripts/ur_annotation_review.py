"""AI visual, sampled annotations kept separate from automated pose diagnostics."""

import argparse
from collections import Counter
from pathlib import Path

from fall_detect.config import load_config
from fall_detect.data.cache import load_pose_cache
from fall_detect.io_utils import ContractError, read_json, sha256, write_csv, write_json
from fall_detect.skeleton.preprocessing import good_frame

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "docs/annotations/ur-train-validation-20261012-v2"
REGIONS = ("head", "shoulders", "elbows", "wrists", "hips", "knees", "ankles")
# Four visually selected source IDs: last pre-descent, first descent,
# last descent/roll, first settled floor posture. Bounds are uncertain.
PHASES = {
    "fall-22": (16, 21, 31, 36),
    "fall-19": (37, 46, 55, 64),
    "fall-16": (26, 30, 45, 50),
    "fall-18": (24, 30, 42, 48),
    "fall-08": (34, 42, 75, 83),
    "fall-13": (47, 54, 70, 77),
    "fall-29": (54, 63, 72, 81),
    "fall-14": (28, 34, 39, 45),
    "fall-09": (118, 135, 168, 185),
    "fall-07": (86, 100, 142, 156),
    "fall-11": (48, 60, 107, 118),
    "fall-25": (47, 54, 70, 77),
    "fall-06": (28, 37, 64, 73),
    "fall-02": (51, 60, 90, 100),
    "fall-30": (32, 39, 51, 57),
}
# Region codes are manually entered from the raw contact sheets reviewed by AI.
# They are NOT inferred from AlphaPose confidence, and NOT left/right coordinates.
VISIBILITY = {
    "adl-33": "VVPPVVV VVPPVVV VVPPVPC VPUCPCC VVPPPPC",
    "adl-08": "NNNNNNN VPPPPPP VPPPPPP VVPPVVV VVPPVVV",
    "adl-40": "NNNNNNN VPPPPPP VVPPVVV VPUUVPP VPUUVPP",
    "adl-19": "UUUUUUU UUUUUUU UUUUUUU UUUUUUU UUUUUUU",
    "adl-05": "VUUUPPP VVPPVVV VPPPPPP VPPPPPP VPPPPPP",
    "adl-38": "NNNNNNN VVPPVVV VPUUVPP VPUUVPC VPUUVPC",
    "adl-17": "UUUUUUU UUUUUUU UUUUUUU UUUUUUU UUUUUUU",
    "adl-06": "NNNNNNN VVPPVVV VPPPPPP VPPPPPP VVPPVVV",
    "adl-26": "VVPPVVV VVPPVVV VPUUVPP VVPPVVV VVPPVVV",
    "adl-29": "VPUUVPP VVPPVVV VPUUVPP VVPPVVV VVPPVVV",
    "adl-22": "UUUUUUU UUUUUUU UUUUUUU UUUUUUU UUUUUUU",
    "adl-21": "UUUUUUU UUUUUUU UUUUUUU UUUUUUU UUUUUUU",
    "adl-36": "NNNNNNN VVPPVVV VVPPVVV VPPPPPP VPPPPPP",
    "adl-25": "VVPPVVV VPPPPPP VVPPVVV VVPPVVV VVPPVVV",
    "adl-11": "NNNNNNN VVPPVVV VPPPPPP VPPPPPP VPPPPPP",
    "adl-28": "VVPPVVV VVPPVVV VPUUVPP VPUUVPP VPUUVPP",
    "adl-32": "VVPPVVV VVPPVVV VPUUVPC VPUCPCC VVPPPPC",
    "adl-31": "VVPPVVV VVPPVVV VPUUVPC VPUCPCC VVPPPPC",
    "adl-27": "VPPPPPP VPPPPPP VVPPVVV VVPPVVV VVPPVVV",
    "adl-07": "VUUUPPP VVPPVVV VPPPPPP VVPPVVV VVPPVVV",
    "adl-30": "NNNNNNN VVPPVVV VPUUVPP VPUUVPC VPUUVPC",
    "adl-12": "UUUUUUU UUUUUUU UUUUUUU UUUUUUU UUUUUUU",
    "adl-24": "VVPPVVV VPUUVPP VPUUVPP VVPPVVV VVPPVVV",
    "adl-10": "NNNNNNN VVPPVVV VPPPPPP VPPPPPP VPPPPPP",
    "adl-37": "NNNNNNN VVPPVVV VVPPVVV VPPPPPP VPPPPPP",
    "adl-23": "UUUUUUU UUUUUUU UUUUUUU UUUUUUU UUUUUUU",
    "adl-20": "UUUUUUU UUUUUUU UUUUUUU UUUUUUU UUUUUUU",
    "adl-04": "VUUUPPP VVPPVVV VPPPPPP VPPPPPP VPPPPPP",
    "adl-01": "NNNNNNN VVPPVVV VPPPPPP VPPPPPP VPUUVPP",
    "adl-16": "UUUUUUU UUUUUUU UUUUUUU UUUUUUU UUUUUUU",
    "adl-18": "UUUUUUU UUUUUUU UUUUUUU UUUUUUU UUUUUUU",
    "adl-35": "VVPPVVV VPUUVPC VPUCPCC VPUUVPC VVPPVVV",
    "fall-22": "VPPPPPP VPPPPPP VPPPPPP VPPPPPP VPPPPPP VPUUPPP VPUCPPP CCCCPPP CCPPPVV CCPPPVV CCPPPVV CCPPPVV",
    "fall-19": "VVPPVVV VVPPVVV VVPPVVV VVPPVVV VVPPVVV VPUUVPP CPCCVPP CPCCPPP CPCCPPP CPCCPPP CPCCPPP CPCCPPP",
    "fall-16": "VPPPPPP VPPPPPP VPPPPPP VPPPPPP VPPPPPP VPPPPPP VPPPPPP VPUUPPP VPUUPPP VPPPPPP VPPPPPP VPPPPPP",
    "fall-18": "VPPPPPP VPPPPPP VPPPPPP VPPPPPP VPPPPPP VPPPPPP VPUUPPP VPUCPPP CCPPVVV CCPPVVV CCPPVVV CCPPVVV",
    "fall-08": "VPPPPPP VPPPPPP VPPPPPP VPPPPPP VPPPPPP VPUUPPP VPUUPPP VPPPPPP VPUUPPP VPPPPPP VPPPPPP VPPPPPP",
    "fall-13": "VVPPVVV VVPPVVV VVPPVVV VVPPVVV VVPPVVV VVPPVVV VVPPVVV VPUUVPP VPUUVPP VPUCPPP CCPPPVV CCCPPVV",
    "fall-29": "VVPPVVV VVPPVVV VVPPVVV VVPPVVV VVPPVVV VVPPVVV VVPPVVV VPUUVPP VPUUVPP CCCCPVV CCCCPVV CCCCPVV",
    "fall-14": "VPPPPPP VPPPPPP VPPPPPP VPPPPPP VPPPPPP VPPPPPP VPUUPPP VPUUPPP VPPPPPP VPPPPPP VPPPPPP VPPPPPP",
    "fall-09": "VVPPVVV VVPPVVV VVPPVVV VPPPPPP VVPPVVV VVPPVVV VPPPPPP VPPPPPP VPUUPPP VPUCPPP CCPPPVV CCPPPVV",
    "fall-07": "VPPPPPP VVPPVVV VVPPVVV VVPPVVV VVPPVVV VPPPPPP VPPPPPP VPUUPPP VPUUPPP VPUUPPP VPPPPPP CCPPPVV",
    "fall-11": "VVPPVVV VVPPVVV VVPPVVV VVPPVVV VVPPVVV VPUUPPP VPUUPPP VPUUPPP VPUUPPP VPPPPPP VPPPPPP VPPPPPP",
    "fall-25": "VVPPVVV VVPPVVV VVPPVVV VVPPVVV VVPPVVV VVPPVVV VVPPVVV VPUUVPP VPUUVPP VPUCPPP CCCCPVV CCCCPVV",
    "fall-06": "VPPPPPP VPPPPPP VPPPPPP VPPPPPP VPUUPPP VPUUPPP VPUUPPP VPUUPPP VPPPPPP VPPPPPP VPPPPPP VPPPPPP",
    "fall-02": "VPPPPPP VPPPPPP VPPPPPP VPPPPPP VPPPPPP VPPPPPP VPUUPPP VPUUPPP VPUUPPP VPPPPPP VPPPPPP VPPPPPP",
    "fall-30": "VPPPPPP VPPPPPP VPPPPPP VPPPPPP VPPPPPP VPPPPPP VPPPPPP VPUUPPP VPUUPPP CCCCPVV CCCCPVV CCCCPVV",
}


def interval_from_frames(frames, bounds):
    by_id = {f["source_frame_id"]: f["timestamp_ms"] for f in frames}
    if (
        len(bounds) != 4
        or list(bounds) != sorted(set(bounds))
        or any(b not in by_id for b in bounds)
    ):
        raise ContractError("Phase boundaries must be four increasing observed frame IDs")
    times = [by_id[b] for b in bounds]
    return {
        "source_frame_ids": list(bounds),
        "onset_uncertainty_ms": times[:2],
        "settled_uncertainty_ms": times[2:],
        "core_descent_ms": [times[1], times[2]],
        "possible_descent_ms": [times[0], times[3]],
    }


def build():
    if DEST.exists():
        raise ContractError("Preserve completed annotation version")
    index = read_json(ROOT / "runs/ur-human-isotropic-001/dataset-index.json")
    cfg = load_config(ROOT / "configs/ur-human-isotropic-001.yaml")
    review = read_json(ROOT / "outputs/annotation-review-20261012/review-frames.json")
    sources = [
        s for s in index["split_manifest"]["sequences"] if s["split"] in ("train", "validation")
    ]
    if set(VISIBILITY) != {s["sequence_id"] for s in sources}:
        raise ContractError("Manual review must cover exactly Train/Validation, never Test")
    samples, phases, automatic, summary = [], [], [], []
    for source in sources:
        sid = source["sequence_id"]
        selected = [r for r in review["frames"] if r["sequence_id"] == sid]
        codes = VISIBILITY[sid].split()
        if len(codes) != len(selected) or any(len(c) != 7 or set(c) - set("VPOCUN") for c in codes):
            raise ContractError("Manual region visibility count/code mismatch")
        for row, code in zip(selected, codes, strict=True):
            samples.append(
                {
                    **row,
                    "image_sha256": sha256(ROOT / row["path"]),
                    "annotation_method": "AI visual review of original contact sheets",
                    "human_reviewed": False,
                    **dict(zip(REGIONS, code, strict=True)),
                }
            )
        entry = index["sequences"][sid]
        observations, meta = load_pose_cache(
            ROOT / entry["pose_path"], {"sequence_id": sid, "split": source["split"]}
        )
        if source["label"]:
            phases.append(
                {
                    "sequence_id": sid,
                    "split": source["split"],
                    "label": 1,
                    **interval_from_frames(meta["frames"], PHASES[sid]),
                    "annotation_method": "AI visual sampled onset/descent/settling brackets",
                    "human_reviewed": False,
                }
            )
        else:
            phases.append(
                {
                    "sequence_id": sid,
                    "split": source["split"],
                    "label": 0,
                    "onset_uncertainty_ms": None,
                    "settled_uncertainty_ms": None,
                    "core_descent_ms": None,
                    "possible_descent_ms": None,
                    "annotation_method": "source ADL label; sampled visual normal-activity check",
                    "human_reviewed": False,
                }
            )
        good = 0
        for f, poses in observations:
            per_frame_good = False
            for p in poses:
                per_frame_good |= good_frame(p.keypoints, cfg["quality"], 13)
                for joint, (x, y, confidence) in enumerate(p.keypoints):
                    automatic.append(
                        {
                            "sequence_id": sid,
                            "split": source["split"],
                            "source_frame_id": f.source_frame_id,
                            "timestamp_ms": f.timestamp_ms,
                            "track_id": p.track_id,
                            "human13_joint_index": joint,
                            "confidence": float(confidence),
                            "confidence_ge_0_2": bool(confidence >= 0.2),
                            "predicted_inside_image": bool(0 <= x < f.width and 0 <= y < f.height),
                            "annotation_method": "model diagnostic; NOT visual visibility ground truth",
                        }
                    )
            good += int(per_frame_good)
        summary.append(
            {
                "sequence_id": sid,
                "split": source["split"],
                "label": source["label"],
                "original_frames": len(observations),
                "sampled_visual_frames": len(selected),
                "any_track_good_pose_frames": good,
                "any_track_good_pose_ratio": good / len(observations),
                "visual_region_counts": dict(Counter("".join(codes))),
            }
        )
    DEST.mkdir(parents=True)
    write_csv(DEST / "sampled-joint-regions.csv", samples)
    write_csv(DEST / "automatic-human13-diagnostics.csv", automatic)
    write_json(
        DEST / "fall-intervals.json",
        {
            "sequences": phases,
            "test_annotations": 0,
            "official_ground_truth": False,
            "sampling_uncertainty_preserved": True,
        },
    )
    write_json(
        DEST / "manifest.json",
        {
            "schema_version": 1,
            "annotator": "Codex AI visual reviewer",
            "human_validated": False,
            "source_index_sha256": sha256(ROOT / "runs/ur-human-isotropic-001/dataset-index.json"),
            "review_frames_sha256": sha256(
                ROOT / "outputs/annotation-review-20261012/review-frames.json"
            ),
            "implementation_sha256": sha256(__file__),
            "split_hash": index["split_manifest"]["split_hash"],
            "training_fall_count": 0,
            "validation_fall_count": 15,
            "sampled_frames": len(samples),
            "automatic_joint_rows": len(automatic),
            "test_reads": 0,
            "regions": list(REGIONS),
            "region_codes": {
                "V": "region visible (not a pixel coordinate or both-side point guarantee)",
                "P": "partly visible / self occlusion or one-sided ambiguity",
                "O": "object occlusion",
                "C": "region substantially cropped out of frame",
                "U": "not reliably resolvable from blur/darkness/small image",
                "N": "target person not visible in sampled image",
            },
            "limitations": [
                "340 sampled images, not all original frames",
                "paired regions do not assign anatomical left/right",
                "not coordinate ground truth; confidence values are separate model diagnostics",
                "reviewer has prior knowledge of model errors; not an independent blind annotation",
                "new timing diagnostic does not rewrite earlier adoption guards or certify alert delay",
            ],
            "summary": summary,
            "files": {p.name: sha256(p) for p in DEST.iterdir() if p.is_file()},
        },
    )
    print(
        {
            "annotated_sequences": len(sources),
            "sampled_frames": len(samples),
            "validation_falls": 15,
            "training_falls": 0,
            "automatic_joint_rows": len(automatic),
            "test_reads": 0,
        }
    )


def status():
    manifest = read_json(DEST / "manifest.json")
    for name, digest in manifest["files"].items():
        if sha256(DEST / name) != digest:
            raise ContractError("Preserved annotation changed")
    print(
        {
            "sampled_frames": manifest["sampled_frames"],
            "human_validated": False,
            "test_annotations": 0,
        }
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("build", "status"))
    globals()[p.parse_args().stage]()
