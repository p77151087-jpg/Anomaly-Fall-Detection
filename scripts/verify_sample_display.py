"""Real Validation replay: independent action display and byte-identical AE outputs."""

import argparse
import csv
import subprocess
from collections import Counter
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import torch

from fall_detect.assets import require_weights
from fall_detect.config import project_path
from fall_detect.data.cache import load_pose_cache
from fall_detect.io_utils import ContractError, read_json, sha256, write_json
from fall_detect.models.stgcn import HumanFeatureExtractor
from fall_detect.skeleton.preprocessing import human_preprocess
from fall_detect.training.experiment import Scorer, verify_artifacts
from fall_detect.visualization.action_status import ACTION_LABELS, SCORE_COLUMNS
from fall_detect.visualization.video import demo

ROOT = Path(__file__).resolve().parents[1]
BASE = "ur-sample-human-s1-001"
RUN = "ur-human-isotropic-001"
CASES = [(BASE, "fall-09")] + [
    (RUN, s) for s in ("adl-35", "fall-09", "fall-19", "fall-18", "fall-30")
]
REPORT = ROOT / "docs/validation/sample-display-integration.json"
BEFORE = ROOT / "docs/validation/sample-display-before.json"


def require(condition, message):
    if not condition:
        raise ContractError(message)


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def preserve():
    before = read_json(BEFORE)
    for name, hashes in before["old_runs"].items():
        require(verify_artifacts(ROOT / "runs" / name) == hashes, f"Old run changed: {name}")
    for group in ("configs", "manifests", "cache_files", "old_media"):
        for name, digest in before[group].items():
            require(sha256(ROOT / name) == digest, f"Old artifact changed: {name}")
    for name, digest in before["source_files"].items():
        saved = ROOT / "docs/validation/sample-display-legacy-source" / name
        require(sha256(saved) == digest, f"Archived source changed: {name}")
    return {
        key: len(before[key])
        for key in ("old_runs", "cache_files", "configs", "manifests", "old_media")
    }


def check_actions(scorer, sid, action_rows):
    entry = scorer.index["sequences"][sid]
    observations, _ = load_pose_cache(project_path(entry["pose_path"]), {"sequence_id": sid})
    observed = {(p.source_frame_id, p.track_id): p for _, poses in observations for p in poses}
    extractor = HumanFeatureExtractor(scorer.config, require_weights(scorer.config), scorer.device)
    extractor.model.eval()
    differences = []
    low_quality = 0
    for row in action_rows:
        require(row["label"] != "UNKNOWN", "Sample display must not emit large UNKNOWN")
        if row["state"] == "unavailable":
            require(
                not row["label"] and not row["head_score"], "Unavailable must not invent an action"
            )
            continue
        ids = [int(v) for v in row["history_frame_ids"].split()]
        require(ids == sorted(set(ids)), "History has duplicate or reordered observations")
        track = int(row["track_id"])
        poses = [observed[(i, track)] for i in ids]
        require(
            ids[-1] == int(row["source_frame_id"]) == int(row["observation_frame_id"]),
            "Wrong source frame",
        )
        start, end = poses[0].timestamp_ms, poses[-1].timestamp_ms
        require(
            float(row["history_start_ms"]) == start and float(row["history_end_ms"]) == end,
            "Wrong history interval",
        )
        require(
            float(row["available_at_ms"]) == end <= float(row["timestamp_ms"]) + 1e-7,
            "Future action leaked",
        )
        age = max(0.0, float(row["timestamp_ms"]) - end)
        require(float(row["score_age_ms"]) == age, "Wrong action freshness")
        if row["state"] == "warming_up":
            require(
                len(ids) < 30 and row["label"] == "PENDING" and not row["head_score"],
                "Invalid warmup",
            )
            continue
        require(row["state"] == "valid" and len(ids) == 30, "Classified without 30 observations")
        # Independent full upstream forward: do not use production action status or its method.
        points, motion = human_preprocess(
            np.stack([p.keypoints for p in poses]),
            poses[-1].frame_size_wh,
            window_size=30,
            normalization="per_frame",
        )
        with torch.inference_mode():
            expected = (
                extractor.model(
                    (
                        torch.from_numpy(points).to(scorer.device),
                        torch.from_numpy(motion).to(scorer.device),
                    )
                )
                .cpu()
                .numpy()[0]
            )
        actual = np.array([float(row[k]) for k in SCORE_COLUMNS])
        differences.append(float(np.max(np.abs(actual - expected))))
        index = int(expected.argmax())
        require(
            int(row["class_index"]) == index and row["label"] == ACTION_LABELS[index],
            "Wrong argmax label",
        )
        require(float(row["head_score"]) == float(expected[index]), "Wrong displayed head score")
        low_quality += row["pose_good"] == "False"
    require(differences and max(differences) == 0, "Fresh action/full cached forward differ")
    return {
        "compared_action_frames": len(differences),
        "max_action_score_difference": max(differences),
        "low_quality_classified_frames": low_quality,
        "action_counts": dict(Counter(r["label"] or "(unavailable)" for r in action_rows)),
    }


def check_video(output, render, meta):
    preview = output.with_name(output.stem + "-preview.mp4")
    if render and not preview.exists():
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-nostdin",
                "-v",
                "error",
                "-n",
                "-i",
                str(output),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(preview),
            ],
            check=True,
        )
    cap = cv2.VideoCapture(str(preview))
    count, contacts = 0, []
    times = (
        [0.3, 4.7, 5.33, 5.93]
        if output.stem == "fall-09"
        else [2.07, 2.87, 3.57, 4.27]
        if output.stem == "adl-35"
        else [0.3, 1.1, 1.6, 2.1]
    )
    wanted = {round(t * meta["fps"]) for t in times}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        require([frame.shape[1], frame.shape[0]] == meta["size_wh"], "Preview size differs")
        if count in wanted:
            contacts.append(cv2.resize(frame, (480, round(frame.shape[0] * 480 / frame.shape[1]))))
        count += 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    require(count == meta["frames"] and abs(fps - meta["fps"]) < 0.01, "Preview decode failed")
    contact = output.with_name(output.stem + "-check.jpg")
    if render and not contact.exists():
        cv2.imencode(".jpg", cv2.hconcat(contacts))[1].tofile(str(contact))
    return str(preview.relative_to(ROOT)), str(contact.relative_to(ROOT))


def main(render=False, write=False):
    require(
        not write or not REPORT.exists(), "Preserve completed integration; omit --write to recheck"
    )
    if REPORT.exists():
        previous = read_json(REPORT)
        require(previous["before_sha256"] == sha256(BEFORE), "Preservation snapshot changed")
        for name, digest in previous["source_files"].items():
            require(sha256(ROOT / name) == digest, f"Source changed: {name}")
        for case in previous["cases"]:
            for name, digest in case["files"].items():
                require(sha256(ROOT / name) == digest, f"New media changed: {name}")
    cases = []
    for name, sid in CASES:
        scorer = Scorer(ROOT / "runs" / name)
        split = next(
            r["split"]
            for r in scorer.index["split_manifest"]["sequences"]
            if r["sequence_id"] == sid
        )
        require(split == "validation", "Display verification must not add Test evaluation")
        output = ROOT / "outputs/sample-display" / name / f"{sid}.mp4"
        if render and not output.exists():
            print(f"Rendering {name}/{sid}", flush=True)
            demo(scorer.run, ROOT / "data/raw/urfall" / sid, output, overlay_mode="auto")
        meta = read_json(output.with_suffix(".json"))
        require(meta["rendering"]["overlay_mode"] == "sample", "Wrong default display")
        require(meta["action_display"]["independent_of_ae_windows"], "Not independent actions")
        require(
            project_path(meta["run"]) == scorer.run
            and meta["recipe_hash"] == scorer.index["recipe_hash"],
            "Wrong run/recipe",
        )
        old = ROOT / "outputs/human-isotropic-streaming" / name
        for suffix in (".csv", "-events.csv"):
            require(
                sha256(output.parent / (sid + suffix)) == sha256(old / (sid + suffix)),
                "AE score/state/events changed",
            )
        ae = rows(output.with_suffix(".csv"))
        events = sorted(
            rows(output.with_name(sid + "-events.csv")),
            key=lambda r: float(r["confirmed_start_ms"]),
        )
        actions = rows(output.with_name(sid + "-actions.csv"))
        display = rows(output.with_name(sid + "-display.csv"))
        for row in display:
            past = [
                e
                for e in events
                if float(e["confirmed_start_ms"]) <= float(row["timestamp_ms"]) + 1e-7
            ]
            require(int(row["history_event_count"]) == len(past), "Event history count differs")
            if past:
                require(
                    row["history_event_start_ms"] == past[-1]["confirmed_start_ms"]
                    and row["history_track_id"] == past[-1]["track_id"],
                    "History is not latest actual AE event",
                )
            else:
                require(not row["history_event_start_ms"], "Invented event history")
        details = check_actions(scorer, sid, actions)
        preview, contact = check_video(output, render, meta)
        case = {
            "run": name,
            "sequence_id": sid,
            "frames": meta["frames"],
            "ae_csv_byte_identical": True,
            "ae_events_csv_byte_identical": True,
            "ae_scored_frames": sum(bool(r["raw_score"]) for r in ae),
            "current_ae_unknown_frames": sum(r["state"] == "unknown" for r in ae),
            "first_ae_history_event_ms": float(events[0]["confirmed_start_ms"]) if events else None,
            **details,
            "preview": preview,
            "contact": contact,
            "files": {
                str(p.relative_to(ROOT)): sha256(p)
                for p in output.parent.glob(sid + "*")
                if p.is_file()
            },
        }
        cases.append(case)
        print(
            {k: v for k, v in case.items() if k not in ("files", "preview", "contact")}, flush=True
        )
    print("Verifying all preserved runs, caches, configs, manifests and old media", flush=True)
    kept = preserve()
    report = {
        "source_kind": "real_end_to_end",
        "cases": cases,
        "preserved": kept,
        "before_sha256": sha256(BEFORE),
        "new_training_epochs": 0,
        "new_test_scored": False,
        "ae_metrics_changed": False,
        "seven_action_accuracy_not_measured": True,
        "source_files": {
            p: sha256(ROOT / p)
            for p in (
                "src/fall_detect/visualization/sample_display.py",
                "src/fall_detect/visualization/video.py",
                "src/fall_detect/models/stgcn.py",
                "src/fall_detect/cli.py",
                "scripts/verify_sample_display.py",
            )
        },
    }
    if write:
        write_json(REPORT, report)
    print({"completed": True, "frames": sum(c["frames"] for c in cases), **kept}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    main(args.render, args.write)
