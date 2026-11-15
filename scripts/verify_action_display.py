"""Verify preserved AE outputs and genuine seven-class display on fixed Validation clips."""

import argparse
import csv
import json
import subprocess
from collections import Counter

import cv2
import imageio_ffmpeg
import numpy as np

from fall_detect.config import project_path
from fall_detect.detection.events import score_sequence
from fall_detect.evaluation.metrics import sequence_metrics
from fall_detect.io_utils import ContractError, read_json, sha256, write_json
from fall_detect.models.stgcn import HumanFeatureExtractor
from fall_detect.training.experiment import Scorer, cache_for, verify_artifacts
from fall_detect.visualization.action_status import ACTION_LABELS, SCORE_COLUMNS
from fall_detect.visualization.video import demo

ROOT = project_path(".")
BEFORE = ROOT / "docs/validation/action-display-before.json"
DESTINATION = ROOT / "outputs/seven-actions-validation"
REPORT = ROOT / "docs/validation/action-display-integration.json"


def require(condition, message):
    if not condition:
        raise ContractError(message)


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_preserved():
    before = read_json(BEFORE)
    for run, hashes in before["old_runs"].items():
        require(verify_artifacts(ROOT / "runs" / run) == hashes, f"Frozen run changed: {run}")
    for group in ("cache_files", "configs", "manifests"):
        for filename, digest in before[group].items():
            require(sha256(ROOT / filename) == digest, f"Preserved input changed: {filename}")
    for filename, digest in before["source_files"].items():
        require(
            sha256(ROOT / "docs/validation/action-display-legacy-source" / filename) == digest,
            "Archived source changed",
        )
    for filename, digest in read_json(
        ROOT / "docs/validation/action-display-before-videos.json"
    ).items():
        require(sha256(ROOT / filename) == digest, "Before demo changed")
    return before


def main(render=False, write=False):
    before = verify_preserved()
    scorer = Scorer(ROOT / "runs" / before["run"])
    for weight in scorer.index["weights"].values():
        require(
            sha256(project_path(weight["local_path"])) == weight["sha256"], "Model weight changed"
        )
    selected = {r["sequence_id"]: r for r in scorer.index["split_manifest"]["sequences"]}
    results = []
    for sid in before["validation_ids"]:
        require(selected[sid]["split"] == "validation", "Do not use Test for display verification")
        output = DESTINATION / f"{sid}.mp4"
        if not output.exists() and render:
            demo(
                ROOT / "runs" / before["run"],
                ROOT / "data/raw/urfall" / sid,
                output,
                overlay_mode="auto",
            )
        require(output.exists(), "Run with --render to produce missing diagnostic videos")
        meta = read_json(output.with_suffix(".json"))
        require(meta["rendering"]["overlay_mode"] == "actions", "Wrong overlay")
        require(meta["recipe_hash"] == scorer.index["recipe_hash"], "Recipe differs")
        require(project_path(meta["run"]) == ROOT / "runs" / before["run"], "Wrong run")
        for suffix in (".csv", "-events.csv"):
            require(
                sha256(DESTINATION / (sid + suffix))
                == sha256(ROOT / "outputs/seven-actions-before" / (sid + suffix)),
                "AE score/state/event CSV changed",
            )
        ae_rows = rows(output.with_suffix(".csv"))
        action_rows = rows(DESTINATION / f"{sid}-actions.csv")
        require(len(ae_rows) == len(action_rows), "Action log timeline differs")
        cache = cache_for(scorer.index, sid)
        extractor = HumanFeatureExtractor(scorer.config, scorer.index["weights"], scorer.device)
        feature_rows = {
            (r.sample_index, r.track_id): r.row_index
            for r in cache.records
            if r.row_index is not None
        }
        differences = []
        for action, ae in zip(action_rows, ae_rows):
            require(
                all(
                    action[k] == ae[k]
                    for k in ("sample_index", "timestamp_ms", "source_frame_id", "track_id")
                ),
                "Action/AE identity differs",
            )
            if action["state"] != "valid":
                require(action["label"] in ("PENDING", "UNKNOWN"), "Missing data has action label")
                require(
                    not action["class_index"]
                    and not action["head_score"]
                    and all(not action[k] for k in SCORE_COLUMNS),
                    "Missing data retains class scores",
                )
                continue
            require(action["track_id"] != "", "Valid action without a track")
            key = (int(action["sample_index"]), int(action["track_id"]))
            require(key in feature_rows, "Valid action without a cached valid feature")
            index = feature_rows[key]
            expected = extractor.action_scores(cache.features[index : index + 1])[0]
            actual = np.array([float(action[k]) for k in SCORE_COLUMNS])
            differences.append(float(np.max(np.abs(actual - expected))))
            chosen = int(expected.argmax())
            require(
                int(action["class_index"]) == chosen and action["label"] == ACTION_LABELS[chosen],
                "Displayed label differs from genuine argmax",
            )
            require(
                abs(float(action["head_score"]) - expected[chosen]) <= 1e-6, "Top score differs"
            )
        require(differences and max(differences) <= 1e-6, "Fresh/cached action head mismatch")
        require(
            all(
                not a["head_score"]
                for a, ae in zip(action_rows, ae_rows)
                if ae["reason"] in ("low_confidence", "track_gap", "no_person", "stale_score")
            ),
            "Quality failure retained classification",
        )
        preview = DESTINATION / f"{sid}-preview.mp4"
        if not preview.exists() and render:
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
        require(preview.exists(), "Missing playback preview")
        cap = cv2.VideoCapture(str(preview))
        count, contacts = 0, []
        wanted = {
            round(t * meta["fps"])
            for t in (
                [0.3, 3.6, 4.8, 5.9]
                if sid == "fall-09"
                else [0.3, 1.4, 1.8, 2.7]
                if sid == "fall-29"
                else [0.3, 1.1, 1.6, 2.1]
            )
        }
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            require(
                [frame.shape[1], frame.shape[0]] == meta["size_wh"], "Playback dimensions differ"
            )
            if count in wanted:
                contacts.append(
                    cv2.resize(frame, (480, round(frame.shape[0] * 480 / frame.shape[1])))
                )
            count += 1
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        require(
            count == meta["frames"] and abs(fps - meta["fps"]) < 0.01, "Playback length differs"
        )
        contact = DESTINATION / f"{sid}-check.jpg"
        if contacts and not contact.exists() and render:
            cv2.imencode(".jpg", cv2.hconcat(contacts))[1].tofile(str(contact))
        counts = dict(Counter(r["label"] for r in action_rows))
        conflicts = [
            float(a["timestamp_ms"])
            for a, ae in zip(action_rows, ae_rows)
            if a["label"] == "FALLING" and ae["state"] == "normal"
        ]
        results.append(
            {
                "sequence_id": sid,
                "split": "validation",
                "frames": count,
                "classification_windows_compared": len(differences),
                "max_class_score_difference": max(differences),
                "ae_csv_byte_identical": True,
                "events_csv_byte_identical": True,
                "visible_action_counts": counts,
                "falling_with_ae_normal_ms": conflicts,
                "preview": str(preview.relative_to(ROOT)),
                "preview_sha256": sha256(preview),
                "raw_video_sha256": sha256(output),
                "action_csv_sha256": sha256(DESTINATION / f"{sid}-actions.csv"),
                "contact_sheet": str(contact.relative_to(ROOT)),
                "playback_fully_decoded": True,
            }
        )
        print({"case": sid, "frames": count, "labels": counts, "AE_unchanged": True}, flush=True)
    predictions = []
    for source in selected.values():
        if source["split"] == "validation":
            cache = cache_for(scorer.index, source["sequence_id"])
            outcome, _, _ = score_sequence(
                cache, scorer.score(cache.features), scorer.threshold["threshold"]
            )
            predictions.append({**outcome, "label": source["label"]})
    metrics = sequence_metrics(predictions)
    original = read_json(ROOT / "docs/validation/human-displacement-validation.json")["rows"][0][
        "validation"
    ]
    require(
        all(original[k] == v for k, v in metrics.items()), "Original Validation metrics changed"
    )
    verify_preserved()
    result = {
        "source_kind": "real_end_to_end",
        "run": before["run"],
        "cases": results,
        "frames": sum(r["frames"] for r in results),
        "classification_windows_compared": sum(
            r["classification_windows_compared"] for r in results
        ),
        "AE_validation_unchanged": metrics,
        "new_test_scored": False,
        "new_training": False,
        "seven_class_accuracy_not_measured": True,
        "old_runs_preserved": len(before["old_runs"]),
        "old_cache_files_preserved": len(before["cache_files"]),
        "source_sha256": {
            name: sha256(ROOT / name)
            for name in [
                *before["source_files"],
                "src/fall_detect/visualization/action_status.py",
                "scripts/verify_action_display.py",
            ]
        },
        "visual_review": "See docs/seven-actions.md; contact sheets require human visual review",
        "not_formal_course_film": True,
    }
    if write:
        require(not REPORT.exists(), "Preserve completed integration report")
        write_json(REPORT, result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "run",
                    "frames",
                    "classification_windows_compared",
                    "old_runs_preserved",
                    "old_cache_files_preserved",
                    "new_test_scored",
                )
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render", action="store_true", help="Produce missing demos and H264 previews"
    )
    parser.add_argument(
        "--write", action="store_true", help="Write immutable verification evidence once"
    )
    args = parser.parse_args()
    main(args.render, args.write)
