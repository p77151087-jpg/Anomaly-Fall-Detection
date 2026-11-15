"""Render Validation diagnostics and compare fresh inference with frozen cached results."""

import argparse
import csv
import subprocess

import cv2
import imageio_ffmpeg
from sample_human import RUNS, path, ranking, verify

from fall_detect.config import project_path
from fall_detect.detection.events import score_sequence
from fall_detect.io_utils import ContractError, read_json, sha256, write_json
from fall_detect.training.experiment import Scorer, cache_for, verify_artifacts
from fall_detect.visualization.video import demo


def rows(filename):
    with filename.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=RUNS)
    parser.add_argument("--case", choices=["adl-04", "fall-09", "fall-25", "fall-29", "fall-19"])
    args = parser.parse_args()
    verify()
    report = read_json(path("validation"))
    chosen = args.run or max((r for r in report["rows"] if r["run"] in RUNS), key=ranking)["run"]
    run = project_path("runs") / chosen
    destination = project_path("outputs/sample-human-validation") / chosen
    destination.mkdir(parents=True, exist_ok=True)
    evidence = path("integration-" + chosen + ("-" + args.case if args.case else ""))
    if evidence.exists():
        raise ContractError("Preserve completed integration evidence")
    scorer = Scorer(run)
    results = []
    for sid in (args.case,) if args.case else ("adl-04", "fall-09", "fall-25"):
        split = next(
            s for s in scorer.index["split_manifest"]["sequences"] if s["sequence_id"] == sid
        )
        if split["split"] != "validation":
            raise ContractError("Demo diagnostics must use Validation")
        video = destination / f"{sid}.mp4"
        if not video.exists():
            demo(run, project_path(f"data/raw/urfall/{sid}"), video, overlay_mode="simple")
        metadata = read_json(video.with_suffix(".json"))
        if (
            metadata["recipe_hash"] != scorer.index["recipe_hash"]
            or project_path(metadata["run"]) != run
        ):
            raise ContractError("Demo belongs to another recipe/run")
        cache = cache_for(scorer.index, sid)
        outcome, windows, expected_events = score_sequence(
            cache, scorer.score(cache.features), scorer.threshold["threshold"]
        )
        log = rows(video.with_suffix(".csv"))
        shown = {(int(r["sample_index"]), int(r["track_id"])): r for r in log if r["track_id"]}
        differences, masked = [], 0
        for row in windows:
            if row["raw_score"] is None:
                continue
            actual = shown.get((row["sample_index"], row["track_id"]))
            if not actual or actual["raw_score"] == "":
                masked += 1
                continue
            differences.append(abs(float(actual["raw_score"]) - row["raw_score"]))
            if actual["state"] != row["state"]:
                raise ContractError("Visible event state differs from cached inference")
        if not differences or max(differences) > 1e-6:
            raise ContractError("Cached/fresh scores differ")
        actual_events = rows(video.with_name(f"{sid}-events.csv"))
        if len(actual_events) != len(expected_events):
            raise ContractError("Event count differs")
        for actual, expected in zip(actual_events, expected_events):
            for key in ("confirmed_start_ms", "last_active_ms", "end_ms", "peak_score"):
                if abs(float(actual[key]) - expected[key]) > 1e-6:
                    raise ContractError(f"Event differs: {key}")
            if actual["end_reason"] != expected["end_reason"]:
                raise ContractError("Event end reason differs")
        if any(r["state"] == "unknown" and (r["raw_score"] or r["event_score"]) for r in log):
            raise ContractError("Unknown displays a retained score")
        preview = destination / f"{sid}-preview.mp4"
        if not preview.exists():
            subprocess.run(
                [
                    imageio_ffmpeg.get_ffmpeg_exe(),
                    "-nostdin",
                    "-v",
                    "error",
                    "-n",
                    "-i",
                    str(video),
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
        count, selected_frames = 0, []
        wanted = {
            round(t * metadata["fps"])
            for t in ((1, 3.6, 4.7, 5.5) if sid == "fall-09" else (1, 1.7, 2.1, 2.4))
        }
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if count in wanted:
                selected_frames.append(frame)
            count += 1
        cap.release()
        if count != metadata["frames"]:
            raise ContractError("H264 decoded frame count differs")
        if selected_frames:
            strip = cv2.hconcat(
                [
                    cv2.resize(f, (480, round(f.shape[0] * 480 / f.shape[1])))
                    for f in selected_frames
                ]
            )
            cv2.imencode(".jpg", strip)[1].tofile(destination / f"{sid}-check.jpg")
        results.append(
            {
                "sequence_id": sid,
                "split": "validation",
                "outcome": outcome,
                "frames": count,
                "compared_windows": len(differences),
                "masked_windows": masked,
                "max_score_difference": max(differences),
                "events_match": True,
                "states_match_where_visible": True,
                "preview": str(preview),
                "preview_sha256": sha256(preview),
                "events": expected_events,
            }
        )
        print({"sequence_id": sid, "frames": count, "compared": len(differences)}, flush=True)
    verify()
    write_json(
        evidence,
        {
            "source_kind": "real_end_to_end",
            "run": chosen,
            "adopted": chosen == report["selected"],
            "cases": results,
            "hashes": verify_artifacts(run),
            "visual_review": "pending",
            "not_formal_course_film": True,
        },
    )


if __name__ == "__main__":
    main()
