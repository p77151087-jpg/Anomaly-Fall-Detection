"""Recheck Train-only clustering, preservation and real streaming for quality experiments."""

import argparse
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np
import torch
from human_quality import ROOT, path, verify
from verify_human_residual import read_csv, require

from fall_detect.config import project_path
from fall_detect.detection.events import score_sequence
from fall_detect.io_utils import read_json, sha256, write_json
from fall_detect.training.experiment import Scorer, cache_for, normal_matrix, verify_artifacts
from fall_detect.training.patterns import pattern_sampler
from fall_detect.training.scaler import FeatureScaler
from fall_detect.visualization.video import demo

CASES = [
    ("ur-sample-human-s1-001", "fall-09"),
    ("ur-human-patterns-001", "adl-35"),
    ("ur-human-patterns-001", "fall-09"),
    ("ur-human-patterns-001", "fall-30"),
]


def main(render=False, write=False):
    lock = verify()
    report_path = path("integration")
    require(not write or not report_path.exists(), "Preserve completed integration")
    if report_path.exists():
        previous = read_json(report_path)
        require(previous["validation_sha256"] == sha256(path("validation")), "Validation changed")
        require(previous["verification_script_sha256"] == sha256(__file__), "Verifier changed")
        for c in previous["cases"]:
            for name, digest in c["files"].items():
                require(sha256(ROOT / name) == digest, "Streaming artifact changed")
    validation = read_json(path("validation"))
    for row in validation["rows"]:
        require(
            verify_artifacts(ROOT / "runs" / row["run"]) == row["hashes"],
            "Frozen candidate changed",
        )
    patterned = Scorer(ROOT / "runs/ur-human-patterns-001")
    train_x, feature_hashes, counts = normal_matrix(patterned.index, "train", return_counts=True)
    fitted = FeatureScaler.fit(train_x)
    np.testing.assert_array_equal(fitted.mean, patterned.scaler.mean)
    np.testing.assert_array_equal(fitted.scale, patterned.scaler.scale)
    _, refitted = pattern_sampler(
        fitted.transform(train_x), counts, torch.Generator().manual_seed(42), 42
    )
    sources = read_json(patterned.run / "sources.json")
    require(
        refitted == sources["training_sampling"]["normal_patterns"], "Train profile refit differs"
    )
    require(feature_hashes == sources["train_feature_hashes"], "Train feature provenance differs")
    baseline = ROOT / "runs" / lock["baseline"]
    require(
        sha256(baseline / "scaler.npz") == sha256(patterned.run / "scaler.npz"),
        "Train scaler changed",
    )
    cases = []
    for name, sid in CASES:
        scorer = Scorer(ROOT / "runs" / name)
        source = next(
            r for r in scorer.index["split_manifest"]["sequences"] if r["sequence_id"] == sid
        )
        require(source["split"] == "validation", "Do not use Test for streaming verification")
        output = ROOT / "outputs/human-quality-streaming" / name / f"{sid}.mp4"
        if render and not output.exists():
            demo(scorer.run, ROOT / "data/raw/urfall" / sid, output, overlay_mode="auto")
        require(output.exists(), "Use --render for missing clips")
        meta = read_json(output.with_suffix(".json"))
        require(meta["rendering"]["overlay_mode"] == "actions", "Seven-class display missing")
        require(project_path(meta["run"]) == scorer.run, "Wrong run")
        require(meta["recipe_hash"] == scorer.index["recipe_hash"], "Recipe differs")
        cache = cache_for(scorer.index, sid)
        _, windows, events = score_sequence(
            cache, scorer.score(cache.features), scorer.threshold["threshold"]
        )
        by_key = {(r["sample_index"], r["track_id"]): r for r in windows}
        ae = read_csv(output.with_suffix(".csv"))
        differences = []
        for r in ae:
            if not r["raw_score"]:
                continue
            expected = by_key[(int(r["sample_index"]), int(r["track_id"]))]
            differences.append(abs(float(r["raw_score"]) - expected["raw_score"]))
            require(r["state"] == expected["state"], "Visible state differs")
            require(
                (float(r["event_score"]) if r["event_score"] else None) == expected["event_score"],
                "K3 differs",
            )
        require(differences and max(differences) == 0, "Fresh/cached scores differ")
        actual_events = read_csv(output.with_name(f"{sid}-events.csv"))
        require(len(actual_events) == len(events), "Event count differs")
        for actual, expected in zip(actual_events, events):
            for key, value in expected.items():
                require(
                    actual[key] == str(value)
                    or (isinstance(value, (int, float)) and float(actual[key]) == value),
                    f"Event differs: {key}",
                )
        action_file = output.with_name(f"{sid}-actions.csv")
        old_action = ROOT / "outputs/seven-actions-validation" / f"{sid}-actions.csv"
        action_parity = None
        if old_action.exists():
            action_parity = sha256(action_file) == sha256(old_action)
            require(action_parity, "Independent action labels changed")
        original_parity = None
        if name == lock["baseline"]:
            original_parity = all(
                sha256(output.parent / (sid + suffix))
                == sha256(ROOT / "outputs/seven-actions-validation" / (sid + suffix))
                for suffix in (".csv", "-events.csv")
            )
            require(original_parity, "Legacy MSE/pose inference changed")
        preview = output.with_name(f"{sid}-preview.mp4")
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
        count = 0
        contacts = []
        times = (
            [0.3, 4.7, 5.33, 5.93]
            if sid == "fall-09"
            else [2.07, 2.87, 3.57, 4.27]
            if sid == "adl-35"
            else [0.3, 1.1, 1.6, 2.1]
        )
        wanted = {round(t * meta["fps"]) for t in times}
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            require([frame.shape[1], frame.shape[0]] == meta["size_wh"], "Preview size differs")
            if count in wanted:
                contacts.append(
                    cv2.resize(frame, (480, round(frame.shape[0] * 480 / frame.shape[1])))
                )
            count += 1
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        require(count == meta["frames"] and abs(fps - meta["fps"]) < 0.01, "Preview decode failed")
        contact = output.with_name(f"{sid}-check.jpg")
        if render and not contact.exists():
            cv2.imencode(".jpg", cv2.hconcat(contacts))[1].tofile(str(contact))
        case = {
            "run": name,
            "sequence_id": sid,
            "frames": count,
            "compared_scores": len(differences),
            "max_score_difference": max(differences),
            "events_equal": True,
            "action_csv_byte_identical": action_parity,
            "original_ae_csv_byte_identical": original_parity,
            "first_visible_alarm_ms": next(
                (float(r["timestamp_ms"]) for r in ae if r["state"] == "suspected_fall"), None
            ),
            "visible_alarm_frames": sum(r["state"] == "suspected_fall" for r in ae),
            "preview": str(preview.relative_to(ROOT)),
            "contact": str(contact.relative_to(ROOT)),
            "files": {
                str(p.relative_to(ROOT)): sha256(p)
                for p in output.parent.glob(sid + "*")
                if p.is_file()
            },
        }
        cases.append(case)
        print(
            {
                k: case[k]
                for k in (
                    "run",
                    "sequence_id",
                    "frames",
                    "compared_scores",
                    "max_score_difference",
                    "first_visible_alarm_ms",
                )
            },
            flush=True,
        )
    verify()
    report = {
        "source_kind": "real_end_to_end",
        "cases": cases,
        "train_only_scaler_and_profile_refit": True,
        "train_sequences": len(counts),
        "train_windows": len(train_x),
        "selected": validation["selected"],
        "target_reached": validation["target_reached"],
        "new_test_scored": False,
        "old_runs_preserved": 39,
        "old_cache_files_preserved": 1165,
        "validation_sha256": sha256(path("validation")),
        "verification_script_sha256": sha256(__file__),
        "pilot_sha256": sha256(path("pilot")),
        "implementation_sha256": sha256(path("implementation")),
    }
    if write:
        write_json(report_path, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    main(args.render, args.write)
