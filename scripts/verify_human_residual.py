"""Real streaming parity for residual profiles; diagnostic clips are not improved models."""

import argparse
import csv
import subprocess

import cv2
import imageio_ffmpeg
from human_residual import path, verify

from fall_detect.config import project_path
from fall_detect.detection.events import score_sequence
from fall_detect.io_utils import ContractError, read_json, sha256, write_json
from fall_detect.training.experiment import Scorer, cache_for, verify_artifacts
from fall_detect.training.residual import fit_residual_profile
from fall_detect.visualization.video import demo

ROOT = project_path(".")
DEST = ROOT / "outputs/human-residual-streaming"
REPORT = path("integration")


def require(value, reason):
    if not value:
        raise ContractError(reason)


def read_csv(filename):
    with filename.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main(render=False, write=False):
    lock = verify()
    if REPORT.exists():
        previous = read_json(REPORT)
        require(
            previous["validation_sha256"] == sha256(path("validation")), "Validation report changed"
        )
        require(
            previous["implementation_sha256"] == sha256(path("implementation")),
            "Locked implementation changed",
        )
        for case in previous["cases"]:
            for filename, digest in case["files"].items():
                require(sha256(ROOT / filename) == digest, "Saved streaming artifact changed")
    results = read_json(path("validation"))
    baseline = Scorer(ROOT / "runs" / lock["baseline"])
    for resource in baseline.index["weights"].values():
        require(
            sha256(project_path(resource["local_path"])) == resource["sha256"], "Weight changed"
        )
    require(not write or not REPORT.exists(), "Preserve existing verification report")
    cases = []
    for row in results["rows"]:
        run = ROOT / "runs" / row["run"]
        scorer = Scorer(run)
        require(verify_artifacts(run) == row["hashes"], "Frozen candidate changed")
        if scorer.residual_weights is not None:
            profile = read_json(run / "residual-profile.json")
            require(
                profile == fit_residual_profile(baseline, profile["mode"]),
                "Train-only profile refit differs",
            )
            for filename in (
                "autoencoder.pt",
                "scaler.npz",
                "training-history.csv",
                "dataset-index.json",
            ):
                require(
                    sha256(run / filename) == sha256(baseline.run / filename),
                    "Inherited training artifact changed",
                )
        sid = "fall-09"
        require(
            next(s for s in scorer.index["split_manifest"]["sequences"] if s["sequence_id"] == sid)[
                "split"
            ]
            == "validation",
            "Must not render Test for selection",
        )
        output = DEST / row["run"] / f"{sid}.mp4"
        if render and not output.exists():
            demo(run, ROOT / "data/raw/urfall" / sid, output, overlay_mode="auto")
        require(output.exists(), "Use --render for missing diagnostic clips")
        meta = read_json(output.with_suffix(".json"))
        require(meta["rendering"]["overlay_mode"] == "actions", "Action UI missing")
        ae_rows = read_csv(output.with_suffix(".csv"))
        cached = cache_for(scorer.index, sid)
        outcome, windows, events = score_sequence(
            cached, scorer.score(cached.features), scorer.threshold["threshold"]
        )
        by_window = {(w["sample_index"], w["track_id"]): w for w in windows}
        differences = []
        for actual in ae_rows:
            if not actual["raw_score"]:
                continue
            expected = by_window[(int(actual["sample_index"]), int(actual["track_id"]))]
            differences.append(abs(float(actual["raw_score"]) - expected["raw_score"]))
            require(actual["state"] == expected["state"], "Visible window state mismatch")
            require(
                (float(actual["event_score"]) if actual["event_score"] else None)
                == expected["event_score"],
                "K3 score mismatch",
            )
        require(differences and max(differences) == 0, "Real streaming scores differ from cache")
        event_rows = read_csv(output.with_name(f"{sid}-events.csv"))
        require(len(event_rows) == len(events), "Event count mismatch")
        for actual, expected in zip(event_rows, events):
            for key, value in expected.items():
                require(
                    actual[key] == str(value)
                    or (isinstance(value, (int, float)) and float(actual[key]) == value),
                    f"Event mismatch: {key}",
                )
        # Both score alternatives must leave the independent genuine action head unchanged.
        action_file = output.with_name(f"{sid}-actions.csv")
        reference = ROOT / "outputs/seven-actions-validation" / f"{sid}-actions.csv"
        require(sha256(action_file) == sha256(reference), "Seven-class action output changed")
        original_csv_equal = None
        if row["run"] == lock["baseline"]:
            original_csv_equal = all(
                sha256(output.parent / (sid + suffix))
                == sha256(ROOT / "outputs/seven-actions-validation" / (sid + suffix))
                for suffix in (".csv", "-events.csv")
            )
            require(original_csv_equal, "Original MSE inference changed")
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
        count, contacts = 0, []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            require([frame.shape[1], frame.shape[0]] == meta["size_wh"], "Preview size mismatch")
            if count in (0, 141, 160, 178):
                contacts.append(
                    cv2.resize(frame, (480, round(frame.shape[0] * 480 / frame.shape[1])))
                )
            count += 1
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        require(
            count == meta["frames"] and abs(fps - meta["fps"]) < 0.01, "Preview decode mismatch"
        )
        contact = output.with_name(f"{sid}-check.jpg")
        if render and not contact.exists():
            cv2.imencode(".jpg", cv2.hconcat(contacts))[1].tofile(str(contact))
        case = {
            "run": row["run"],
            "sequence_id": sid,
            "split": "validation",
            "frames": count,
            "compared_visible_scores": len(differences),
            "max_score_difference": max(differences),
            "events_equal": True,
            "action_csv_byte_identical": True,
            "original_ae_csv_byte_identical": original_csv_equal,
            "first_window_alarm_ms": outcome["first_alarm_ms"],
            "first_visible_alarm_ms": next(
                (float(r["timestamp_ms"]) for r in ae_rows if r["state"] == "suspected_fall"), None
            ),
            "visible_alert_frames": sum(r["state"] == "suspected_fall" for r in ae_rows),
            "preview": str(preview.relative_to(ROOT)),
            "contact": str(contact.relative_to(ROOT)),
            "files": {
                str(p.relative_to(ROOT)): sha256(p) for p in output.parent.iterdir() if p.is_file()
            },
        }
        cases.append(case)
        print(
            {
                k: case[k]
                for k in (
                    "run",
                    "frames",
                    "compared_visible_scores",
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
        "selected": results["selected"],
        "target_reached": results["target_reached"],
        "profiles_refitted_train_only": True,
        "train_sequences": 24,
        "train_windows": 3638,
        "old_runs_preserved": 37,
        "old_cache_files_preserved": 1165,
        "new_test_scored": False,
        "accuracy_improved": False,
        "implementation_sha256": sha256(path("implementation")),
        "validation_sha256": sha256(path("validation")),
        "verification_script_sha256": sha256(__file__),
        "test_fixture_sha256": sha256(ROOT / "tests/test_openpose18.py"),
    }
    if write:
        write_json(REPORT, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    main(args.render, args.write)
