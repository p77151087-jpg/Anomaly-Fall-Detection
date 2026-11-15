"""Render the fall-19 comparison from the frozen inputs and verified experiment outputs."""

import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from fall19_shoulder_descent import (
    OUT,
    ROOT,
    TARGET,
    VAL,
    WIDE,
    normalize_wide,
    require,
    save_json,
    sha,
)

from fall_detect.config import load_config
from fall_detect.data.cache import load_pose_cache
from fall_detect.skeleton.windows import CausalResampler
from fall_detect.visualization.cached_replay import CachedDisplayReplay, parse_cached_actions
from fall_detect.visualization.sample_display import EventHistory, ae_hint, draw_sample

DEST = OUT / "visual-review"
REVIEW = ROOT / "outputs/ur-fall19-review-20261012-001"
ACTIONS = ROOT / "outputs/sample-display/ur-human-isotropic-001/fall-19-actions.csv"
WIDTH, HEIGHT, TOP = 1280, 720, 44


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def table(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def group(rows):
    by_sample = defaultdict(list)
    for row in rows:
        by_sample[row["sample_index"]].append(row)
    return by_sample


def active(states):
    return sorted(track for track, state in states.items() if state["state"] == "suspected_fall")


def format_score(value):
    return "unavailable" if value is None else f"{value:.6f}"


def add_panel(canvas, image, x, title, label, detail):
    canvas[TOP : TOP + image.shape[0], x : x + 640] = image
    ink = (232, 240, 249)
    cv2.putText(canvas, title, (x + 15, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.78, ink, 2, cv2.LINE_AA)
    cv2.putText(canvas, label, (x + 15, 667), cv2.FONT_HERSHEY_SIMPLEX, 0.59, ink, 1, cv2.LINE_AA)
    cv2.putText(canvas, detail, (x + 15, 696), cv2.FONT_HERSHEY_SIMPLEX, 0.47, ink, 1, cv2.LINE_AA)


def build():
    if DEST.exists():
        raise FileExistsError(f"Preserve existing video review: {DEST}")
    lock = read(OUT / "lock.json")
    source = lock["source_sha256"]
    artifacts = read(OUT / "artifact-manifest.json")["files"]
    for name in ("summary.json", "branch-windows.csv", "validation-events.csv", "display-grid.csv"):
        require(sha(OUT / name) == artifacts[name], f"Experiment output changed: {name}")
    for path in (VAL / "windows.csv", WIDE / "resolved-config.yaml"):
        require(
            sha(path) == source[path.relative_to(ROOT).as_posix()], "Frozen display input changed"
        )
    summary = read(OUT / "summary.json")
    entry = lock["allowed_sequences"][TARGET]
    require(
        entry["split"] == "validation" and summary["accepted_on_this_validation"],
        "No verified Validation rescue to render",
    )
    pose_path = ROOT / entry["pose_path"]
    for path in (pose_path, pose_path.with_suffix(".json")):
        require(sha(path) == source[path.relative_to(ROOT).as_posix()], "Original pose changed")
    poses, _ = load_pose_cache(
        pose_path,
        {
            "sequence_id": TARGET,
            "split": "validation",
            "source_hash": entry["source_hash"],
            "recipe_hash": lock["recipe"]["wide_recipe_hash"],
            "split_hash": lock["recipe"]["split_hash"],
        },
    )
    config = load_config(WIDE / "resolved-config.yaml")
    resampler = CausalResampler(30, config["skeleton"]["resampling_hold_policy"])
    samples = {
        s.sample_index: s
        for frame, observations in poses
        for s in resampler.push(frame, observations)
    }
    require(len(samples) == entry["n_samples"] == 100, "Unexpected source grid")
    action_meta = ACTIONS.with_name("fall-19.json")
    original_action = read(action_meta)
    require(
        original_action["recipe_hash"] == lock["recipe"]["wide_recipe_hash"]
        and original_action["source_kind"] == "real",
        "Unexpected cached classifier source",
    )
    actions = parse_cached_actions(table(ACTIONS), samples, poses, config)
    old = group(
        normalize_wide([r for r in table(VAL / "windows.csv") if r["sequence_id"] == TARGET])[
            TARGET
        ]
    )
    branch = group(
        normalize_wide(
            [
                {**r, "window_id": ""}
                for r in table(OUT / "branch-windows.csv")
                if r["sequence_id"] == TARGET
            ]
        )[TARGET]
    )
    events = [r for r in table(OUT / "validation-events.csv") if r["sequence_id"] == TARGET]
    grid = {
        int(r["sample_index"]): r
        for r in table(OUT / "display-grid.csv")
        if r["sequence_id"] == TARGET
    }
    require(len(events) == 1 and set(samples) == set(grid), "Unexpected event or display grid")
    first = summary["fall19"]["first_current_alert"]["sample_index"]
    review_manifest = REVIEW / "source-manifest.json"
    review = read(review_manifest)
    require(
        review["sequence_id"] == TARGET
        and review["split"] == "validation"
        and review["source_hash"] == entry["source_hash"],
        "Unexpected RGB source",
    )
    images = {}
    source_times = {frame.source_frame_id: frame.timestamp_ms for frame, _ in poses}
    pins = {
        p.relative_to(ROOT).as_posix(): sha(p)
        for p in (
            Path(__file__),
            ACTIONS,
            action_meta,
            review_manifest,
            ROOT / "src/fall_detect/visualization/sample_display.py",
            ROOT / "src/fall_detect/visualization/cached_replay.py",
            ROOT / "src/fall_detect/visualization/video.py",
        )
    }
    for row in review["frames"]:
        fid = row["source_frame_id"]
        path = REVIEW / "raw" / f"{fid:03}.png"
        digest = sha(path)
        require(
            digest == row["source_sha256"] and row["timestamp_ms"] == source_times[fid],
            "Archived RGB bytes or time differ",
        )
        pins[path.relative_to(ROOT).as_posix()] = digest
        images[fid] = cv2.imdecode(
            np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR
        )
    require(
        len(images) == 100 and all(im.shape == (480, 640, 3) for im in images.values()),
        "Unexpected RGB dimensions",
    )

    DEST.mkdir(exist_ok=False)
    video = DEST / "fall-19-wide-vs-shoulder.mp4"
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-n",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{WIDTH}x{HEIGHT}",
        "-framerate",
        "30",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-threads",
        "2",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(video),
    ]
    tau, cue_tau = summary["original_wide_threshold"], summary["normal_threshold"]
    old_state, shoulder_state = (
        CachedDisplayReplay(config, tau),
        CachedDisplayReplay(config, cue_tau),
    )
    right_history = EventHistory()
    expected_headers, verified_alert_frames = [], []
    with subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    ) as encoder:
        for index, sample in samples.items():
            old_windows, cue_windows = old.get(index, []), branch.get(index, [])
            _, left = old_state.step(sample, old_windows)
            _, cue = shoulder_state.step(sample, cue_windows)
            right = {**left, **{t: cue[t] for t in active(cue)}}
            stored = grid[index]
            require(
                active(left) == json.loads(stored["wide_current_alert_tracks"])
                and active(right) == json.loads(stored["combined_current_alert_tracks"]),
                "Rendered current alerts differ from verified CSV",
            )
            for event in events:
                if float(event["confirmed_start_ms"]) == sample.timestamp_ms:
                    right_history.observe(
                        int(event["track_id"]), sample.timestamp_ms, {"state": "suspected_fall"}
                    )
            require(
                right_history.count == int(stored["combined_history_count"]),
                "Rendered history differs from CSV",
            )
            if active(right):
                require(
                    ae_hint(right[1]) == "Current: alert" and 1 in actions[index],
                    "Native view lacks the current alert",
                )
                verified_alert_frames.append(index)
            source_image = images[sample.source_frame_id]
            left_image = draw_sample(
                source_image,
                sample.poses,
                actions[index],
                left,
                sample.timestamp_ms,
                old_state.history,
                config["quality"],
            )
            right_image = draw_sample(
                source_image,
                sample.poses,
                actions[index],
                right,
                sample.timestamp_ms,
                right_history,
                config["quality"],
            )
            output = np.full((HEIGHT, WIDTH, 3), (36, 26, 19), dtype=np.uint8)
            baseline_score = old_windows[0]["event_score"] if old_windows else None
            shoulder_score = cue_windows[0]["event_score"] if cue_windows else None
            add_panel(
                output,
                left_image,
                0,
                "fall-19 | Original wide",
                f"Wide K3: {format_score(baseline_score)} / {tau:.6f}",
                f"Grid {sample.timestamp_ms / 1000:.3f}s | original RGB frame {sample.source_frame_id}",
            )
            add_panel(
                output,
                right_image,
                640,
                "fall-19 | Wide + shoulder descent",
                f"Shoulder K3: {format_score(shoulder_score)} / {cue_tau:.6f}",
                "Independent K3 cue | original track and quality gates",
            )
            expected_headers.append(output[TOP + 25 : TOP + 106, 640 + 330 : 640 + 606].copy())
            encoder.stdin.write(output.tobytes())
        encoder.stdin.close()
        error = encoder.stderr.read().decode("utf-8", "replace")
        require(encoder.wait(timeout=30) == 0, f"FFmpeg encoding failed: {error}")

    cap = cv2.VideoCapture(str(video))
    require(
        cap.isOpened() and cap.get(cv2.CAP_PROP_FPS) == 30,
        "Encoded video could not be decoded at 30 fps",
    )
    count, max_mae = 0, 0.0
    first_frame = None
    while True:
        okay, frame = cap.read()
        if not okay:
            break
        require(
            frame.shape == (HEIGHT, WIDTH, 3) and count < len(expected_headers),
            "Wrong decoded frame shape or count",
        )
        header = frame[TOP + 25 : TOP + 106, 640 + 330 : 640 + 606]
        max_mae = max(
            max_mae, float(np.abs(header.astype(np.int16) - expected_headers[count]).mean())
        )
        red = header[:32].astype(np.int16)
        red_count = np.count_nonzero(
            (red[:, :, 2] - red[:, :, 1] > 55) & (red[:, :, 2] - red[:, :, 0] > 55)
        )
        require(
            (red_count > 35) == (int(grid[count]["combined_history_count"]) > 0),
            "Encoded history badge differs from CSV",
        )
        if count == first:
            first_frame = frame.copy()
        count += 1
    cap.release()
    require(
        count == 100 and max_mae < 5.0 and first_frame is not None, "Decoded verification failed"
    )
    okay, png = cv2.imencode(".png", first_frame)
    require(okay, "Could not encode first-alert evidence")
    still = DEST / "first-visible-alert.png"
    with still.open("xb") as stream:
        stream.write(png.tobytes())
    for path, digest in pins.items():
        require(sha(ROOT / path) == digest, f"Original visual source changed: {path}")
    save_json(
        DEST / "verification.json",
        {
            "decoded_frames": count,
            "fps": 30,
            "width": WIDTH,
            "height": HEIGHT,
            "first_visible_alert_sample_index": first,
            "current_alert_sample_indices": verified_alert_frames,
            "max_decoded_native_header_mean_absolute_error": max_mae,
            "history_badge_pixel_checks": count,
            "source_sha256": pins,
            "video_sha256": sha(video),
            "first_alert_png_sha256": sha(still),
            "ffmpeg_command": command,
            "evidence_kind": "existing RGB and pose to real encoded replay",
        },
    )
    print(
        f"Verified {count} encoded frames, first alert index {first}, current alert count {len(verified_alert_frames)}"
    )


if __name__ == "__main__":
    build()
