"""Render cached Validation scores with the unchanged sample renderer; no model calls."""

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import validation_alert_quality as prior
import yaml
from PIL import Image, ImageDraw, ImageFont

from fall_detect.data.cache import load_pose_cache
from fall_detect.evaluation.human_timing import frame_time_bounds, temporal_relation
from fall_detect.io_utils import ContractError
from fall_detect.skeleton.preprocessing import good_frame
from fall_detect.skeleton.windows import CausalResampler
from fall_detect.visualization.cached_replay import CachedDisplayReplay, parse_cached_actions
from fall_detect.visualization.sample_display import HEADER, draw_sample, unavailable

ROOT = Path(__file__).resolve().parents[1]
TAG = "ur-validation-display-20261105-001"
OUT = ROOT / "outputs" / TAG
QA = ROOT / "docs/validation" / TAG
HUMAN = ROOT / "outputs/ur-validation-human-timing-20261105-001"
HUMAN_QA = ROOT / "docs/validation/ur-validation-human-timing-20261105-001"
REVIEW = ROOT / "outputs/ur-validation-alert-quality-20261013-001"
ACTIONS = ROOT / "outputs/sample-display/ur-human-isotropic-001"
SELECTED = ("fall-18", "fall-09", "fall-30")
MODELS = {
    "demo": ("ur-sample-human-s1-001", "outputs/human-isotropic-validation/ur-sample-human-s1-001"),
    "wide": (
        "ur-isotropic-wide-20261012-001",
        "outputs/isotropic-refinement-20261012/ur-isotropic-wide-20261012-001",
    ),
}
MUTABLE = {"docs/current-progress.md", "docs/acceptance-results.md"}
TITLE, FOOTER, FPS = 64, 160, 30
FONT = Path("C:/Windows/Fonts/msjh.ttc")
read, save, sha = prior.read, prior.save, prior.digest


def normalize(name):
    return str(name).replace("\\", "/")


def inventory():
    result = {}
    for name, digest in read(HUMAN_QA / "preservation.json")["files"].items():
        name = normalize(name)
        path = HUMAN_QA / "prior-contracts" / name if name in MUTABLE else ROOT / name
        result[path.relative_to(ROOT).as_posix()] = digest
    result.update(
        {normalize(k): v for k, v in read(HUMAN_QA / "final-qa.json")["artifact_sha256"].items()}
    )
    result.update(
        {
            (HUMAN / name).relative_to(ROOT).as_posix(): digest
            for name, digest in read(HUMAN / "artifact-manifest.json")["files"].items()
        }
    )
    result.update({normalize(k): v for k, v in read(HUMAN_QA / "inputs.json")["files"].items()})
    result[(HUMAN_QA / "final-qa.json").relative_to(ROOT).as_posix()] = sha(
        HUMAN_QA / "final-qa.json"
    )
    return result


def check_files(files, archived=False):
    for name, digest in files.items():
        name = normalize(name)
        path = QA / "prior-contracts" / name if archived and name in MUTABLE else ROOT / name
        if sha(path) != digest:
            raise ContractError(f"Preserved source changed: {name}")


class H264Output:
    def __init__(self, path, size):
        self.path, self.size, self.count = Path(path), size, 0
        if self.path.exists():
            raise ContractError("Preserve existing encoded video")
        self.command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-n",
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgr24",
            "-video_size",
            f"{size[0]}x{size[1]}",
            "-framerate",
            str(FPS),
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
            str(path),
        ]
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    def write(self, image):
        if image.shape != (self.size[1], self.size[0], 3) or image.dtype != np.uint8:
            raise ContractError("Video dimensions/type changed")
        self.process.stdin.write(np.ascontiguousarray(image).tobytes())
        self.count += 1

    def close(self):
        self.process.stdin.close()
        status = self.process.wait(timeout=30)
        error = self.process.stderr.read().decode("utf8", "replace")
        if status or self.count == 0 or not self.path.exists() or self.path.stat().st_size == 0:
            raise ContractError(f"H264 encoding failed: {error}")
        return {
            "file": self.path.relative_to(OUT).as_posix(),
            "frames": self.count,
            "fps": FPS,
            "size_wh": list(self.size),
            "codec": "h264/yuv420p",
            "command": self.command,
            "ffmpeg_stderr": error,
        }


def headline(actions):
    return (
        max(
            actions,
            key=lambda k: (
                {"FALLING": 3, "LYING": 2}.get(actions[k]["label"], 0),
                actions[k]["state"] == "valid",
                -k,
            ),
        )
        if actions
        else None
    )


@lru_cache(maxsize=4)
def diagnostic_font(size):
    return ImageFont.truetype(str(FONT), size)


def audit_canvas(native, row, reviewed, timing, model, run):
    height, width = native.shape[:2]
    canvas = np.full((TITLE + height + FOOTER, width, 3), (26, 21, 17), dtype=np.uint8)
    canvas[TITLE : TITLE + height] = native
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    f22, f16 = diagnostic_font(22), diagnostic_font(16)
    title = "預設模型 demo" if model == "demo" else "wide 參照模型"
    draw.text((10, 4), f"{title}｜影片判定 {row['video_outcome']}", font=f22, fill=(231, 237, 243))
    draw.text(
        (10, 34), f"{run}｜固定門檻 {row['threshold']:.6f}／K3", font=f16, fill=(184, 199, 211)
    )
    base = TITLE + height + 5
    labels = {
        "unknown": "無法判定",
        "normal": "正常",
        "suspected_fall": "警示",
        "warming_up": "收集中",
        "no_window_yet": "尚無窗口",
    }
    lines = [
        f"格點 {row['timestamp_ms'] / 1000:.6f}s｜原幀 {row['source_frame_id']} @ {row['source_timestamp_ms'] / 1000:.3f}s",
        f"CSV事件：{labels[row['csv_state']]}｜當前AE：{labels[row['current_ae_state']]}｜歷史事件：{row['history_count']}",
    ]
    n = lambda v: "—" if v is None else f"{v:.5f}"
    lines.append(
        f"CSV原分數 {n(row['csv_raw_score'])}／K3 {n(row['csv_event_score'])}（非機率；非當前可用分數）"
    )
    lines.append(
        f"本人開始 {timing['onset'][0] / 1000:.3f}–{timing['onset'][1] / 1000:.3f}｜落地 {timing['impact'][0] / 1000:.3f}–{timing['impact'][1] / 1000:.3f}s"
    )
    relation = {
        "before_onset": "開始之前",
        "onset_uncertainty": "開始時刻不確定範圍",
        "guaranteed_fall_interval": "兩界限都涵蓋區間",
        "impact_uncertainty": "落地時刻不確定範圍",
        "after_impact": "最晚落地之後",
    }
    lines.append(
        f"{relation.get(row['human_relation'], '時間未知')}｜本人骨架：大多符合／備註部分出框"
    )
    lines.append("左側 ACTION 為七類對照；右側 AE 歷史紀錄不等於當前警示")
    for i, text in enumerate(lines):
        draw.text((10, base + 24 * i), text, font=f16, fill=(203, 213, 221))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def load_inputs():
    models, inputs = {}, {}
    for label, (name, formal) in MODELS.items():
        run = ROOT / "runs" / name
        cfg = yaml.safe_load((run / "resolved-config.yaml").read_text("utf8"))
        index, threshold = read(run / "dataset-index.json"), read(run / "threshold.json")
        allowed = {
            s["sequence_id"]
            for s in index["split_manifest"]["sequences"]
            if s["split"] == "validation"
        }
        if len(allowed) != 23 or not set(SELECTED).issubset(allowed):
            raise ContractError("Only original Validation may be replayed")
        if (
            cfg["quality"]
            != {"min_joint_confidence": 0.2, "min_valid_joints": 9, "min_good_frame_ratio": 0.8}
            or cfg["skeleton"]["window_size"] != 30
            or cfg["skeleton"]["stride"] != 1
        ):
            raise ContractError("Frozen quality/window rules changed")
        if threshold["comparison"] != "gt" or threshold["consecutive_windows"] != 3:
            raise ContractError("Frozen threshold rule changed")
        folder = ROOT / formal
        seq = {r["sequence_id"]: r for r in prior.csv_read(folder / "sequences.csv")}
        windows = prior.parse_windows(prior.csv_read(folder / "windows.csv"))
        events = prior.csv_read(folder / "events.csv")
        if set(seq) != allowed or not all(w["sequence_id"] in allowed for w in windows + events):
            raise ContractError("Formal scores are not Validation-only")
        models[label] = {
            "config": cfg,
            "index": index,
            "threshold": threshold["threshold"],
            "sequence": seq,
            "windows": windows,
            "events": events,
        }
        for path in [
            run / "resolved-config.yaml",
            run / "dataset-index.json",
            run / "threshold.json",
            run / "autoencoder.pt",
            run / "scaler.npz",
            *[folder / n for n in ("sequences.csv", "windows.csv", "events.csv", "metrics.json")],
        ]:
            inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    for sid in SELECTED:
        for label, data in models.items():
            entry = data["index"]["sequences"][sid]
            path = prior.local_path(entry["pose_path"])
            observations, meta = load_pose_cache(
                path,
                {
                    "sequence_id": sid,
                    "split": "validation",
                    "source_hash": entry["source_hash"],
                    "recipe_hash": data["index"]["recipe_hash"],
                    "split_hash": data["index"]["split_manifest"]["split_hash"],
                },
            )
            data.setdefault("poses", {})[sid] = observations
            data.setdefault("pose_meta", {})[sid] = meta
            for p in (path, path.with_suffix(".json")):
                inputs[p.relative_to(ROOT).as_posix()] = sha(p)
        a, b = models["demo"]["pose_meta"][sid], models["wide"]["pose_meta"][sid]
        if (
            a["array_sha256"] != b["array_sha256"]
            or a["frames"] != b["frames"]
            or a["poses"] != b["poses"]
            or a["source_hash"] != b["source_hash"]
        ):
            raise ContractError("Demo and wide original observations differ")
    return models, inputs


def build():
    if OUT.exists() or QA.exists():
        raise ContractError("Preserve existing display replay outputs/QA; refuse overwrite")
    preserved = inventory()
    check_files(preserved)
    models, inputs = load_inputs()
    if models["demo"]["config"]["quality"] != models["wide"]["config"]["quality"]:
        raise ContractError("Model quality thresholds differ")
    bundle = read(REVIEW / "review-bundle.json")
    annotation = read(HUMAN / "human-review-original.json")
    prior.validate_review(annotation, bundle)
    expected = read(HUMAN / "annotation-audit.json")
    if sha(HUMAN / "human-review-original.json") != expected["annotation_sha256"]:
        raise ContractError("Archived self-reported human review changed")
    QA.mkdir(parents=True)
    OUT.mkdir(parents=True)
    for sub in ("videos", "frames"):
        (OUT / sub).mkdir()
    for name in MUTABLE:
        p = QA / "prior-contracts" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("xb") as stream:
            stream.write((ROOT / name).read_bytes())
    save(QA / "preservation.json", {"files": preserved})
    all_trace, summaries, video_info, reference_hashes = [], [], [], []
    font_hash = sha(FONT)
    for sid in SELECTED:
        observations = models["wide"]["poses"][sid]
        cfg = models["wide"]["config"]
        resampler = CausalResampler(
            cfg["skeleton"]["target_fps"],
            cfg["skeleton"]["resampling_hold_policy"],
            cfg["skeleton"]["timestamp_tolerance_ms"],
        )
        samples = {s.sample_index: s for f, poses in observations for s in resampler.push(f, poses)}
        action_path = ACTIONS / f"{sid}-actions.csv"
        action_meta = read(ACTIONS / f"{sid}.json")
        if (
            action_meta["recipe_hash"] != models["wide"]["index"]["recipe_hash"]
            or action_meta["action_display"]["normalization"] != "original30/per_frame"
            or action_meta["rendering"]["overlay_mode"] != "sample"
        ):
            raise ContractError("Cached independent classifier provenance mismatch")
        actions = parse_cached_actions(prior.csv_read(action_path), samples, observations, cfg)
        inputs[action_path.relative_to(ROOT).as_posix()] = sha(action_path)
        inputs[(ACTIONS / f"{sid}.json").relative_to(ROOT).as_posix()] = sha(
            ACTIONS / f"{sid}.json"
        )
        clip = next(c for c in bundle["clips"] if c["sequence_id"] == sid)
        reviewed = next(c for c in annotation["clips"] if c["sequence_id"] == sid)
        timing = {k: frame_time_bounds(reviewed[k], clip["frames"]) for k in ("onset", "impact")}
        frame_meta = {f["source_frame_id"]: f for f in clip["frames"]}
        source_meta = {f.source_frame_id: f for f, _ in observations}
        lo, hi = clip["frames"][0]["timestamp_ms"], clip["frames"][-1]["timestamp_ms"]
        ticks = [s for s in samples.values() if lo <= s.timestamp_ms <= hi]
        start_index, end_index = ticks[0].sample_index, ticks[-1].sample_index
        by_window, engines = {}, {}
        for label, model in models.items():
            grouped = defaultdict(list)
            for w in model["windows"]:
                if w["sequence_id"] == sid:
                    grouped[w["sample_index"]].append(w)
            by_window[label] = grouped
            engines[label] = CachedDisplayReplay(model["config"], model["threshold"])
        size = (640, TITLE + HEADER + 480 + FOOTER)
        writers = {
            label: H264Output(OUT / "videos" / f"{sid}-{label}.mp4", size) for label in MODELS
        }
        paired = H264Output(OUT / "videos" / f"{sid}-compare.mp4", (1280, size[1]))
        clip_rows = {label: [] for label in MODELS}
        for sample in samples.values():
            panels = []
            for label, model in models.items():
                windows = by_window[label].get(sample.sample_index, [])
                raw, visible = engines[label].step(sample, windows)
                action = actions[sample.sample_index]
                selected_track = headline(action)
                ae = visible.get(
                    selected_track,
                    {
                        "state": "unknown",
                        "reason": "no_person",
                        "raw_score": None,
                        "event_score": None,
                    },
                )
                source = model["sequence"][sid]
                latest = engines[label].history.latest or {}
                w = windows[0] if windows else {}
                if len(windows) > 1:
                    raise ContractError(
                        "This bounded three-clip audit expects one window track per tick"
                    )
                in_clip = start_index <= sample.sample_index <= end_index
                row = {
                    "sequence_id": sid,
                    "model": label,
                    "run": MODELS[label][0],
                    "sample_index": sample.sample_index,
                    "timestamp_ms": sample.timestamp_ms,
                    "source_frame_id": sample.source_frame_id,
                    "source_timestamp_ms": source_meta[sample.source_frame_id].timestamp_ms
                    if sample.source_frame_id is not None
                    else None,
                    "rendered": in_clip,
                    "media_frame_index": sample.sample_index - start_index if in_clip else None,
                    "media_time_ms": (sample.sample_index - start_index) * 1000 / FPS
                    if in_clip
                    else None,
                    "headline_track_id": selected_track,
                    "csv_state": w.get("state", "no_window_yet"),
                    "csv_raw_score": w.get("raw_score"),
                    "csv_event_score": w.get("event_score"),
                    "current_ae_state": ae["state"],
                    "current_ae_reason": ae.get("reason"),
                    "current_raw_score": ae.get("raw_score"),
                    "current_event_score": ae.get("event_score"),
                    "history_count": engines[label].history.count,
                    "history_event_start_ms": latest.get("confirmed_start_ms"),
                    "action_label": action.get(selected_track, unavailable())["label"],
                    "good_pose_tracks": [
                        p.track_id
                        for p in sample.poses
                        if good_frame(p.keypoints, model["config"]["quality"])
                    ],
                    "visible_states": visible,
                    "raw_statuses_before_mask": raw,
                    "video_prediction": int(source["prediction"]) if source["prediction"] else None,
                    "video_outcome": "TP" if source["prediction"] == "1" else "FN",
                    "threshold": model["threshold"],
                    "human_relation": temporal_relation(
                        sample.timestamp_ms, timing["onset"], timing["impact"]
                    )["relation"],
                }
                all_trace.append(row)
                if not in_clip:
                    continue
                if sample.source_frame_id not in frame_meta:
                    raise ContractError(
                        "Clip would require an unreviewed/out-of-scope source image"
                    )
                image_path = REVIEW / "review/frames" / f"{sid}-{sample.source_frame_id:03}.png"
                image_sha = sha(image_path)
                if image_sha != frame_meta[sample.source_frame_id]["source_sha256"]:
                    raise ContractError("Original review image hash changed")
                inputs[image_path.relative_to(ROOT).as_posix()] = image_sha
                image = cv2.imdecode(
                    np.frombuffer(image_path.read_bytes(), np.uint8), cv2.IMREAD_COLOR
                )
                if image is None or image.shape != (480, 640, 3):
                    raise ContractError("Cannot decode original Validation RGB")
                native = draw_sample(
                    image,
                    sample.poses,
                    action,
                    visible,
                    sample.timestamp_ms,
                    engines[label].history,
                    model["config"]["quality"],
                )
                panel = audit_canvas(native, row, reviewed, timing, label, MODELS[label][0])
                reference_hashes.append(
                    {
                        "sequence_id": sid,
                        "model": label,
                        "media_frame_index": row["media_frame_index"],
                        "native_bgr_sha256": hashlib.sha256(native.tobytes()).hexdigest(),
                        "panel_bgr_sha256": hashlib.sha256(panel.tobytes()).hexdigest(),
                    }
                )
                writers[label].write(panel)
                panels.append(panel)
                clip_rows[label].append(row)
            if panels:
                pair = np.hstack(panels)
                paired.write(pair)
                points = {
                    "fall-18": {42, 43, 44, 64},
                    "fall-09": {176, 177, 178, 180, 184},
                    "fall-30": {55, 56, 57, 69},
                }[sid]
                if sample.sample_index in points:
                    path = OUT / "frames" / f"{sid}-sample-{sample.sample_index:03}.png"
                    ok, png = cv2.imencode(".png", pair)
                    if not ok:
                        raise ContractError("Cannot preserve lossless paired evidence")
                    with path.open("xb") as stream:
                        stream.write(png.tobytes())
        for label, writer in writers.items():
            info = writer.close()
            info.update(
                sequence_id=sid,
                model=label,
                grid_start_ms=ticks[0].timestamp_ms,
                grid_end_ms=ticks[-1].timestamp_ms,
                review_source_bounds_ms=[lo, hi],
            )
            video_info.append(info)
        info = paired.close()
        info.update(
            sequence_id=sid,
            model="compare",
            grid_start_ms=ticks[0].timestamp_ms,
            grid_end_ms=ticks[-1].timestamp_ms,
            review_source_bounds_ms=[lo, hi],
        )
        video_info.append(info)
        for label, engine in engines.items():
            replayed = engine.events.finish(observations[-1][0].timestamp_ms)
            formal = [e for e in models[label]["events"] if e["sequence_id"] == sid]
            expected = [
                {
                    **e,
                    "track_id": int(e["track_id"]),
                    **{
                        k: float(e[k])
                        for k in ("confirmed_start_ms", "last_active_ms", "peak_score", "end_ms")
                    },
                }
                for e in formal
            ]
            if replayed != expected:
                raise ContractError("Cached rendering changed formal AE events")
            visible = clip_rows[label]
            summaries.append(
                {
                    "sequence_id": sid,
                    "model": label,
                    "run": MODELS[label][0],
                    "video_outcome": visible[0]["video_outcome"],
                    "events": len(replayed),
                    "first_event_ms": replayed[0]["confirmed_start_ms"] if replayed else None,
                    "rendered_frames": len(visible),
                    "current_ae_alert_frames": sum(
                        r["current_ae_state"] == "suspected_fall" for r in visible
                    ),
                    "history_recorded_frames": sum(r["history_count"] > 0 for r in visible),
                    "first_history_render_ms": next(
                        (r["timestamp_ms"] for r in visible if r["history_count"]), None
                    ),
                    "first_current_ae_alert_ms": next(
                        (
                            r["timestamp_ms"]
                            for r in visible
                            if r["current_ae_state"] == "suspected_fall"
                        ),
                        None,
                    ),
                    "action_FALLING_frames": sum(r["action_label"] == "FALLING" for r in visible),
                    "current_ae_states": dict(Counter(r["current_ae_state"] for r in visible)),
                    "csv_active_frames": sum(r["csv_state"] == "suspected_fall" for r in visible),
                    "masked_csv_active_frames": sum(
                        r["csv_state"] == "suspected_fall"
                        and r["current_ae_state"] != "suspected_fall"
                        for r in visible
                    ),
                    "onset_ms": timing["onset"],
                    "impact_ms": timing["impact"],
                }
            )
    prior.csv_save(OUT / "display-frames.csv", all_trace)
    prior.csv_save(OUT / "summary.csv", summaries)
    prior.csv_save(OUT / "preencode-frame-hashes.csv", reference_hashes)
    save(
        OUT / "summary.json",
        {
            "id": TAG,
            "rows": summaries,
            "videos": video_info,
            "source": "existing Validation RGB/pose/window scores/action heads/human uncertainty bounds",
            "view": "unmodified sample renderer (auto view for Human models)",
            "new_model_inference_training_calibration": 0,
            "test_access_evaluation": 0,
            "accuracy_improvement_claimed": False,
            "demo_default_changed": False,
            "real_feature_smoke": "not_run",
            "real_end_to_end_new_RGB_model": "not_run",
            "real_cached_render_replay": "completed",
        },
    )
    sources = [
        Path(__file__),
        ROOT / "src/fall_detect/visualization/cached_replay.py",
        ROOT / "tests/test_cached_display_replay.py",
        ROOT / "docs/ur-validation-display-20261105-plan.md",
        HUMAN / "human-review-original.json",
        HUMAN / "annotation-audit.json",
        REVIEW / "review-bundle.json",
        ROOT / "src/fall_detect/visualization/video.py",
        ROOT / "src/fall_detect/visualization/sample_display.py",
        ROOT / "src/fall_detect/visualization/action_status.py",
        ROOT / "src/fall_detect/skeleton/preprocessing.py",
        ROOT / "src/fall_detect/skeleton/windows.py",
        ROOT / "src/fall_detect/detection/events.py",
        ROOT / "src/fall_detect/data/cache.py",
    ]
    inputs.update({p.relative_to(ROOT).as_posix(): sha(p) for p in sources})
    save(
        QA / "inputs.json",
        {
            "files": inputs,
            "font_path": str(FONT),
            "font_sha256": font_hash,
            "ffmpeg_path": imageio_ffmpeg.get_ffmpeg_exe(),
            "ffmpeg_sha256": sha(imageio_ffmpeg.get_ffmpeg_exe()),
            "loaded_model_modules": [
                name
                for name in sys.modules
                if name
                in {
                    "fall_detect.training.experiment",
                    "fall_detect.models.stgcn",
                    "fall_detect.models.autoencoder",
                }
            ],
            "model_weights_or_features_loaded": False,
            "allowed_sequences": list(SELECTED),
        },
    )
    print(
        json.dumps(
            {"completed": True, "rows": summaries, "videos": len(video_info)}, ensure_ascii=False
        )
    )


def verify():
    check_files(read(QA / "preservation.json")["files"], archived=True)
    check_files(read(QA / "inputs.json")["files"])
    if (OUT / "artifact-manifest.json").exists():
        for name, digest in read(OUT / "artifact-manifest.json")["files"].items():
            if sha(OUT / name) != digest:
                raise ContractError(f"Rendered artifact changed: {name}")
    print(
        json.dumps(
            {"verified": True, "summary": read(OUT / "summary.json")["rows"]}, ensure_ascii=False
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("build", "verify", "serve"))
    parser.add_argument("--port", type=int, default=8840)
    args = parser.parse_args()
    if args.stage == "build":
        build()
    else:
        verify()
        if args.stage == "serve":
            from functools import partial
            from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

            handler = partial(SimpleHTTPRequestHandler, directory=str(OUT))
            with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
                print(f"http://127.0.0.1:{args.port}/index.html", flush=True)
                server.serve_forever()


if __name__ == "__main__":
    main()
