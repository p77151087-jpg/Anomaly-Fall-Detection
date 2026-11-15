"""Independent saved-CSV/NPZ timeline and decoded native-header pixel audit."""

import argparse
import bisect
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import validation_alert_quality as io
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
TAG = "ur-validation-display-20261105-001"
OUT = ROOT / "outputs" / TAG
QA = ROOT / "docs/validation" / TAG
SELECTED = ("fall-18", "fall-09", "fall-30")
TITLE, HEIGHT, WIDTH, NATIVE = 64, 820, 640, 596
FONT = Path("C:/Windows/Fonts/msjh.ttc")
BG, GREY, RED, AMBER, GREEN = (
    (26, 21, 17),
    (182, 193, 202),
    (103, 101, 255),
    (96, 207, 245),
    (132, 223, 101),
)


def cv_template(text, xy, scale, color, thickness, crop):
    image = np.full((116, 640, 3), BG, np.uint8)
    while cv2.getTextSize(text, 0, scale, thickness)[0][0] > 296 and scale > 0.18:
        scale -= 0.02
    cv2.putText(image, text, xy, 0, scale, color, thickness, cv2.LINE_AA)
    y0, y1, x0, x1 = crop
    return image[y0:y1, x0:x1]


def mae(a, b):
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def timeline():
    """Compute expectations directly from original scores, actual source metadata and arrays."""
    trace = io.csv_read(OUT / "display-frames.csv")
    manifest = io.read(QA / "inputs.json")
    summary = io.read(OUT / "summary.json")
    annotation = io.read(
        ROOT / "outputs/ur-validation-human-timing-20261105-001/human-review-original.json"
    )
    expected, global_counts = {}, Counter()
    for sid in SELECTED:
        action_rows = io.csv_read(
            ROOT / f"outputs/sample-display/ur-human-isotropic-001/{sid}-actions.csv"
        )
        action_by_index = {int(r["sample_index"]): r for r in action_rows}
        contexts = {}
        for model in ("demo", "wide"):
            name = "ur-sample-human-s1-001" if model == "demo" else "ur-isotropic-wide-20261012-001"
            formal = ROOT / (
                "outputs/human-isotropic-validation/ur-sample-human-s1-001"
                if model == "demo"
                else "outputs/isotropic-refinement-20261012/ur-isotropic-wide-20261012-001"
            )
            cfg = yaml.safe_load((ROOT / "runs" / name / "resolved-config.yaml").read_text("utf8"))
            index = io.read(ROOT / "runs" / name / "dataset-index.json")
            split = {
                r["sequence_id"]
                for r in index["split_manifest"]["sequences"]
                if r["split"] == "validation"
            }
            assert sid in split and len(split) == 23
            pose_path = ROOT / index["sequences"][sid]["pose_path"]
            meta = io.read(pose_path.with_suffix(".json"))
            assert meta["split"] == "validation" and meta["sequence_id"] == sid
            with np.load(pose_path, allow_pickle=False) as arrays:
                points = arrays["keypoints"]
            assert np.isfinite(points).all()
            frames = meta["frames"]
            times = [f["timestamp_ms"] for f in frames]
            source_by_id = {f["source_frame_id"]: f for f in frames}
            windows = [r for r in io.csv_read(formal / "windows.csv") if r["sequence_id"] == sid]
            assert len({r["sample_index"] for r in windows}) == len(windows)
            by_window = {int(r["sample_index"]): r for r in windows}
            formal_events = [
                r for r in io.csv_read(formal / "events.csv") if r["sequence_id"] == sid
            ]
            threshold = io.read(ROOT / "runs" / name / "threshold.json")["threshold"]
            sequence = next(
                r for r in io.csv_read(formal / "sequences.csv") if r["sequence_id"] == sid
            )
            selected = [r for r in trace if r["model"] == model and r["sequence_id"] == sid]
            n = int(np.floor((times[-1] - times[0]) * 30 / 1000 + 1e-7)) + 1
            assert len(selected) == n == len(action_rows)
            history, prior_index, was_active, event_starts = [], None, False, []
            sticky = None
            outputs = []
            for i, row in enumerate(selected):
                tick = times[0] + i * (1000 / 30)
                assert int(row["sample_index"]) == i and float(row["timestamp_ms"]) == tick
                j = bisect.bisect_right(times, tick + 1e-7) - 1
                frame = frames[j]
                assert (
                    frame["timestamp_ms"] <= tick + 1e-7
                    and tick - frame["timestamp_ms"] <= 50 + 1e-7
                )
                fid = frame["source_frame_id"]
                assert (
                    int(row["source_frame_id"]) == fid
                    and float(row["source_timestamp_ms"]) == frame["timestamp_ms"]
                )
                good_tracks = [
                    meta["poses"][p]["track_id"]
                    for p in frame["pose_indices"]
                    if np.count_nonzero(points[p, :, 2] >= 0.2) >= 9
                ]
                assert json.loads(row["good_pose_tracks"]) == good_tracks
                assert all(meta["poses"][p]["track_id"] == 1 for p in frame["pose_indices"])
                w = by_window.get(i)
                if w:
                    raw = float(w["raw_score"]) if w["raw_score"] else None
                    if prior_index is not None and i != prior_index + 1:
                        history.clear()
                    prior_index = i
                    if raw is None:
                        history.clear()
                        score, state = None, "unknown"
                    else:
                        history.append(raw)
                        history = history[-3:]
                        score = min(history) if len(history) == 3 else None
                        state = (
                            "warming_up"
                            if score is None
                            else "suspected_fall"
                            if score > threshold
                            else "normal"
                        )
                    assert state == w["state"] and score == (
                        float(w["event_score"]) if w["event_score"] else None
                    )
                    active = state == "suspected_fall"
                    if active and not was_active:
                        event_starts.append(tick)
                    was_active = active
                    sticky = {
                        "state": state,
                        "updated": float(w["available_at_ms"]),
                        "reason": w["reason"],
                    }
                if not frame["pose_indices"]:
                    current = "unknown"
                    sticky = {"state": "unknown", "updated": tick, "reason": "track_gap"}
                else:
                    current = sticky["state"] if sticky else "warming_up"
                    age = tick - sticky["updated"] if sticky else 0
                    if (
                        not good_tracks
                        or age > cfg["demo"]["score_hold_stride_multiplier"] * 1000 / 30
                    ):
                        current = "unknown"
                        sticky = {
                            "state": "unknown",
                            "updated": sticky["updated"] if sticky else tick,
                            "reason": "low_confidence" if not good_tracks else "stale_score",
                        }
                action = action_by_index[i]
                assert int(action["source_frame_id"]) == fid
                if action["state"] == "valid":
                    values = [
                        float(action[k])
                        for k in (
                            "standing",
                            "walking",
                            "sitting",
                            "lying",
                            "stand_up",
                            "sit_down",
                            "fall_down",
                        )
                    ]
                    assert np.isfinite(values).all() and int(np.argmax(values)) == int(
                        action["class_index"]
                    )
                    assert values[int(action["class_index"])] == float(action["head_score"])
                assert row["action_label"] == action["label"]
                assert row["current_ae_state"] == current
                if current == "unknown":
                    assert row["current_raw_score"] == row["current_event_score"] == ""
                assert int(row["history_count"]) == len(event_starts)
                assert io.number(row["history_event_start_ms"]) == (
                    event_starts[-1] if event_starts else None
                )
                rendered = row["rendered"] == "True"
                if rendered:
                    outputs.append(
                        {
                            "row": row,
                            "current": current,
                            "history": bool(event_starts),
                            "latest_event": event_starts[-1] if event_starts else None,
                            "action": action,
                            "source": frame,
                            "tick": tick,
                        }
                    )
                global_counts["full_model_grid_frames"] += 1
                global_counts["saved_window_rows"] += int(w is not None)
            assert event_starts == [float(e["confirmed_start_ms"]) for e in formal_events]
            review = next(c for c in annotation["clips"] if c["sequence_id"] == sid)
            onset = [source_by_id[x]["timestamp_ms"] for x in review["onset"]["bounds_frame_ids"]]
            impact = [source_by_id[x]["timestamp_ms"] for x in review["impact"]["bounds_frame_ids"]]
            counts = {
                "rendered_frames": len(outputs),
                "current_ae_alert_frames": sum(o["current"] == "suspected_fall" for o in outputs),
                "history_recorded_frames": sum(o["history"] for o in outputs),
                "action_FALLING_frames": sum(o["action"]["label"] == "FALLING" for o in outputs),
            }
            item = next(
                c for c in summary["rows"] if c["model"] == model and c["sequence_id"] == sid
            )
            for key, value in counts.items():
                assert item[key] == value
            assert item["onset_ms"] == onset and item["impact_ms"] == impact
            assert item["first_event_ms"] == (event_starts[0] if event_starts else None)
            assert (
                sequence["first_alarm_ms"] == ""
                if not event_starts
                else float(sequence["first_alarm_ms"]) == event_starts[0]
            )
            for k, o in enumerate(outputs):
                assert int(o["row"]["media_frame_index"]) == k
                assert float(o["row"]["media_time_ms"]) == k * 1000 / 30
                t = o["tick"]
                relation = (
                    "before_onset"
                    if t < onset[0]
                    else "after_impact"
                    if t > impact[1]
                    else "onset_uncertainty"
                    if t < onset[1]
                    else "impact_uncertainty"
                    if t >= impact[0]
                    else "guaranteed_fall_interval"
                )
                assert o["row"]["human_relation"] == relation
            expected[sid, model] = {
                "frames": outputs,
                "sequence": sequence,
                "threshold": threshold,
                "config": cfg,
                "run": name,
            }
            contexts[model] = points
        np.testing.assert_array_equal(contexts["demo"], contexts["wide"])
    assert (
        manifest["loaded_model_modules"] == []
        and manifest["model_weights_or_features_loaded"] is False
    )
    return expected, dict(global_counts)


def title_template(context, model):
    canvas = Image.new("RGB", (WIDTH, TITLE), (17, 21, 26))
    draw = ImageDraw.Draw(canvas)
    kind = "預設模型 demo" if model == "demo" else "wide 參照模型"
    outcome = "TP" if context["sequence"]["prediction"] == "1" else "FN"
    draw.text(
        (10, 4),
        f"{kind}｜影片判定 {outcome}",
        font=ImageFont.truetype(str(FONT), 22),
        fill=(231, 237, 243),
    )
    draw.text(
        (10, 34),
        f"{context['run']}｜固定門檻 {context['threshold']:.6f}／K3",
        font=ImageFont.truetype(str(FONT), 16),
        fill=(184, 199, 211),
    )
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def audit_pixels(image, expected, title_images):
    e = expected
    native = image[TITLE : TITLE + NATIVE]
    red = native[30:60, 334:638]
    red_count = int(((red[:, :, 2] > 190) & (red[:, :, 1] < 170) & (red[:, :, 0] < 170)).sum())
    assert (red_count > 30) == e["history"], (e["row"], red_count)
    hints = {
        "unknown": "Current: unavailable (pose / tracking)",
        "warming_up": "Current: collecting motion",
        "normal": "Current: normal",
        "suspected_fall": "Current: alert",
    }
    scores = {
        state: mae(
            native[84:105, 334:638], cv_template(text, (334, 97), 0.4, GREY, 1, (84, 105, 334, 638))
        )
        for state, text in hints.items()
    }
    best = min(scores, key=scores.get)
    assert best == e["current"] and scores[best] < 8, (e["row"], scores)
    action = e["action"]
    label = action["label"]
    color = (
        RED
        if label == "FALLING"
        else AMBER
        if label in {"LYING", "PENDING"}
        else GREEN
        if action["state"] == "valid"
        else GREY
    )
    act = cv_template(label, (14, 52), 0.9, color, 2, (28, 61, 14, 318))
    action_error = mae(native[28:61, 14:318], act)
    assert action_error < 8, (e["row"], action_error)
    title_scores = {name: mae(image[:TITLE], template) for name, template in title_images.items()}
    assert (
        min(title_scores, key=title_scores.get) == e["row"]["model"]
        and title_scores[e["row"]["model"]] < 8
    )
    if e["history"]:
        text = f"ID 1 | event at {e['latest_event'] / 1000:.2f}s"
        error = mae(
            native[60:82, 334:638], cv_template(text, (334, 72), 0.44, AMBER, 1, (60, 82, 334, 638))
        )
        assert error < 8
    return {
        "history_red_pixels": red_count,
        "current_hint_best_match": best,
        "current_hint_mae": scores[best],
        "cached_action_text_mae": action_error,
        "model_title_mae": title_scores[e["row"]["model"]],
    }


def video_audit(expected):
    pixel_rows, video_rows = [], []
    summary = io.read(OUT / "summary.json")
    for sid in SELECTED:
        titles = {m: title_template(expected[sid, m], m) for m in ("demo", "wide")}
        for model in ("demo", "wide"):
            path = OUT / "videos" / f"{sid}-{model}.mp4"
            cap = cv2.VideoCapture(str(path))
            count = 0
            for e in expected[sid, model]["frames"]:
                ok, image = cap.read()
                assert ok and image.shape == (HEIGHT, WIDTH, 3)
                checks = audit_pixels(image, e, titles)
                pixel_rows.append(
                    {"sequence_id": sid, "model": model, "media_frame_index": count, **checks}
                )
                count += 1
            assert not cap.read()[0]
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            assert abs(fps - 30) < 1e-8
            info = next(
                v for v in summary["videos"] if v["sequence_id"] == sid and v["model"] == model
            )
            assert count == info["frames"] and info["size_wh"] == [WIDTH, HEIGHT]
            video_rows.append(
                {
                    "sequence_id": sid,
                    "model": model,
                    "decoded_frames": count,
                    "fps": fps,
                    "native_pixels_verified": True,
                }
            )
        caps = [
            cv2.VideoCapture(str(OUT / "videos" / f"{sid}-{kind}.mp4"))
            for kind in ["compare", "demo", "wide"]
        ]
        count = 0
        max_pair_error = 0
        for i in range(len(expected[sid, "demo"]["frames"])):
            results = [cap.read() for cap in caps]
            assert all(r[0] for r in results)
            pair, left, right = [r[1] for r in results]
            assert pair.shape == (HEIGHT, WIDTH * 2, 3)
            for panel, single, model in [
                (pair[:, :WIDTH], left, "demo"),
                (pair[:, WIDTH:], right, "wide"),
            ]:
                audit_pixels(panel, expected[sid, model]["frames"][i], titles)
                err = mae(panel, single)
                assert err < 12
                max_pair_error = max(max_pair_error, err)
            if i == next(
                (j for j, e in enumerate(expected[sid, "wide"]["frames"]) if e["history"]), -1
            ):
                target = OUT / "frames" / f"{sid}-decoded-first-event.png"
                if not target.exists():
                    ok, data = cv2.imencode(".png", pair)
                    assert ok
                    with target.open("xb") as stream:
                        stream.write(data.tobytes())
            count += 1
        assert all(not cap.read()[0] for cap in caps)
        for cap in caps:
            cap.release()
        video_rows.append(
            {
                "sequence_id": sid,
                "model": "compare",
                "decoded_frames": count,
                "fps": 30,
                "native_pixels_verified": True,
                "max_panel_vs_single_mae": max_pair_error,
            }
        )
    return pixel_rows, video_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    expected, counts = timeline()
    pixels, videos = video_audit(expected)
    result = {
        "passed": True,
        "method": "Independent original CSV/NPZ causal timeline and every decoded native header glyph/color; pair vs individual panel checks",
        **counts,
        "individual_native_frames_audited": len(pixels),
        "unique_decoded_video_frames": sum(v["decoded_frames"] for v in videos),
        "videos": videos,
        "changed_quality_rules": False,
        "new_model_inference_training": 0,
        "test_access": 0,
    }
    path = QA / "independent-verification.json"
    if path.exists():
        assert io.read(path) == result
    else:
        io.save(path, result)
        io.csv_save(QA / "decoded-native-pixels.csv", pixels)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
