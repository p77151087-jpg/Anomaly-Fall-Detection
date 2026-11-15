"""Fresh RGB acceptance of the frozen optional shoulder demo; Validation only."""

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from fall_detect.config import load_config
from fall_detect.data.cache import load_pose_cache
from fall_detect.data.readers import source_identity
from fall_detect.io_utils import ContractError, read_json, sha256, write_csv, write_json
from fall_detect.skeleton.windows import CausalResampler
from fall_detect.visualization.cached_replay import parse_cached_actions
from fall_detect.visualization.sample_display import EventHistory, draw_sample

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/ur-shoulder-demo-20261112-001"
CAND = ROOT / "outputs/ur-fall19-shoulder-descent-20261107-001"
RUN = ROOT / "runs/ur-isotropic-wide-20261012-001"
PILOT = ("fall-19", "adl-35")


def require(ok, message):
    if not ok:
        raise ContractError(message)


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def number(value):
    return float(value) if value not in (None, "") else None


def base(sid):
    return OUT / ("pilot" if sid in PILOT else "validation") / sid


def protect():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "contract.json"
    if path.exists():
        return read_json(path)
    lock = read_json(CAND / "lock.json")
    entries = {sid: e for sid, e in lock["allowed_sequences"].items() if e["split"] == "validation"}
    require(
        len(entries) == 23 and sum(e["label"] for e in entries.values()) == 15,
        "Original Validation manifest differs",
    )
    protected = set(p for p in CAND.rglob("*") if p.is_file())
    protected.update(p for p in RUN.iterdir() if p.is_file())
    for e in lock["allowed_sequences"].values():
        require(e["split"] in {"train", "validation"}, "Test is outside this command")
        for key in ("pose_path", "feature_path"):
            p = ROOT / e[key]
            protected.update((p.with_suffix(".json"), p.with_suffix(".npz")))
    # Only existing Train/Validation assets and pretrained weights are protected here.
    for name in lock["source_sha256"]:
        if name.startswith("weights/"):
            protected.add(ROOT / name)
    sources = [
        ROOT / p
        for p in (
            "src/fall_detect/detection/shoulder_demo.py",
            "src/fall_detect/cli.py",
            "src/fall_detect/visualization/video.py",
            "src/fall_detect/visualization/sample_display.py",
            "scripts/validate_shoulder_demo.py",
            "docs/ur-shoulder-demo-20261112-plan.md",
        )
    ]
    source_hashes = {}
    for sid, e in entries.items():
        raw = ROOT / "data/raw/urfall" / sid
        sync = list(raw.glob("*-data.csv"))
        identity = source_identity(raw, sync[0] if len(sync) == 1 else None)
        require(identity["source_hash"] == e["source_hash"], f"Original RGB changed: {sid}")
        source_hashes[sid] = identity["source_hash"]
    contract = {
        "id": OUT.name,
        "evidence": "real_end_to_end",
        "no_fit": True,
        "test_access": False,
        "entries": entries,
        "source_hashes": source_hashes,
        "protected_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sorted(protected)},
        "implementation_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        "pilot": list(PILOT),
        "thresholds": [0.5341113805770874, 0.18564587694466197],
        "feature_tolerance": {"atol": 1e-5, "rtol": 1e-5},
        "pixel_mae_limit": 8.0,
    }
    write_json(path, contract)
    return contract


def check_protected(contract):
    for collection in ("protected_sha256", "implementation_sha256"):
        for name, expected in contract[collection].items():
            require(sha256(ROOT / name) == expected, f"Protected artifact changed: {name}")


def execute_one(sid):
    p = base(sid)
    if p.with_suffix(".json").exists():
        print(f"REUSE completed fresh RGB: {sid}", flush=True)
        return
    require(not p.with_suffix(".mp4").exists(), f"Incomplete output requires a new attempt: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-B",
        "-X",
        "utf8",
        "-m",
        "fall_detect",
        "demo",
        "--run",
        str(RUN),
        "--input",
        str(ROOT / "data/raw/urfall" / sid),
        "--output",
        str(p.with_suffix(".mp4")),
        "--shoulder-candidate",
        str(CAND),
        "--device",
        "cuda",
        "--overlay",
        "sample",
        "--no-display",
    ]
    print(f"START fresh RGB: {sid}", flush=True)
    with p.with_suffix(".log").open("x", encoding="utf-8") as log:
        result = subprocess.run(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    require(result.returncode == 0, f"Demo failed: {sid}; see {p.with_suffix('.log')}")


def independent_events(windows, thresholds, end_ms):
    """CSV-only rolling minimum and union, independent of production EventState."""
    histories, active, events, last = {}, {}, [], {}
    for w in windows:
        track, tick, now = w["track_id"], int(w["sample_index"]), float(w["available_at_ms"])
        h = histories.setdefault(track, [deque(maxlen=3), deque(maxlen=3)])
        if track in last and tick != last[track] + 1:
            if track in active:
                events.append({**active.pop(track), "end_ms": now, "end_reason": "time_gap"})
            for q in h:
                q.clear()
        last[track] = tick
        scores = []
        for index, name in enumerate(("wide", "shoulder")):
            raw = number(w[name + "_raw"])
            if raw is None:
                h[index].clear()
            else:
                h[index].append(raw)
            k3 = min(h[index]) if len(h[index]) == 3 else None
            require(k3 == number(w[name + "_k3"]), f"Independent {name} K3 differs")
            if k3 is not None:
                scores.append(k3 / thresholds[index])
        ratio = max(scores, default=None)
        require(ratio == number(w["combined_k3_ratio"]), "Fusion score differs")
        alert = ratio is not None and ratio > 1
        require(alert == (w["combined_state"] == "suspected_fall"), "Fusion decision differs")
        if alert:
            event = active.setdefault(
                track,
                {
                    "track_id": track,
                    "confirmed_start_ms": now,
                    "last_active_ms": now,
                    "peak_score": ratio,
                },
            )
            event["last_active_ms"] = now
            event["peak_score"] = max(event["peak_score"], ratio)
        elif track in active:
            events.append(
                {
                    **active.pop(track),
                    "end_ms": now,
                    "end_reason": w["wide_reason"] or w["shoulder_reason"] or "low_score",
                }
            )
    events.extend({**e, "end_ms": end_ms, "end_reason": "sequence_end"} for e in active.values())
    return sorted(events, key=lambda e: (e["confirmed_start_ms"], e["track_id"]))


def verify_one(sid, contract):
    require(sid in contract["entries"], "Only locked Validation sequences are allowed")
    entry, p = contract["entries"][sid], base(sid)
    metadata = read_json(p.with_suffix(".json"))
    require(
        metadata["source_identity"]["source_hash"] == entry["source_hash"], "RGB identity differs"
    )
    require(all(metadata["forward_verified"].values()), "Real forward was not verified")
    config = load_config(RUN / "resolved-config.yaml", device="cpu")
    observed, pose_meta = load_pose_cache(p.with_name(p.name + "-poses.npz"), {"split": "demo"})
    original, _ = load_pose_cache(ROOT / entry["pose_path"], {"split": "validation"})
    require(len(observed) == len(original), "Raw frame count differs")
    pose_max = 0.0
    for (a, ap), (b, bp) in zip(observed, original, strict=True):
        require(
            (a.source_frame_id, a.timestamp_ms) == (b.source_frame_id, b.timestamp_ms),
            "Frame timing differs",
        )
        require([v.track_id for v in ap] == [v.track_id for v in bp], "Fresh tracker differs")
        for x, y in zip(ap, bp, strict=True):
            for key in ("keypoints", "bbox_xyxy"):
                u, v = getattr(x, key), getattr(y, key)
                pose_max = max(pose_max, float(np.max(np.abs(u - v))))
                require(
                    np.allclose(u, v, atol=1e-5, rtol=1e-5),
                    "Fresh Pose differs beyond numerical tolerance",
                )
    fresh_feature = p.with_name(p.name + "-features.npz")
    with (
        np.load(fresh_feature, allow_pickle=False) as f,
        np.load((ROOT / entry["feature_path"]).with_suffix(".npz"), allow_pickle=False) as g,
    ):
        x, y = f["features"], g["features"]
        require(
            x.shape == y.shape and x.shape[1] == 512 and np.isfinite(x).all(),
            "Fresh features invalid",
        )
        feature_max = float(np.max(np.abs(x - y))) if x.size else 0.0
        require(
            np.allclose(x, y, **contract["feature_tolerance"]),
            "Fresh features differ beyond tolerance",
        )
    windows = rows(p.with_name(p.name + "-windows.csv"))
    old = [r for r in rows(CAND / "validation-windows.csv") if r["sequence_id"] == sid]
    records = read_json(ROOT / entry["feature_path"])["records"]
    require(len(windows) == len(old) == len(records), "Scheduled window count differs")
    max_score_difference = 0.0
    for a, b, record in zip(windows, old, records, strict=True):
        for field in ("sample_index", "track_id", "available_at_ms", "combined_state"):
            require(a[field] == b[field], f"Cached candidate {field} differs: {sid}")
        require(
            a["quality"] == record["quality"] and number(a["row_index"]) == record["row_index"],
            "Original quality mask differs",
        )
        for field in ("wide_raw", "wide_k3", "shoulder_raw", "shoulder_k3", "combined_k3_ratio"):
            u, v = number(a[field]), number(b[field])
            require((u is None) == (v is None), f"Missing score differs: {field}")
            if u is not None:
                max_score_difference = max(max_score_difference, abs(u - v))
                require(
                    np.isclose(u, v, atol=1e-6, rtol=1e-5), f"Fresh score differs: {sid}/{field}"
                )
    events = rows(p.with_name(p.name + "-events.csv"))
    independent = independent_events(windows, contract["thresholds"], observed[-1][0].timestamp_ms)
    cached_events = [r for r in rows(CAND / "validation-events.csv") if r["sequence_id"] == sid]
    require(len(events) == len(independent) == len(cached_events), "Event count differs")
    for a, b, c in zip(events, independent, cached_events, strict=True):
        for field in ("confirmed_start_ms", "last_active_ms", "end_ms", "peak_score"):
            old_key = "peak_relative_k3" if field == "peak_score" else field
            require(
                np.isclose(float(a[field]), b[field], atol=1e-8, rtol=0),
                "Independent event differs",
            )
            require(
                np.isclose(float(a[field]), float(c[old_key]), atol=1e-6, rtol=1e-5),
                "Cached event timing/peak differs",
            )
        require(a["end_reason"] == b["end_reason"] == c["end_reason"], "Event end reason differs")
    resampler = CausalResampler(
        30,
        config["skeleton"]["resampling_hold_policy"],
        config["skeleton"]["timestamp_tolerance_ms"],
    )
    samples = [s for frame, poses in observed for s in resampler.push(frame, poses)]
    actions = parse_cached_actions(
        rows(p.with_name(p.name + "-actions.csv")),
        {s.sample_index: s for s in samples},
        observed,
        config,
    )
    current, display = rows(p.with_suffix(".csv")), rows(p.with_name(p.name + "-display.csv"))
    grouped = defaultdict(list)
    for row in current:
        grouped[int(row["sample_index"])].append(row)
    old_display = {
        int(r["sample_index"]): r
        for r in rows(CAND / "display-grid.csv")
        if r["sequence_id"] == sid
    }
    capture = cv2.VideoCapture(str(p.with_suffix(".mp4")))
    pixel_max, pixel_sum, alerts, first_alert = 0.0, 0.0, 0, None
    require(len(samples) == len(display) == metadata["frames"], "Decoded grid length differs")
    for s, d in zip(samples, display, strict=True):
        logs = grouped[s.sample_index]
        actual_tracks = sorted(int(r["track_id"]) for r in logs if r["state"] == "suspected_fall")
        require(
            actual_tracks
            == json.loads(old_display[s.sample_index]["combined_current_alert_tracks"]),
            "Current display gate differs",
        )
        if actual_tracks:
            alerts += 1
            first_alert = s.timestamp_ms if first_alert is None else first_alert
        occurred = [e for e in events if float(e["confirmed_start_ms"]) <= s.timestamp_ms]
        require(
            int(d["history_event_count"]) == len(occurred), "History CSV differs from event CSV"
        )
        history = EventHistory()
        if occurred:
            latest = max(
                occurred, key=lambda e: (float(e["confirmed_start_ms"]), -int(e["track_id"]))
            )
            history.latest = {
                "track_id": int(latest["track_id"]),
                "confirmed_start_ms": float(latest["confirmed_start_ms"]),
            }
            require(
                number(d["history_event_start_ms"]) == history.latest["confirmed_start_ms"],
                "History timestamp differs",
            )
        visible = {int(r["track_id"]): {"state": r["state"]} for r in logs if r["track_id"]}
        blank = np.zeros((s.frame_size_wh[1], s.frame_size_wh[0], 3), np.uint8)
        expected = draw_sample(
            blank,
            s.poses,
            actions[s.sample_index],
            visible,
            s.timestamp_ms,
            history,
            config["quality"],
            history_title="AE + SHOULDER EVENTS",
        )
        ok, decoded = capture.read()
        require(ok and decoded.shape == expected.shape, "Video decode dimensions/count differ")
        # The entire right header includes history, timestamp, and current alert/unknown text.
        delta = np.abs(decoded[8:108, 330:].astype(float) - expected[8:108, 330:].astype(float))
        error = float(delta.mean())
        require(
            error < contract["pixel_mae_limit"],
            "Decoded video warning panel differs from CSV/events",
        )
        pixel_max, pixel_sum = max(pixel_max, error), pixel_sum + error
    require(not capture.read()[0], "Video has unexpected trailing frames")
    capture.release()
    if sid == "fall-19":
        require(first_alert == 1500.0 and alerts == 7 and len(events) == 1, "fall-19 pilot failed")
    if sid == "adl-35":
        require(alerts == 0 and not events, "adl-35 pilot false alert")
    peak = max(
        (number(r["combined_k3_ratio"]) for r in windows if r["combined_k3_ratio"]), default=None
    )
    prediction = int(peak > 1) if peak is not None else None
    require(bool(events) == (prediction == 1), "Video decision and events differ")
    valid = len({int(r["sample_index"]) for r in windows if r["row_index"]})
    total = len({int(r["sample_index"]) for r in windows})
    return {
        "sequence_id": sid,
        "label": entry["label"],
        "prediction": prediction,
        "event_count": len(events),
        "first_alarm_ms": number(events[0]["confirmed_start_ms"]) if events else None,
        "current_alert_frames": alerts,
        "first_current_alert_ms": first_alert,
        "n_valid_endpoints": valid,
        "n_scheduled_endpoints": total,
        "frames": len(samples),
        "pose_max_abs_difference": pose_max,
        "feature_max_abs_difference": feature_max,
        "score_max_abs_difference": max_score_difference,
        "pixel_panel_max_mae": pixel_max,
        "pixel_panel_mean_mae": pixel_sum / len(samples),
        "output_sha256": {
            str(x.relative_to(OUT)): sha256(x) for x in p.parent.glob(p.name + "*") if x.is_file()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["pilot", "validation", "verify"])
    args = parser.parse_args()
    contract = protect()
    check_protected(contract)
    if args.phase == "validation":
        pilot = read_json(OUT / "pilot-acceptance.json")
        require(pilot["passed"] and len(pilot["sequences"]) == 2, "Pilot must pass first")
        for result in pilot["sequences"]:
            for name, digest in result["output_sha256"].items():
                require(sha256(OUT / name) == digest, "Pilot output changed")
    selected = list(PILOT) if args.phase == "pilot" else list(contract["entries"])
    results = []
    for sid in selected:
        if args.phase != "verify":
            execute_one(sid)
        print(f"VERIFY video/CSV: {sid}", flush=True)
        results.append(verify_one(sid, contract))
    check_protected(contract)
    target = OUT / (
        "pilot-acceptance.json" if args.phase == "pilot" else "validation-acceptance.json"
    )
    tp = sum(r["label"] == 1 and r["prediction"] == 1 for r in results)
    tn = sum(r["label"] == 0 and r["prediction"] == 0 for r in results)
    fp = sum(r["label"] == 0 and r["prediction"] == 1 for r in results)
    fn = sum(r["label"] == 1 and r["prediction"] == 0 for r in results)
    unknown = sum(r["prediction"] is None for r in results)
    report = {
        "passed": True,
        "evidence": "real_end_to_end",
        "sequences": results,
        "metrics": {
            "n_total": len(results),
            "n_decidable": len(results) - unknown,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "unknown": unknown,
            "overall_accuracy": (tp + tn) / len(results),
            "decidable_accuracy": (tp + tn) / (len(results) - unknown),
            "fall_f1": 2 * tp / (2 * tp + fp + fn),
            "fall_precision": tp / (tp + fp),
            "fall_recall": tp / (tp + fn),
            "window_coverage": sum(r["n_valid_endpoints"] for r in results)
            / sum(r["n_scheduled_endpoints"] for r in results),
            "frames_pixel_verified": sum(r["frames"] for r in results),
        },
        "protected_artifacts": len(contract["protected_sha256"]),
        "test": "unknown; not accessed/evaluated",
        "training": "not performed",
    }
    if target.exists():
        require(read_json(target) == report, "Prior acceptance differs; use a new version")
    else:
        write_json(target, report)
        if args.phase != "pilot":
            write_csv(
                OUT / "validation-sequences.csv",
                [{k: v for k, v in r.items() if k != "output_sha256"} for r in results],
            )
    print(json.dumps(report["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
