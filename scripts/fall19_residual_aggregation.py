"""Run the locked, frozen-wide residual aggregation comparison on Train/Validation."""

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from analyze_fall19_score_overlap import (
    BASE,
    HUMAN,
    HUMAN_ID,
    MOTION,
    MOTION_ID,
    QA,
    QUALITY,
    ROOT,
    VAL,
    WIDE,
    Inputs,
    require,
)

from fall_detect.data.cache import load_pose_cache
from fall_detect.detection.events import score_sequence
from fall_detect.evaluation.metrics import sequence_metrics
from fall_detect.recipes import recipe_spec
from fall_detect.skeleton.preprocessing import good_frame
from fall_detect.skeleton.windows import CausalResampler
from fall_detect.training.experiment import Scorer, cache_for
from fall_detect.visualization.cached_replay import CachedDisplayReplay

TAG = "ur-fall19-residual-aggregation-20261107-001"
OUT = ROOT / "outputs" / TAG
PLAN = ROOT / "docs/ur-fall19-residual-aggregation-20261107-plan.md"
TIMING = ROOT / "outputs/ur-validation-human-timing-20261105-001"
DISPLAY = ROOT / "outputs/ur-validation-display-20261105-001"
TARGET = "fall-19"
TAIL = 26
EARLY = range(33, 44)
K = 3


def relative(path):
    return path.relative_to(ROOT).as_posix()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def save_csv(path, rows, fieldnames=None):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def csv_strings(rows):
    return [{k: "" if v is None else str(v) for k, v in row.items()} for row in rows]


def pin_sources():
    inputs = Inputs()
    frozen = inputs.json(WIDE / "frozen-artifacts.json")
    for name, digest in frozen.items():
        inputs.read(WIDE / name, digest)
    index = inputs.json(WIDE / "dataset-index.json", frozen["dataset-index.json"])
    threshold = inputs.json(WIDE / "threshold.json", frozen["threshold.json"])
    require(frozen["autoencoder.pt"] == threshold["model_sha256"], "Model hash mismatch")
    require(frozen["scaler.npz"] == threshold["scaler_sha256"], "Scaler hash mismatch")
    selected = [
        row
        for row in index["split_manifest"]["sequences"]
        if row["split"] in {"train", "validation"}
    ]
    require(
        Counter((r["split"], r["label"]) for r in selected)
        == {("train", 0): 24, ("validation", 0): 8, ("validation", 1): 15},
        "Unexpected split membership",
    )
    require(len({r["sequence_id"] for r in selected}) == 47, "Duplicate sequence")

    archive = inputs.json(MOTION / "frozen-artifacts.json")
    source = inputs.json(MOTION / "sources.json", archive["sources.json"])
    scoped_index = inputs.json(MOTION / "dataset-index.json", archive["dataset-index.json"])
    motion_lock = inputs.json(QA / MOTION_ID / "lock.json", source["lock_sha256"])
    inputs.read(MOTION / "ae-scores.npz", archive["ae-scores.npz"])
    require(
        source["baseline_model"] == BASE
        and motion_lock["feature_recipe_hash"] == index["recipe_hash"],
        "Wrong archived baseline",
    )
    require(
        [r["sequence_id"] for r in selected if r["split"] == "train"] == source["fit_sequences"],
        "Train source differs",
    )
    entries = {}
    for row in selected:
        sid = row["sequence_id"]
        old = index["sequences"][sid]
        copy = scoped_index["sequences"][sid]
        require(
            old["status"] == copy["status"] == "prepared"
            and old["source_hash"] == copy["source_hash"],
            f"Invalid cache source: {sid}",
        )
        entries[sid] = {
            **row,
            "source_hash": old["source_hash"],
            "valid_features": old["valid_features"],
            "n_samples": old["n_samples"],
        }
        kinds = ("features", "poses") if row["split"] == "validation" else ("features",)
        for kind in kinds:
            key = "feature_path" if kind == "features" else "pose_path"
            path = (ROOT / old[key]).resolve()
            comparison = ROOT / copy[key]
            require(
                path.parent == ROOT / "data/cache/ur-human-isotropic-001" / kind
                and path.name == comparison.name,
                f"Unexpected original cache path: {sid}",
            )
            for suffix in (".npz", ".json"):
                expected = motion_lock["input_hashes"][relative(comparison.with_suffix(suffix))]
                inputs.read(path.with_suffix(suffix), expected)
            entries[sid][key] = relative(path)

    display_pins = inputs.json(QA / "ur-validation-display-20261105-001/inputs.json")["files"]
    original = {
        name: inputs.csv(VAL / f"{name}.csv", display_pins[relative(VAL / f"{name}.csv")])
        for name in ("windows", "sequences", "events")
    }
    original["metrics"] = inputs.json(
        VAL / "metrics.json", display_pins[relative(VAL / "metrics.json")]
    )
    old_display = inputs.json(DISPLAY / "artifact-manifest.json")["files"]
    prior_display = inputs.json(DISPLAY / "summary.json", old_display["summary.json"])
    quality_pins = inputs.json(QUALITY / "artifact-manifest.json")["files"]
    quality_rows = inputs.csv(
        QUALITY / "windows-with-quality.csv", quality_pins["windows-with-quality.csv"]
    )
    human_pins = inputs.json(QA / HUMAN_ID / "final-qa.json")["artifacts"]
    human_path = HUMAN / "human-window-diagnosis.csv"
    early = [
        r
        for r in inputs.csv(human_path, human_pins[relative(human_path)])
        if r["method"] == "wide30"
        and r["sequence_id"] == TARGET
        and r["track_id"] == "1"
        and int(r["sample_index"]) in EARLY
    ]
    require([int(r["sample_index"]) for r in early] == list(EARLY), "Different human early block")
    require(
        all(
            r["quality"] == "valid"
            and r["current_track_good"] == "True"
            and r["human_pose_check_at_last_source"] == "matches_person"
            and r["human_phase_at_last_source"] == "transition"
            for r in early
        ),
        "Early block quality differs",
    )
    timing_pins = inputs.json(TIMING / "artifact-manifest.json")["files"]
    timing = inputs.json(TIMING / "comparisons.json", timing_pins["comparisons.json"])
    for path in (
        PLAN,
        Path(__file__),
        ROOT / "scripts/analyze_fall19_score_overlap.py",
        ROOT / "docs/ur-fall19-score-overlap-20261107.md",
        ROOT / "src/fall_detect/training/experiment.py",
        ROOT / "src/fall_detect/training/scaler.py",
        ROOT / "src/fall_detect/training/residual.py",
        ROOT / "src/fall_detect/models/autoencoder.py",
        ROOT / "src/fall_detect/models/weights.py",
        ROOT / "src/fall_detect/config.py",
        ROOT / "src/fall_detect/recipes.py",
        ROOT / "src/fall_detect/data/cache.py",
        ROOT / "src/fall_detect/detection/events.py",
        ROOT / "src/fall_detect/evaluation/metrics.py",
        ROOT / "src/fall_detect/skeleton/windows.py",
        ROOT / "src/fall_detect/skeleton/preprocessing.py",
        ROOT / "src/fall_detect/visualization/cached_replay.py",
        ROOT / "src/fall_detect/visualization/sample_display.py",
    ):
        inputs.read(path)
    contract = {
        "wide_feature_recipe_hash": index["recipe_hash"],
        "split_hash": index["split_manifest"]["split_hash"],
        "scaler_sha256": frozen["scaler.npz"],
        "ae_sha256": frozen["autoencoder.pt"],
        "mean512": "torch float32 mean of 512 squared residuals",
        "top26": "float64 mean of largest 26 from the same float32 squared residual vector",
        "forward_batch_size": 1,
        "window_size": 30,
        "stride": 1,
        "k": K,
        "comparison": "gt",
        "threshold_rule": "max of per-video K3 maxima for original eight Validation ADL",
        "early_sample_indices": list(EARLY),
        "early_track_id": 1,
        "early_internal_k3_indices": list(EARLY)[K - 1 :],
        "pilot": {
            "adl-05": "first three legal rows",
            "adl-04": "first three legal rows",
            TARGET: "first three rows of the locked early block",
        },
    }
    lock = {
        "id": TAG,
        "locked_at_utc": datetime.now(UTC).isoformat(),
        "baseline": BASE,
        "comparison_contract": contract,
        "comparison_recipe_sha256": hashlib.sha256(
            json.dumps(contract, sort_keys=True).encode()
        ).hexdigest(),
        "allowed_sequences": entries,
        "legacy_coarse_intervals": motion_lock["coarse_intervals"],
        "source_sha256": inputs.hashes,
    }
    return lock, threshold, original, quality_rows, early, timing, prior_display


def restrict_io(locked_paths):
    """Reject extra data, run or output paths after the inventory has been pinned."""
    allowed = {ROOT / path for path in locked_paths}
    protected = [ROOT / name for name in ("data", "runs", "outputs")]
    seen = set()
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND

    def audit(event, args):
        if event != "open" or isinstance(args[0], int):
            return
        raw, _, flags = args
        path = Path(os.fsdecode(raw)).absolute()
        if not any(path.is_relative_to(root) for root in protected) or path.is_relative_to(OUT):
            return
        path = path.resolve()
        if path not in allowed or flags & write_flags:
            raise PermissionError(
                f"Artifact I/O outside the frozen Train/Validation contract: {path}"
            )
        seen.add(relative(path))

    sys.addaudithook(audit)
    return seen


@torch.inference_mode()
def squared_residuals(scorer, features):
    normalized = scorer.scaler.transform(features)
    result, means = [], []
    for start in range(len(normalized)):
        batch = torch.as_tensor(
            normalized[start : start + 1], dtype=torch.float32, device=scorer.device
        )
        error = (batch - scorer.model(batch)) ** 2
        result.append(error)
        means.append(error.mean(dim=1))
    errors = torch.cat(result).cpu().numpy()
    scores = torch.cat(means).cpu().numpy().astype(np.float64)
    require(
        errors.shape == (len(features), 512)
        and np.isfinite(errors).all()
        and np.isfinite(scores).all(),
        "Non-finite or incorrectly shaped residuals",
    )
    return errors, scores


def early_feature_rows(cache):
    rows = {
        r.sample_index: r.row_index
        for r in cache.records
        if r.track_id == 1 and r.sample_index in EARLY
    }
    require(
        set(rows) == set(EARLY) and all(value is not None for value in rows.values()),
        "Locked early feature rows are unavailable",
    )
    return [rows[sample] for sample in EARLY]


def score_all(scorer, caches, archived):
    pilot = {}
    for sid in ("adl-05", "adl-04", TARGET):
        indices = early_feature_rows(caches[sid])[:3] if sid == TARGET else [0, 1, 2]
        _, obtained = squared_residuals(scorer, caches[sid].features[indices])
        expected = archived[sid][indices]
        pilot[sid] = {
            "feature_rows": indices,
            "archived_mean512": expected.tolist(),
            "recomputed_mean512": obtained.tolist(),
            "exact": bool(np.array_equal(expected, obtained)),
        }
    save_json(OUT / "pilot.json", pilot)
    require(
        all(row["exact"] for row in pilot.values()), "Pilot MSE does not exactly match the archive"
    )
    print("Pilot: 9/9 frozen mean512 scores match exactly", flush=True)
    residuals, mean_scores = {}, {}
    mismatches = []
    for sid, cache in caches.items():
        error, raw = squared_residuals(scorer, cache.features)
        expected = archived[sid]
        require(raw.shape == expected.shape, f"Different score count: {sid}")
        if not np.array_equal(expected, raw):
            mismatches.append(
                {
                    "sequence_id": sid,
                    "different_rows": int(np.count_nonzero(expected != raw)),
                    "max_abs_delta": float(np.max(np.abs(expected - raw))),
                }
            )
        residuals[sid], mean_scores[sid] = error, raw
    save_json(
        OUT / "mean512-parity.json",
        {
            "exact": not mismatches,
            "score_rows": sum(map(len, mean_scores.values())),
            "mismatches": mismatches,
        },
    )
    require(not mismatches, "Full MSE parity failed")
    return residuals, mean_scores


def evaluate(caches, scores, validation):
    normal_peaks = {}
    for sid, entry in validation.items():
        if entry["label"] == 0:
            peak, _, _ = score_sequence(caches[sid], scores[sid], 0.0, K)
            require(
                peak["sequence_score"] is not None,
                f"Normal threshold source has no legal K3: {sid}",
            )
            normal_peaks[sid] = peak["sequence_score"]
    require(len(normal_peaks) == 8, "Expected eight normal Validation threshold sources")
    threshold = max(normal_peaks.values())
    videos, windows, events = [], [], []
    for sid, entry in validation.items():
        row, timeline, sequence_events = score_sequence(caches[sid], scores[sid], threshold, K)
        videos.append({**row, "label": entry["label"]})
        windows.extend(timeline)
        events.extend(sequence_events)
    return {
        "threshold": threshold,
        "normal_video_k3": normal_peaks,
        "sequences": videos,
        "windows": windows,
        "events": events,
        "metrics": sequence_metrics(videos),
    }


def replay_validation(config, lock, validation, results, quality_rows):
    spec = recipe_spec(config)
    skeleton = config["skeleton"]
    quality = config["quality"]
    archived_good = {
        (
            r["sequence_id"],
            int(r["sample_index"]),
            int(r["track_id"]) if r["track_id"] else None,
        ): r["current_same_track_confidence_good"] == "True"
        for r in quality_rows
    }
    by_mode = {}
    for mode, result in results.items():
        grouped = defaultdict(list)
        for row in result["windows"]:
            grouped[(row["sequence_id"], row["sample_index"])].append(row)
        by_mode[mode] = grouped
    grid, quality_checks = [], 0
    for sid, entry in validation.items():
        expected = {
            "sequence_id": sid,
            "split": "validation",
            "recipe_hash": lock["comparison_contract"]["wide_feature_recipe_hash"],
            "split_hash": lock["comparison_contract"]["split_hash"],
            "source_hash": entry["source_hash"],
        }
        observations, _ = load_pose_cache(ROOT / entry["pose_path"], expected)
        resampler = CausalResampler(skeleton["target_fps"], skeleton["resampling_hold_policy"])
        samples = [
            sample for frame, poses in observations for sample in resampler.push(frame, poses)
        ]
        require(len(samples) == entry["n_samples"], f"Pose replay grid differs: {sid}")
        replays = {
            mode: CachedDisplayReplay(config, result["threshold"])
            for mode, result in results.items()
        }
        for sample in samples:
            key = (sid, sample.sample_index)
            good_tracks = {
                p.track_id
                for p in sample.poses
                if good_frame(p.keypoints, quality, spec["joints"], spec.get("derived_joints", ()))
            }
            for row in by_mode["mean512"].get(key, []):
                require(
                    archived_good[(*key, row["track_id"])] == (row["track_id"] in good_tracks),
                    f"Original confidence mask differs: {key}",
                )
                quality_checks += 1
            for mode, replay in replays.items():
                saved = by_mode[mode].get(key, [])
                _, current = replay.step(sample, saved)
                raw_alerts = sorted(w["track_id"] for w in saved if w["state"] == "suspected_fall")
                current_alerts = sorted(
                    track for track, state in current.items() if state["state"] == "suspected_fall"
                )
                require(
                    set(current_alerts) <= good_tracks, "Current alert bypassed confidence gate"
                )
                grid.append(
                    {
                        "mode": mode,
                        "sequence_id": sid,
                        "sample_index": sample.sample_index,
                        "timestamp_ms": sample.timestamp_ms,
                        "source_frame_id": sample.source_frame_id,
                        "good_tracks": json.dumps(sorted(good_tracks)),
                        "raw_active_tracks": json.dumps(raw_alerts),
                        "current_alert_tracks": json.dumps(current_alerts),
                        "history_events": replay.history.count,
                    }
                )
    require(
        quality_checks == len(quality_rows), "Some archived window quality gates were not checked"
    )
    return grid, quality_checks


def display_summary(grid, results, validation, coarse):
    groups = defaultdict(list)
    for row in grid:
        groups[(row["mode"], row["sequence_id"])].append(row)
    result = {}
    for mode, evaluation in results.items():
        result[mode] = {}
        for sid in validation:
            rows = groups[mode, sid]
            raw = [r for r in rows if json.loads(r["raw_active_tracks"])]
            current = [r for r in rows if json.loads(r["current_alert_tracks"])]
            events = [e for e in evaluation["events"] if e["sequence_id"] == sid]
            first = events[0] if events else None
            first_visible = None
            if first:
                frame = next(r for r in rows if r["timestamp_ms"] == first["confirmed_start_ms"])
                first_visible = first["track_id"] in json.loads(frame["current_alert_tracks"])
            interval = coarse.get(sid)
            phase_events, phase_visible, early_events = None, None, None
            if interval:
                lo, hi = interval["fall_phase_ms"]
                phase_events = sum(lo <= e["confirmed_start_ms"] <= hi for e in events)
                phase_visible = sum(lo <= r["timestamp_ms"] <= hi for r in current)
                early_events = sum(
                    e["confirmed_start_ms"] < interval["normal_until_ms"] for e in events
                )
            result[mode][sid] = {
                "grid_frames": len(rows),
                "raw_active_frames": len(raw),
                "current_alert_frames": len(current),
                "masked_active_frames": sum(not json.loads(r["current_alert_tracks"]) for r in raw),
                "history_recorded_frames": sum(r["history_events"] > 0 for r in rows),
                "events": len(events),
                "first_current_alert_ms": current[0]["timestamp_ms"] if current else None,
                "first_event_current_visible": first_visible,
                "legacy_phase_events": phase_events,
                "legacy_phase_current_alert_frames": phase_visible,
                "legacy_early_events": early_events,
            }
    return result


def target_report(early_human, results, grid):
    target = {}
    table = []
    for mode, result in results.items():
        timeline = [
            r for r in result["windows"] if r["sequence_id"] == TARGET and r["track_id"] == 1
        ]
        selected = {r["sample_index"]: r for r in timeline if r["sample_index"] in EARLY}
        display = {
            r["sample_index"]: r for r in grid if r["mode"] == mode and r["sequence_id"] == TARGET
        }
        starts = {
            e["confirmed_start_ms"]
            for e in result["events"]
            if e["sequence_id"] == TARGET and e["track_id"] == 1
        }
        for annotation in early_human:
            sample = int(annotation["sample_index"])
            window = selected[sample]
            require(
                display[sample]["source_frame_id"] == int(annotation["last_actual_source_frame_id"])
                and 1 in json.loads(display[sample]["good_tracks"]),
                "Early pose alignment differs",
            )
            internal = sample >= EARLY.start + K - 1
            table.append(
                {
                    "mode": mode,
                    "sample_index": sample,
                    "source_frame_id": int(annotation["last_actual_source_frame_id"]),
                    "available_at_ms": window["available_at_ms"],
                    "raw_score": window["raw_score"],
                    "event_score": window["event_score"],
                    "threshold": result["threshold"],
                    "internal_k3": internal,
                    "internal_k3_over_threshold": internal
                    and window["event_score"] > result["threshold"],
                    "internal_event_confirmed": internal and window["available_at_ms"] in starts,
                    "current_alert": 1 in json.loads(display[sample]["current_alert_tracks"]),
                }
            )
        block = [r for r in table if r["mode"] == mode]
        internal_rows = [r for r in block if r["internal_k3"]]
        early_peak = max(r["event_score"] for r in internal_rows)
        full = next(r for r in result["sequences"] if r["sequence_id"] == TARGET)
        target[mode] = {
            "early_raw_windows": len(block),
            "internal_k3_windows": len(internal_rows),
            "early_raw_peak": max(r["raw_score"] for r in block),
            "early_internal_k3_peak": early_peak,
            "early_internal_k3_to_threshold": early_peak / result["threshold"],
            "early_internal_k3_above": sum(r["internal_k3_over_threshold"] for r in block),
            "early_internal_confirmations": sum(r["internal_event_confirmed"] for r in block),
            "early_internal_current_alerts": sum(r["current_alert"] for r in internal_rows),
            "full_raw_peak": max(r["raw_score"] for r in timeline if r["raw_score"] is not None),
            "full_k3_peak": full["sequence_score"],
            "full_k3_to_threshold": full["sequence_score"] / result["threshold"],
            "full_prediction": full["prediction"],
            "full_events": len(starts),
            "first_alarm_ms": full["first_alarm_ms"],
            "valid_endpoints": full["n_valid_endpoints"],
            "scheduled_endpoints": full["n_scheduled_endpoints"],
            "normal_videos_at_or_above_early_k3": sum(
                v >= early_peak for v in result["normal_video_k3"].values()
            ),
            "normal_videos_at_or_above_full_k3": sum(
                v >= full["sequence_score"] for v in result["normal_video_k3"].values()
            ),
        }
    return target, table


def check_old_clip_display(prior, grid):
    for row in prior["rows"]:
        if row["model"] != "wide":
            continue
        media = next(
            v
            for v in prior["videos"]
            if v["model"] == "wide" and v["sequence_id"] == row["sequence_id"]
        )
        selected = [
            g
            for g in grid
            if g["mode"] == "mean512"
            and g["sequence_id"] == row["sequence_id"]
            and media["grid_start_ms"] - 1e-7 <= g["timestamp_ms"] <= media["grid_end_ms"] + 1e-7
        ]
        require(len(selected) == row["rendered_frames"], "Original comparison clip grid differs")
        require(
            sum(bool(json.loads(r["current_alert_tracks"])) for r in selected)
            == row["current_ae_alert_frames"]
            and sum(r["history_events"] > 0 for r in selected) == row["history_recorded_frames"],
            "Baseline display differs from the verified short clip",
        )


def timed_cases(timing, results):
    rows = []
    for case in timing:
        sid = case["sequence_id"]
        for mode, result in results.items():
            first = next(
                r["first_alarm_ms"] for r in result["sequences"] if r["sequence_id"] == sid
            )
            lo, hi = case["impact_bounds_ms"]
            relation = (
                "no_event"
                if first is None
                else "before_impact_interval"
                if first < lo
                else "after_impact_interval"
                if first > hi
                else "inside_impact_uncertainty"
            )
            rows.append(
                {
                    "mode": mode,
                    "sequence_id": sid,
                    "first_alarm_ms": first,
                    "impact_earliest_ms": lo,
                    "impact_latest_ms": hi,
                    "relation": relation,
                }
            )
    return rows


def compare_videos(results, display):
    candidate = {r["sequence_id"]: r for r in results["top26"]["sequences"]}
    comparison = []
    for old in results["mean512"]["sequences"]:
        sid = old["sequence_id"]
        new = candidate[sid]
        require(
            (new["n_scheduled_endpoints"], new["n_valid_endpoints"])
            == (old["n_scheduled_endpoints"], old["n_valid_endpoints"]),
            "Quality coverage changed",
        )
        row = {
            "sequence_id": sid,
            "label": old["label"],
            "scheduled_endpoints": old["n_scheduled_endpoints"],
            "valid_endpoints": old["n_valid_endpoints"],
        }
        for mode, sequence in (("mean512", old), ("top26", new)):
            row.update(
                {
                    f"{mode}_{key}": value
                    for key, value in sequence.items()
                    if key in {"sequence_score", "prediction", "first_alarm_ms"}
                }
            )
            row.update({f"{mode}_{key}": value for key, value in display[mode][sid].items()})
        comparison.append(row)
    return comparison


def group_tail_share(residuals, tail, validation, locked):
    groups = {"train_adl": [], "validation_adl": [], "validation_fall": []}
    for sid in residuals:
        totals = residuals[sid].sum(axis=1, dtype=np.float64)
        share = np.divide(tail[sid] * TAIL, totals, out=np.zeros_like(totals), where=totals > 0)
        group = (
            "train_adl"
            if sid not in validation
            else "validation_adl"
            if validation[sid]["label"] == 0
            else "validation_fall"
        )
        groups[group].extend(share.tolist())
        if sid == TARGET:
            groups["fall19_early"] = share[locked].tolist()
    return {
        group: {
            "rows": len(values),
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
        }
        for group, values in groups.items()
    }


def check_hashes(files, root=ROOT):
    for path, digest in files.items():
        require(sha(root / path) == digest, f"Frozen file changed: {path}")


def build():
    if OUT.exists():
        raise FileExistsError(f"Preserve existing experiment: {OUT}")
    lock, threshold, original, quality_rows, early, timing, prior_display = pin_sources()
    OUT.mkdir(exist_ok=False)
    save_json(OUT / "lock.json", lock)
    observed_io = restrict_io(lock["source_sha256"])
    scorer = Scorer(WIDE)
    config = scorer.config
    require(
        scorer.residual_weights is None and config["anomaly_score"]["mode"] == "mse",
        "The original wide score is not plain MSE",
    )
    require(
        config["skeleton"]["window_size"] == 30
        and config["skeleton"]["stride"] == 1
        and config["skeleton"]["coordinate_normalization"] == "per_frame_isotropic"
        and config["calibration"]["consecutive_windows"] == K,
        "Different frozen scoring configuration",
    )
    entries = lock["allowed_sequences"]
    validation = {sid: row for sid, row in entries.items() if row["split"] == "validation"}
    scoped = {**scorer.index, "preparation_scope": ["train", "validation"]}
    caches = {sid: cache_for(scoped, sid) for sid in entries}
    for sid, cache in caches.items():
        require(
            cache.features.shape == (entries[sid]["valid_features"], 512),
            "Feature count differs from frozen index",
        )
        if sid in validation:
            require(
                cache.metadata["array_sha256"] == threshold["feature_hashes"][sid],
                "Validation feature hash differs",
            )
    with np.load(MOTION / "ae-scores.npz", allow_pickle=False) as archived:
        require(
            set(archived.files) == set(entries),
            "Archive must contain exactly Train/Validation scores",
        )
        residuals, mean_scores = score_all(scorer, caches, archived)
    require(sum(map(len, mean_scores.values())) == 6001, "Unexpected total score count")
    baseline = evaluate(caches, mean_scores, validation)
    require(baseline["threshold"] == threshold["threshold"], "Original threshold not reproduced")
    require(baseline["metrics"] == original["metrics"], "Original metrics not reproduced")
    for name in ("windows", "sequences", "events"):
        require(
            csv_strings(baseline[name]) == original[name], f"Original {name} CSV not reproduced"
        )
    print(
        "All 6,001 scores and the original Validation threshold, windows and events match",
        flush=True,
    )

    tail = {
        sid: np.sort(error, axis=1)[:, -TAIL:].mean(axis=1, dtype=np.float64)
        for sid, error in residuals.items()
    }
    results = {"mean512": baseline, "top26": evaluate(caches, tail, validation)}
    grid, checked_quality = replay_validation(config, lock, validation, results, quality_rows)
    check_old_clip_display(prior_display, grid)
    display = display_summary(grid, results, validation, lock["legacy_coarse_intervals"])
    case, early_table = target_report(early, results, grid)
    comparison = compare_videos(results, display)
    timing_comparison = timed_cases(timing, results)
    protected = [r for r in comparison if r["label"] == 1 and r["mean512_prediction"] == 1]
    require(len(protected) == 14, "Original fall true positives differ")
    old_tp_lost = [r["sequence_id"] for r in protected if r["top26_prediction"] != 1]
    guards = {
        mode: {
            "legacy_phase_hit_videos": sum(
                (r["legacy_phase_events"] or 0) > 0 for r in rows.values()
            ),
            "legacy_phase_current_alert_videos": sum(
                (r["legacy_phase_current_alert_frames"] or 0) > 0 for r in rows.values()
            ),
            "legacy_early_event_videos": [
                sid for sid, r in rows.items() if r["legacy_early_events"]
            ],
            "new_fall19_has_current_alert_in_legacy_phase": bool(
                rows[TARGET]["legacy_phase_current_alert_frames"]
            ),
        }
        for mode, rows in display.items()
    }
    early_recovered = (
        case["top26"]["early_internal_confirmations"] > 0
        and case["top26"]["early_internal_current_alerts"] > 0
    )
    candidate = results["top26"]["metrics"]
    passes = (
        early_recovered
        and not old_tp_lost
        and candidate["FP"] == candidate["n_unknown"] == 0
        and guards["top26"]["legacy_phase_hit_videos"] >= 4
        and not guards["top26"]["legacy_early_event_videos"]
        and guards["top26"]["new_fall19_has_current_alert_in_legacy_phase"]
    )

    for mode, result in results.items():
        folder = OUT / mode
        folder.mkdir(exist_ok=False)
        for name in ("sequences", "windows", "events"):
            save_csv(folder / f"{name}.csv", result[name], list(original[name][0]))
        save_json(folder / "metrics.json", result["metrics"])
    save_csv(OUT / "validation-comparison.csv", comparison)
    save_csv(OUT / "display-grid.csv", grid)
    save_csv(OUT / "fall19-early.csv", early_table)
    save_csv(OUT / "human-impact-comparison.csv", timing_comparison)
    with (OUT / "residuals.npz").open("xb") as stream:
        np.savez_compressed(stream, **residuals)
    with (OUT / "scores.npz").open("xb") as stream:
        np.savez_compressed(
            stream,
            **{
                f"{sid}__{mode}": values
                for mode, group in (("mean512", mean_scores), ("top26", tail))
                for sid, values in group.items()
            },
        )
    summary = {
        "id": TAG,
        "device": str(scorer.device),
        "torch": torch.__version__,
        "comparison_recipe_sha256": lock["comparison_recipe_sha256"],
        "mean512_exact_score_rows": 6001,
        "strict_frozen_ae_load": True,
        "scores": {
            mode: {
                "threshold": result["threshold"],
                "normal_video_k3": result["normal_video_k3"],
                "metrics": result["metrics"],
                "event_count": len(result["events"]),
            }
            for mode, result in results.items()
        },
        "fall19": case,
        "early_recovered": early_recovered,
        "old_tp_preserved": len(protected) - len(old_tp_lost),
        "old_tp_lost": old_tp_lost,
        "display": display,
        "legacy_display_guards": guards,
        "human_impact_comparison": timing_comparison,
        "quality_window_rows_checked": checked_quality,
        "display_grid_rows": len(grid),
        "top26_residual_share": group_tail_share(
            residuals, tail, validation, early_feature_rows(caches[TARGET])
        ),
        "recognition_and_display_guard_pass": passes,
        "training_epochs": 0,
        "test_evaluations": 0,
        "evidence_kind": "frozen_real_feature_inference_and_cached_display_state_replay",
        "real_rgb_end_to_end": "not_run",
    }
    save_json(OUT / "summary.json", summary)
    check_hashes(lock["source_sha256"])
    save_json(
        OUT / "artifact-io.json",
        {
            "protected_artifact_reads": sorted(observed_io),
            "old_source_hashes_unchanged": len(lock["source_sha256"]),
            "test_data_reads": 0,
            "writes_outside_new_output_in_protected_roots": 0,
        },
    )
    files = {p.relative_to(OUT).as_posix(): sha(p) for p in OUT.rglob("*") if p.is_file()}
    save_json(OUT / "artifact-manifest.json", {"files": files})
    print(
        json.dumps(
            {
                "output": str(OUT),
                "fall19": case,
                "old_tp_preserved": summary["old_tp_preserved"],
                "mean512_threshold": baseline["threshold"],
                "top26_threshold": results["top26"]["threshold"],
                "candidate_metrics": candidate,
                "display_guards": guards["top26"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def verify():
    lock = json.loads((OUT / "lock.json").read_text(encoding="utf-8"))
    manifest = json.loads((OUT / "artifact-manifest.json").read_text(encoding="utf-8"))
    check_hashes(lock["source_sha256"])
    check_hashes(manifest["files"], OUT)
    print(
        f"Verified {len(lock['source_sha256'])} frozen inputs and {len(manifest['files'])} outputs"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("build", "verify"))
    args = parser.parse_args()
    (build if args.stage == "build" else verify)()
