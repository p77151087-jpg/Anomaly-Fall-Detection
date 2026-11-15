"""Audit frozen wide AE score overlap using the original Train and Validation evidence."""

import argparse
import csv
import hashlib
import io
import json
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = "ur-isotropic-wide-20261012-001"
MOTION_ID = "ur-motion-novelty-20261012-001"
HUMAN_ID = "ur-fall19-human-amendment-20261013-001"
WIDE = ROOT / "runs" / BASE
MOTION = ROOT / "runs" / MOTION_ID
VAL = ROOT / "outputs/isotropic-refinement-20261012" / BASE
QUALITY = ROOT / "outputs/ur-validation-alert-quality-20261013-001"
HUMAN = ROOT / "outputs" / HUMAN_ID
QA = ROOT / "docs/validation"
OUTPUT = ROOT / "outputs/ur-fall19-score-overlap-20261107-001"
TARGET = "fall-19"
K = 3


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Inputs:
    def __init__(self):
        self.hashes = {}

    def read(self, path, expected=None):
        relative = path.relative_to(ROOT).as_posix()
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        require(expected is None or digest == expected, f"Pinned input changed: {relative}")
        require(self.hashes.setdefault(relative, digest) == digest, f"Input changed: {relative}")
        return content

    def json(self, path, expected=None):
        return json.loads(self.read(path, expected))

    def csv(self, path, expected):
        return list(csv.DictReader(io.StringIO(self.read(path, expected).decode("utf-8-sig"))))


@dataclass
class Window:
    sequence_id: str
    track: int | None
    sample: int
    time_ms: float
    raw: float | None
    quality: str
    k3: float | None = None
    current_good: bool | None = None
    source_frame: int | None = None
    phase: str | None = None
    human_pose: str | None = None

    @property
    def key(self):
        return self.sequence_id, self.track, self.sample


def legal_runs(windows):
    tracks = defaultdict(list)
    for window in windows:
        tracks[window.track].append(window)
    for track_windows in tracks.values():
        run = []
        for window in sorted(track_windows, key=lambda w: w.sample):
            if run and (window.raw is None or window.sample != run[-1].sample + 1):
                yield run
                run = []
            if window.raw is not None:
                run.append(window)
        if run:
            yield run


def key(row):
    return (
        row["sequence_id"],
        int(row["track_id"]) if row["track_id"] else None,
        int(row["sample_index"]),
    )


def number(value):
    return float(value) if value != "" else None


def boolean(value):
    require(value in {"True", "False"}, f"Invalid archived Boolean: {value}")
    return value == "True"


def load_archived_windows(inputs):
    frozen = inputs.json(MOTION / "frozen-artifacts.json")
    index = inputs.json(MOTION / "dataset-index.json", frozen["dataset-index.json"])
    source = inputs.json(MOTION / "sources.json", frozen["sources.json"])
    scale = inputs.json(MOTION / "normalization.json", frozen["normalization.json"])
    lock = inputs.json(QA / MOTION_ID / "lock.json", source["lock_sha256"])
    pinned = lock["input_hashes"]
    threshold_path = WIDE / "threshold.json"
    threshold = inputs.json(threshold_path, pinned[threshold_path.relative_to(ROOT).as_posix()])
    manifest = index["split_manifest"]
    selected = {
        row["sequence_id"]: row
        for row in manifest["sequences"]
        if row["split"] in {"train", "validation"}
    }
    train_ids = [sid for sid, row in selected.items() if row["split"] == "train"]
    val_ids = [sid for sid, row in selected.items() if row["split"] == "validation"]
    require(index["preparation_scope"] == ["train", "validation"], "Unexpected preparation scope")
    require(
        len(train_ids) == 24 and all(selected[sid]["label"] == 0 for sid in train_ids),
        "Expected original 24 Train ADL",
    )
    require(
        Counter(selected[sid]["label"] for sid in val_ids) == {0: 8, 1: 15},
        "Unexpected Validation membership",
    )
    require(
        train_ids == source["fit_sequences"] == scale["fit_sequences"], "Train source order changed"
    )
    require(
        source["baseline_model"] == BASE and scale["fit_split"] == "train" and scale["normal_only"],
        "Unexpected AE score source",
    )
    require(
        threshold["split"] == "validation"
        and threshold["comparison"] == "gt"
        and threshold["consecutive_windows"] == K,
        "Unexpected event rule",
    )
    require(threshold["split_hash"] == manifest["split_hash"], "Different split hash")
    require(
        threshold["recipe_hash"]
        == index["recipe_hash"]
        == source["feature_recipe_hash"]
        == lock["feature_recipe_hash"],
        "Different recipe",
    )
    for filename, field in [("autoencoder.pt", "model_sha256"), ("scaler.npz", "scaler_sha256")]:
        require(
            pinned[(WIDE / filename).relative_to(ROOT).as_posix()] == threshold[field],
            "Different model or Scaler provenance",
        )

    windows = {}
    arrays = {}
    archive = inputs.read(MOTION / "ae-scores.npz", frozen["ae-scores.npz"])
    with np.load(io.BytesIO(archive), allow_pickle=False) as values:
        require(
            set(values.files) == set(selected),
            "Score archive must contain exactly the 47 allowed sequences",
        )
        for sid, row in selected.items():
            entry = index["sequences"][sid]
            metadata_path = (ROOT / entry["feature_path"]).with_suffix(".json").resolve()
            require(
                metadata_path.parent == ROOT / "data/cache" / MOTION_ID / "features",
                "Unexpected metadata path",
            )
            metadata = inputs.json(
                metadata_path, pinned[metadata_path.relative_to(ROOT).as_posix()]
            )
            require(
                entry["status"] == "prepared"
                and metadata["sequence_id"] == sid
                and metadata["split"] == row["split"],
                "Mismatched window identity",
            )
            require(
                metadata["source_hash"] == entry["source_hash"]
                and metadata["split_hash"] == threshold["split_hash"]
                and metadata["recipe_hash"] == threshold["recipe_hash"],
                "Mismatched feature provenance",
            )
            require(
                metadata["window_size"] == 30
                and metadata["stride"] == 1
                and metadata["feature_dim"] == 512,
                "Unexpected window contract",
            )
            if row["split"] == "validation":
                require(
                    metadata["array_sha256"] == threshold["feature_hashes"][sid],
                    "Different Validation feature hash",
                )
            scores = arrays[sid] = values[sid]
            require(
                scores.shape == (entry["valid_features"],)
                and np.isfinite(scores).all()
                and (scores >= 0).all(),
                "Invalid archived scores",
            )
            used = []
            windows[sid] = sequence = []
            for record in metadata["records"]:
                require(
                    record["sequence_id"] == sid and record["split"] == row["split"],
                    "Mismatched window membership",
                )
                index_in_archive = record["row_index"]
                require(
                    (index_in_archive is not None) == (record["quality"] == "valid"),
                    "Quality and score presence disagree",
                )
                raw = None
                if index_in_archive is not None:
                    used.append(index_in_archive)
                    raw = float(scores[index_in_archive])
                sequence.append(
                    Window(
                        sid,
                        record["track_id"],
                        record["sample_index"],
                        record["available_at_ms"],
                        raw,
                        record["quality"],
                    )
                )
            require(
                sorted(used) == list(range(len(scores)))
                and len({w.key for w in sequence}) == len(sequence),
                "Invalid window row mapping",
            )
            for run in legal_runs(sequence):
                for left, middle, right in zip(run, run[1:], run[2:], strict=False):
                    right.k3 = min(left.raw, middle.raw, right.raw)

    train_archive = inputs.read(MOTION / "train-scale-scores.npz", frozen["train-scale-scores.npz"])
    with np.load(io.BytesIO(train_archive), allow_pickle=False) as values:
        train_scores = np.concatenate([arrays[sid] for sid in train_ids])
        train_groups = np.concatenate([np.repeat(sid, len(arrays[sid])) for sid in train_ids])
        require(
            len(train_scores) == source["train_windows"] == 3638, "Unexpected Train score count"
        )
        require(
            np.array_equal(train_scores, values["ae"])
            and np.array_equal(train_groups, values["groups"]),
            "Train archives disagree",
        )
        require(
            float(np.quantile(train_scores, 0.95)) == scale["ae_p95"], "Archived Train p95 mismatch"
        )
    return windows, selected, train_ids, val_ids, threshold


def validate_formal_outputs(inputs, windows, val_ids, selected, threshold):
    prior = inputs.json(QA / "ur-validation-display-20261105-001/inputs.json")["files"]
    wide_windows = inputs.csv(
        VAL / "windows.csv", prior[(VAL / "windows.csv").relative_to(ROOT).as_posix()]
    )
    formal = {key(row): row for row in wide_windows}
    quality_hashes = inputs.json(QUALITY / "artifact-manifest.json")["files"]
    enriched = {
        key(row): row
        for row in inputs.csv(
            QUALITY / "windows-with-quality.csv", quality_hashes["windows-with-quality.csv"]
        )
    }
    val_windows = {w.key: w for sid in val_ids for w in windows[sid]}
    require(
        formal.keys() == enriched.keys() == val_windows.keys() and len(wide_windows) == len(formal),
        "Different Validation timelines",
    )
    event_starts = []
    for sid in val_ids:
        for run in legal_runs(windows[sid]):
            active = False
            for w in run:
                next_active = w.k3 is not None and w.k3 > threshold
                if next_active and not active:
                    event_starts.append((sid, w.track, w.time_ms))
                active = next_active
    for identity, row in formal.items():
        w = val_windows[identity]
        state = (
            "unknown"
            if w.raw is None
            else "warming_up"
            if w.k3 is None
            else "suspected_fall"
            if w.k3 > threshold
            else "normal"
        )
        for saved in (row, enriched[identity]):
            require(
                (
                    number(saved["raw_score"]),
                    number(saved["event_score"]),
                    float(saved["available_at_ms"]),
                    saved["state"],
                )
                == (w.raw, w.k3, w.time_ms, state),
                f"Validation score or state differs: {identity}",
            )
            require(
                w.raw is not None or saved["reason"] == w.quality,
                f"Validation quality differs: {identity}",
            )
        w.current_good = boolean(enriched[identity]["current_same_track_confidence_good"])

    events_path = VAL / "events.csv"
    events = inputs.csv(events_path, prior[events_path.relative_to(ROOT).as_posix()])
    require(
        sorted(event_starts)
        == sorted(
            (row["sequence_id"], int(row["track_id"]), float(row["confirmed_start_ms"]))
            for row in events
        ),
        "CSV event starts disagree",
    )
    path = VAL / "sequences.csv"
    rows = inputs.csv(path, prior[path.relative_to(ROOT).as_posix()])
    require(
        len(rows) == len(val_ids) and {row["sequence_id"] for row in rows} == set(val_ids),
        "Different Validation video set",
    )
    for row in rows:
        sid = row["sequence_id"]
        scores = [w.k3 for w in windows[sid] if w.k3 is not None]
        require(
            int(row["label"]) == selected[sid]["label"]
            and float(row["sequence_score"]) == max(scores),
            "Validation sequence label or score differs",
        )
        require(
            int(row["prediction"]) == (max(scores) > threshold), "Validation prediction differs"
        )
    return {
        "window_rows": len(formal),
        "raw_scores": sum(w.raw is not None for w in val_windows.values()),
        "event_starts": len(events),
        "sequence_rows": len(rows),
    }


def attach_human_phases(inputs, target):
    pinned = inputs.json(QA / HUMAN_ID / "final-qa.json")["artifacts"]
    frame_path = HUMAN / "human-frame-diagnosis.csv"
    frames = inputs.csv(frame_path, pinned[frame_path.relative_to(ROOT).as_posix()])
    path = HUMAN / "human-window-diagnosis.csv"
    rows = {
        key(row): row
        for row in inputs.csv(path, pinned[path.relative_to(ROOT).as_posix()])
        if row["method"] == "wide30"
    }
    require(
        len(frames) == 100 and set(rows) == {w.key for w in target},
        "Human timeline does not cover the target",
    )
    timestamps = [float(row["timestamp_ms"]) for row in frames]
    require(all(a < b for a, b in pairwise(timestamps)), "Human timestamps out of order")
    for w in target:
        row = rows[w.key]
        index = bisect_right(timestamps, w.time_ms) - 1
        require(index >= 0, "No causal source frame")
        frame = frames[index]
        w.source_frame = int(frame["source_frame_id"])
        w.phase = frame["phase"]
        w.human_pose = frame["pose_check"]
        require(
            int(row["last_actual_source_frame_id"]) == w.source_frame
            and row["human_phase_at_last_source"] == w.phase
            and row["human_pose_check_at_last_source"] == w.human_pose,
            "Human phase alignment changed",
        )
        require(
            (
                number(row["raw_score"]),
                number(row["event_score"]),
                boolean(row["current_track_good"]),
            )
            == (w.raw, w.k3, w.current_good),
            "Human score or quality disagrees",
        )
    return {
        phase: [min(values), max(values)]
        for phase in ("upright", "transition", "ground")
        if (values := [float(row["timestamp_ms"]) for row in frames if row["phase"] == phase])
    }


def describe(values):
    if not values:
        return {"n": 0}
    levels = {"min": 0, "p50": 0.5, "p95": 0.95, "p99": 0.99, "max": 1}
    return {"n": len(values), **{name: float(np.quantile(values, q)) for name, q in levels.items()}}


def scores(windows, field):
    return [value for w in windows if (value := getattr(w, field)) is not None]


def peak(windows, field):
    candidates = [w for w in windows if getattr(w, field) is not None]
    return vars(max(candidates, key=lambda w: getattr(w, field))).copy() if candidates else None


def phase_summary(windows):
    good = [w for w in windows if w.current_good]
    matches = [w for w in good if w.human_pose == "matches_person"]
    runs = list(legal_runs(windows))
    return {
        "scheduled": len(windows),
        "legal_raw": len(scores(windows, "raw")),
        "legal_k3": len(scores(windows, "k3")),
        "longest_legal_run": max(map(len, runs), default=0),
        "current_good_legal_raw": len(scores(good, "raw")),
        "current_good_legal_k3": len(scores(good, "k3")),
        "human_matches_good_legal_raw": len(scores(matches, "raw")),
        "invalid_reasons": dict(Counter(w.quality for w in windows if w.raw is None)),
        "raw_peak": peak(windows, "raw"),
        "k3_peak": peak(windows, "k3"),
        "raw_peak_when_human_matches_and_current_good": peak(matches, "raw"),
        "k3_peak_when_human_matches_and_current_good": peak(matches, "k3"),
    }


def compare_blocks(target, controls):
    """Give each same-track block the same number of chances for an extreme score."""
    run = max(legal_runs(target), key=len)
    count = len(run)
    require(count >= K, "Matched analysis requires three consecutive legal windows")
    target_raw, target_k3 = max(scores(run, "raw")), max(scores(run[K - 1 :], "k3"))
    result = []
    for sid, split, windows in controls:
        blocks = [r[i : i + count] for r in legal_runs(windows) for i in range(len(r) - count + 1)]
        raw = [max(scores(block, "raw")) for block in blocks]
        k3 = [max(scores(block[K - 1 :], "k3")) for block in blocks]
        result.append(
            {
                "sequence_id": sid,
                "split": split,
                "n_blocks": len(blocks),
                "raw_block_peaks": describe(raw),
                "k3_block_peaks": describe(k3),
                "raw_blocks_at_least_target": sum(v >= target_raw for v in raw),
                "k3_blocks_at_least_target": sum(v >= target_k3 for v in k3),
            }
        )
    return {
        "n_raw_windows_per_block": count,
        "n_k3_windows_per_block": count - K + 1,
        "target_first_ms": run[0].time_ms,
        "target_last_ms": run[-1].time_ms,
        "target_raw_peak": target_raw,
        "target_k3_peak": target_k3,
        "controls": result,
    }


def analyze():
    inputs = Inputs()
    windows, selected, train_ids, val_ids, threshold_data = load_archived_windows(inputs)
    threshold = threshold_data["threshold"]
    verification = validate_formal_outputs(inputs, windows, val_ids, selected, threshold)
    target = windows[TARGET]
    phase_ranges = attach_human_phases(inputs, target)
    normal_val = [sid for sid in val_ids if selected[sid]["label"] == 0]
    phase_windows = {phase: [w for w in target if w.phase == phase] for phase in phase_ranges}
    transition = phase_windows["transition"]
    good_matches = [w for w in transition if w.current_good and w.human_pose == "matches_person"]
    train = [w for sid in train_ids for w in windows[sid]]
    validation = [w for sid in normal_val for w in windows[sid]]
    populations = {
        "train_adl": train,
        "validation_adl": validation,
        "fall19_transition": transition,
        "fall19_transition_good_and_human_matches": good_matches,
    }
    distribution = {
        group: {field: describe(scores(items, field)) for field in ("raw", "k3")}
        for group, items in populations.items()
    }
    ranking = {}
    for group in ("train_adl", "validation_adl"):
        ranking[group] = {}
        for field in ("raw", "k3"):
            reference = scores(populations[group], field)
            ranking[group][field] = {
                name: {
                    "reference_at_or_below_target": sum(score <= value for score in reference),
                    "reference_n": len(reference),
                    "target": value,
                }
                for name, value in (
                    ("transition_median", distribution["fall19_transition"][field]["p50"]),
                    ("transition_peak", distribution["fall19_transition"][field]["max"]),
                    (
                        "good_matches_peak",
                        distribution["fall19_transition_good_and_human_matches"][field]["max"],
                    ),
                )
            }

    video_scores = []
    for sid, items in windows.items():
        row = selected[sid]
        video_scores.append(
            {
                "sequence_id": sid,
                "split": row["split"],
                "label": row["label"],
                "legal_raw": len(scores(items, "raw")),
                "legal_k3": len(scores(items, "k3")),
                "raw_peak": max(scores(items, "raw"), default=None),
                "k3_peak": max(scores(items, "k3"), default=None),
            }
        )
    target_video = next(row for row in video_scores if row["sequence_id"] == TARGET)
    val_adl_scores = [row for row in video_scores if row["sequence_id"] in normal_val]
    normal_ceiling = max(row["k3_peak"] for row in val_adl_scores)
    controls = [(sid, selected[sid]["split"], windows[sid]) for sid in train_ids + normal_val]
    score_boundaries = {
        "strict_gt_threshold": threshold,
        "lowest_zero_fp_k3_threshold_on_these_eight_adl": normal_ceiling,
        "fall19_raw_above_threshold": sum(v > threshold for v in scores(target, "raw")),
        "fall19_k3_above_threshold": sum(v > threshold for v in scores(target, "k3")),
        "normal_val_ids_flagged_by_k1_at_frozen_threshold": [
            row["sequence_id"] for row in val_adl_scores if row["raw_peak"] > threshold
        ],
        "normal_val_ids_at_or_above_fall19_k3_peak": [
            row["sequence_id"]
            for row in val_adl_scores
            if row["k3_peak"] >= target_video["k3_peak"]
        ],
        "normal_val_ids_at_or_above_fall19_raw_peak": [
            row["sequence_id"]
            for row in val_adl_scores
            if row["raw_peak"] >= target_video["raw_peak"]
        ],
        "normal_train_ids_at_or_above_fall19_k3_peak": [
            row["sequence_id"]
            for row in video_scores
            if row["split"] == "train"
            and row["k3_peak"] is not None
            and row["k3_peak"] >= target_video["k3_peak"]
        ],
    }
    summary = {
        "baseline": BASE,
        "split_hash": threshold_data["split_hash"],
        "recipe_hash": threshold_data["recipe_hash"],
        "source_sha256": inputs.hashes,
        "baseline_provenance_hashes": {
            "model": threshold_data["model_sha256"],
            "scaler": threshold_data["scaler_sha256"],
        },
        "train_adl_videos": len(train_ids),
        "validation_adl_videos": len(normal_val),
        "validation_fall_videos": len(val_ids) - len(normal_val),
        "formal_validation_recheck": verification,
        "fall19_coverage": phase_summary(target),
        "fall19_phase_ranges_ms": phase_ranges,
        "fall19_phases": {phase: phase_summary(items) for phase, items in phase_windows.items()},
        "window_distributions": distribution,
        "window_empirical_ranks": ranking,
        "video_scores": video_scores,
        "equal_window_budget": compare_blocks(transition, controls),
        "equal_window_budget_current_good": compare_blocks(
            good_matches,
            [
                (sid, "validation", [w for w in windows[sid] if w.current_good])
                for sid in normal_val
            ],
        ),
        "score_boundaries": score_boundaries,
        "training_inference_calibration_test_runs": 0,
    }
    return summary, target, populations, val_adl_scores


def render_figure(summary, target, populations, normal_videos):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    threshold = summary["score_boundaries"]["strict_gt_threshold"]
    blue, teal, red, ink = "#205887", "#237a69", "#ad422f", "#25374b"
    fig, axes = plt.subplot_mosaic(
        "AB\nCC", figsize=(12, 8.4), layout="constrained", height_ratios=[1, 0.86]
    )
    fig.suptitle(
        "fall-19 | Where the frozen wide AE loses separation",
        fontsize=17,
        fontweight="bold",
        color=ink,
    )
    for ax in axes.values():
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.16)
    ax = axes["A"]
    for group, title, color in (
        ("train_adl", "Train ADL", blue),
        ("validation_adl", "Validation ADL", teal),
        ("fall19_transition", "fall-19 transition", red),
    ):
        values = np.sort(scores(populations[group], "raw"))
        ax.step(
            values,
            np.arange(1, len(values) + 1) / len(values),
            where="post",
            label=f"{title} ({len(values):,})",
            color=color,
            lw=1.8,
        )
    ax.axvline(threshold, color=ink, linestyle="--", lw=1)
    ax.set(
        xlabel="Raw reconstruction MSE",
        ylabel="Cumulative share of legal windows",
        title="A  Window scores overlap before K3",
        xlim=(0, 0.62),
        ylim=(0, 1.03),
    )
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    ax = axes["B"]
    target_k3 = summary["fall19_phases"]["transition"]["k3_peak"]["k3"]
    rows = [
        ("fall-19", target_k3, red),
        *[
            (row["sequence_id"], row["k3_peak"], teal)
            for row in sorted(normal_videos, key=lambda r: r["k3_peak"])
        ],
    ]
    for y, (_, value, color) in enumerate(rows):
        ax.hlines(y, 0, value, color=color, alpha=0.34, linewidth=2)
        ax.scatter(value, y, color=color, s=35, zorder=3)
    ax.set_yticks(range(len(rows)), [row[0] for row in rows])
    ax.invert_yaxis()
    ax.axvline(threshold, color=ink, linestyle="--", lw=1, label=f"Threshold {threshold:.3f}")
    ax.set(
        xlabel="Highest legal K3 per video",
        title="B  All eight normal Val videos rank higher",
        xlim=(0, 0.60),
    )
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    ax = axes["C"]
    for phase, color in (("transition", "#f3d8ba"), ("ground", "#e3e8ed")):
        start, end = summary["fall19_phase_ranges_ms"][phase]
        ax.axvspan(start / 1000, end / 1000, color=color, alpha=0.55)
        ax.text(
            (start + end) / 2000, 0.32, f"Human {phase} phase", ha="center", fontsize=9, color=ink
        )
    times = [w.time_ms / 1000 for w in target]
    ax.plot(
        times,
        [np.nan if w.raw is None else w.raw for w in target],
        color=red,
        marker=".",
        lw=1.6,
        label="Raw MSE",
    )
    ax.plot(times, [np.nan if w.k3 is None else w.k3 for w in target], color=blue, lw=2, label="K3")
    missing = [w.time_ms / 1000 for w in target if w.raw is None]
    ax.scatter(
        missing, [-0.027] * len(missing), marker="x", s=13, color="#707c89", label="Unavailable"
    )
    ax.axhline(threshold, color=ink, linestyle="--", lw=1, label="Original threshold")
    ax.set(
        xlabel="Original sequence time (seconds)",
        ylabel="Anomaly score",
        title="C  fall-19 scores remain low before availability drops",
        xlim=(0.88, 3.34),
        ylim=(-0.055, 0.58),
    )
    ax.legend(loc="center right", frameon=False, fontsize=9, ncols=2)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, facecolor="white")
    plt.close(fig)
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Recompute and compare the existing numerical summary"
    )
    args = parser.parse_args()
    if not args.check and OUTPUT.exists():
        raise FileExistsError(f"Preserve existing analysis: {OUTPUT}")
    summary, target, populations, normal_videos = analyze()
    if args.check:
        require(
            json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8")) == summary,
            "Saved numerical summary differs",
        )
        print("Archived inputs and saved score overlap summary verified.")
        return
    figure = render_figure(summary, target, populations, normal_videos)
    OUTPUT.mkdir(exist_ok=False)
    with (OUTPUT / "score-overlap.png").open("xb") as stream:
        stream.write(figure)
    with (OUTPUT / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
