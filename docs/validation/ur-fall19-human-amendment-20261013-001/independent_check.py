"""Independent CSV/provenance checks; standard library, no model or Test decoding."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
E = Path(__file__).resolve().parent
O = ROOT / "outputs" / E.name
PARENT = ROOT / "outputs/ur-fall19-human-review-20261012-001"
OLD = ROOT / "outputs/ur-fall19-review-20261012-001"
DIGEST = "f7dbda11fcf42dbcf4f2a203ea0d0c22b70d7fa0b0c7915051be8cc540b88f3c"
ORIGINAL = ROOT / "docs/validation/ur-fall19-review-20261012-001/human-annotations" / DIGEST


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def cell(value):
    return "" if value is None else str(value)


def verify():
    assert digest(ORIGINAL / "original-export.json") == DIGEST
    assert digest(Path("C:/Users/Huayi/Downloads/fall-19-human-self-reported.json")) == DIGEST
    original = read(ORIGINAL / "original-export.json")
    derived = read(O / "derived-annotation.json")
    report = read(O / "diagnosis.json")
    source = read(OLD / "source-manifest.json")
    data = read(OLD / "review-data.json")
    excluded = {"schema_version", "frames", "onset_bounds_frame_ids", "impact_bounds_frame_ids"}
    assert all(derived[k] == v for k, v in original.items() if k not in excluded)
    new_keys = {
        "original_schema_version",
        "derived_from_sha256",
        "amendment_origin",
        "temporal_semantics",
        "phase_ranges_frame_ids",
        "precise_time_ground_truth",
        "used_for_training",
        "used_for_calibration",
    }
    assert set(derived) == set(original) | new_keys
    assert derived["onset_bounds_frame_ids"] is None
    assert derived["impact_bounds_frame_ids"] is None
    assert derived["phase_ranges_frame_ids"] == {"transition": [33, 61], "ground": [62, 100]}
    changed_cells = []
    for a, b, f in zip(original["frames"], derived["frames"], source["frames"], strict=True):
        assert set(a) == set(b)
        for key in a:
            if a[key] != b[key]:
                changed_cells.append((a["source_frame_id"], key, a[key], b[key]))
        assert all(b[k] == f[k] for k in ["source_frame_id", "timestamp_ms", "source_sha256"])
        frame_id = f["source_frame_id"]
        rgb = (
            ROOT
            / "data/raw/urfall/fall-19/fall-19-cam0-rgb"
            / f"fall-19-cam0-rgb-{frame_id:03d}.png"
        )
        assert digest(rgb) == f["source_sha256"]
    assert changed_cells == [
        (1, "pose_check", "matches_person", "missing"),
        (2, "pose_check", "matches_person", "missing"),
    ]
    assert [r["source_frame_id"] for r in derived["frames"] if r["pose_check"] == "missing"] == [
        1,
        2,
        54,
        55,
        59,
    ]
    new_frames, old_frames = (
        rows(O / "human-frame-diagnosis.csv"),
        rows(PARENT / "human-frame-diagnosis.csv"),
    )
    differences = [
        (int(a["source_frame_id"]), k, a[k], b[k])
        for a, b in zip(old_frames, new_frames, strict=True)
        for k in a
        if a[k] != b[k]
    ]
    assert differences == changed_cells
    assert digest(O / "human-window-diagnosis.csv") == digest(PARENT / "human-window-diagnosis.csv")
    windows = rows(O / "human-window-diagnosis.csv")
    grouped = {}
    count = 0
    for name, frozen in data["windows"].items():
        selected = [w for w in windows if w["method"] == name]
        for w, raw in zip(selected, frozen, strict=True):
            actual = max(
                (f for f in data["frames"] if f["timestamp_ms"] <= raw["available_at_ms"]),
                key=lambda f: f["timestamp_ms"],
            )
            label = next(
                f for f in derived["frames"] if f["source_frame_id"] == actual["source_frame_id"]
            )
            assert int(w["last_actual_source_frame_id"]) == actual["source_frame_id"]
            assert float(w["last_actual_source_ms"]) == actual["timestamp_ms"]
            assert all(w[k] == cell(v) for k, v in raw.items())
            for k in ["phase", "visibility", "pose_check"]:
                assert w[f"human_{k}_at_last_source"] == label[k]
            matching = [p for p in actual["poses"] if p["track_id"] == raw["track_id"]]
            assert w["current_track_pose_present"] == cell(bool(matching))
            assert w["current_track_good"] == cell(any(p["current_good_pose"] for p in matching))
            threshold = data["models"][name]["threshold"]
            assert float(w["threshold"]) == threshold
            for key, score_key in [
                ("above_raw_threshold", "raw_score"),
                ("above_k3_threshold", "event_score"),
            ]:
                assert w[key] == cell(raw[score_key] is not None and raw[score_key] > threshold)
            grouped.setdefault((name, label["phase"]), []).append(raw)
            count += 1
    for (name, phase), group in grouped.items():
        summary = report["methods"][name][phase]
        valid = [r for r in group if r["quality"] == "valid"]
        assert summary["scheduled"] == len(group) and summary["valid"] == len(valid)
        assert summary["coverage"] == len(valid) / len(group)
        assert summary["quality_counts"] == dict(Counter(r["quality"] for r in group))
        for key, raw_key in [("max_raw", "raw_score"), ("max_k3", "event_score")]:
            scores = [r[raw_key] for r in valid if r[raw_key] is not None]
            assert summary[key] == (max(scores) if scores else None)
        assert summary["raw_crossings"] == summary["k3_crossings"] == 0
    assert count == 223 and len(new_frames) == 100
    assert report["methods"] == read(PARENT / "diagnosis.json")["methods"]
    assert report["wrong_but_confidence_good_frames"] == list(range(44, 52))
    assert not report["semantic_warnings"] and not report["precise_time_ground_truth"]
    assert not report["precise_onset_latency_computed"] and not report["used_for_calibration"]
    before = read(E / "before.json")
    for name, value in before["assets"].items():
        assert digest(ROOT / name) == value
    for name, value in before["original_contracts"].items():
        assert digest(E / "prior-contract" / name) == value
    return {
        "passed": True,
        "evidence_level": "real_feature_smoke",
        "original_sha256": DIGEST,
        "derived_sha256": digest(O / "derived-annotation.json"),
        "changed_human_cells": changed_cells,
        "source_rgb_hashes_checked": 100,
        "frame_rows_checked": 100,
        "independent_causal_window_joins": count,
        "window_csv_bytes_unchanged": True,
        "method_summaries_unchanged": True,
        "parent_artifacts_preserved": len(before["assets"]),
        "contracts_archived": 2,
        "precise_time_ground_truth": False,
        "new_training_epochs": 0,
        "test_feature_reads": 0,
        "new_test_evaluations": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    result = verify()
    target = E / "independent-verification.json"
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.save:
        with target.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
    else:
        assert target.read_text(encoding="utf-8") == serialized
    print(serialized)
