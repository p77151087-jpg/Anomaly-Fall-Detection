"""Independent numerical, sequence/event and preservation audit; never reads Test arrays."""

import argparse
import csv

import joblib
import numpy as np
from ur_motion_novelty import (
    BASE,
    ROOT,
    RUN,
    TAG,
    C,
    E,
    O,
    original_windows,
    save,
    validated_index,
    verify_lock,
)
from verify_normal_methods import independent_events, independent_metrics, number

from fall_detect.io_utils import ContractError, read_json, sha256, write_csv
from fall_detect.training.experiment import Scorer, cache_for


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def independent_distances(values, bank, groups=None, bank_groups=None):
    result = []
    # Direct coordinate subtraction, independently of production scipy.cdist.
    for start in range(0, len(values), 16):
        distance = np.sqrt(np.sum((values[start : start + 16, None] - bank[None]) ** 2, axis=2))
        if groups is not None:
            distance[groups[start : start + 16, None] == bank_groups[None]] = np.inf
        result.extend(np.sort(distance, axis=1)[:, :5].mean(1) / np.sqrt(90))
    return np.asarray(result)


def verify(write=False):
    if write and (E / "verification.json").exists():
        raise ContractError("Preserve completed verification")
    lock = verify_lock()
    index = validated_index()
    reference = Scorer(ROOT / "runs" / BASE)
    arrays, caches, provenance = {}, {}, {}
    for source in index["split_manifest"]["sequences"]:
        if source["split"] == "test":
            with np.testing.assert_raises(ContractError):
                cache_for(index, source["sequence_id"])
            assert index["sequences"][source["sequence_id"]]["status"] == "not_prepared"
            continue
        sid = source["sequence_id"]
        path = C / "motion" / f"{sid}.npz"
        with np.load(path, allow_pickle=False) as saved:
            raw, motion = saved["original_windows"], saved["motion"]
        caches[sid], actual_raw, prov = original_windows(index, sid, reference.config)
        np.testing.assert_array_equal(raw, actual_raw)
        provenance[sid] = prov
        assert read_json(path.with_suffix(".json"))["rows"] == prov
        # Independent min/max trajectory, not calling motion_trajectory.
        xy = raw[..., :2].astype(np.float64)
        lo, hi = np.min(xy, axis=2), np.max(xy, axis=2)
        center = (hi + lo) * 0.5
        span = np.max(hi - lo, axis=2)
        wanted = np.concatenate(
            ((center - center[:, :1]) / span[:, :1, None], np.log(span / span[:, :1])[..., None]),
            axis=2,
        ).reshape(-1, 90)
        np.testing.assert_array_equal(motion, wanted)
        assert len(motion) == len(caches[sid].features)
        for row in prov:
            assert row["start_ms"] <= row["end_ms"] <= row["available_at_ms"]
            assert row["quality"] == "valid" and row["track_id"] is not None
        arrays[sid] = motion
    train_ids = [
        s["sequence_id"] for s in index["split_manifest"]["sequences"] if s["split"] == "train"
    ]
    assert len(train_ids) == 24 and all(
        s["label"] == 0 for s in index["split_manifest"]["sequences"] if s["split"] == "train"
    )
    train = np.concatenate([arrays[sid] for sid in train_ids])
    groups = np.concatenate([np.repeat(sid, len(arrays[sid])) for sid in train_ids])
    mean, scale = train.mean(0), train.std(0)
    scale[scale == 0] = 1
    model = joblib.load(RUN / "model.joblib")
    np.testing.assert_array_equal(model.scaler.mean, mean)
    np.testing.assert_array_equal(model.scaler.scale, scale)
    bank = ((train - mean) / scale).astype(np.float32).astype(np.float64)
    np.testing.assert_array_equal(model.bank, bank)
    np.testing.assert_array_equal(model.groups, groups)
    assert model.neighbors == 5
    scores = {name: {} for name in ["ae", "motion", "fused"]}
    for name, filename in [("ae", "ae-scores"), ("motion", "motion-scores"), ("fused", "scores")]:
        with np.load(RUN / f"{filename}.npz", allow_pickle=False) as saved:
            assert set(saved.files) == set(arrays)
            scores[name] = {sid: saved[sid] for sid in saved.files}
    train_motion = independent_distances(bank, bank, groups, groups)
    with np.load(RUN / "train-scale-scores.npz", allow_pickle=False) as saved:
        np.testing.assert_allclose(
            saved["motion_leave_video_out"], train_motion, atol=1e-12, rtol=1e-12
        )
        np.testing.assert_array_equal(saved["groups"], groups)
        np.testing.assert_array_equal(
            saved["ae"], np.concatenate([scores["ae"][sid] for sid in train_ids])
        )
    scales = read_json(RUN / "normalization.json")
    ae_p95 = float(np.quantile(np.concatenate([scores["ae"][sid] for sid in train_ids]), 0.95))
    motion_p95 = float(np.quantile(train_motion, 0.95))
    assert scales["ae_p95"] == ae_p95
    np.testing.assert_allclose(scales["motion_p95"], motion_p95, atol=1e-12, rtol=1e-12)
    assert scales["fit_sequences"] == train_ids and scales["fit_split"] == "train"
    for sid, values in arrays.items():
        standardized = ((values - mean) / scale).astype(np.float32).astype(np.float64)
        independent = independent_distances(standardized, bank)
        np.testing.assert_allclose(scores["motion"][sid], independent, atol=1e-12, rtol=1e-12)
        # Reloaded production scores must reproduce strict-threshold values exactly.
        np.testing.assert_array_equal(model.score(values), scores["motion"][sid])
        np.testing.assert_array_equal(reference.score(caches[sid].features), scores["ae"][sid])
        wanted = np.maximum(
            scores["ae"][sid] / ae_p95, scores["motion"][sid] / scales["motion_p95"]
        )
        np.testing.assert_array_equal(scores["fused"][sid], wanted)
    actual = csv_rows(O / "sequence-results.csv")
    windows, events = csv_rows(O / "window-results.csv"), csv_rows(O / "events.csv")
    metrics = independent_metrics(actual)
    formal = read_json(O / "metrics.json")
    for key, value in metrics.items():
        assert formal[key] == value, key
    assert metrics["n_total"] == 23
    threshold = read_json(RUN / "threshold.json")["threshold"]
    expected_events = []
    by_sid = {row["sequence_id"]: row for row in actual}
    comparison = []
    fall19 = []
    from fall_detect.detection.events import score_sequence

    baseline_phases_rows, baseline_windows, baseline_events = [], [], []
    for s in index["split_manifest"]["sequences"]:
        if s["split"] != "validation":
            continue
        sid = s["sequence_id"]
        selected = [w for w in windows if w["sequence_id"] == sid]
        es, seq_score, valid, scheduled = independent_events(
            caches[sid], scores["fused"][sid], threshold, selected
        )
        expected_events.extend(es)
        r = by_sid[sid]
        assert number(r["sequence_score"]) == seq_score
        assert int(r["n_valid_endpoints"]) == valid and int(r["n_scheduled_endpoints"]) == scheduled
        assert (None if r["prediction"] == "" else int(r["prediction"])) == (
            int(seq_score > threshold) if seq_score is not None else None
        )
        assert number(r["first_alarm_ms"]) == (es[0]["confirmed_start_ms"] if es else None)
        baseline, ws, bs = score_sequence(
            caches[sid], scores["ae"][sid], reference.threshold["threshold"]
        )
        baseline_phases_rows.append({**baseline, "label": s["label"]})
        baseline_windows.extend(ws)
        baseline_events.extend(bs)
        comparison.append(
            {
                "sequence_id": sid,
                "label": s["label"],
                "baseline_prediction": baseline["prediction"],
                "new_prediction": r["prediction"],
                "baseline_first_alarm_ms": baseline["first_alarm_ms"],
                "new_first_alarm_ms": number(r["first_alarm_ms"]),
                "baseline_k3_score": baseline["sequence_score"],
                "new_k3_score": seq_score,
                "window_coverage": valid / scheduled,
            }
        )
        if sid == "fall-19":
            low, high = lock["coarse_intervals"][sid]["fall_phase_ms"]
            good = {tuple(x) for x in lock["current_good_pose_keys"][sid]}
            for i, p in enumerate(provenance[sid]):
                if low <= p["available_at_ms"] <= high:
                    fall19.append(
                        {
                            **p,
                            "ae_score": float(scores["ae"][sid][i]),
                            "motion_score": float(scores["motion"][sid][i]),
                            "ae_scaled": float(scores["ae"][sid][i] / ae_p95),
                            "motion_scaled": float(scores["motion"][sid][i] / scales["motion_p95"]),
                            "fused_score": float(scores["fused"][sid][i]),
                            "current_good_pose": (p["sample_index"], p["track_id"]) in good,
                            "center_dx_dy_logspan_last": arrays[sid][i].reshape(30, 3)[-1].tolist(),
                        }
                    )
    assert len(expected_events) == len(events)
    for expected, row in zip(expected_events, events):
        for key, value in expected.items():
            assert (row[key] if isinstance(value, str) else number(row[key])) == value, key
    calibration = read_json(RUN / "calibration.json")
    values = np.unique([number(r["sequence_score"]) for r in actual if r["sequence_score"] != ""])
    candidates = []
    for t in [np.nextafter(values[0], -np.inf), *values.tolist()]:
        predictions = [
            {
                **r,
                "prediction": int(number(r["sequence_score"]) > t)
                if r["sequence_score"] != ""
                else None,
            }
            for r in actual
        ]
        m = independent_metrics(predictions)
        candidates.append(
            (
                m["operational_accuracy"],
                m["sequence_fall_f1"],
                m["end_to_end_fall_recall"],
                -m["FP"],
                float(t),
            )
        )
    assert max(candidates)[-1] == threshold and len(calibration["candidates"]) == len(candidates)
    for candidate, t in zip(calibration["candidates"], candidates):
        assert candidate["threshold"] == t[-1]
        m = candidate["metrics"]
        assert (
            m["operational_accuracy"],
            m["sequence_fall_f1"],
            m["end_to_end_fall_recall"],
            -m["FP"],
        ) == t[:-1]
    from ur_normal_methods import phases_for

    baseline_phases = phases_for(baseline_phases_rows, baseline_windows, baseline_events, lock)
    old_metrics = independent_metrics(baseline_phases_rows)
    for key, value in old_metrics.items():
        assert lock["reference_metrics"][key] == value
    result = read_json(RUN / "result.json")
    phases = phases_for(
        actual,
        [
            {
                **w,
                "sample_index": int(w["sample_index"]),
                "track_id": number(w["track_id"]),
                "available_at_ms": float(w["available_at_ms"]),
                "raw_score": number(w["raw_score"]),
                "event_score": number(w["event_score"]),
            }
            for w in windows
        ],
        expected_events,
        lock,
    )
    assert result["phases"] == phases
    # Useful scalar threshold bound, without choosing another model/threshold.
    fall_score = number(by_sid["fall-19"]["sequence_score"])
    min_fp_for_fall19 = (
        sum(
            int(r["label"]) == 0
            and number(r["sequence_score"]) is not None
            and number(r["sequence_score"]) >= fall_score
            for r in actual
        )
        if fall_score is not None
        else None
    )
    for name, digest in read_json(RUN / "frozen-artifacts.json").items():
        assert sha256(RUN / name) == digest, name
    before = read_json(E / "before.json")
    for n, digest in before["assets"].items():
        assert sha256(ROOT / n) == digest, n
    for n, digest in before["contracts"].items():
        assert sha256(E / "prior-contract" / n) == digest
    record = {
        "passed": True,
        "candidate": TAG,
        "sequences": len(arrays),
        "train_sequences": len(train_ids),
        "train_rows": len(train),
        "motion_rows": sum(len(x) for x in arrays.values()),
        "three_branch_scores_verified": 3 * sum(len(x) for x in arrays.values()),
        "train_leave_video_scores_verified": len(train),
        "validation_rows": len(actual),
        "window_rows": len(windows),
        "events": len(events),
        "metrics": metrics,
        "old_assets_unchanged": len(before["assets"]),
        "test_feature_reads": 0,
        "test_evaluations": 0,
        "new_neural_epochs": 0,
        "normal_fit_completed": True,
        "min_normal_fp_to_recover_fall19_with_same_score": min_fp_for_fall19,
        "independent_distance_tolerance": 1e-12,
        "reloaded_production_scores_exact": True,
        "original_quality_and_timing_exact": True,
        "baseline_phases": baseline_phases,
        "new_phases": phases,
        "fall19_phase": fall19,
        "scales": scales,
        "baseline_metrics": old_metrics,
        "baseline_events": len(baseline_events),
        "first_alarm_changes": sum(
            r["baseline_first_alarm_ms"] != r["new_first_alarm_ms"] for r in comparison
        ),
        "window_state_changes": sum(
            a["state"] != b["state"] for a, b in zip(baseline_windows, windows)
        ),
        "event_timing_and_tracks_match_baseline": [
            {k: v for k, v in r.items() if k != "peak_score"} for r in baseline_events
        ]
        == [{k: v for k, v in r.items() if k != "peak_score"} for r in expected_events],
        "motion_normal_val_p95": float(
            np.quantile(
                np.concatenate(
                    [
                        scores["motion"][s["sequence_id"]]
                        for s in index["split_manifest"]["sequences"]
                        if s["split"] == "validation" and s["label"] == 0
                    ]
                ),
                0.95,
            )
        ),
    }
    if write:
        save(E / "verification.json", record)
        write_csv(O / "baseline-comparison.csv", comparison)
        write_csv(O / "fall19-phase-components.csv", fall19)
    print(
        {
            k: v
            for k, v in record.items()
            if k not in ["baseline_phases", "new_phases", "fall19_phase"]
        },
        flush=True,
    )
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    verify(parser.parse_args().write)
