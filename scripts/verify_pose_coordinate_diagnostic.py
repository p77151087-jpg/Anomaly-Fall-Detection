"""Independent NumPy scaling / AE forward / manual K3 audit; never reads Test arrays."""

import argparse
import csv
import json

import numpy as np
import torch

from fall_detect.config import ROOT, project_path
from fall_detect.data.cache import load_feature_cache, load_pose_cache
from fall_detect.evaluation.pose_coordinates import DIAGNOSTIC, JOINT, intervention_recipe
from fall_detect.io_utils import read_json, sha256, write_json
from fall_detect.models.autoencoder import Autoencoder


def verify():
    out = ROOT / "outputs" / DIAGNOSTIC
    qa = ROOT / "docs/validation" / DIAGNOSTIC
    base = ROOT / "runs/ur-isotropic-wide-20261012-001"
    lock = read_json(qa / "lock.json")
    for name, expected in lock["files"].items():
        assert sha256(ROOT / name) == expected, name
    for name, expected in lock["reference_artifacts"].items():
        assert sha256(base / name) == expected, name
    assert sha256(out / "ai-provisional-coordinates.json") == lock["proposal_sha256"]
    assert (
        sha256(ROOT / "outputs/ur-fall19-human-amendment-20261013-001/derived-annotation.json")
        == lock["human_phase_annotation_sha256"]
    )
    with np.load(base / "scaler.npz", allow_pickle=False) as z:
        mean, scale = z["mean"], z["scale"]
    checkpoint = torch.load(base / "autoencoder.pt", map_location="cpu", weights_only=True)
    device = "cuda"
    model = Autoencoder(checkpoint["input_dim"], checkpoint["hidden_dim"], checkpoint["latent_dim"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    threshold = read_json(base / "threshold.json")["threshold"]
    index = read_json(base / "dataset-index.json")
    cached = load_feature_cache(project_path(index["sequences"]["fall-19"]["feature_path"]))
    original, _ = load_pose_cache(project_path(index["sequences"]["fall-19"]["pose_path"]), {})
    total_scores = 0
    total_windows = 0
    total_events = 0
    stages = []
    for stage in sorted(p for p in out.iterdir() if p.is_dir() and (p / "summary.json").exists()):
        summary = read_json(stage / "summary.json")
        annotation = read_json(stage / "coordinates-used.json")
        assert summary["recipe_hash"] == intervention_recipe(annotation)
        cache_dir = ROOT / "data/cache" / DIAGNOSTIC / stage.name
        derived, meta = load_pose_cache(cache_dir / "derived-pose.npz", {})
        assert meta["recipe_hash"] == summary["recipe_hash"] != lock["base_recipe_hash"]
        applied = set(summary["applied_frame_ids"])
        points = {r["source_frame_id"]: r for r in annotation["points"]}
        for (f, ps), (g, qs) in zip(original, derived):
            assert f == g
            assert len(ps) == len(qs)
            for p, q in zip(ps, qs):
                for field in (
                    "sequence_id",
                    "source_frame_id",
                    "sample_index",
                    "timestamp_ms",
                    "track_id",
                    "frame_size_wh",
                    "detection_confidence",
                    "quality",
                    "layout_id",
                    "recipe_id",
                ):
                    assert getattr(p, field) == getattr(q, field)
                np.testing.assert_array_equal(p.bbox_xyxy, q.bbox_xyxy)
                expected = p.keypoints.copy()
                if f.source_frame_id in applied:
                    expected[JOINT, :2] = points[f.source_frame_id]["xy"]
                np.testing.assert_array_equal(expected, q.keypoints)
        sources = read_json(stage / "causal-window-sources.json")
        sm = {(r["sample_index"], r["track_id"]): r["source_frame_ids"] for r in sources}
        # Independent timestamp lookup: past observation only, no production resampler.
        past_sources = []
        period = 1000 / 30
        for sample_index in range(cached.metadata["n_samples"]):
            tick = original[0][0].timestamp_ms + sample_index * period
            frame, poses = max(
                ((f, ps) for f, ps in original if f.timestamp_ms <= tick + 1e-7),
                key=lambda x: x[0].timestamp_ms,
            )
            present = tick - frame.timestamp_ms <= 1.5 * period + 1e-7 and not frame.source_missing
            past_sources.append(
                (
                    frame.source_frame_id if present else None,
                    {p.track_id for p in poses} if present else set(),
                )
            )
        for r in cached.records:
            expected_ids = [
                fid
                for fid, tracks in past_sources[r.sample_index - 29 : r.sample_index + 1]
                if r.track_id in tracks
            ]
            assert sm[(r.sample_index, r.track_id)] == expected_ids
        for name in ("cached", "sham", "intervention"):
            cache = (
                cached
                if name == "cached"
                else load_feature_cache(
                    cache_dir
                    / ("sham-features.npz" if name == "sham" else "intervention-features.npz")
                )
            )
            assert cache.records == cached.records
            scaled = ((cache.features - mean) / scale).astype(np.float32)
            manual = []
            with torch.inference_mode():
                for row in scaled:
                    x = torch.tensor(row[None], device=device)
                    manual.append(float(((model(x) - x) ** 2).mean().cpu()))
            scores = np.load(stage / f"{name}-scores.npy", allow_pickle=False)
            np.testing.assert_allclose(scores, manual, atol=1e-6, rtol=0)
            with (stage / f"{name}-windows.csv").open(encoding="utf-8-sig", newline="") as h:
                rows = list(csv.DictReader(h))
            histories = {}
            last = {}
            active = {}
            events = []
            k3 = []
            for r, w in zip(
                sorted(cache.records, key=lambda x: (x.sample_index, x.track_id or -1)), rows
            ):
                key = r.track_id
                hist = histories.setdefault(key, [])
                t = float(w["available_at_ms"])
                if key in last and r.sample_index != last[key] + cache.metadata["stride"]:
                    if key in active:
                        events.append({**active.pop(key), "end_ms": t, "end_reason": "time_gap"})
                    hist.clear()
                last[key] = r.sample_index
                if r.row_index is None:
                    assert w["raw_score"] == w["event_score"] == "" and w["state"] == "unknown"
                    if key in active:
                        events.append({**active.pop(key), "end_ms": t, "end_reason": r.quality})
                    hist.clear()
                    continue
                value = float(scores[r.row_index])
                hist.append(value)
                hist[:] = hist[-3:]
                assert float(w["raw_score"]) == value
                event = min(hist) if len(hist) == 3 else None
                if event is None:
                    assert w["event_score"] == "" and w["state"] == "warming_up"
                else:
                    assert float(w["event_score"]) == event
                    k3.append(event)
                    assert w["state"] == ("suspected_fall" if event > threshold else "normal")
                if event is not None and event > threshold:
                    if key not in active:
                        active[key] = {
                            "sequence_id": r.sequence_id,
                            "track_id": key,
                            "confirmed_start_ms": t,
                            "last_active_ms": t,
                            "peak_score": value,
                        }
                    else:
                        active[key]["last_active_ms"] = t
                        active[key]["peak_score"] = max(active[key]["peak_score"], value)
                elif key in active:
                    events.append({**active.pop(key), "end_ms": t, "end_reason": "low_score"})
            for a in active.values():
                events.append(
                    {**a, "end_ms": cache.metadata["end_ms"], "end_reason": "sequence_end"}
                )
            assert events == read_json(stage / f"{name}-events.json")
            result = summary["results"][name]
            assert result["sequence_score"] == (max(k3) if k3 else None)
            assert result["prediction"] == (int(max(k3) > threshold) if k3 else None)
            assert result["n_events"] == len(events)
            assert result["n_scheduled_endpoints"] == 71 and result["n_valid_endpoints"] == 23
            assert result["window_coverage"] == 23 / 71
            assert result["max_raw_score"] == float(scores.max())
            total_scores += len(scores)
            total_windows += len(rows)
            total_events += len(events)
        sham = load_feature_cache(cache_dir / "sham-features.npz")
        intervention = load_feature_cache(cache_dir / "intervention-features.npz")
        np.testing.assert_allclose(sham.features, cached.features, atol=1e-6, rtol=1e-5)
        affected = 0
        for r in cached.records:
            if r.row_index is None:
                continue
            if r.track_id == 1 and set(sm[(r.sample_index, r.track_id)]).intersection(applied):
                affected += 1
            else:
                np.testing.assert_allclose(
                    sham.features[r.row_index],
                    intervention.features[r.row_index],
                    atol=1e-6,
                    rtol=1e-5,
                )
        assert affected == summary["affected_legal_windows"]
        assert summary["fit_calls"] == summary["calibration_calls"] == 0
        assert summary["test"] == "not_loaded_not_evaluated" and summary["adopted"] is False
        stages.append(stage.name)
    assert stages
    before = read_json(qa / "old-assets-before.json")
    print(f"Verifying {len(before)} preserved file hashes (integrity only)...", flush=True)
    for name, expected in before.items():
        assert sha256(ROOT / name) == expected, name
    for name in (
        "AGENTS.md",
        "Fall_Detection_Codex_Implementation_Spec.md",
        "docs/current-progress.md",
        "docs/acceptance-results.md",
        "docs/course-alignment-plan.md",
    ):
        assert (qa / "prior-contracts" / name).exists()
    result = {
        "verified": True,
        "stages": stages,
        "independent_scores": total_scores,
        "independent_windows": total_windows,
        "independent_events": total_events,
        "preserved_asset_hashes": len(before),
        "test_arrays_loaded": 0,
        "human_stage": "pending" if stages == ["ai-provisional"] else "self_reported",
        "formal_validation_accuracy_changed": False,
    }
    target = qa / ("independent-verification-" + str(len(stages)) + ".json")
    if target.exists():
        assert read_json(target) == result
    else:
        write_json(target, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    verify()
