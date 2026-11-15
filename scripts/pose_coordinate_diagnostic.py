"""Controlled frozen-model diagnosis; no training, threshold search, or Test inference."""

import argparse
import base64
import json
import shutil
from pathlib import Path

import numpy as np

from fall_detect.config import ROOT, project_path
from fall_detect.data.cache import load_feature_cache, load_pose_cache, save_feature_cache
from fall_detect.detection.events import score_sequence
from fall_detect.evaluation.pose_coordinates import (
    BASE_RECIPE,
    DIAGNOSTIC,
    FRAMES,
    JOINT,
    apply_coordinates,
    assert_same_records,
    intervention_recipe,
    validate_annotation,
    window_sources,
)
from fall_detect.io_utils import (
    ContractError,
    read_json,
    sha256,
    write_csv,
    write_json,
)
from fall_detect.models.stgcn import HumanFeatureExtractor
from fall_detect.pipeline import features_from_observations
from fall_detect.skeleton.windows import CausalResampler, WindowBuilder
from fall_detect.training.experiment import Scorer, verify_artifacts

OUT = ROOT / "outputs" / DIAGNOSTIC
QA = ROOT / "docs" / "validation" / DIAGNOSTIC
CACHE = ROOT / "data" / "cache" / DIAGNOSTIC
RUN = ROOT / "runs" / DIAGNOSTIC
BASE = ROOT / "runs/ur-isotropic-wide-20261012-001"
HUMAN = ROOT / "outputs/ur-fall19-human-amendment-20261013-001/derived-annotation.json"
SOURCE = ROOT / "data/raw/urfall/fall-19/fall-19-cam0-rgb"
MUTABLE = [
    "AGENTS.md",
    "Fall_Detection_Codex_Implementation_Spec.md",
    "docs/current-progress.md",
    "docs/acceptance-results.md",
    "docs/course-alignment-plan.md",
]
LOCK_FILES = [
    "scripts/pose_coordinate_diagnostic.py",
    "scripts/verify_pose_coordinate_diagnostic.py",
    "src/fall_detect/evaluation/pose_coordinates.py",
    "tests/test_pose_coordinates.py",
    "src/fall_detect/visualization/pose_coordinate_review.html",
    "docs/ur-pose-coordinate-diagnostic-20261013-plan.md",
]
# One pre-score RGB interpretation. Not human truth; do not search around these points.
AI_XY = [
    (376, 354),
    (377, 369),
    (377, 385),
    (376, 401),
    (378, 417),
    (378, 432),
    (379, 448),
    (378, 449),
]


def old_inventory():
    """File integrity only: does not decode raw Test or load Test features."""
    paths = set()
    for folder in ("runs", "data/cache", "weights", "configs", "outputs"):
        for p in (ROOT / folder).rglob("*"):
            if p.is_file() and DIAGNOSTIC not in p.parts and "__pycache__" not in p.parts:
                paths.add(p)
    paths.add(HUMAN)
    paths.update(SOURCE.glob("*.png"))  # Only Validation fall-19 RGB is opened.
    prior = read_json(ROOT / "docs/validation/ur-pose-resolution-20261013-001/final-qa.json")
    for name, expected in prior["artifact_sha256"].items():
        path = ROOT / name
        if sha256(path) != expected:
            raise ContractError(f"Prior finalized artifact already changed: {name}")
        if path.relative_to(ROOT).as_posix() not in MUTABLE:
            paths.add(path)
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(paths)}


def original_inputs(sid, index):
    if sid not in {"fall-19", "adl-05", "adl-04"}:
        raise ContractError("Unapproved sequence; no Test access")
    meta = index["sequences"][sid]
    feature = load_feature_cache(project_path(meta["feature_path"]), {"recipe_hash": BASE_RECIPE})
    expected_split = "train" if sid == "adl-05" else "validation"
    if feature.metadata["split"] != expected_split:
        raise ContractError("Diagnostic input split mismatch")
    observations, pose_meta = load_pose_cache(
        project_path(meta["pose_path"]), {"split": expected_split}
    )
    return feature, observations, pose_meta


def prepare():
    if any(p.exists() for p in (OUT, QA, CACHE, RUN)):
        raise ContractError("Diagnostic directory exists; preparation never overwrites")
    for p in (OUT, QA, CACHE, RUN):
        p.mkdir(parents=True, exist_ok=False)
    for name in MUTABLE:
        dest = QA / "prior-contracts" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    index = read_json(BASE / "dataset-index.json")
    verify_artifacts(BASE)
    if index["recipe_hash"] != BASE_RECIPE or index["source_kind"] != "real":
        raise ContractError("Wrong reference run")
    _feature, observations, pose_meta = original_inputs("fall-19", index)
    pose_path = project_path(index["sequences"]["fall-19"]["pose_path"])
    annotation = {
        "schema_version": 1,
        "diagnostic_id": DIAGNOSTIC,
        "sequence_id": "fall-19",
        "split": "validation",
        "source_hash": pose_meta["source_hash"],
        "base_recipe_hash": BASE_RECIPE,
        "pose_array_sha256": sha256(pose_path),
        "pose_metadata_sha256": sha256(pose_path.with_suffix(".json")),
        "author_kind": "ai_provisional",
        "human_confirmed": False,
        "coordinate_basis": "original_RGB_visible_elbow_estimate; not joint ground truth",
        "points": [],
    }
    poses, images = [], []
    for fid, xy in zip(FRAMES, AI_XY):
        frame, records = next((f, p) for f, p in observations if f.source_frame_id == fid)
        pose = next(p for p in records if p.track_id == 1)
        img = SOURCE / f"fall-19-cam0-rgb-{fid:03d}.png"
        annotation["points"].append(
            {
                "source_frame_id": fid,
                "track_id": 1,
                "joint_index": JOINT,
                "joint_name": "left_elbow",
                "timestamp_ms": frame.timestamp_ms,
                "frame_size_wh": [frame.width, frame.height],
                "image_sha256": sha256(img),
                "original_xy": pose.keypoints[JOINT, :2].astype(float).tolist(),
                "original_confidence": float(pose.keypoints[JOINT, 2]),
                "visibility": "visible",
                "xy": list(xy),
            }
        )
        poses.append(pose.keypoints.astype(float).tolist())
        images.append("data:image/png;base64," + base64.b64encode(img.read_bytes()).decode())
    validate_annotation(annotation, annotation)
    write_json(OUT / "ai-provisional-coordinates.json", annotation)
    ui = OUT / "review"
    ui.mkdir()
    data = {
        "annotation": annotation,
        "original_poses": poses,
        "images": images,
        "proposal_sha256": sha256(OUT / "ai-provisional-coordinates.json"),
    }
    template = (ROOT / LOCK_FILES[4]).read_text(encoding="utf-8")
    (ui / "index.html").write_text(
        template.replace("__DATA__", json.dumps(data, ensure_ascii=False)), encoding="utf-8"
    )
    shutil.copyfile(BASE / "resolved-config.yaml", RUN / "frozen-reference-config.yaml")
    write_json(
        RUN / "diagnostic-reference.json",
        {
            "kind": "frozen_inference_diagnostic_only",
            "training": "not_run",
            "calibration": "not_run",
            "reference_run": str(BASE),
            "reference_artifacts": verify_artifacts(BASE),
            "train_control": "adl-05",
            "validation_diagnosis": "fall-19",
            "validation_control": "adl-04",
            "test": "not_prepared_not_loaded_not_evaluated",
            "formal_validation_metrics_changed": False,
        },
    )
    print("Snapshotting preserved files (integrity hashes only, no Test inference)...", flush=True)
    before = old_inventory()
    write_json(QA / "old-assets-before.json", before)
    lock = {
        "schema_version": 1,
        "diagnostic_id": DIAGNOSTIC,
        "base_recipe_hash": BASE_RECIPE,
        "ai_recipe_hash": intervention_recipe(annotation),
        "human_coordinates": "pending",
        "reference_artifacts": verify_artifacts(BASE),
        "human_phase_annotation_sha256": sha256(HUMAN),
        "files": {name: sha256(ROOT / name) for name in LOCK_FILES},
        "proposal_sha256": sha256(OUT / "ai-provisional-coordinates.json"),
        "old_asset_count": len(before),
        "test_inference": False,
    }
    write_json(QA / "lock.json", lock)
    print(
        json.dumps(
            {"prepared": DIAGNOSTIC, "locked_points": 8, "old_asset_count": len(before)},
            ensure_ascii=False,
        ),
        flush=True,
    )


def check_lock():
    lock = read_json(QA / "lock.json")
    if {name: sha256(ROOT / name) for name in LOCK_FILES} != lock["files"]:
        raise ContractError("Pre-score source/plan lock mismatch")
    if sha256(OUT / "ai-provisional-coordinates.json") != lock["proposal_sha256"]:
        raise ContractError("Provisional coordinates changed after locking")
    if (
        verify_artifacts(BASE) != lock["reference_artifacts"]
        or sha256(HUMAN) != lock["human_phase_annotation_sha256"]
    ):
        raise ContractError("Reference model/calibration/annotation changed")
    return lock


def first_three_controls(sid, index, config, extractor, scorer):
    cached, observations, _ = original_inputs(sid, index)
    c = config["skeleton"]
    rs = CausalResampler(c["target_fps"], c["resampling_hold_policy"], c["timestamp_tolerance_ms"])
    wb = WindowBuilder(config)
    values, identities = [], []
    for frame, poses in observations:
        for sample in rs.push(frame, poses):
            for window in wb.push(sample):
                if window.quality == "valid":
                    values.append(
                        extractor.extract_features(window.keypoints, window.frame_size_wh)[0]
                    )
                    identities.append((window.sample_index, window.track_id))
                    if len(values) == 3:
                        break
            if len(values) == 3:
                break
        if len(values) == 3:
            break
    expected_records = [r for r in cached.records if r.row_index is not None][:3]
    if identities != [(r.sample_index, r.track_id) for r in expected_records]:
        raise ContractError("Control window identity mismatch")
    expected = np.array([cached.features[r.row_index] for r in expected_records])
    values = np.asarray(values, dtype=np.float32)
    np.testing.assert_allclose(values, expected, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(scorer.score(values), scorer.score(expected), atol=1e-6, rtol=0)
    return {
        "sequence_id": sid,
        "split": cached.metadata["split"],
        "windows": identities,
        "feature_max_abs_delta": float(np.max(np.abs(values - expected))),
        "scores": scorer.score(values).astype(float).tolist(),
        "fit_called": False,
    }


def diagnose(human_path=None):
    lock = check_lock()
    reference = read_json(OUT / "ai-provisional-coordinates.json")
    annotation = read_json(human_path) if human_path else reference
    validate_annotation(annotation, reference, require_human=bool(human_path))
    recipe = intervention_recipe(annotation)
    if human_path and annotation.get("parent_proposal_sha256") != lock["proposal_sha256"]:
        raise ContractError("Human file refers to another proposal")
    mode = "human-" + recipe[:12] if human_path else "ai-provisional"
    dest, cache_dir = OUT / mode, CACHE / mode
    # mkdir is an exclusive claim; parallel/repeated calls cannot share a writable stage.
    if dest.exists() or cache_dir.exists():
        raise ContractError("Inference output already exists; read status/verify instead")
    dest.mkdir(exist_ok=False)
    cache_dir.mkdir(exist_ok=False)
    write_json(dest / "coordinates-used.json", annotation)
    scorer = Scorer(BASE, device="cuda")
    if scorer.residual_weights is not None or scorer.config["anomaly_score"]["mode"] != "mse":
        raise ContractError("Diagnostic assumes frozen wide MSE")
    config, index = scorer.config, scorer.index
    cached, observations, pose_meta = original_inputs("fall-19", index)
    extractor = HumanFeatureExtractor(
        config, read_json(BASE / "sources.json")["weights"], scorer.device
    )
    metadata = {k: v for k, v in cached.metadata.items() if k not in {"array_sha256", "records"}}
    sham = features_from_observations(observations, extractor, config, metadata)
    assert_same_records(cached, sham)
    np.testing.assert_allclose(sham.features, cached.features, atol=1e-6, rtol=1e-5)
    corrected, applied = apply_coordinates(observations, annotation, reference, bool(human_path))
    # Persist a distinct derived observation, without mutating original confidence or cache.
    from fall_detect.data.cache import save_pose_cache

    derived_meta = {
        k: v for k, v in pose_meta.items() if k not in {"frames", "poses", "array_sha256"}
    }
    derived_meta.update(
        recipe_hash=recipe,
        base_recipe_hash=BASE_RECIPE,
        diagnostic_only=True,
        coordinate_author=annotation["author_kind"],
        human_confirmed=annotation["human_confirmed"],
    )
    save_pose_cache(cache_dir / "derived-pose.npz", corrected, derived_meta)
    metadata.update(
        recipe_hash=recipe,
        base_recipe_hash=BASE_RECIPE,
        diagnostic_only=True,
        coordinate_author=annotation["author_kind"],
        human_confirmed=annotation["human_confirmed"],
    )
    derived = features_from_observations(corrected, extractor, config, metadata)
    assert_same_records(cached, derived)
    save_feature_cache(cache_dir / "sham-features.npz", sham)
    save_feature_cache(cache_dir / "intervention-features.npz", derived)
    memberships = window_sources(observations, config)
    write_json(dest / "causal-window-sources.json", memberships)
    membership = {(r["sample_index"], r["track_id"]): r for r in memberships}
    for r in cached.records:
        if r.row_index is not None and not (
            r.track_id == 1
            and set(membership[(r.sample_index, r.track_id)]["source_frame_ids"]).intersection(
                applied
            )
        ):
            np.testing.assert_allclose(
                derived.features[r.row_index], sham.features[r.row_index], atol=1e-6, rtol=1e-5
            )
    controls = [
        first_three_controls(sid, index, config, extractor, scorer) for sid in ("adl-05", "adl-04")
    ]
    threshold = scorer.threshold["threshold"]
    results = {}
    for name, cache in (("cached", cached), ("sham", sham), ("intervention", derived)):
        scores = scorer.score(cache.features)
        result, windows, events = score_sequence(cache, scores, threshold, k=3)
        np.save(dest / f"{name}-scores.npy", scores, allow_pickle=False)
        write_csv(dest / f"{name}-windows.csv", windows)
        write_json(dest / f"{name}-events.json", events)
        results[name] = {
            **result,
            "max_raw_score": float(scores.max()),
            "n_events": len(events),
            "raw_above_threshold": int((scores > threshold).sum()),
            "k3_above_threshold": sum(
                w["event_score"] is not None and w["event_score"] > threshold for w in windows
            ),
        }
    np.testing.assert_allclose(
        np.load(dest / "cached-scores.npy"), np.load(dest / "sham-scores.npy"), atol=1e-6, rtol=0
    )
    baseline_scores = np.load(dest / "sham-scores.npy")
    new_scores = np.load(dest / "intervention-scores.npy")
    changes = []
    for r in cached.records:
        if r.row_index is None:
            continue
        sources = membership[(r.sample_index, r.track_id)]["source_frame_ids"]
        changes.append(
            {
                "sample_index": r.sample_index,
                "track_id": r.track_id,
                "start_ms": r.start_ms,
                "available_at_ms": r.available_at_ms,
                "contains_corrected_source": bool(set(sources).intersection(applied)),
                "baseline_raw": float(baseline_scores[r.row_index]),
                "intervention_raw": float(new_scores[r.row_index]),
                "raw_delta": float(new_scores[r.row_index] - baseline_scores[r.row_index]),
                "feature_l2_delta": float(
                    np.linalg.norm(derived.features[r.row_index] - sham.features[r.row_index])
                ),
            }
        )
    write_csv(dest / "window-deltas.csv", changes)
    summary = {
        "diagnostic_id": DIAGNOSTIC,
        "kind": "frozen_input_sensitivity_not_formal_validation",
        "coordinate_author": annotation["author_kind"],
        "human_confirmed": annotation["human_confirmed"],
        "human_stage": "self_reported_coordinates" if human_path else "pending",
        "recipe_hash": recipe,
        "base_recipe_hash": BASE_RECIPE,
        "applied_frame_ids": applied,
        "joint_index": JOINT,
        "threshold": threshold,
        "comparison": "gt",
        "k": 3,
        "results": results,
        "controls": controls,
        "strict_stgcn_load": extractor.load_report,
        "forward_verified": extractor.forward_verified,
        "feature_smoke": {
            "shape": list(derived.features.shape),
            "finite": bool(np.isfinite(derived.features).all()),
            "sham_max_abs_delta": float(np.max(np.abs(sham.features - cached.features))),
        },
        "affected_legal_windows": sum(c["contains_corrected_source"] for c in changes),
        "score_increases": sum(c["raw_delta"] > 1e-6 for c in changes),
        "score_decreases": sum(c["raw_delta"] < -1e-6 for c in changes),
        "phase_timing": "whole posture phases; no exact onset/impact GT",
        "test": "not_loaded_not_evaluated",
        "fit_calls": 0,
        "calibration_calls": 0,
        "adopted": False,
        "formal_validation_accuracy_changed": False,
    }
    write_json(dest / "summary.json", summary)
    if not human_path:
        write_json(OUT / "summary.json", summary)
    verify_artifacts(BASE)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def status():
    print(
        (OUT / "summary.json").read_text(encoding="utf-8")
        if (OUT / "summary.json").exists()
        else "Not yet evaluated"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "diagnose", "status", "serve"))
    p.add_argument("--human-coordinates", type=Path)
    p.add_argument("--port", type=int, default=8810)
    args = p.parse_args()
    if args.human_coordinates and args.action != "diagnose":
        p.error("human coordinates only valid for diagnose")
    if args.action == "prepare":
        prepare()
    elif args.action == "diagnose":
        diagnose(args.human_coordinates)
    elif args.action == "status":
        status()
    else:
        from functools import partial
        from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

        if not (OUT / "review/index.html").exists():
            raise ContractError("Prepare review assets first")
        print(f"http://127.0.0.1:{args.port}/review/index.html", flush=True)
        ThreadingHTTPServer(
            ("127.0.0.1", args.port), partial(SimpleHTTPRequestHandler, directory=str(OUT))
        ).serve_forever()
