"""Exclusive, frozen Train/Validation human XY sensitivity experiment."""

import argparse
import base64
import json
from pathlib import Path

import numpy as np
from pose_coordinate_diagnostic import first_three_controls

from fall_detect.config import ROOT, project_path
from fall_detect.data.cache import (
    load_feature_cache,
    load_pose_cache,
    save_feature_cache,
    save_pose_cache,
)
from fall_detect.detection.events import score_sequence
from fall_detect.evaluation.pose_coordinates import BASE_RECIPE, assert_same_records, window_sources
from fall_detect.evaluation.whole_pose_coordinates import (
    ID,
    PARENT,
    apply_review,
    recipe_hash,
    validate_review,
)
from fall_detect.io_utils import ContractError, read_json, sha256, write_csv, write_json
from fall_detect.models.stgcn import HumanFeatureExtractor
from fall_detect.pipeline import features_from_observations
from fall_detect.training.experiment import Scorer, verify_artifacts

OUT = ROOT / "outputs" / ID
QA = ROOT / "docs/validation" / ID
CACHE = ROOT / "data/cache" / ID
BASE = ROOT / "runs/ur-isotropic-wide-20261012-001"
BUNDLE = ROOT / "outputs" / PARENT / "review/review-data.json"
UPLOAD_SHA = "9b4f3216924a04896578187249f85d7dab4d855068e961620816d516a38f7429"
PHASE = ROOT / "outputs/ur-fall19-human-amendment-20261013-001/derived-annotation.json"
MUTABLE = (
    "AGENTS.md",
    "Fall_Detection_Codex_Implementation_Spec.md",
    "docs/current-progress.md",
    "docs/acceptance-results.md",
)
LOCKED = (
    "docs/ur-fall19-whole-coordinate-20261013-plan.md",
    "scripts/whole_pose_coordinate_diagnostic.py",
    "scripts/verify_whole_pose_coordinate_diagnostic.py",
    "src/fall_detect/evaluation/whole_pose_coordinates.py",
    "tests/test_whole_pose_coordinates.py",
)


def exclusive_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(value)


def original_inputs(sid, index):
    if sid not in {"fall-19", "adl-05", "adl-04"}:
        raise ContractError("Only locked Train/Validation controls; no Test")
    meta = index["sequences"][sid]
    expected = "train" if sid == "adl-05" else "validation"
    paths = [project_path(meta[key]) for key in ("feature_path", "pose_path")]
    for path in paths:
        if read_json(path.with_suffix(".json")).get("split") != expected:
            raise ContractError("Reject unexpected split before opening arrays")
    cached = load_feature_cache(paths[0], {"recipe_hash": BASE_RECIPE, "split": expected})
    observations, pose_meta = load_pose_cache(
        paths[1], {"recipe_hash": BASE_RECIPE, "split": expected}
    )
    return cached, observations, pose_meta


def prepare(upload):
    # Refuse before reads/writes if another process already owns this stage.
    if any(p.exists() for p in (OUT, QA, CACHE)):
        raise ContractError("Stage exists; use status/verify. Never overwrite")
    if sha256(upload) != UPLOAD_SHA:
        raise ContractError("This protocol locks exactly the received human file")
    annotation, bundle = read_json(upload), read_json(BUNDLE)
    audit = validate_review(annotation, bundle)
    if sorted((r["source_frame_id"], r["joint_index"]) for r in audit["corrections"]) != [
        (47, 5),
        (47, 6),
        (47, 9),
        (47, 10),
        (47, 11),
        (49, 1),
    ]:
        raise ContractError("Six prelocked corrections only")
    index = read_json(BASE / "dataset-index.json")
    if index["recipe_hash"] != BASE_RECIPE:
        raise ContractError("Wrong reference recipe")
    _, observations, _ = original_inputs("fall-19", index)
    apply_review(observations, annotation, bundle)
    source = project_path(index["sequences"]["fall-19"]["pose_path"])
    manifest = ROOT / "outputs/ur-fall19-review-20261012-001/source-manifest.json"
    for path, key in (
        (source, "pose_array_sha256"),
        (source.with_suffix(".json"), "pose_metadata_sha256"),
        (manifest, "source_manifest_sha256"),
    ):
        if sha256(path) != bundle[key]:
            raise ContractError("Original reference changed")
    for f in bundle["frames"]:
        raw = ROOT / f["image_path"]
        from hashlib import sha256 as bytes_sha

        if (
            sha256(raw) != f["image_sha256"]
            or bytes_sha(base64.b64decode(f["image"].split(",", 1)[1])).hexdigest()
            != f["image_sha256"]
        ):
            raise ContractError("Original RGB checksum mismatch")
    prior = read_json(ROOT / "docs/validation" / PARENT / "final-qa.json")
    for name, expected in prior["artifact_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ContractError(f"Prior final artifact changed before import: {name}")
    paths = {ROOT / n for n in prior["artifact_sha256"]}
    paths.add(ROOT / "docs/validation" / PARENT / "final-qa.json")
    # Bounded integrity inventory, without opening any Test feature arrays or images.
    for folder in (
        "outputs/ur-pose-coordinate-diagnostic-20261013-001",
        "docs/validation/ur-pose-coordinate-diagnostic-20261013-001",
        "data/cache/ur-pose-coordinate-diagnostic-20261013-001",
    ):
        paths.update(p for p in (ROOT / folder).rglob("*") if p.is_file())
    for sid in ("fall-19", "adl-05", "adl-04"):
        original_inputs(sid, index)
        for key in ("pose_path", "feature_path"):
            path = project_path(index["sequences"][sid][key])
            paths.update((path, path.with_suffix(".json")))
    paths.update(ROOT / name for name in MUTABLE)
    paths.update((PHASE, manifest))
    paths.update(ROOT / f["image_path"] for f in bundle["frames"])
    paths.update(p for p in BASE.iterdir() if p.is_file())
    paths.update(
        p for p in (ROOT / "src/fall_detect").rglob("*.py") if p.name != "whole_pose_coordinates.py"
    )
    paths.update((ROOT / "scripts/pose_coordinate_diagnostic.py",))
    inventory = {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(paths)}
    # The first mkdir claims the entire stage. An interrupted stage is preserved.
    OUT.mkdir(parents=True, exist_ok=False)
    QA.mkdir(parents=True, exist_ok=False)
    CACHE.mkdir(parents=True, exist_ok=False)
    for name in MUTABLE:
        exclusive_bytes(QA / "prior-contracts" / name, (ROOT / name).read_bytes())
    exclusive_bytes(OUT / "human-review-original.json", Path(upload).read_bytes())
    write_json(OUT / "annotation-audit.json", audit)
    write_csv(OUT / "corrections.csv", audit["corrections"])
    write_json(QA / "old-assets-before.json", inventory)
    lock = {
        "id": ID,
        "source_sha256": UPLOAD_SHA,
        "parent_bundle_sha256": bundle["bundle_sha256"],
        "recipe_hash": recipe_hash(annotation, UPLOAD_SHA),
        "base_recipe_hash": BASE_RECIPE,
        "files": {n: sha256(ROOT / n) for n in LOCKED},
        "reference_artifacts": verify_artifacts(BASE),
        "phase_sha256": sha256(PHASE),
        "old_inventory_sha256": sha256(QA / "old-assets-before.json"),
        "allowed_sequences": {"fall-19": "validation", "adl-05": "train", "adl-04": "validation"},
        "fit": False,
        "calibration": False,
        "test": "not_loaded_not_evaluated",
    }
    write_json(QA / "lock.json", lock)
    print(
        json.dumps(
            {
                "id": ID,
                "prepared": True,
                "review": audit,
                "old_files": len(inventory),
                "recipe_hash": lock["recipe_hash"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def check_lock():
    lock = read_json(QA / "lock.json")
    if {n: sha256(ROOT / n) for n in LOCKED} != lock["files"]:
        raise ContractError("Pre-inference source/plan lock changed")
    if sha256(OUT / "human-review-original.json") != lock["source_sha256"]:
        raise ContractError("Archived human file changed")
    if (
        verify_artifacts(BASE) != lock["reference_artifacts"]
        or sha256(PHASE) != lock["phase_sha256"]
        or sha256(QA / "old-assets-before.json") != lock["old_inventory_sha256"]
    ):
        raise ContractError("Reference artifact changed")
    for name, expected in read_json(QA / "old-assets-before.json").items():
        path = ROOT / name
        if name in MUTABLE:
            path = QA / "prior-contracts" / name
        if sha256(path) != expected:
            raise ContractError(f"Preserved old asset changed: {name}")
    return lock


def diagnose():
    lock = check_lock()
    if (OUT / "inference").exists():
        raise ContractError("Inference stage exists; never overwrite or rerun")
    dest = OUT / "inference"
    dest.mkdir(exist_ok=False)
    scorer = Scorer(BASE, device="cuda")
    if scorer.residual_weights is not None or scorer.config["anomaly_score"]["mode"] != "mse":
        raise ContractError("Frozen plain wide MSE required")
    config, index = scorer.config, scorer.index
    cached, observations, pose_meta = original_inputs("fall-19", index)
    extractor = HumanFeatureExtractor(
        config, read_json(BASE / "sources.json")["weights"], scorer.device
    )
    metadata = {k: v for k, v in cached.metadata.items() if k not in {"array_sha256", "records"}}
    sham = features_from_observations(observations, extractor, config, metadata)
    assert_same_records(cached, sham)
    np.testing.assert_allclose(sham.features, cached.features, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(
        scorer.score(sham.features), scorer.score(cached.features), atol=1e-6, rtol=0
    )
    controls = [
        first_three_controls(sid, index, config, extractor, scorer) for sid in ("adl-05", "adl-04")
    ]
    annotation = read_json(OUT / "human-review-original.json")
    corrected, audit = apply_review(observations, annotation, read_json(BUNDLE))
    recipe = lock["recipe_hash"]
    derived_meta = {
        k: v for k, v in pose_meta.items() if k not in {"array_sha256", "poses", "frames"}
    }
    change_meta = {
        "recipe_hash": recipe,
        "base_recipe_hash": BASE_RECIPE,
        "diagnostic_only": True,
        "coordinate_author": "human_self_reported",
        "human_confirmed": True,
        "annotation_sha256": UPLOAD_SHA,
        "uncertain_policy": "retain_original",
    }
    derived_meta.update(change_meta)
    save_pose_cache(CACHE / "derived-pose.npz", corrected, derived_meta)
    metadata.update(change_meta)
    derived = features_from_observations(corrected, extractor, config, metadata)
    assert_same_records(cached, derived)
    save_feature_cache(CACHE / "sham-features.npz", sham)
    save_feature_cache(CACHE / "intervention-features.npz", derived)
    memberships = window_sources(observations, config)
    write_json(dest / "causal-window-sources.json", memberships)
    membership = {(r["sample_index"], r["track_id"]): r for r in memberships}
    applied = {r["source_frame_id"] for r in audit["corrections"]}
    for r in cached.records:
        if r.row_index is not None and not set(
            membership[(r.sample_index, r.track_id)]["source_frame_ids"]
        ).intersection(applied):
            np.testing.assert_array_equal(derived.features[r.row_index], sham.features[r.row_index])
    results, arrays = {}, {}
    threshold = scorer.threshold["threshold"]
    for name, cache in (("cached", cached), ("sham", sham), ("intervention", derived)):
        scores = scorer.score(cache.features)
        arrays[name] = scores
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
            "legal_last_available_ms": max(
                r.available_at_ms for r in cache.records if r.row_index is not None
            ),
        }
    deltas = []
    for r in cached.records:
        if r.row_index is None:
            continue
        row = r.row_index
        deltas.append(
            {
                "sample_index": r.sample_index,
                "track_id": r.track_id,
                "start_ms": r.start_ms,
                "available_at_ms": r.available_at_ms,
                "contains_corrected_source": bool(
                    set(membership[(r.sample_index, r.track_id)]["source_frame_ids"]).intersection(
                        applied
                    )
                ),
                "baseline_raw": float(arrays["sham"][row]),
                "intervention_raw": float(arrays["intervention"][row]),
                "raw_delta": float(arrays["intervention"][row] - arrays["sham"][row]),
                "feature_l2_delta": float(
                    np.linalg.norm(derived.features[row] - sham.features[row])
                ),
            }
        )
    write_csv(dest / "window-deltas.csv", deltas)
    summary = {
        "id": ID,
        "kind": "frozen_human_input_control_not_formal_validation",
        "recipe_hash": recipe,
        "base_recipe_hash": BASE_RECIPE,
        "annotation_sha256": UPLOAD_SHA,
        "review": audit,
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
        "affected_legal_windows": sum(d["contains_corrected_source"] for d in deltas),
        "score_increases": sum(d["raw_delta"] > 1e-6 for d in deltas),
        "score_decreases": sum(d["raw_delta"] < -1e-6 for d in deltas),
        "score_unchanged_tolerance": sum(abs(d["raw_delta"]) <= 1e-6 for d in deltas),
        "unknown_window_records": sum(r.row_index is None for r in cached.records),
        "test": "not_loaded_not_evaluated",
        "fit_calls": 0,
        "calibration_calls": 0,
        "adopted": False,
        "formal_validation_accuracy_changed": False,
        "phase_timing": "whole posture phases, not exact onset/impact ground truth",
    }
    write_json(dest / "summary.json", summary)
    check_lock()
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "diagnose", "status", "verify"))
    parser.add_argument("--human-review", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.human_review is None:
            parser.error("prepare requires --human-review")
        prepare(args.human_review)
    elif args.human_review is not None:
        parser.error("--human-review is only for prepare")
    elif args.action == "diagnose":
        diagnose()
    elif args.action == "verify":
        check_lock()
        from verify_whole_pose_coordinate_diagnostic import verify

        print(json.dumps(verify(), ensure_ascii=False, indent=2))
    else:
        print((OUT / "inference/summary.json").read_text(encoding="utf-8"))
