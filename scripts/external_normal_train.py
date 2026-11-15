"""Two preregistered mixed-normal fits; unchanged original UR Validation, Test sealed."""

import argparse
import os
from contextlib import contextmanager

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from unittest.mock import patch

import human_isotropic_refinement as prior
import numpy as np
import yaml
from external_normal_data import ROOT, save_new
from prepare_external_normal import EVIDENCE as PREPARED

from fall_detect.config import load_config
from fall_detect.data.cache import load_feature_cache
from fall_detect.io_utils import ContractError, read_json, sha256, write_json, write_npz
from fall_detect.training import experiment

PREFIX = "external-normal-20261012"
PROFILE = ROOT / f"docs/validation/{PREFIX}-training-inputs.json"
LOCK = ROOT / f"docs/validation/{PREFIX}-model-lock.json"
RESULT = ROOT / f"docs/validation/{PREFIX}-validation.json"
ANNOTATIONS = ROOT / "docs/annotations/ur-train-validation-20261012-v2"
INDEX = ROOT / "runs/ur-human-isotropic-001/dataset-index.json"
ORIGINAL_MATRIX = experiment.normal_matrix
# These whole continuous camera intervals were selected from raw visual review,
# before any candidate fitting/calibration. Titles and sensor plots are excluded.
UMA_BOUNDS = {
    1: (375, 900),
    2: (307, 614),
    3: (386, 900),
    4: (380, 900),
    5: (378, 900),
    6: (392, 900),
    7: (378, 900),
    8: (378, 900),
}


def selected_rows(cache, interval):
    rows = sorted(
        {
            r.row_index
            for r in cache.records
            if r.quality == "valid"
            and (
                interval is None
                or (interval[0] <= r.start_ms + 1e-7 and r.end_ms <= interval[1] + 1e-7)
            )
        }
    )
    return rows


def curate():
    if PROFILE.exists():
        raise ContractError("Normal input curation completed; preserve this version")
    prepared = read_json(PREPARED)
    old_hashes = set()
    for name in ("le2i-extension", "caucafall-extension"):
        for s in read_json(ROOT / f"data/manifests/{name}/split-v1.json")["sequences"]:
            if "source_sha256" in s:
                old_hashes.add(s["source_sha256"])
    rows = []
    for source in prepared["entries"]:
        if (
            source["label"] != 0
            or source["split"] != "external_train"
            or source["source_sha256"] in old_hashes
        ):
            raise ContractError("Normal source label/assignment/old-split duplicate contract")
        path = ROOT / source["feature_path"]
        cache = load_feature_cache(
            path,
            {
                "sequence_id": source["sequence_id"],
                "source_kind": "real",
                "recipe_hash": prepared["recipe_hash"],
                "label": 0,
                "split": "train",
            },
        )
        interval = None
        if source["dataset"] == "umafall_v8":
            bounds = UMA_BOUNDS[int(source["sequence_id"].split("-")[-1])]
            frames = read_json((ROOT / source["pose_path"]).with_suffix(".json"))["frames"]
            by_id = {f["source_frame_id"]: f["timestamp_ms"] for f in frames}
            interval = [by_id[b] for b in bounds]
        indices = selected_rows(cache, interval)
        dest = path.parent / "curated-training-matrix.npz"
        if dest.exists():
            raise ContractError("Curated normal matrix already exists")
        write_npz(dest, features=cache.features[indices])
        rows.append(
            {
                **source,
                "curated_interval_ms": interval,
                "curated_row_indices": indices,
                "curated_count": len(indices),
                "eligible": bool(indices),
                "matrix_path": dest.relative_to(ROOT).as_posix(),
                "matrix_sha256": sha256(dest),
            }
        )
    save_new(
        PROFILE,
        {
            "schema_version": 1,
            "source_kind": "real",
            "prepared_sha256": sha256(PREPARED),
            "recipe_hash": prepared["recipe_hash"],
            "entries": rows,
            "curation": "raw sampled visual review; only complete windows inside normal RGB camera intervals; zero usable rows excluded",
            "ursul_actions_visual": {
                "13": "walk away",
                "18": "walk then bend / pick-up",
                "24": "walk then step-up",
            },
            "uma_demonstrations_not_raw_unedited_sessions": True,
            "explicit_test_exclusion": True,
            "ur_validation_score_used_for_data_selection": False,
            "whole_video_duplicate_hash_matches_to_prior_external_splits": 0,
            "person_independence_from_ur_not_proven": True,
            "ai_visual_source_review_human_validated": False,
        },
    )
    print(
        {
            "acquired": len(rows),
            "usable_real_normal_sources": sum(r["eligible"] for r in rows),
            "curated_feature_windows": sum(r["curated_count"] for r in rows),
        },
        flush=True,
    )


def verified_array(row, recipe):
    if row["split"] != "external_train" or row["label"] != 0:
        raise ContractError("External matrix must be assigned Train ADL")
    for field, digest in (("source_path", "source_sha256"), ("matrix_path", "matrix_sha256")):
        if sha256(ROOT / row[field]) != row[digest]:
            raise ContractError("Frozen external RGB/matrix changed")
    cache = load_feature_cache(
        ROOT / row["feature_path"],
        {
            "sequence_id": row["sequence_id"],
            "recipe_hash": recipe,
            "source_kind": "real",
            "label": 0,
            "split": "train",
        },
    )
    with np.load(ROOT / row["matrix_path"], allow_pickle=False) as data:
        array = data["features"].copy()
    if array.shape != (row["curated_count"], 512) or not np.isfinite(array).all():
        raise ContractError("External Train feature matrix shape/finite contract")
    expected_indices = selected_rows(cache, row["curated_interval_ms"])
    if expected_indices != row["curated_row_indices"] or not np.array_equal(
        array, cache.features[expected_indices]
    ):
        raise ContractError("External matrix not an exact audited real-feature subset")
    return array


@contextmanager
def mixed_normal_inputs(profile):
    def matrix(index, split, return_counts=False):
        if split == "validation":
            return ORIGINAL_MATRIX(index, split, return_counts)
        if split != "train":
            raise ContractError("Test data forbidden in mixed-normal workflow")
        if index["recipe_hash"] != profile["recipe_hash"]:
            raise ContractError("Original and external feature recipes differ")
        original, hashes, counts = ORIGINAL_MATRIX(index, "train", True)
        arrays = [original]
        for row in profile["entries"]:
            if not row["eligible"]:
                continue
            array = verified_array(row, profile["recipe_hash"])
            arrays.append(array)
            sid = row["sequence_id"]
            if sid in hashes:
                raise ContractError("External source namespace duplicates UR source")
            hashes[sid], counts[sid] = row["matrix_sha256"], len(array)
        answer = (np.concatenate(arrays), hashes)
        return (*answer, counts) if return_counts else answer

    with patch.object(experiment, "normal_matrix", side_effect=matrix):
        yield


def freeze():
    profile = read_json(PROFILE)
    original = read_json(INDEX)
    # Preserve earlier outputs/configs/weights/source bytes. Reading old Test hashes
    # for integrity is never model inference or score-based selection.
    before = prior.frozen_files()
    for folder in (
        "outputs/ur-isotropic-wide-test-20261012-001",
        "docs/annotations/ur-train-validation-20261012-v2",
    ):
        for path in (ROOT / folder).rglob("*"):
            if path.is_file():
                before[path.relative_to(ROOT).as_posix()] = sha256(path)
    old = read_json(ROOT / "docs/validation/isotropic-refinement-20261012-lock.json")
    wide = read_json(
        ROOT
        / "docs/validation/isotropic-refinement-20261012-result-ur-isotropic-wide-20261012-001.json"
    )
    candidates = []
    for hidden, latent in ((128, 32), (256, 64)):
        name = f"ur-external{hidden}-20261012-001"
        path = ROOT / f"configs/{name}.yaml"
        if path.exists() or (ROOT / "runs" / name).exists():
            raise ContractError("Candidate run/config ID must be new")
        cfg = load_config(ROOT / "configs/ur-human-isotropic-001.yaml")
        cfg["autoencoder"].update(hidden_dim=hidden, latent_dim=latent)
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        load_config(path)
        candidates.append(
            {
                "run": name,
                "config": path.relative_to(ROOT).as_posix(),
                "config_sha256": sha256(path),
            }
        )
    save_new(
        LOCK,
        {
            "schema_version": 1,
            "source_sha256": sha256(__file__),
            "files_before": before,
            "profile_sha256": sha256(PROFILE),
            "annotations_sha256": sha256(ANNOTATIONS / "manifest.json"),
            "prepared_sha256": sha256(PREPARED),
            "ur_split_hash": original["split_manifest"]["split_hash"],
            "training_protocol": "original UR Train24 ADL plus externally assigned real ADL; unchanged UR-only evaluation",
            "train_normal_sources": 24 + sum(r["eligible"] for r in profile["entries"]),
            "external_train_windows": sum(r["curated_count"] for r in profile["entries"]),
            "candidates": candidates,
            "reference": wide["run"],
            "reference_metrics": wide["metrics"],
            "reference_fn": [
                r["sequence_id"] for r in wide["predictions"] if r["label"] and r["prediction"] != 1
            ],
            "coarse_intervals": old["coarse_intervals"],
            "selection": "strictly exceed 22/23 original UR Validation Accuracy; F1 not lower, no more FP/unknown, old phase guard remains; ties retain reference",
            "stop": "Exactly these two capacities, original seed42, max100/patience10; no extra data/seed/quality/threshold scans after results",
            "test_reads_for_training_or_inference": 0,
            "historical_test_seen": True,
            "annotations_use": "new timing/visibility diagnostics only; no supervised falls in Scaler/AE; do not rewrite old adoption guards",
        },
    )
    print(
        {"candidates": 2, "reference_validation_accuracy": wide["metrics"]["operational_accuracy"]},
        flush=True,
    )


def verify_lock():
    lock = read_json(LOCK)
    if sha256(__file__) != lock["source_sha256"] or sha256(PROFILE) != lock["profile_sha256"]:
        raise ContractError("Frozen implementation or Train data selection changed")
    for relative, expected in lock["files_before"].items():
        if sha256(ROOT / relative) != expected:
            raise ContractError(f"Preserved prior file changed: {relative}")
    for c in lock["candidates"]:
        if sha256(ROOT / c["config"]) != c["config_sha256"]:
            raise ContractError("Frozen candidate config changed")
    return lock


def timing_diagnostic(run):
    from human_residual import visible_pose_keys

    from fall_detect.detection.events import score_sequence

    scorer = experiment.Scorer(ROOT / "runs" / run)
    phases = read_json(ANNOTATIONS / "fall-intervals.json")["sequences"]
    rows = []
    for phase in phases:
        if phase["label"] != 1 or phase["split"] != "validation":
            continue
        sid = phase["sequence_id"]
        cache = experiment.cache_for(scorer.index, sid)
        _, windows, events = score_sequence(
            cache, scorer.score(cache.features), scorer.threshold["threshold"]
        )
        lo, hi = phase["possible_descent_ms"]
        core_lo, core_hi = phase["core_descent_ms"]
        good = visible_pose_keys(scorer, sid)
        starts = [e["confirmed_start_ms"] for e in events]
        rows.append(
            {
                "sequence_id": sid,
                "possible_descent_ms": [lo, hi],
                "core_descent_ms": [core_lo, core_hi],
                "event_in_possible_phase": any(lo <= t <= hi for t in starts),
                "event_in_core_phase": any(core_lo <= t <= core_hi for t in starts),
                "pre_phase_event": any(t < lo for t in starts),
                "legal_windows_in_possible_phase": sum(
                    w["raw_score"] is not None and lo <= w["available_at_ms"] <= hi for w in windows
                ),
                "current_good_pose_alerts_in_possible_phase": sum(
                    w["state"] == "suspected_fall"
                    and lo <= w["available_at_ms"] <= hi
                    and (w["sample_index"], w["track_id"]) in good
                    for w in windows
                ),
            }
        )
    return rows


def training():
    if RESULT.exists():
        raise ContractError("Completed two-candidate experiment; no additional fitting")
    lock = verify_lock()
    profile = read_json(PROFILE)
    accesses, results = [], []
    with prior.train_validation_only(accesses):
        for c in lock["candidates"]:
            name = c["run"]
            path = ROOT / f"docs/validation/{PREFIX}-result-{name}.json"
            if path.exists():
                experiment.verify_artifacts(ROOT / "runs" / name)
                results.append(read_json(path))
                continue
            run = ROOT / "runs" / name
            if run.exists():
                raise ContractError("Preserve incomplete run; requires a separate audit")
            with mixed_normal_inputs(profile):
                experiment.train(load_config(ROOT / c["config"]), INDEX, run)
            source = read_json(run / "sources.json")
            source["experimental_training_inputs"] = {
                "protocol": lock["training_protocol"],
                "profile": PROFILE.relative_to(ROOT).as_posix(),
                "profile_sha256": sha256(PROFILE),
                "model_lock_sha256": sha256(LOCK),
                "workflow_sha256": sha256(__file__),
                "scaler_ae_split": "Train ADL only",
                "external_source_ids": [
                    r["sequence_id"] for r in profile["entries"] if r["eligible"]
                ],
                "original_ur_split_retained": True,
                "validation_only_ur": True,
                "test_used": False,
            }
            write_json(run / "sources.json", source)
            write_json(
                run / "training-artifacts.json", experiment.artifact_hashes(run, training=True)
            )
            experiment.calibrate(run)
            result = prior.validation_result(name, lock, export=False)
            result["new_ai_annotation_timing_diagnostic"] = timing_diagnostic(name)
            save_new(path, result)
            # Save canonical original-Val CSVs with original denominator, no external
            # examples added to boost accuracy.
            folder = ROOT / "outputs" / PREFIX / name
            folder.mkdir(parents=True, exist_ok=False)
            from fall_detect.detection.events import score_sequence
            from fall_detect.io_utils import write_csv

            scorer = experiment.Scorer(run)
            windows, events = [], []
            for s in scorer.index["split_manifest"]["sequences"]:
                if s["split"] != "validation":
                    continue
                cache = experiment.cache_for(scorer.index, s["sequence_id"])
                _, w, e = score_sequence(
                    cache, scorer.score(cache.features), scorer.threshold["threshold"]
                )
                windows.extend(w)
                events.extend(e)
            write_json(folder / "metrics.json", result["metrics"])
            write_csv(folder / "sequences.csv", result["predictions"])
            write_csv(folder / "windows.csv", windows)
            write_csv(folder / "events.csv", events)
            results.append(result)
            print(
                {"run": name, "metrics": result["metrics"], "eligible": result["eligible"]},
                flush=True,
            )
    eligible = [r for r in results if r["eligible"]]
    save_new(
        RESULT,
        {
            "results": results,
            "selected_candidate": max(eligible, key=prior.rank)["run"] if eligible else None,
            "reference": lock["reference"],
            "feature_accesses": accesses,
            "test_feature_reads": 0,
            "new_test_evaluations": 0,
            "default_run_retained": "ur-sample-human-s1-001",
            "historical_test_seen": True,
            "repeated_validation_not_independent": True,
            "evidence_tier": "real_feature_smoke",
            "lock_sha256": sha256(LOCK),
        },
    )
    verify_lock()


def status():
    lock = verify_lock()
    result = read_json(RESULT) if RESULT.exists() else None
    print(
        {
            "complete": result is not None,
            "preserved_files": len(lock["files_before"]),
            "test_feature_reads": 0,
            "selected_candidate": result["selected_candidate"] if result else None,
            "results": [{"run": r["run"], "metrics": r["metrics"]} for r in result["results"]]
            if result
            else [],
        }
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("curate", "freeze", "training", "status"))
    globals()[p.parse_args().stage]()
