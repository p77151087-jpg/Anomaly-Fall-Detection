"""UR-only, preregistered Train joint-position augmentation; never access Test features."""

import argparse
import hashlib
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import human_isotropic_refinement as prior
import normal_data_support as support
import numpy as np
import retire_external_normal as retired
import yaml

from fall_detect.config import load_config, project_path
from fall_detect.data.cache import load_pose_cache, save_feature_cache
from fall_detect.detection.events import score_sequence
from fall_detect.io_utils import ContractError, read_json, sha256, write_csv, write_json, write_npz
from fall_detect.models.stgcn import HumanFeatureExtractor
from fall_detect.models.weights import select_device
from fall_detect.pipeline import features_from_observations
from fall_detect.skeleton.preprocessing import human_preprocess
from fall_detect.training import experiment

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "ur-train-shift-20261012"
RUN = "ur-joint-shift-20261012-001"
INDEX = ROOT / "runs/ur-human-isotropic-001/dataset-index.json"
CACHE = ROOT / "data/cache/ur-train-joint-shift-20261012-001"
LOCK = ROOT / f"docs/validation/{PREFIX}-lock.json"
PROFILE = ROOT / f"docs/validation/{PREFIX}-inputs.json"
RESULT = ROOT / f"docs/validation/{PREFIX}-validation.json"
FRACTION = 0.01


def save_new(path, value):
    if path.exists():
        raise ContractError("Preserve completed UR-only shift experiment")
    write_json(path, value)


def observed_points(points, size):
    points = np.asarray(points)
    if (
        points.shape != (30, 13, 3)
        or not np.issubdtype(points.dtype, np.floating)
        or not np.isfinite(points).all()
        or min(size) <= 0
    ):
        raise ContractError("Require finite observed floating Human13 and positive image size")
    return points


def global_shift(points, size):
    points = observed_points(points, size)
    result = points.copy()
    result[:, :, :2] += np.asarray([size[0] * 0.05, -size[1] * 0.05], dtype=points.dtype)
    return result


def joint_shift(points, size, seed, fraction=FRACTION):
    points = observed_points(points, size)
    if not 0 < fraction <= 0.01:
        raise ContractError("Only a fixed small normal-preserving shift is allowed")
    result = points.copy()
    xy = points[:, :, :2]
    inside = (xy >= 0).all(axis=2) & (xy[:, :, 0] < size[0]) & (xy[:, :, 1] < size[1])
    visible = (points[:, :, 2] >= 0.2) & inside
    spans = [
        float(np.ptp(row[mask], axis=0).max())
        for row, mask in zip(xy, visible)
        if np.count_nonzero(mask) >= 2
    ]
    if not spans:
        return result
    span = float(np.median(spans))
    offsets = np.random.default_rng(seed).uniform(-fraction, fraction, size=(13, 2)) * span
    proposed = xy + offsets[None]
    admissible = (
        visible.all(axis=0)
        & (proposed >= 0).all(axis=(0, 2))
        & (proposed[:, :, 0] < size[0]).all(axis=0)
        & (proposed[:, :, 1] < size[1]).all(axis=0)
    )
    # One offset per joint per complete window, no per-frame artificial shaking.
    # Low-confidence, occluded/cropped or newly out-of-image joints remain untouched.
    result[:, admissible, :2] = proposed[:, admissible]
    return result


def seed_for(sid, row):
    return int.from_bytes(hashlib.sha256(f"42:{sid}:{row}".encode()).digest()[:8], "little")


def freeze():
    if LOCK.exists():
        raise ContractError("One candidate already frozen")
    retired.status()
    index = experiment.load_index(INDEX)
    train = [s for s in index["split_manifest"]["sequences"] if s["split"] == "train"]
    if len(train) != 24 or any(s["label"] != 0 for s in train):
        raise ContractError("Require original24 UR Train ADL")
    preserved = prior.frozen_files()
    old = read_json(retired.LOCK)
    for path, digest in old["files"].items():
        preserved[retired.mapped_path(path, old["mappings"])] = digest
    for p in [
        retired.LOCK,
        retired.RESULT,
        Path(retired.__file__),
        Path(support.__file__),
        Path(prior.__file__),
    ]:
        preserved[p.relative_to(ROOT).as_posix()] = sha256(p)
    config_path = ROOT / f"configs/{RUN}.yaml"
    if config_path.exists() or (ROOT / "runs" / RUN).exists():
        raise ContractError("Require a new config and run")
    config = load_config(ROOT / "configs/ur-isotropic-wide-20261012-001.yaml")
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    load_config(config_path)
    reference = read_json(
        ROOT
        / "docs/validation/isotropic-refinement-20261012-result-ur-isotropic-wide-20261012-001.json"
    )
    save_new(
        LOCK,
        {
            "source_sha256": sha256(__file__),
            "files_before": preserved,
            "index_sha256": sha256(INDEX),
            "split_hash": index["split_manifest"]["split_hash"],
            "run": RUN,
            "config": config_path.relative_to(ROOT).as_posix(),
            "config_sha256": sha256(config_path),
            "train_ids": [s["sequence_id"] for s in train],
            "reference": reference["run"],
            "reference_metrics": reference["metrics"],
            "reference_fn": [
                r["sequence_id"]
                for r in reference["predictions"]
                if r["label"] == 1 and r["prediction"] == 0
            ],
            "coarse_intervals": read_json(ROOT / "docs/validation/human-isotropic-lock.json")[
                "coarse_intervals"
            ],
            "transform": "Original Train normal observed Human13 only. One seeded constant XY offset per fully confident inside-image joint per30frame window, <=1% median visible-body span per axis. Preserve low-confidence/cropped points and all confidences/timestamps. Discard identical feature rows, originals:eligible-derived1:1; no synthetic missing joints or invented real recordings.",
            "pilot_gate": "48 original feature parity checks atol/rtol1e-5, unchanged scheduled records. At least90% Train feature rows measurably different (>1e-6). Pure translation normalized tensor delta <=2e-5; diagnostic only, not duplicate training data.",
            "selection": "Same previous guard: strictly better than22/23 URVal Accuracy, F1 not lower, no more FP/unknown, >=4 old coarse hits, zero early events, good current pose alert for recovered reference FN. Retain original default on failure.",
            "stop": "Exactly one256/64 AE, original seed42/max100/patience10/sequence sampling/wd.001/noise0. No further seeds/capacities/magnitudes/quality/threshold sweeps. No Test.",
            "validation_unchanged": True,
            "external_training_sources": 0,
            "new_test_evaluations": 0,
            "historical_test_seen": True,
        },
    )
    print(
        {"candidate": RUN, "UR_Train_ADL": 24, "external_inputs": 0, "stop": "one fixed candidate"}
    )


def verify_lock():
    lock = read_json(LOCK)
    if sha256(__file__) != lock["source_sha256"] or sha256(INDEX) != lock["index_sha256"]:
        raise ContractError("Frozen workflow or original index changed")
    if sha256(ROOT / lock["config"]) != lock["config_sha256"]:
        raise ContractError("Frozen candidate config changed")
    for path, digest in lock["files_before"].items():
        if sha256(ROOT / path) != digest:
            raise ContractError(f"Preserved old file changed: {path}")
    for old in retired.MAPPINGS:
        if (ROOT / old).exists():
            raise ContractError("External inputs reactivated")
    return lock


class ShiftExtractor:
    feature_dim = 512

    def __init__(self, extractor, original, sid):
        self.extractor, self.original, self.sid = extractor, original, sid
        self.position = self.parity_checks = 0
        self.parity_max = self.translation_tensor_max = self.translation_feature_max = 0.0

    def extract_features(self, points, size):
        snapshot = points.copy()
        if self.position in (0, len(self.original) - 1):
            feature = self.extractor.extract_features(points, size)[0]
            np.testing.assert_allclose(feature, self.original[self.position], atol=1e-5, rtol=1e-5)
            self.parity_max = max(
                self.parity_max, float(np.max(np.abs(feature - self.original[self.position])))
            )
            self.parity_checks += 1
            original_tensor = human_preprocess(points, size, normalization="per_frame_isotropic")
            shifted = global_shift(points, size)
            translated_tensor = human_preprocess(shifted, size, normalization="per_frame_isotropic")
            delta = max(
                float(np.max(np.abs(a - b))) for a, b in zip(original_tensor, translated_tensor)
            )
            if delta > 2e-5:
                raise ContractError("Unexpected translation change under isotropic normalization")
            self.translation_tensor_max = max(self.translation_tensor_max, delta)
            shifted_feature = self.extractor.extract_features(shifted, size)[0]
            self.translation_feature_max = max(
                self.translation_feature_max, float(np.max(np.abs(feature - shifted_feature)))
            )
        derived = joint_shift(points, size, seed_for(self.sid, self.position))
        np.testing.assert_array_equal(derived[:, :, 2], points[:, :, 2])
        np.testing.assert_array_equal(points, snapshot)
        result = self.extractor.extract_features(derived, size)
        self.position += 1
        return result


def derive():
    if PROFILE.exists() or CACHE.exists():
        raise ContractError("Preserve existing derived Train inputs")
    verify_lock()
    index = experiment.load_index(INDEX)
    config = load_config(ROOT / f"configs/{RUN}.yaml")
    extractor = HumanFeatureExtractor(
        config, index["weights"], select_device(config["runtime"]["device"], 4)
    )
    accesses, entries = [], {}
    with prior.train_validation_only(accesses):
        for source in index["split_manifest"]["sequences"]:
            if source["split"] != "train":
                continue
            sid = source["sequence_id"]
            support.input_source(index, sid, train=True)
            original = experiment.cache_for(index, sid)
            observations, _ = load_pose_cache(
                project_path(index["sequences"][sid]["pose_path"]),
                {
                    "sequence_id": sid,
                    "source_hash": index["sequences"][sid]["source_hash"],
                    "recipe_hash": index["recipe_hash"],
                },
            )
            wrapped = ShiftExtractor(extractor, original.features, sid)
            derived = features_from_observations(observations, wrapped, config, original.metadata)
            if (
                [asdict(r) for r in derived.records] != [asdict(r) for r in original.records]
                or wrapped.position != len(original.features)
                or wrapped.parity_checks != 2
            ):
                raise ContractError(
                    "Derived data changed original timing/quality or parity incomplete"
                )
            difference = np.max(np.abs(derived.features - original.features), axis=1)
            changed = np.flatnonzero(difference > 1e-6)
            derived.metadata.update(
                augmentation="UR Train joint-shift only",
                original_features_sha256=original.metadata["array_sha256"],
            )
            folder = CACHE / sid
            folder.mkdir(parents=True)
            path, combined = folder / "derived-features.npz", folder / "training-matrix.npz"
            save_feature_cache(path, derived)
            write_npz(
                combined, features=np.concatenate([original.features, derived.features[changed]])
            )
            entries[sid] = {
                "original_sha256": original.metadata["array_sha256"],
                "original_count": len(original.features),
                "derived_path": path.relative_to(ROOT).as_posix(),
                "derived_sha256": sha256(path),
                "eligible_derived_rows": changed.tolist(),
                "derived_count": len(changed),
                "combined_path": combined.relative_to(ROOT).as_posix(),
                "combined_sha256": sha256(combined),
                "combined_count": len(original.features) + len(changed),
                "parity_checks": wrapped.parity_checks,
                "original_parity_max": wrapped.parity_max,
                "translation_tensor_max": wrapped.translation_tensor_max,
                "translation_feature_max": wrapped.translation_feature_max,
                "joint_shift_feature_max": float(difference.max()),
            }
            print(
                f"{sid}: {len(original.features)} original / {len(changed)} changed derived rows",
                flush=True,
            )
    original_count = sum(e["original_count"] for e in entries.values())
    changed_count = sum(e["derived_count"] for e in entries.values())
    save_new(
        PROFILE,
        {
            "recipe_hash": index["recipe_hash"],
            "sequences": entries,
            "source_kind": "real observed UR skeletons + explicit coordinate augmentation",
            "original_Train_videos": len(entries),
            "original_windows": original_count,
            "changed_derived_windows": changed_count,
            "training_matrix_count": original_count + changed_count,
            "pilot_passed": changed_count >= 0.9 * original_count,
            "strict_weight_load": extractor.load_report,
            "new_original_recordings": 0,
            "external_training_sources": 0,
            "feature_accesses": accesses,
            "test_feature_arrays_loaded": 0,
            "evidence_tier": "real_feature_smoke",
            "new_raw_RGB_pose_extraction": False,
            "lock_sha256": sha256(LOCK),
        },
    )
    verify_lock()


def checked_profile(lock):
    profile = read_json(PROFILE)
    if profile["lock_sha256"] != sha256(LOCK) or set(profile["sequences"]) != set(
        lock["train_ids"]
    ):
        raise ContractError("Derived profile must match exactly original UR Train24")
    if not profile["pilot_passed"]:
        raise ContractError("Pilot failed; no candidate fitting permitted")
    for e in profile["sequences"].values():
        for path, digest in [
            (e["derived_path"], e["derived_sha256"]),
            (e["combined_path"], e["combined_sha256"]),
        ]:
            if sha256(ROOT / path) != digest:
                raise ContractError("Frozen derived input hash changed")
    return profile


def training():
    if RESULT.exists() or (ROOT / "runs" / RUN).exists():
        raise ContractError("Never overwrite or extend this one-candidate fit")
    lock = verify_lock()
    profile = checked_profile(lock)
    index = experiment.load_index(INDEX)
    accesses = []
    with prior.train_validation_only(accesses):
        with support.train_matrix_inputs(index, profile=profile):
            experiment.train(load_config(ROOT / lock["config"]), INDEX, ROOT / "runs" / RUN)
        sources_path = ROOT / "runs" / RUN / "sources.json"
        sources = read_json(sources_path)
        sources["experimental_training_inputs"] = {
            "profile": PROFILE.relative_to(ROOT).as_posix(),
            "profile_sha256": sha256(PROFILE),
            "lock_sha256": sha256(LOCK),
            "workflow_sha256": sha256(__file__),
            "UR_Train_ADL_only": True,
            "external_sources": 0,
            "validation_inputs_unchanged": True,
            "test_used": False,
            "input_description": "UR Train original features + explicit observed-joint shift features; no additional original films",
        }
        write_json(sources_path, sources)
        write_json(
            ROOT / "runs" / RUN / "training-artifacts.json",
            experiment.artifact_hashes(ROOT / "runs" / RUN, training=True),
        )
        experiment.calibrate(ROOT / "runs" / RUN)
        result = prior.validation_result(RUN, lock, export=False)
        folder = ROOT / "outputs" / PREFIX / RUN
        folder.mkdir(parents=True, exist_ok=False)
        write_json(folder / "metrics.json", result["metrics"])
        write_csv(folder / "sequences.csv", result["predictions"])
        scorer = experiment.Scorer(ROOT / "runs" / RUN)
        windows, events = [], []
        for s in index["split_manifest"]["sequences"]:
            if s["split"] != "validation":
                continue
            cache = experiment.cache_for(index, s["sequence_id"])
            _, w, e = score_sequence(
                cache, scorer.score(cache.features), scorer.threshold["threshold"]
            )
            windows.extend(w)
            events.extend(e)
        write_csv(folder / "windows.csv", windows)
        write_csv(folder / "events.csv", events)
        result["exports"] = {p.relative_to(ROOT).as_posix(): sha256(p) for p in folder.iterdir()}
    save_new(
        RESULT,
        {
            "result": result,
            "profile_sha256": sha256(PROFILE),
            "lock_sha256": sha256(LOCK),
            "feature_accesses": accesses,
            "selected_candidate": RUN if result["eligible"] else None,
            "default_run_retained": "ur-sample-human-s1-001",
            "external_sources": 0,
            "test_feature_arrays_loaded": 0,
            "new_test_evaluations": 0,
            "historical_test_seen": True,
            "repeated_validation_not_independent": True,
            "evidence_tier": "real_feature_smoke",
        },
    )
    verify_lock()
    print({"metrics": result["metrics"], "eligible": result["eligible"]})


def status():
    lock = verify_lock()
    result = read_json(RESULT) if RESULT.exists() else None
    print(
        {
            "complete": result is not None,
            "preserved_files": len(lock["files_before"]),
            "external_sources": 0,
            "new_test_evaluations": 0,
            "selected_candidate": result["selected_candidate"] if result else None,
            "metrics": result["result"]["metrics"] if result else None,
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "derive", "training", "status"))
    globals()[parser.parse_args().stage]()
