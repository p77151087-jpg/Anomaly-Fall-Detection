"""Train/Validation-only data sufficiency audit and frozen-feature mirror augmentation."""

import argparse
import os
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import human_isotropic_refinement as prior
import numpy as np
import yaml

from fall_detect.config import load_config, project_path
from fall_detect.data.cache import load_pose_cache, save_feature_cache
from fall_detect.io_utils import ContractError, read_json, sha256, write_csv, write_json, write_npz
from fall_detect.models.stgcn import HumanFeatureExtractor
from fall_detect.models.weights import select_device
from fall_detect.pipeline import features_from_observations
from fall_detect.training import experiment

ROOT = project_path(".")
PREFIX = "normal-data-support-20261012"
INDEX = ROOT / "runs/ur-human-isotropic-001/dataset-index.json"
SMALL = "ur-human-isotropic-001"
REFERENCE = "ur-isotropic-wide-20261012-001"
CACHE = ROOT / "data/cache/ur-train-mirror-20261012-001"
ORIGINAL_MATRIX = experiment.normal_matrix
MIRROR13 = np.asarray([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11])


def evidence(name):
    return ROOT / "docs/validation" / f"{PREFIX}-{name}.json"


def save_new(path, value):
    if path.exists():
        raise ContractError(f"Preserve completed evidence: {path}")
    write_json(path, value)


def input_source(index, sid, train=False):
    source = prior.allowed_source(index, sid)
    if train and (source["split"] != "train" or source["label"] != 0):
        raise ContractError("Derived training inputs require original Train ADL only")
    return source


def mirror_human13(points, width):
    points = np.asarray(points)
    if points.ndim != 3 or points.shape[1:] != (13, 3) or width <= 0:
        raise ContractError("Mirror requires observed Human13 windows and positive width")
    if not np.isfinite(points).all():
        raise ContractError("Nonfinite mirror input")
    mirrored = points[:, MIRROR13].copy()
    mirrored[:, :, 0] = (width - 1) - mirrored[:, :, 0]
    return mirrored


@contextmanager
def train_matrix_inputs(index, selected=None, profile=None):
    """Reuse the canonical train loop; replace only its explicit, audited Train matrix."""

    def matrix(current, split, return_counts=False):
        if split == "validation":
            return ORIGINAL_MATRIX(current, split, return_counts)
        if split != "train":
            raise ContractError("No Test matrices in this workflow")
        arrays, hashes, counts = [], {}, {}
        for source in current["split_manifest"]["sequences"]:
            if source["split"] != "train" or (
                selected is not None and source["sequence_id"] not in selected
            ):
                continue
            sid = source["sequence_id"]
            input_source(current, sid, train=True)
            if profile is None:
                cache = experiment.cache_for(current, sid)
                array, digest = cache.features, cache.metadata["array_sha256"]
            else:
                if current["recipe_hash"] != profile["recipe_hash"]:
                    raise ContractError("Augmented matrix recipe mismatch")
                entry = profile["sequences"][sid]
                path = ROOT / entry["combined_path"]
                if sha256(path) != entry["combined_sha256"]:
                    raise ContractError("Derived Train matrix hash mismatch")
                with np.load(path, allow_pickle=False) as data:
                    array = data["features"].copy()
                digest = entry["combined_sha256"]
                if array.shape != (entry["combined_count"], 512) or not np.isfinite(array).all():
                    raise ContractError("Derived Train matrix shape/finite contract")
            arrays.append(array)
            hashes[sid], counts[sid] = digest, len(array)
        if not arrays:
            raise ContractError("No allowed Train matrices")
        result = (np.concatenate(arrays), hashes)
        return (*result, counts) if return_counts else result

    with patch.object(experiment, "normal_matrix", side_effect=matrix):
        yield


def freeze():
    if evidence("diagnostic-lock").exists():
        raise ContractError("Diagnostic protocol already frozen")
    # Preservation hashes are not Test feature arrays used for learning or diagnosis.
    before = read_json(ROOT / "docs/validation/ur-isotropic-wide-test-20261012-before.json")
    hashes = before["preserved_files"].copy()
    for relative, expected in hashes.items():
        if sha256(ROOT / relative) != expected:
            raise ContractError(f"Historical artifact changed: {relative}")
    for folder in ["outputs/ur-isotropic-wide-test-20261012-001"]:
        for path in (ROOT / folder).iterdir():
            if path.is_file():
                hashes[path.relative_to(ROOT).as_posix()] = sha256(path)
    index = experiment.load_index(INDEX)
    train_ids = [
        s["sequence_id"] for s in index["split_manifest"]["sequences"] if s["split"] == "train"
    ]
    curves = []
    for subset_seed in (42, 43):
        order = np.random.default_rng(subset_seed).permutation(train_ids).tolist()
        for count in (8, 16):
            curves.append(
                {
                    "name": f"subset-{subset_seed}-n{count}",
                    "subset_seed": subset_seed,
                    "n_train_videos": count,
                    "train_ids": order[:count],
                }
            )
    save_new(
        evidence("diagnostic-lock"),
        {
            "created_at": datetime.now(UTC).isoformat(),
            "source_sha256": sha256(Path(__file__)),
            "preserved_files": hashes,
            "index_sha256": sha256(INDEX),
            "curves": curves,
            "curve_protocol": "Exactly four uncalibrated learning-curve fits: two nested Train-only orders, n8/n16. AE128/32, modelseed42, wd.001, sequence sampling, max100/patience10; normal original Validation selects checkpoint. Full24 existing AE is a descriptive reference. No Fall classification ranking or Test scoring of curve models, no resplit of original protocol.",
            "known_test_observation_only": "Saved wide Test lost one additional Fall while cache recipe/split/window coverage stayed identical. This implicates AE scores plus its Validation-calibrated threshold, not changed pose inputs. No new Test data are inspected for candidate choice.",
            "new_test_evaluations": 0,
            "historical_test_seen": True,
        },
    )
    print("Four finite learning-curve fits preregistered", flush=True)


def verify_freeze():
    record = read_json(evidence("diagnostic-lock"))
    if sha256(Path(__file__)) != record["source_sha256"]:
        raise ContractError("Workflow changed after diagnostic preregistration")
    for relative, expected in record["preserved_files"].items():
        if sha256(ROOT / relative) != expected:
            raise ContractError(f"Preserved artifact changed: {relative}")
    return record


def diagnose():
    if evidence("diagnosis").exists():
        raise ContractError("Diagnosis completed")
    record = verify_freeze()
    index = experiment.load_index(INDEX)
    accesses, audit, models = [], [], []
    with prior.train_validation_only(accesses):
        for source in index["split_manifest"]["sequences"]:
            if source["split"] != "train":
                continue
            sid = source["sequence_id"]
            cache = experiment.cache_for(index, sid)
            endpoints = sorted({r.sample_index for r in cache.records if r.quality == "valid"})
            chosen, last = [], -(10**9)
            for endpoint in endpoints:
                if endpoint >= last + 30:
                    chosen.append(endpoint)
                    last = endpoint
            audit.append(
                {
                    "sequence_id": sid,
                    "valid_windows": len(cache.features),
                    "nonoverlapping_30frame_endpoints": len(chosen),
                    "source_hash": index["sequences"][sid]["source_hash"],
                }
            )
        for name in (SMALL, REFERENCE):
            scorer = experiment.Scorer(ROOT / "runs" / name)
            train_x, _ = ORIGINAL_MATRIX(index, "train")
            val_x, _ = ORIGINAL_MATRIX(index, "validation")
            train_mean, val_mean = (
                float(scorer.score(train_x).mean()),
                float(scorer.score(val_x).mean()),
            )
            calibration = read_json(ROOT / "runs" / name / "calibration.json")
            normal = [r for r in calibration["sequences"] if r["label"] == 0]
            threshold_sensitivity = []
            for omitted in normal:
                rows = [
                    r
                    for r in calibration["sequences"]
                    if r["sequence_id"] != omitted["sequence_id"]
                ]
                chosen, _ = experiment.choose_threshold(rows, "balanced_sequence_accuracy")
                threshold_sensitivity.append(
                    {"omitted_normal": omitted["sequence_id"], "threshold": chosen["threshold"]}
                )
            models.append(
                {
                    "run": name,
                    "parameter_count": sum(p.numel() for p in scorer.model.parameters()),
                    "train_clean_mean": train_mean,
                    "normal_val_mean": val_mean,
                    "val_train_ratio": val_mean / train_mean,
                    "threshold": scorer.threshold["threshold"],
                    "normal_sequence_scores": normal,
                    "normal_omit_threshold_sensitivity": threshold_sensitivity,
                }
            )
        curves = []
        for choice in record["curves"]:
            folder = ROOT / "outputs" / PREFIX / "learning" / choice["name"]
            if folder.exists():
                raise ContractError("Preserve learning-curve fit; do not overwrite or add seeds")
            print(f"Learning curve {choice['name']}", flush=True)
            with train_matrix_inputs(index, selected=set(choice["train_ids"])):
                experiment.train(
                    load_config(ROOT / "configs/ur-human-isotropic-001.yaml"), INDEX, folder
                )
            scorer = experiment.Scorer(folder, calibrated=False)
            train_x = np.concatenate(
                [experiment.cache_for(index, sid).features for sid in choice["train_ids"]]
            )
            val_x, _ = ORIGINAL_MATRIX(index, "validation")
            train_mean, val_mean = (
                float(scorer.score(train_x).mean()),
                float(scorer.score(val_x).mean()),
            )
            curves.append(
                {
                    **choice,
                    "status": read_json(folder / "status.json"),
                    "train_clean_mean": train_mean,
                    "normal_val_mean": val_mean,
                    "val_train_ratio": val_mean / train_mean,
                    "train_windows": len(train_x),
                    "artifacts": experiment.verify_artifacts(folder, calibrated=False),
                }
            )
            print({"curve": choice["name"], "normal_val_mean": val_mean}, flush=True)
    save_new(
        evidence("diagnosis"),
        {
            "created_at": datetime.now(UTC).isoformat(),
            "train_audit": audit,
            "models": models,
            "learning_curves": curves,
            "train_video_count": len(audit),
            "train_valid_windows": sum(r["valid_windows"] for r in audit),
            "nonoverlapping_endpoints": sum(r["nonoverlapping_30frame_endpoints"] for r in audit),
            "feature_accesses": accesses,
            "test_feature_arrays_loaded": 0,
            "new_test_evaluations": 0,
            "evidence_tier": "real_feature_smoke",
            "limitations": "Nonoverlapping windows are still from 24 films, not independent people. Two subset orders are a small descriptive learning curve, not a causal proof of how much Test error comes from data scarcity. Validation has been repeatedly viewed; no new blind data.",
        },
    )
    verify_freeze()
    print(
        {
            "train_videos": len(audit),
            "nonoverlapping_endpoints": sum(r["nonoverlapping_30frame_endpoints"] for r in audit),
        },
        flush=True,
    )


def lock():
    if evidence("candidate-lock").exists():
        raise ContractError("Candidate choices already frozen")
    verify_freeze()
    diagnosis = read_json(evidence("diagnosis"))
    previous = read_json(ROOT / "docs/validation/isotropic-refinement-20261012-validation.json")
    reference = next(r for r in previous["results"] if r["run"] == REFERENCE)
    candidates = []
    for name, hidden, latent in [
        ("ur-mirror128-20261012-001", 128, 32),
        ("ur-mirror256-20261012-001", 256, 64),
    ]:
        config = load_config(ROOT / "configs/ur-human-isotropic-001.yaml")
        config["autoencoder"].update(hidden_dim=hidden, latent_dim=latent)
        config["experiment_training_inputs"] = {
            "workflow": "scripts/normal_data_support.py training",
            "profile": f"docs/validation/{PREFIX}-mirror.json",
            "transform": "Train-only horizontal reflection and anatomical left/right permutation before frozen ST-GCN; originals+mirrors1:1, no extra original films",
        }
        path = ROOT / "configs" / f"{name}.yaml"
        if path.exists() or (ROOT / "runs" / name).exists():
            raise ContractError("Require new config/run IDs")
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        candidates.append(
            {
                "run": name,
                "config": path.relative_to(ROOT).as_posix(),
                "config_sha256": sha256(path),
            }
        )
    save_new(
        evidence("candidate-lock"),
        {
            "created_at": datetime.now(UTC).isoformat(),
            "diagnosis_sha256": sha256(evidence("diagnosis")),
            "source_sha256": sha256(Path(__file__)),
            "candidates": candidates,
            "reference": REFERENCE,
            "reference_metrics": reference["metrics"],
            "reference_fn": [
                r["sequence_id"]
                for r in reference["predictions"]
                if r["label"] == 1 and r["prediction"] == 0
            ],
            "coarse_intervals": read_json(ROOT / "docs/validation/human-isotropic-lock.json")[
                "coarse_intervals"
            ],
            "rationale": "Original Train supplies only24 films and overlapping windows; investigate missing normal orientation variation with real-weight feature augmentation, independently of Test identities/scores. Keep the highest Validation wide22/23 as reference despite its viewed Test result.",
            "pilot_gate": "Only24 original Train ADL; same window records/quality/timestamps; finite genuine512D mirrors; 48 original feature parity checks atol/rtol1e-5; each mirror differs measurably. One fixed reflection, no extra transformations if gate fails.",
            "selection": "Strictly better overall Val Accuracy than22/23, F1 not lower, no extra FP/unknown, >=4 coarse phase hits, zero early events, current-good-pose phase alert for each newly recovered reference FN. Rank by Val Accuracy/F1/recall/fewer FP; ties retain reference. Original default remains if guards fail.",
            "stop": "Exactly two augmented AEs, <=100 epochs each/patience10, fixedseed42/sequence sampling/wd.001/noise0. No additional seeds, capacities, transforms, quality thresholds, calibration objectives or Test evaluations, even on success.",
            "learning_curve_diagnosis_only": len(diagnosis["learning_curves"]) == 4,
            "historical_test_seen": True,
        },
    )
    print(
        "Two Train-mirror candidates locked before mirror extraction or recognition results",
        flush=True,
    )


class MirrorExtractor:
    feature_dim = 512

    def __init__(self, extractor, original):
        self.extractor, self.original, self.position = extractor, original, 0
        self.parity_checks, self.max_parity_difference = 0, 0.0

    def extract_features(self, points, frame_size_wh):
        if self.position in (0, len(self.original) - 1):
            actual = self.extractor.extract_features(points, frame_size_wh)[0]
            np.testing.assert_allclose(actual, self.original[self.position], atol=1e-5, rtol=1e-5)
            self.max_parity_difference = max(
                self.max_parity_difference,
                float(np.max(np.abs(actual - self.original[self.position]))),
            )
            self.parity_checks += 1
        mirrored = mirror_human13(points, frame_size_wh[0])
        result = self.extractor.extract_features(mirrored, frame_size_wh)
        self.position += 1
        return result


def mirror():
    if evidence("mirror").exists() or CACHE.exists():
        raise ContractError("Preserve mirror preparation")
    verify_freeze()
    record = read_json(evidence("candidate-lock"))
    if sha256(evidence("diagnosis")) != record["diagnosis_sha256"]:
        raise ContractError("Diagnosis changed")
    index = experiment.load_index(INDEX)
    config = load_config(ROOT / "configs/ur-human-isotropic-001.yaml")
    extractor = HumanFeatureExtractor(
        config, index["weights"], select_device(config["runtime"]["device"], 4)
    )
    accesses, prepared = [], {}
    with prior.train_validation_only(accesses):
        for source in index["split_manifest"]["sequences"]:
            if source["split"] != "train":
                continue
            sid = source["sequence_id"]
            input_source(index, sid, train=True)
            original = experiment.cache_for(index, sid)
            observations, _ = load_pose_cache(
                project_path(index["sequences"][sid]["pose_path"]),
                {
                    "sequence_id": sid,
                    "source_hash": index["sequences"][sid]["source_hash"],
                    "recipe_hash": index["recipe_hash"],
                },
            )
            wrapped = MirrorExtractor(extractor, original.features)
            derived = features_from_observations(observations, wrapped, config, original.metadata)
            if [asdict(r) for r in derived.records] != [asdict(r) for r in original.records]:
                raise ContractError("Mirror altered original scheduled records")
            if wrapped.position != len(original.features) or wrapped.parity_checks != 2:
                raise ContractError("Missing original parity or mirror windows")
            difference = float(np.max(np.abs(derived.features - original.features)))
            if difference <= 1e-6:
                raise ContractError("Mirror supplied no new feature variation")
            derived.metadata.update(
                augmentation="horizontal_mirror_train_only",
                original_features_sha256=original.metadata["array_sha256"],
            )
            folder = CACHE / sid
            folder.mkdir(parents=True)
            path = folder / "mirror-features.npz"
            save_feature_cache(path, derived)
            combined = folder / "training-matrix.npz"
            write_npz(combined, features=np.concatenate([original.features, derived.features]))
            prepared[sid] = {
                "original_sha256": original.metadata["array_sha256"],
                "original_count": len(original.features),
                "mirror_path": path.relative_to(ROOT).as_posix(),
                "mirror_sha256": sha256(path),
                "combined_path": combined.relative_to(ROOT).as_posix(),
                "combined_sha256": sha256(combined),
                "combined_count": 2 * len(original.features),
                "parity_checks": wrapped.parity_checks,
                "max_original_parity_difference": wrapped.max_parity_difference,
                "max_mirror_difference": difference,
            }
            print(f"Mirror {sid}: {len(original.features)} real observed windows", flush=True)
    save_new(
        evidence("mirror"),
        {
            "created_at": datetime.now(UTC).isoformat(),
            "recipe_hash": index["recipe_hash"],
            "sequences": prepared,
            "strict_weight_load": extractor.load_report,
            "source_kind": "real observations with explicit derived Train augmentation",
            "original_video_count": len(prepared),
            "training_matrix_count": sum(r["combined_count"] for r in prepared.values()),
            "feature_accesses": accesses,
            "test_feature_arrays_loaded": 0,
            "new_test_evaluations": 0,
            "evidence_tier": "real_feature_smoke",
            "new_original_recordings": 0,
        },
    )
    verify_freeze()


def training():
    if evidence("validation").exists():
        raise ContractError("Completed finite round; no additional training")
    verify_freeze()
    record, profile = read_json(evidence("candidate-lock")), read_json(evidence("mirror"))
    index = experiment.load_index(INDEX)
    results, accesses = [], []
    with prior.train_validation_only(accesses):
        for candidate in record["candidates"]:
            name = candidate["run"]
            path = ROOT / candidate["config"]
            if sha256(path) != candidate["config_sha256"]:
                raise ContractError("Candidate config changed")
            if (ROOT / "runs" / name).exists():
                raise ContractError(
                    "Existing candidate run requires explicit audit; never overwrite"
                )
            config = load_config(path)
            config["experiment_training_inputs"]["profile_sha256"] = sha256(evidence("mirror"))
            with train_matrix_inputs(index, profile=profile):
                experiment.train(config, INDEX, ROOT / "runs" / name)
            experiment.calibrate(ROOT / "runs" / name)
            result = prior.validation_result(name, record, export=False)
            folder = ROOT / "outputs" / PREFIX / name
            folder.mkdir(parents=True)
            write_json(folder / "metrics.json", result["metrics"])
            write_csv(folder / "sequences.csv", result["predictions"])
            # Export all actual Val windows/events with the same scorer and core event path.
            scorer = experiment.Scorer(ROOT / "runs" / name)
            all_windows, all_events = [], []
            from fall_detect.detection.events import score_sequence

            for source in index["split_manifest"]["sequences"]:
                if source["split"] != "validation":
                    continue
                cache = experiment.cache_for(index, source["sequence_id"])
                _, windows, events = score_sequence(
                    cache, scorer.score(cache.features), scorer.threshold["threshold"]
                )
                all_windows.extend(windows)
                all_events.extend(events)
            write_csv(folder / "windows.csv", all_windows)
            write_csv(folder / "events.csv", all_events)
            result["exports"] = {
                p.relative_to(ROOT).as_posix(): sha256(p) for p in folder.iterdir()
            }
            results.append(result)
            save_new(evidence(f"result-{name}"), result)
            print(
                {"run": name, "metrics": result["metrics"], "eligible": result["eligible"]},
                flush=True,
            )
    eligible = [r for r in results if r["eligible"]]
    save_new(
        evidence("validation"),
        {
            "created_at": datetime.now(UTC).isoformat(),
            "results": results,
            "selected_candidate": max(eligible, key=prior.rank)["run"] if eligible else None,
            "highest_validation_candidate": max(results, key=prior.rank)["run"],
            "candidate_lock_sha256": sha256(evidence("candidate-lock")),
            "profile_sha256": sha256(evidence("mirror")),
            "feature_accesses": accesses,
            "test_feature_arrays_loaded": 0,
            "new_test_evaluations": 0,
            "default_run": "ur-sample-human-s1-001",
            "evidence_tier": "real_feature_smoke",
            "historical_test_seen": True,
            "formal_timing_ground_truth": False,
        },
    )
    verify_freeze()


def status():
    record = verify_freeze()
    complete = evidence("validation").exists()
    print(
        {
            "complete": complete,
            "preserved_files": len(record["preserved_files"]),
            "selected_candidate": read_json(evidence("validation"))["selected_candidate"]
            if complete
            else None,
            "new_test_evaluations": 0,
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["freeze", "diagnose", "lock", "mirror", "training", "status"]
    )
    command = parser.parse_args().command
    {
        "freeze": freeze,
        "diagnose": diagnose,
        "lock": lock,
        "mirror": mirror,
        "training": training,
        "status": status,
    }[command]()
