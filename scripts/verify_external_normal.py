"""Independent mixed-normal input, Scaler, CSV/K3 and preservation audit."""

import argparse
import csv
from collections import defaultdict

import external_normal_train as workflow
import human_isotropic_refinement as prior
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from verify_isotropic_refinement import independent_k3

from fall_detect.io_utils import ContractError, read_json, sha256
from fall_detect.training import experiment
from fall_detect.training.scaler import FeatureScaler


def require(value, reason):
    if not value:
        raise ContractError(reason)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def common_scale_normal_loss(scorer, index, reference_scaler):
    errors = []
    for s in index["split_manifest"]["sequences"]:
        if s["split"] != "validation" or s["label"] != 0:
            continue
        cache = experiment.cache_for(index, s["sequence_id"])
        x = scorer.scaler.transform(cache.features)
        # Batch one reproduces the actual scoring path; all models expressed in
        # one original Train24 scaler space for this diagnostic only.
        with torch.inference_mode():
            for row in x:
                tensor = torch.from_numpy(row[None]).to(scorer.device)
                reconstruction = scorer.model(tensor).cpu().numpy()[0]
                err = (
                    (reconstruction.astype(np.float64) - row)
                    * scorer.scaler.scale
                    / reference_scaler.scale
                )
                errors.append(float(np.mean(err**2)))
    return float(np.mean(errors))


def verify(write=False):
    root = workflow.ROOT
    target = root / f"docs/validation/{workflow.PREFIX}-verification.json"
    if write and target.exists():
        raise ContractError("Preserve completed verification")
    lock = workflow.verify_lock()
    data = read_json(workflow.RESULT)
    profile = read_json(workflow.PROFILE)
    require(data["lock_sha256"] == sha256(workflow.LOCK), "Model selection lock changed")
    original_index = experiment.load_index(workflow.INDEX)
    accesses, checks = [], []
    with prior.train_validation_only(accesses):
        original_x, _ = experiment.normal_matrix(original_index, "train")
        common_scaler = FeatureScaler.fit(original_x)
        with workflow.mixed_normal_inputs(profile):
            train_x, hashes, counts = experiment.normal_matrix(original_index, "train", True)
            _, val_hashes = experiment.normal_matrix(original_index, "validation")
        require(len(train_x) == 7469 and len(hashes) == 45, "Mixed-normal actual count mismatch")
        scaler_expected = FeatureScaler.fit(train_x)
        common = {
            "ur-human-isotropic-001": common_scale_normal_loss(
                experiment.Scorer(root / "runs/ur-human-isotropic-001"),
                original_index,
                common_scaler,
            ),
            "ur-isotropic-wide-20261012-001": common_scale_normal_loss(
                experiment.Scorer(root / "runs/ur-isotropic-wide-20261012-001"),
                original_index,
                common_scaler,
            ),
        }
        expected_ids = {
            s["sequence_id"]
            for s in original_index["split_manifest"]["sequences"]
            if s["split"] == "validation"
        }
        for result in data["results"]:
            name = result["run"]
            run = root / "runs" / name
            require(
                experiment.verify_artifacts(run) == result["artifact_hashes"],
                "Frozen new run changed",
            )
            scaler = FeatureScaler.load(run / "scaler.npz")
            np.testing.assert_array_equal(scaler.mean, scaler_expected.mean)
            np.testing.assert_array_equal(scaler.scale, scaler_expected.scale)
            sources = read_json(run / "sources.json")
            require(
                sources["train_feature_hashes"] == hashes, "AE actual Train input hashes differ"
            )
            require(
                sources["training_sampling"]["train_normal_feature_counts"] == counts,
                "AE source row counts differ",
            )
            require(
                sources["val_normal_feature_hashes"] == val_hashes,
                "Checkpoint used changed/non-normal Val",
            )
            require(
                sources["experimental_training_inputs"]["profile_sha256"]
                == sha256(workflow.PROFILE),
                "Additional Train provenance missing",
            )
            require(
                read_json(run / "split-manifest.json") == original_index["split_manifest"],
                "UR evaluation split changed",
            )
            folder = root / "outputs" / workflow.PREFIX / name
            rows = read_csv(folder / "sequences.csv")
            windows = read_csv(folder / "windows.csv")
            events = read_csv(folder / "events.csv")
            require(
                {r["sequence_id"] for r in rows} == expected_ids and len(rows) == 23,
                "Evaluation denominator changed",
            )
            require({r["sequence_id"] for r in windows} == expected_ids, "Window scope changed")
            usable = [r for r in rows if r["prediction"] != ""]
            truth = [int(r["label"]) for r in usable]
            predictions = [int(r["prediction"]) for r in usable]
            tn, fp, fn, tp = confusion_matrix(truth, predictions, labels=[0, 1]).ravel().tolist()
            metrics = result["metrics"]
            require(
                [tp, tn, fp, fn] == [metrics[k] for k in ("TP", "TN", "FP", "FN")],
                "Independent confusion mismatch",
            )
            require(
                accuracy_score(truth, predictions) == metrics["sequence_accuracy"],
                "Accuracy mismatch",
            )
            require(f1_score(truth, predictions) == metrics["sequence_fall_f1"], "F1 mismatch")
            require(
                (tp + tn) / len(rows) == metrics["operational_accuracy"]
                and len(rows) - len(usable) == metrics["n_unknown"],
                "Overall/unknown denominator mismatch",
            )
            grouped = defaultdict(list)
            for w in windows:
                grouped[w["sequence_id"]].append(w)
            for row in rows:
                score = independent_k3(grouped[row["sequence_id"]])
                expected = None if score is None else int(score > result["threshold"])
                actual = None if row["prediction"] == "" else int(row["prediction"])
                require(expected == actual, "Independent strict K3 mismatch")
                require(score is None or score == float(row["sequence_score"]), "K3 score mismatch")
                require(
                    any(e["sequence_id"] == row["sequence_id"] for e in events) == bool(expected),
                    "Event/outcome mismatch",
                )
            recomputed = prior.validation_result(name, lock, export=False)
            for key in ("metrics", "phases", "adoption_checks", "eligible"):
                require(recomputed[key] == result[key], f"Recomputed {key} differs")
            require(
                workflow.timing_diagnostic(name) == result["new_ai_annotation_timing_diagnostic"],
                "New annotation diagnostic changed",
            )
            history = read_csv(run / "training-history.csv")
            require(
                min(float(r["val_normal_loss"]) for r in history)
                == result["status"]["best_val_normal_loss"],
                "Checkpoint not minimum normal-Val loss",
            )
            common[name] = common_scale_normal_loss(
                experiment.Scorer(run), original_index, common_scaler
            )
            checks.append(
                {
                    "run": name,
                    "metrics": metrics,
                    "epochs": len(history),
                    "train_only_scaler_exact": True,
                    "ae_train_hashes_exact": True,
                    "csv_independent_k3": 23,
                    "new_timing_diagnostic_recomputed": True,
                    "adopted": result["eligible"],
                    "common_train24_units_normal_val_mse": common[name],
                    "output_hashes": {p.name: sha256(p) for p in folder.iterdir()},
                }
            )
        baseline_timing = workflow.timing_diagnostic(lock["reference"])
    annotation = read_json(workflow.ANNOTATIONS / "manifest.json")
    for name, digest in annotation["files"].items():
        require(sha256(workflow.ANNOTATIONS / name) == digest, "Annotation bytes changed")
    require(
        all(a["split"] != "test" for a in accesses + data["feature_accesses"]),
        "Test feature access in workflow",
    )
    require(
        data["selected_candidate"] is None and not any(c["adopted"] for c in checks),
        "Incorrect adoption",
    )
    evidence = {
        "source_sha256": sha256(__file__),
        "lock_sha256": sha256(workflow.LOCK),
        "evidence_tier": "real_feature_smoke",
        "preserved_old_files": len(lock["files_before"]),
        "train_sources": len(hashes),
        "train_windows": len(train_x),
        "checks": checks,
        "common_units_normal_val_mse": common,
        "baseline_new_ai_annotation_timing": baseline_timing,
        "new_test_evaluations": 0,
        "test_feature_arrays_loaded": 0,
        "feature_accesses": accesses,
        "human13_visibility_ground_truth": False,
        "ai_sampled_annotation_manifest_sha256": sha256(workflow.ANNOTATIONS / "manifest.json"),
    }
    if write:
        workflow.save_new(target, evidence)
    print(
        {
            "verified_candidates": len(checks),
            "preserved": len(lock["files_before"]),
            "epochs": sum(c["epochs"] for c in checks),
            "common_units_normal_val_mse": common,
            "test_feature_arrays_loaded": 0,
        }
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--write", action="store_true")
    verify(p.parse_args().write)
