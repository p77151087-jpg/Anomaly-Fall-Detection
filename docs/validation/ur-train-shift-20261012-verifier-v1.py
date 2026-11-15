"""Independent Train-only matrix/Scaler, unchanged URVal, CSV/K3 and archive audit."""

import argparse
from collections import defaultdict

import numpy as np
import ur_train_shift as workflow
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from verify_isotropic_refinement import independent_k3, read_csv

from fall_detect.data.cache import load_feature_cache
from fall_detect.io_utils import ContractError, read_json, sha256
from fall_detect.training import experiment
from fall_detect.training.scaler import FeatureScaler


def check(condition, message):
    if not condition:
        raise ContractError(message)


def verify(write=False):
    root = workflow.ROOT
    target = root / f"docs/validation/{workflow.PREFIX}-verification.json"
    if write and target.exists():
        raise ContractError("Preserve completed independent verification")
    lock = workflow.verify_lock()
    profile = workflow.checked_profile(lock)
    evidence = read_json(workflow.RESULT)
    result = evidence["result"]
    check(evidence["lock_sha256"] == sha256(workflow.LOCK), "Lock changed")
    check(evidence["profile_sha256"] == sha256(workflow.PROFILE), "Profile changed")
    accesses = []
    with workflow.prior.train_validation_only(accesses):
        index = experiment.load_index(workflow.INDEX)
        check(index["split_manifest"]["split_hash"] == lock["split_hash"], "Original split changed")
        for sid, entry in profile["sequences"].items():
            workflow.support.input_source(index, sid, train=True)
            original = experiment.cache_for(index, sid)
            check(
                entry["original_sha256"] == original.metadata["array_sha256"],
                "Original features changed",
            )
            derived = load_feature_cache(root / entry["derived_path"])
            difference = np.max(np.abs(derived.features - original.features), axis=1)
            selected = np.flatnonzero(difference > 1e-6)
            check(
                selected.tolist() == entry["eligible_derived_rows"], "Derived row selection changed"
            )
            check(entry["derived_count"] == len(selected), "Derived count changed")
            with np.load(root / entry["combined_path"], allow_pickle=False) as saved:
                np.testing.assert_array_equal(
                    saved["features"],
                    np.concatenate([original.features, derived.features[selected]]),
                )
            check(
                [vars(r) for r in derived.records] == [vars(r) for r in original.records],
                "Timing or quality changed",
            )
        with workflow.support.train_matrix_inputs(index, profile=profile):
            train, train_hashes, counts = experiment.normal_matrix(index, "train", True)
            _, val_hashes = experiment.normal_matrix(index, "validation")
        check(
            set(train_hashes) == set(lock["train_ids"]) and len(train_hashes) == 24,
            "Non-UR Train input",
        )
        check(len(train) == profile["training_matrix_count"], "Training denominator changed")
        scaler = FeatureScaler.load(root / "runs" / workflow.RUN / "scaler.npz")
        fitted = FeatureScaler.fit(train)
        np.testing.assert_array_equal(scaler.mean, fitted.mean)
        np.testing.assert_array_equal(scaler.scale, fitted.scale)
        sources = read_json(root / "runs" / workflow.RUN / "sources.json")
        check(sources["train_feature_hashes"] == train_hashes, "AE Train sources differ")
        check(sources["train_sequence_feature_counts"] == counts, "AE Train counts differ")
        check(
            sources["val_normal_feature_hashes"] == val_hashes and len(val_hashes) == 8,
            "Normal Val sources differ",
        )
        check(
            sources["experimental_training_inputs"]["profile_sha256"] == sha256(workflow.PROFILE),
            "Explicit augmentation provenance missing",
        )
        check(
            read_json(root / "runs" / workflow.RUN / "split-manifest.json")
            == index["split_manifest"],
            "UR split changed in run",
        )
        check(
            experiment.verify_artifacts(root / "runs" / workflow.RUN) == result["artifact_hashes"],
            "Model artifacts changed",
        )
        folder = root / "outputs" / workflow.PREFIX / workflow.RUN
        rows, windows, events = [
            read_csv(folder / name) for name in ["sequences.csv", "windows.csv", "events.csv"]
        ]
        ids = {
            s["sequence_id"]
            for s in index["split_manifest"]["sequences"]
            if s["split"] == "validation"
        }
        check({r["sequence_id"] for r in rows} == ids and len(rows) == 23, "Not original URVal23")
        check({r["sequence_id"] for r in windows} == ids, "Wrong window scope")
        check({r["sequence_id"] for r in events} <= ids, "Wrong event scope")
        valid = [r for r in rows if r["prediction"] != ""]
        truth = [int(r["label"]) for r in valid]
        predicted = [int(r["prediction"]) for r in valid]
        tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel().tolist()
        metrics = result["metrics"]
        check(
            [tp, tn, fp, fn] == [metrics[k] for k in ["TP", "TN", "FP", "FN"]],
            "Confusion matrix mismatch",
        )
        check(accuracy_score(truth, predicted) == metrics["sequence_accuracy"], "Accuracy mismatch")
        check(f1_score(truth, predicted) == metrics["sequence_fall_f1"], "F1 mismatch")
        check(
            (tp + tn) / len(rows) == metrics["operational_accuracy"], "Overall denominator mismatch"
        )
        check(len(rows) - len(valid) == metrics["n_unknown"], "Unknown denominator mismatch")
        grouped = defaultdict(list)
        for row in windows:
            grouped[row["sequence_id"]].append(row)
        for row in rows:
            score = independent_k3(grouped[row["sequence_id"]])
            expected = None if score is None else int(score > result["threshold"])
            actual = None if row["prediction"] == "" else int(row["prediction"])
            check(expected == actual, "Independent strict K3 prediction differs")
            check(
                score is None or score == float(row["sequence_score"]),
                "Independent K3 score differs",
            )
            check(
                any(e["sequence_id"] == row["sequence_id"] for e in events) == bool(expected),
                "Event/outcome differs",
            )
        recomputed = workflow.prior.validation_result(workflow.RUN, lock, export=False)
        for key in ["metrics", "phases", "recovered_fn", "adoption_checks", "eligible"]:
            check(recomputed[key] == result[key], f"Recomputed {key} differs")
        history = read_csv(root / "runs" / workflow.RUN / "training-history.csv")
        check(len(history) <= 100, "Exceeded fixed training budget")
        check(
            min(float(r["val_normal_loss"]) for r in history)
            == result["status"]["best_val_normal_loss"],
            "Not minimum normalVal checkpoint",
        )
    check(
        all(
            a["split"] != "test"
            for a in accesses + profile["feature_accesses"] + evidence["feature_accesses"]
        ),
        "Forbidden Test access",
    )
    check(
        evidence["selected_candidate"] == (workflow.RUN if result["eligible"] else None),
        "Adoption changed",
    )
    workflow.verify_lock()
    output = {
        "source_sha256": sha256(__file__),
        "lock_sha256": sha256(workflow.LOCK),
        "profile_sha256": sha256(workflow.PROFILE),
        "evidence_tier": "real_feature_smoke",
        "preserved_files": len(lock["files_before"]),
        "Train_original_videos": 24,
        "Train_original_windows": profile["original_windows"],
        "Train_derived_windows": profile["changed_derived_windows"],
        "Train_combined_windows": len(train),
        "Scaler_Train_only_exact": True,
        "AE_Train_sources_exact": True,
        "UR_split_unchanged": True,
        "original_parity_checks": sum(e["parity_checks"] for e in profile["sequences"].values()),
        "original_parity_max": max(e["original_parity_max"] for e in profile["sequences"].values()),
        "translation_tensor_max": max(
            e["translation_tensor_max"] for e in profile["sequences"].values()
        ),
        "translation_feature_max": max(
            e["translation_feature_max"] for e in profile["sequences"].values()
        ),
        "csv_independent_K3_sequences": 23,
        "metrics": metrics,
        "epochs": len(history),
        "adopted": result["eligible"],
        "external_Train_sources": 0,
        "new_Test_evaluations": 0,
        "Test_feature_arrays_loaded": 0,
        "feature_accesses": accesses,
    }
    if write:
        workflow.save_new(target, output)
    print({k: v for k, v in output.items() if k != "feature_accesses"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    verify(parser.parse_args().write)
