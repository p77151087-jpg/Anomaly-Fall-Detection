"""Independently verify Train augmentation and Validation outputs, without Test arrays."""

import argparse
import csv
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import normal_data_support as workflow
from verify_isotropic_refinement import independent_k3

from fall_detect.data.cache import load_feature_cache
from fall_detect.io_utils import ContractError, read_json, sha256
from fall_detect.training import experiment
from fall_detect.training.scaler import FeatureScaler


def check(value, message):
    if not value:
        raise ContractError(message)


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def verify(write=False):
    root = workflow.ROOT
    frozen = workflow.verify_freeze()
    lock = read_json(workflow.evidence("candidate-lock"))
    results = read_json(workflow.evidence("validation"))
    diagnosis = read_json(workflow.evidence("diagnosis"))
    profile = read_json(workflow.evidence("mirror"))
    repair = read_json(workflow.evidence("implementation-repair"))
    check(
        results["candidate_lock_sha256"] == sha256(workflow.evidence("candidate-lock")),
        "Lock changed",
    )
    check(results["profile_sha256"] == sha256(workflow.evidence("mirror")), "Profile changed")
    index = experiment.load_index(workflow.INDEX)
    train_ids = {
        s["sequence_id"] for s in index["split_manifest"]["sequences"] if s["split"] == "train"
    }
    val_ids = {
        s["sequence_id"] for s in index["split_manifest"]["sequences"] if s["split"] == "validation"
    }
    check(
        set(profile["sequences"]) == train_ids and len(train_ids) == 24,
        "Derived inputs not original Train24",
    )
    arrays, hashes, counts, accesses, checks = [], {}, {}, [], []
    with workflow.prior.train_validation_only(accesses):
        for source in index["split_manifest"]["sequences"]:
            if source["split"] != "train":
                continue
            sid = source["sequence_id"]
            workflow.input_source(index, sid, train=True)
            original = experiment.cache_for(index, sid)
            entry = profile["sequences"][sid]
            mirror = load_feature_cache(
                root / entry["mirror_path"],
                {
                    "augmentation": "horizontal_mirror_train_only",
                    "split": "train",
                    "recipe_hash": index["recipe_hash"],
                    "original_features_sha256": original.metadata["array_sha256"],
                },
            )
            check(original.metadata["array_sha256"] == entry["original_sha256"], "Original hash")
            check(sha256(root / entry["mirror_path"]) == entry["mirror_sha256"], "Mirror hash")
            combined_path = root / entry["combined_path"]
            check(sha256(combined_path) == entry["combined_sha256"], "Combined hash")
            with np.load(combined_path, allow_pickle=False) as loaded:
                combined = loaded["features"].copy()
            np.testing.assert_array_equal(
                combined, np.concatenate([original.features, mirror.features])
            )
            check(mirror.features.shape == original.features.shape, "Mirror shape")
            check(
                [(r.sample_index, r.track_id, r.quality) for r in original.records]
                == [(r.sample_index, r.track_id, r.quality) for r in mirror.records],
                "Mirror changed windows",
            )
            arrays.append(combined)
            hashes[sid], counts[sid] = entry["combined_sha256"], len(combined)
        expected_scaler = FeatureScaler.fit(np.concatenate(arrays))
        check(sum(counts.values()) == 7276, "Train augmented row count")
        for candidate, result in zip(lock["candidates"], results["results"], strict=True):
            name = result["run"]
            check(name == candidate["run"], "Candidate order/choice changed")
            run = root / "runs" / name
            check(
                experiment.verify_artifacts(run) == result["artifact_hashes"], "Frozen run changed"
            )
            source = read_json(run / "sources.json")
            check(source["train_feature_hashes"] == hashes, "AE Train hash manifest mismatch")
            check(
                source["training_sampling"]["train_normal_feature_counts"] == counts,
                "Sampling counts",
            )
            binding = source["experimental_training_inputs"]
            check(
                binding["profile_sha256"] == sha256(workflow.evidence("mirror")),
                "Input profile not bound",
            )
            check(
                binding["workflow_sha256"] == sha256(root / "scripts/normal_data_support.py"),
                "Workflow binding",
            )
            check(
                binding["implementation_repair_sha256"]
                == sha256(workflow.evidence("implementation-repair")),
                "Repair binding",
            )
            scaler = FeatureScaler.load(run / "scaler.npz")
            np.testing.assert_array_equal(scaler.mean, expected_scaler.mean)
            np.testing.assert_array_equal(scaler.scale, expected_scaler.scale)
            original_options = workflow.yaml.safe_load(
                (root / candidate["config"]).read_text(encoding="utf-8")
            )["autoencoder"]
            check(
                experiment.Scorer(run).config["autoencoder"] == original_options,
                "Repair changed model parameters",
            )
            folder = root / "outputs" / workflow.PREFIX / name
            rows = csv_rows(folder / "sequences.csv")
            windows = csv_rows(folder / "windows.csv")
            events = csv_rows(folder / "events.csv")
            check({r["sequence_id"] for r in rows} == val_ids, "Not original Val23")
            evaluable = [r for r in rows if r["prediction"]]
            truth, predicted = (
                [int(r["label"]) for r in evaluable],
                [int(r["prediction"]) for r in evaluable],
            )
            tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel().tolist()
            m = result["metrics"]
            check(
                [tp, tn, fp, fn] == [m[k] for k in ("TP", "TN", "FP", "FN")], "Confusion mismatch"
            )
            check(
                float(accuracy_score(truth, predicted)) == m["sequence_accuracy"], "Binary Accuracy"
            )
            check(float(f1_score(truth, predicted)) == m["sequence_fall_f1"], "Binary Fall F1")
            check((tp + tn) / len(rows) == m["operational_accuracy"], "Overall denominator")
            check(len(rows) - len(evaluable) == m["n_unknown"], "Unknown denominator")
            grouped = defaultdict(list)
            for row in windows:
                grouped[row["sequence_id"]].append(row)
            check(set(grouped) == val_ids, "Not Val windows")
            for row in rows:
                score = independent_k3(grouped[row["sequence_id"]])
                check(score == float(row["sequence_score"]), "K3 score mismatch")
                check(
                    int(score > result["threshold"]) == int(row["prediction"]), "K3 strict decision"
                )
                check(
                    any(e["sequence_id"] == row["sequence_id"] for e in events)
                    == bool(int(row["prediction"])),
                    "Events mismatch",
                )
            recomputed = workflow.prior.validation_result(name, lock, export=False)
            for key in ("metrics", "phases", "adoption_checks", "eligible"):
                check(recomputed[key] == result[key], f"Recomputed {key} mismatch")
            history = csv_rows(run / "training-history.csv")
            check(len(history) <= 100, "Exceeded fixed epochs")
            checks.append(
                {
                    "run": name,
                    "metrics": m,
                    "scaler_exact_train_only": True,
                    "input_matrix_hashes_match": True,
                    "independent_k3_sequences": len(rows),
                    "eligible": result["eligible"],
                    "epochs": len(history),
                }
            )
        for curve in diagnosis["learning_curves"]:
            folder = root / "outputs" / workflow.PREFIX / "learning" / curve["name"]
            check(
                experiment.verify_artifacts(folder, calibrated=False) == curve["artifacts"],
                "Curve changed",
            )
            check(not (folder / "threshold.json").exists(), "Curve used for classification")
            source = read_json(folder / "sources.json")
            check(
                set(source["train_feature_hashes"]) == set(curve["train_ids"]) <= train_ids,
                "Curve non-Train inputs",
            )
    for collection in [
        diagnosis["feature_accesses"],
        profile["feature_accesses"],
        results["feature_accesses"],
        accesses,
    ]:
        check(all(a["split"] in ("train", "validation") for a in collection), "Test feature access")
    xml = root / "docs/validation/pytest-normal-data-support-20261012.xml"
    suites = ET.parse(xml).getroot().findall("testsuite")
    check(
        all(int(s.attrib["errors"]) == 0 and int(s.attrib["failures"]) == 0 for s in suites),
        "Risk tests failed",
    )
    rejected = {}
    for command in ("freeze", "diagnose", "lock", "mirror", "training"):
        response = subprocess.run(
            [sys.executable, str(root / "scripts/normal_data_support.py"), command],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        check(response.returncode != 0, "Completed work was not protected")
        rejected[command] = response.stderr.splitlines()[-1]
    output = {
        "checks": checks,
        "preserved_files": len(frozen["preserved_files"]),
        "learning_curve_fit_count": len(diagnosis["learning_curves"]),
        "learning_curve_epochs": sum(c["status"]["epochs"] for c in diagnosis["learning_curves"]),
        "original_feature_parity_checks": sum(
            s["parity_checks"] for s in profile["sequences"].values()
        ),
        "max_original_feature_difference": max(
            s["max_original_parity_difference"] for s in profile["sequences"].values()
        ),
        "original_train_films": 24,
        "train_rows_original": 3638,
        "train_rows_augmented": 7276,
        "synthetic_tests": {
            "passed": sum(int(s.attrib["tests"]) for s in suites),
            "xml_sha256": sha256(xml),
        },
        "config_transport_repair": repair,
        "completed_mutations_rejected": rejected,
        "evidence_tier": "real_feature_smoke",
        "fresh_raw_video_pose_extraction": False,
        "test_feature_arrays_loaded": 0,
        "new_test_evaluations": 0,
        "verifier_source_sha256": sha256(Path(__file__)),
    }
    if write:
        workflow.save_new(workflow.evidence("verification"), output)
        figure(root, diagnosis, results)
    print(
        {
            "verified_candidates": len(checks),
            "preserved_files": len(frozen["preserved_files"]),
            "genuine_feature_parity_max": output["max_original_feature_difference"],
            "new_test_evaluations": 0,
        }
    )


def figure(root, diagnosis, results):
    common = read_json(workflow.evidence("common-scale"))
    values = {r["name"]: r["normal_val_mse_common_train24_scale"] for r in common["results"]}
    folder = root / "outputs" / workflow.PREFIX
    path = folder / "data-support-and-validation.png"
    check(not path.exists(), "Preserve figure")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for seed in (42, 43):
        axes[0].plot(
            [8, 16, 24],
            [values[f"subset-{seed}-n8"], values[f"subset-{seed}-n16"], values[workflow.SMALL]],
            marker="o",
            label=f"Nested Train order {seed}",
        )
    axes[0].set(
        xlabel="Original normal Train films",
        ylabel="Normal Val MSE in fixed Train24 units",
        xticks=[8, 16, 24],
        title="Data support diagnostic: same8 normal Val films",
    )
    axes[0].legend()
    reference = read_json(workflow.evidence("candidate-lock"))["reference_metrics"]
    metrics = [reference] + [r["metrics"] for r in results["results"]]
    x = np.arange(3)
    axes[1].bar(
        x - 0.17,
        [m["operational_accuracy"] * 100 for m in metrics],
        0.34,
        label="Overall Accuracy %",
    )
    axes[1].bar(
        x + 0.17, [m["sequence_fall_f1"] * 100 for m in metrics], 0.34, label="Fall F1 x100"
    )
    axes[1].set(
        xticks=x,
        xticklabels=["Wide reference", "Train mirror128", "Train mirror256"],
        ylim=[0, 110],
        title="Original23 Val films: no Test evaluation",
    )
    for i, m in enumerate(metrics):
        axes[1].text(
            i,
            102,
            f"TP{m['TP']} TN{m['TN']}\nFP{m['FP']} FN{m['FN']} U{m['n_unknown']}",
            ha="center",
            fontsize=9,
        )
    axes[1].legend(loc="lower right")
    fig.suptitle("Train-only mirror augments24 films; it does not add new independent recordings")
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    verify(parser.parse_args().write)
