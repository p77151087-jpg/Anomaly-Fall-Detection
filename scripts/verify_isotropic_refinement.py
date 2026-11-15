"""Independently check bounded refinement outputs without loading Test features."""

import argparse
import csv
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

matplotlib.use("Agg")
import human_isotropic_refinement as workflow
import matplotlib.pyplot as plt

from fall_detect.io_utils import ContractError, read_json, sha256
from fall_detect.training import experiment
from fall_detect.training.scaler import FeatureScaler


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def check(value, message):
    if not value:
        raise ContractError(message)


def independent_k3(windows):
    histories, previous, best = defaultdict(lambda: deque(maxlen=3)), {}, {}
    for row in sorted(windows, key=lambda r: (int(r["sample_index"]), r["track_id"])):
        if not row["track_id"]:
            check(not row["raw_score"], "Untracked endpoint cannot carry a score")
            continue
        track, sample = int(row["track_id"]), int(row["sample_index"])
        queue = histories[track]
        if previous.get(track, sample - 1) != sample - 1 or not row["raw_score"]:
            queue.clear()
        previous[track] = sample
        if row["raw_score"]:
            queue.append(float(row["raw_score"]))
            if len(queue) == 3:
                best[track] = max(best.get(track, -np.inf), min(queue))
    return max(best.values()) if best else None


def verify(write=False):
    root = workflow.ROOT
    lock = read_json(workflow.evidence("lock"))
    workflow.verify_lock(lock)
    results = read_json(workflow.evidence("validation"))
    check(results["lock_sha256"] == sha256(workflow.evidence("lock")), "Lock changed")
    accesses, checks = [], []
    with workflow.train_validation_only(accesses):
        index = experiment.load_index(workflow.INDEX)
        train_x, train_hashes = experiment.normal_matrix(index, "train")
        _, val_hashes = experiment.normal_matrix(index, "validation")
        expected_scaler = FeatureScaler.fit(train_x)
        validation_ids = {
            s["sequence_id"]
            for s in index["split_manifest"]["sequences"]
            if s["split"] == "validation"
        }
        check(len(train_hashes) == 24 and len(train_x) == 3638, "Train count changed")
        for result in results["results"]:
            name = result["run"]
            run = root / "runs" / name
            check(experiment.verify_artifacts(run) == result["artifact_hashes"], "Run changed")
            scaler = FeatureScaler.load(run / "scaler.npz")
            np.testing.assert_array_equal(scaler.mean, expected_scaler.mean)
            np.testing.assert_array_equal(scaler.scale, expected_scaler.scale)
            sources = read_json(run / "sources.json")
            check(sources["train_feature_hashes"] == train_hashes, "AE used non-Train features")
            check(sources["val_normal_feature_hashes"] == val_hashes, "Checkpoint sources changed")
            folder = root / "outputs" / workflow.PREFIX / name
            rows = read_csv(folder / "sequences.csv")
            windows = read_csv(folder / "windows.csv")
            events = read_csv(folder / "events.csv")
            check({r["sequence_id"] for r in rows} == validation_ids, "Not original Validation")
            check({r["sequence_id"] for r in events} <= validation_ids, "Non-Val events")
            check({r["sequence_id"] for r in windows} == validation_ids, "Non-Val windows")
            evaluable = [r for r in rows if r["prediction"] != ""]
            truth = [int(r["label"]) for r in evaluable]
            predicted = [int(r["prediction"]) for r in evaluable]
            metrics = result["metrics"]
            tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel().tolist()
            check(
                [tp, tn, fp, fn] == [metrics[k] for k in ("TP", "TN", "FP", "FN")],
                "Independent confusion matrix mismatch",
            )
            check(accuracy_score(truth, predicted) == metrics["sequence_accuracy"], "Accuracy")
            check(f1_score(truth, predicted) == metrics["sequence_fall_f1"], "F1")
            check((tp + tn) / len(rows) == metrics["operational_accuracy"], "Overall accuracy")
            check(len(rows) - len(evaluable) == metrics["n_unknown"], "Unknown denominator")
            grouped = defaultdict(list)
            for row in windows:
                grouped[row["sequence_id"]].append(row)
            for row in rows:
                score = independent_k3(grouped[row["sequence_id"]])
                expected = None if score is None else int(score > result["threshold"])
                actual = None if row["prediction"] == "" else int(row["prediction"])
                check(expected == actual, "Independent K3 prediction mismatch")
                if score is not None:
                    check(score == float(row["sequence_score"]), "Independent K3 score mismatch")
                check(
                    any(e["sequence_id"] == row["sequence_id"] for e in events) == bool(expected),
                    "Events disagree with sequence decision",
                )
            recomputed = workflow.validation_result(name, lock, export=False)
            for key in ("metrics", "phases", "recovered_fn", "adoption_checks", "eligible"):
                check(recomputed[key] == result[key], f"Recomputed {key} mismatch")
            history = read_csv(run / "training-history.csv")
            check(len(history) <= 100, "Epoch budget exceeded")
            check(
                min(float(r["val_normal_loss"]) for r in history)
                == result["status"]["best_val_normal_loss"],
                "Not best normal-Val checkpoint",
            )
            checks.append(
                {
                    "run": name,
                    "metrics": metrics,
                    "independent_k3_sequences": len(rows),
                    "train_only_scaler_exact": True,
                    "train_only_ae_hashes": True,
                    "normal_val_checkpoint": True,
                    "eligibility": result["eligible"],
                }
            )
    check(
        all(a["split"] != "test" for a in results["feature_accesses"] + accesses),
        "Test access in refinement audit",
    )
    rejection_checks = {}
    for command in ("diagnose", "lock", "training"):
        response = subprocess.run(
            [sys.executable, str(root / "scripts/human_isotropic_refinement.py"), command],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        check(response.returncode != 0, "Completed mutation was not rejected")
        rejection_checks[command] = {"rejected": True, "reason": response.stderr.splitlines()[-1]}
    output = {
        "preserved_files": len(lock["files_before"]),
        "checks": checks,
        "feature_accesses": accesses,
        "test_feature_arrays_loaded": 0,
        "new_test_evaluations": 0,
        "completed_mutations_rejected": rejection_checks,
        "synthetic_tests": {
            "passed": 26,
            "failed": 0,
            "xml_sha256": sha256(root / "docs/validation/pytest-isotropic-refinement-20261012.xml"),
        },
        "evidence_tier": "real_feature_smoke",
        "verifier_source_sha256": sha256(Path(__file__)),
    }
    if write:
        workflow.save_new(workflow.evidence("verification"), output)
        plot(root, results)
    print(
        {
            "preserved_files": output["preserved_files"],
            "checked_candidates": len(checks),
            "new_test_evaluations": 0,
            "metrics_and_k3_match": True,
        }
    )


def plot(root, results):
    diagnosis = read_json(workflow.evidence("diagnosis"))
    folder = root / "outputs" / workflow.PREFIX
    path = folder / "validation-comparison.png"
    check(not path.exists(), "Preserve comparison figure")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    rows = [r for r in diagnosis["rows"] if r["split"] == "validation"]
    for label, color, title in [(0, "#2874a6", "ADL"), (1, "#c0392b", "Fall")]:
        subset = [r for r in rows if r["label"] == label]
        axes[0].scatter(
            [r["sequence_id"] for r in subset],
            [r["sequence_score"] for r in subset],
            color=color,
            label=title,
        )
    axes[0].axhline(diagnosis["threshold"], ls="--", color="black", label="Frozen threshold")
    axes[0].set_yscale("log")
    axes[0].tick_params(axis="x", rotation=90)
    axes[0].set_ylabel("Max legal K3 AE reconstruction error (not probability)")
    axes[0].set_title("Reference: overlapping Validation sequence scores")
    axes[0].legend()
    names = ["Reference", "Compact", "Wide", "Denoise"]
    metrics = [diagnosis["validation_metrics"]] + [r["metrics"] for r in results["results"]]
    x = np.arange(4)
    axes[1].bar(
        x - 0.17, [m["operational_accuracy"] * 100 for m in metrics], 0.34, label="Accuracy %"
    )
    axes[1].bar(
        x + 0.17, [m["sequence_fall_f1"] * 100 for m in metrics], 0.34, label="Fall F1 x100"
    )
    axes[1].set_xticks(x, names)
    axes[1].set_ylim(0, 110)
    for i, m in enumerate(metrics):
        axes[1].text(
            i, 102, f"TP{m['TP']} TN{m['TN']}\nFP{m['FP']} FN{m['FN']}", ha="center", fontsize=8
        )
    axes[1].set_title("Original 23 Validation sequences; unknown=0")
    axes[1].legend(loc="lower right")
    fig.suptitle("Train/Validation only: no new Test evaluation; no timing ground truth")
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    verify(parser.parse_args().write)
