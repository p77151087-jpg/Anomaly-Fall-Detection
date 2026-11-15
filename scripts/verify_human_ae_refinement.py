"""Read-only model verification and Train/Validation diagnostic figure export."""

import csv
import subprocess
import sys
import xml.etree.ElementTree as ET

import matplotlib
import numpy as np
from human_ae_refinement import BASELINE, ROOT, path, predictions, save, verify

from fall_detect.evaluation.metrics import sequence_metrics
from fall_detect.io_utils import ContractError, read_json, sha256
from fall_detect.training.experiment import Scorer, normal_matrix, verify_artifacts
from fall_detect.training.scaler import FeatureScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render_figure(diagnosis, output):
    figure, axes = plt.subplots(3, 2, figsize=(12, 8), constrained_layout=True)
    figure.suptitle(
        "UR Validation: pose loss and low anomaly scores in three missed clips", fontsize=15
    )
    for i, sid in enumerate(("fall-19", "fall-18", "fall-30")):
        row = next(r for r in diagnosis["rows"] if r["sequence_id"] == sid)
        frames, windows = row["pose_timeline"], row["score_timeline"]
        x = [r["timestamp_ms"] / 1000 for r in frames]
        joints = [max((p["valid_measured_joints"] for p in r["poses"]), default=0) for r in frames]
        axes[i, 0].plot(x, joints, color="#15806a", label="Confident measured joints")
        absent = [r["timestamp_ms"] / 1000 for r in frames if not r["poses"]]
        axes[i, 0].scatter(
            absent,
            np.zeros(len(absent)),
            color="#667085",
            marker="x",
            s=18,
            label="No pose observation",
        )
        axes[i, 0].axhline(9, ls="--", color="#d17a16", label="Per-frame minimum: 9")
        axes[i, 0].set(ylim=(-1, 14), title=f"{sid}: current pose quality", ylabel="Joint count")
        times = [w["available_at_ms"] / 1000 for w in windows]
        for key, color, label in [
            ("raw_score", "#90a4c8", "Raw AE error"),
            ("event_score", "#264d8c", "K3 event score"),
        ]:
            scores = [w[key] if w[key] is not None else np.nan for w in windows]
            axes[i, 1].plot(times, scores, color=color, label=label)
        axes[i, 1].axhline(
            diagnosis["threshold"], ls="--", color="#bf3a3a", label="Frozen threshold"
        )
        axes[i, 1].set(
            title=f"{sid}: valid scores only; blank = no score",
            ylabel="Error (not probability)",
            ylim=(0, 0.9),
        )
        for ax in axes[i]:
            ax.set(xlim=(0, max(x)), xlabel="Source time (s)")
            ax.grid(alpha=0.15)
            if i == 0:
                ax.legend(fontsize=7, loc="lower left")
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main():
    if path("verification").exists():
        raise ContractError("Preserve completed verification")
    locked = verify()
    report = read_json(path("validation"))
    diagnosis = read_json(path("diagnosis"))
    baseline = Scorer(ROOT / "runs" / BASELINE)
    train, hashes, counts = normal_matrix(baseline.index, "train", return_counts=True)
    scaler = FeatureScaler.fit(train)
    results = []
    for row in report["rows"]:
        run = ROOT / "runs" / row["run"]
        if verify_artifacts(run) != row["hashes"]:
            raise ContractError("Frozen result changed")
        stored_scaler = FeatureScaler.load(run / "scaler.npz")
        np.testing.assert_array_equal(stored_scaler.mean, scaler.mean)
        np.testing.assert_array_equal(stored_scaler.scale, scaler.scale)
        sources = read_json(run / "sources.json")
        if sources["train_feature_hashes"] != hashes:
            raise ContractError("Training source hashes differ from Train ADL")
        actual_rows = predictions(row["run"])
        metrics = sequence_metrics(actual_rows)
        for key, value in metrics.items():
            if row["validation"][key] != value:
                raise ContractError(f"Metric mismatch: {key}")
        with (run / "training-history.csv").open(encoding="utf-8-sig", newline="") as handle:
            history = list(csv.DictReader(handle))
        if len(history) != row["status"]["epochs"]:
            raise ContractError("History/status epoch mismatch")
        results.append(
            {
                "run": row["run"],
                "epochs": len(history),
                "best_epoch": min(history, key=lambda r: float(r["val_normal_loss"]))["epoch"],
                "validation_recomputed": True,
                "train_only_scaler_exact": True,
                "missed": [
                    r["sequence_id"]
                    for r in actual_rows
                    if r["label"] == 1 and r["prediction"] != 1
                ],
                "validation": metrics,
            }
        )
    # These completed-experiment guards must reject before writing or Test scoring.
    guards = []
    for stage, expected in [
        ("lock", "Experiment already locked"),
        ("training", "Completed experiment"),
        ("test", "Validation targets/timing guard not reached"),
    ]:
        if stage == "test" and report["target_reached"]:
            continue
        process = subprocess.run(
            [sys.executable, str(ROOT / "scripts/human_ae_refinement.py"), stage],
            capture_output=True,
            text=True,
            errors="replace",
            cwd=ROOT,
            check=False,
        )
        if process.returncode == 0 or expected not in process.stderr:
            raise ContractError(f"Guard failed: {stage}")
        guards.append({"stage": stage, "rejected": True, "reason": expected})
    if not report["target_reached"] and (path("before-test").exists() or path("test").exists()):
        raise ContractError("Unexpected Test evidence before gate")
    test_file = ROOT / "docs/validation/pytest-human-ae-refinement.xml"
    suites = ET.parse(test_file).getroot().findall("testsuite")
    test_counts = {
        k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")
    }
    if test_counts["failures"] or test_counts["errors"]:
        raise ContractError("Risk tests did not pass")

    output = ROOT / "outputs/human-optimization-diagnosis/missed-cases.png"
    render_figure(diagnosis, output)
    verify()
    save(
        "verification",
        {
            "source_kind": "real_end_to_end",
            "rows": results,
            "synthetic_risk_tests": {
                **test_counts,
                "xml": str(test_file.relative_to(ROOT)),
                "sha256": sha256(test_file),
            },
            "new_training_epochs": sum(r["epochs"] for r in results if r["run"] != BASELINE),
            "train_adl_sequences": len(counts),
            "train_features": len(train),
            "preserved_old_runs": len(locked["preserved_runs"]),
            "all_hashes_unchanged": True,
            "guards": guards,
            "new_test_scored": path("test").exists(),
            "selected": report["selected"],
            "recognition_improved": report["selected"] != BASELINE,
            "figure": str(output.relative_to(ROOT)),
            "figure_sha256": sha256(output),
            "visual_review": "pending",
            "formal_notebook_or_film_updated": False,
        },
    )
    print(
        {
            "epochs": sum(r["epochs"] for r in results if r["run"] != BASELINE),
            "risk_tests": test_counts,
            "preserved_runs": len(locked["preserved_runs"]),
            "selected": report["selected"],
        }
    )


if __name__ == "__main__":
    main()
