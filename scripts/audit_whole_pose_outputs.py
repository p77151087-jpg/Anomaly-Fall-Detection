"""Additional bounded preservation, historical Validation CSV and repeat-run audit."""

import csv
import json
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fall_detect.config import ROOT
from fall_detect.evaluation.whole_pose_coordinates import ID
from fall_detect.io_utils import read_json, sha256


def main():
    out, qa = ROOT / "outputs" / ID, ROOT / "docs/validation" / ID
    destination = qa / "supplemental-qa.json"
    if destination.exists():
        raise RuntimeError("Audit output already exists; never overwrite")
    s = read_json(out / "inference/summary.json")
    history = ROOT / "outputs/isotropic-refinement-20261012/ur-isotropic-wide-20261012-001"
    csv_path = history / "sequences.csv"
    old_inventory = read_json(
        ROOT / "docs/validation/ur-pose-coordinate-diagnostic-20261013-001/old-assets-before.json"
    )
    name = csv_path.relative_to(ROOT).as_posix()
    assert sha256(csv_path) == old_inventory[name]
    with csv_path.open(encoding="utf-8-sig", newline="") as h:
        rows = list(csv.DictReader(h))
    matrix = [[0, 0, 0], [0, 0, 0]]
    for row in rows:
        y = int(row["label"])
        pred = int(row["prediction"]) if row["prediction"] else 2
        matrix[y][pred] += 1
    tn, fp, un = matrix[0]
    fn, tp, uf = matrix[1]
    n, valid = len(rows), len(rows) - un - uf
    baseline = {
        "matrix_true_normal_fall_by_pred_normal_fall_unknown": matrix,
        "all_accuracy": (tp + tn) / n,
        "all_denominator": n,
        "decidable_accuracy": (tp + tn) / valid,
        "decidable_denominator": valid,
        "fall_precision": tp / (tp + fp),
        "fall_recall": tp / (tp + fn),
        "all_fall_recall": tp / (tp + fn + uf),
        "fall_f1": 2 * tp / (2 * tp + fp + fn),
        "unknown": un + uf,
        "false_positives": fp,
        "false_negatives": fn,
        "historical_csv_sha256": sha256(csv_path),
    }
    assert matrix == [[8, 0, 0], [1, 14, 0]] and n == valid == 23
    metrics = read_json(history / "metrics.json")
    for key, expected in (
        ("operational_accuracy", baseline["all_accuracy"]),
        ("sequence_accuracy", baseline["decidable_accuracy"]),
        ("sequence_fall_f1", baseline["fall_f1"]),
    ):
        assert metrics[key] == expected
    # New diagnostic only; do not overwrite or relabel this formal CSV.
    fall = next(row for row in rows if row["sequence_id"] == "fall-19")
    assert (
        int(fall["prediction"])
        == s["results"]["cached"]["prediction"]
        == s["results"]["intervention"]["prediction"]
    )
    threshold = s["threshold"]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    for model, style in (("cached", "--"), ("intervention", "-")):
        with (out / f"inference/{model}-windows.csv").open(encoding="utf-8-sig", newline="") as h:
            ws = list(csv.DictReader(h))
        # NaNs create real gaps rather than imply scores for unknown windows.
        x = [float(w["available_at_ms"]) / 1000 for w in ws]
        y = [float(w["event_score"]) if w["event_score"] else float("nan") for w in ws]
        ax.plot(x, y, style, label="Original K3" if model == "cached" else "Human six-point K3")
    ax.axhline(threshold, color="firebrick", linestyle=":", label="Frozen threshold")
    ax.axvspan(1.067, 2.001, color="orange", alpha=0.10, label="Whole transition posture phase")
    ax.axvspan(2.035, 3.303, color="gray", alpha=0.10, label="Whole ground posture phase")
    ax.set(
        xlabel="Window available at (s)",
        ylabel="Anomaly score (not probability)",
        title="fall-19: visible human XY correction, frozen AE / K3",
    )
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    image = out / "k3-comparison.png"
    if image.exists():
        raise RuntimeError("Plot already exists")
    fig.savefig(image, dpi=160)
    plt.close(fig)
    # Read-only digest before/after actual repeated calls, excluding QA being written.
    paths = [
        p for folder in (out, ROOT / "data/cache" / ID) for p in folder.rglob("*") if p.is_file()
    ]
    before = {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}
    attempts = []
    for args in (
        ("prepare", "--human-review", str(out / "human-review-original.json")),
        ("diagnose",),
    ):
        process = subprocess.run(
            [sys.executable, "-X", "utf8", "scripts/whole_pose_coordinate_diagnostic.py", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
            check=False,
        )
        assert (
            process.returncode != 0
            and "never overwrite" in (process.stdout + process.stderr).lower()
        )
        attempts.append(
            {
                "args": list(args),
                "exit_code": process.returncode,
                "stdout": process.stdout,
                "stderr": process.stderr,
            }
        )
    after = {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}
    assert before == after
    cached, new = s["results"]["cached"], s["results"]["intervention"]
    value = {
        "id": ID,
        "pass": True,
        "formal_wide_validation_independently_recomputed": baseline,
        "formal_csv_unchanged_from_prior_locked_inventory": True,
        "controlled_fall19_prediction_unchanged": True,
        "k3_delta": new["sequence_score"] - cached["sequence_score"],
        "k3_delta_percent": (new["sequence_score"] / cached["sequence_score"] - 1) * 100,
        "raw_max_delta": new["max_raw_score"] - cached["max_raw_score"],
        "remaining_k3_threshold_gap": threshold - new["sequence_score"],
        "repeated_actions_refused": attempts,
        "output_cache_files_unchanged": before,
        "test": "not_loaded_not_evaluated",
        "candidate_added": False,
        "figure_phase_basis": "whole human posture phases; not onset/impact truth",
    }
    with destination.open("x", encoding="utf-8") as h:
        json.dump(value, h, ensure_ascii=False, indent=2, allow_nan=False)
        h.write("\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in value.items()
                if k not in {"repeated_actions_refused", "output_cache_files_unchanged"}
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
