"""Read-only rechecks and an immutable report for the completed Human experiments."""

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from dataclasses import asdict

import numpy as np
from human_displacement import (
    BASELINE,
    INDEX,
    ROOT,
    lock,
    path,
    predictions,
    test,
    training,
    verify,
)

from fall_detect.config import project_path
from fall_detect.evaluation.metrics import sequence_metrics
from fall_detect.io_utils import ContractError, read_json, sha256, write_json
from fall_detect.training.experiment import cache_for, normal_matrix, verify_artifacts
from fall_detect.training.scaler import FeatureScaler


def check(condition, message):
    if not condition:
        raise ContractError(message)


def main(write=False):
    verify()
    before = read_json(path("before"))
    report = read_json(path("validation"))
    context = read_json(ROOT / "docs/validation/human-context-validation.json")
    rows = [*context["rows"], report["rows"][1]]
    verified = []
    for row in rows:
        run = ROOT / "runs" / row["run"]
        check(verify_artifacts(run) == row["hashes"], "Frozen run changed")
        metrics = sequence_metrics(predictions(row["run"]))
        metrics["balanced_sequence_accuracy"] = (
            metrics["TP"] / (metrics["TP"] + metrics["FN"])
            + metrics["TN"] / (metrics["TN"] + metrics["FP"])
        ) / 2
        check(metrics == row["validation"], "Recomputed Validation differs")
        index, sources = read_json(run / "dataset-index.json"), read_json(run / "sources.json")
        features, hashes, counts = normal_matrix(index, "train", return_counts=True)
        expected, saved = FeatureScaler.fit(features), FeatureScaler.load(run / "scaler.npz")
        check(np.array_equal(expected.mean, saved.mean), "Train scaler mean differs")
        check(np.array_equal(expected.scale, saved.scale), "Train scaler scale differs")
        check(hashes == sources["train_feature_hashes"], "Train feature sources differ")
        check(
            counts == sources["training_sampling"]["train_normal_feature_counts"], "Counts differ"
        )
        train_ids = {
            s["sequence_id"]
            for s in index["split_manifest"]["sequences"]
            if s["split"] == "train" and s["label"] == 0
        }
        val_ids = {
            s["sequence_id"]
            for s in index["split_manifest"]["sequences"]
            if s["split"] == "validation"
        }
        check(set(hashes) == train_ids, "Scaler contains non-Train ADL sources")
        threshold = read_json(run / "threshold.json")
        check(threshold["split"] == "validation", "Threshold split differs")
        check(set(threshold["feature_hashes"]) == val_ids, "Threshold sources differ")
        with (run / "training-history.csv").open(encoding="utf-8-sig", newline="") as stream:
            history = list(csv.DictReader(stream))
        check(len(history) == row["status"]["epochs"], "Epoch count differs")
        best = min(history, key=lambda r: float(r["val_normal_loss"]))
        check(
            float(best["val_normal_loss"]) == row["status"]["best_val_normal_loss"], "Loss differs"
        )
        verified.append(
            {
                "run": row["run"],
                "epochs": len(history),
                "best_epoch": int(best["epoch"]),
                "best_val_normal_loss": float(best["val_normal_loss"]),
                "threshold": threshold["threshold"],
                "train_adl_sequences": len(counts),
                "train_windows": len(features),
                "scaler_exactly_train_only": True,
                "validation": metrics,
                "frozen_artifacts_verified": True,
            }
        )
    old, new = (
        read_json(ROOT / INDEX),
        read_json(ROOT / "runs/ur-human-displacement-001/dataset-index.json"),
    )
    pose_checks = 0
    schedule_checks = 0
    for source in new["split_manifest"]["sequences"]:
        sid = source["sequence_id"]
        check(
            sha256(project_path(old["sequences"][sid]["pose_path"]))
            == sha256(project_path(new["sequences"][sid]["pose_path"])),
            f"Observed pose changed: {sid}",
        )
        pose_checks += 1
        # Test is hash-preservation only: do not inspect its scores or validity for selection.
        if source["split"] == "test":
            continue

        def signature(cache):
            return [
                {k: v for k, v in asdict(r).items() if k != "recipe_hash"} for r in cache.records
            ]

        check(
            signature(cache_for(old, sid)) == signature(cache_for(new, sid)),
            f"Schedule changed: {sid}",
        )
        schedule_checks += 1
    guards = {}
    for name, action, message in [
        ("lock", lock, "Experiment already locked"),
        ("training", training, "Completed experiment; preserve results"),
        ("test", test, "Validation targets/timing guard not reached; no Test scoring"),
    ]:
        try:
            action()
        except ContractError as exc:
            check(str(exc) == message, f"Unexpected guard rejection: {exc}")
            guards[name] = str(exc)
        else:
            raise ContractError(f"Completed experiment unexpectedly allowed {name}")
    check(
        not path("test").exists() and not path("before-test").exists(), "Unexpected Test evidence"
    )
    integrations = []
    for filename in [
        "human-displacement-integration-ur-sample-human-s1-001-fall-29.json",
        "human-displacement-integration-ur-human-displacement-001.json",
    ]:
        evidence = read_json(ROOT / "docs/validation" / filename)
        check(
            verify_artifacts(ROOT / "runs" / evidence["run"]) == evidence["hashes"],
            "Demo model changed",
        )
        for case in evidence["cases"]:
            check(sha256(case["preview"]) == case["preview_sha256"], "Preview changed")
            check(
                case["max_score_difference"] <= 1e-6
                and case["events_match"]
                and case["states_match_where_visible"],
                "Demo parity failed",
            )
        integrations.append(evidence)
    suite = (
        ET.parse(ROOT / "docs/validation/pytest-human-displacement-full.xml")
        .getroot()
        .find("testsuite")
    )
    check(
        suite is not None and int(suite.get("errors")) == 0 and int(suite.get("failures")) == 0,
        "Full tests failed",
    )
    cases = [c for r in integrations for c in r["cases"]]
    verify()
    result = {
        "source_kind": "real_end_to_end",
        "rows": verified,
        "new_epochs_this_turn": sum(r["epochs"] for r in verified if r["run"] != BASELINE),
        "selected": report["selected"],
        "recognition_improved": False,
        "target_reached": report["target_reached"],
        "new_test_scored": False,
        "historical_test_seen": True,
        "repeated_validation_not_independent": True,
        "old_runs_preserved": len(before["old_runs"]),
        "old_cache_files_preserved": len(before["cache_files"]),
        "observed_pose_hashes_unchanged": pose_checks,
        "train_validation_window_records_identical": schedule_checks,
        "guards": guards,
        "full_tests_passed": int(suite.get("tests")) - int(suite.get("skipped")),
        "full_tests_skipped": int(suite.get("skipped")),
        "integration_frames": sum(c["frames"] for c in cases),
        "integration_compared_windows": sum(c["compared_windows"] for c in cases),
        "max_score_difference": max(c["max_score_difference"] for c in cases),
        "integration_cases": integrations,
        "visual_review": {
            "method": "Contact sheets and targeted event-frame extracts inspected; H264 fully decoded by integration verifier.",
            "adl-04": "Warmup UNKNOWN, followed by NORMAL; no visible false alert.",
            "fall-30": "New window event at 1800ms is masked by low-confidence current pose; zero visible alert rows. No practical alert improvement demonstrated.",
            "fall-09": "Window events at 4833ms and 5100ms are masked by current pose quality. First visible alert remains 5900ms; 8 visible rows.",
            "legacy-fall-29": "Original model preserved, CSV/events byte-identical; only 2 visible alert rows.",
        },
        "timing_scope_correction": "Six coarse Validation cases, not official temporal ground truth. Historical helper's scope string says four; six actual rows were evaluated. Guard rejected for Accuracy/F1 regression, despite coarse hits 2 to 3 and zero early events.",
        "format_check": "Ruff lint passes; formatter suggests only wrapping one expression in locked windows.py. Kept exact trained source bytes; 76 other checked files were formatted. This report script checked separately.",
        "evidence_tiers": {
            "synthetic": "Translation invariance proof and unit tests only, no recognition claim.",
            "real_feature_smoke": "10-case genuine feature/preserved-pose pilot, no Accuracy claim.",
            "real_end_to_end": "Fixed Train/Validation training, metrics and fresh raw-image demos.",
        },
        "verification_script_sha256": sha256(ROOT / "scripts/verify_human_optimization.py"),
    }
    if write:
        check(not path("verification").exists(), "Preserve completed verification")
        write_json(path("verification"), result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "selected",
                    "new_epochs_this_turn",
                    "old_runs_preserved",
                    "old_cache_files_preserved",
                    "integration_frames",
                    "integration_compared_windows",
                    "max_score_difference",
                    "new_test_scored",
                ]
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write", action="store_true", help="Write once; default performs read-only rechecks"
    )
    main(parser.parse_args().write)
