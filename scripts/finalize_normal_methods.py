"""Final QA: unique test cases, immutable stages, reported counts and artifacts."""

import sys
import xml.etree.ElementTree as ET

from ur_normal_methods import ROOT, C, E, O, ae, ae_lock, freeze, save, svm, verify_lock

from fall_detect.io_utils import ContractError, read_json, sha256


def cases(name):
    result = {}
    for case in ET.parse(E / name).getroot().iter("testcase"):
        identity = (case.attrib["classname"], case.attrib["name"])
        assert identity not in result
        result[identity] = (
            "failed"
            if case.find("failure") is not None or case.find("error") is not None
            else "skipped"
            if case.find("skipped") is not None
            else "passed"
        )
    return result


def main():
    if (E / "final-qa.json").exists():
        raise ContractError("Final QA exists; preserve completed evidence")
    lock = verify_lock()
    verification = read_json(E / "verification.json")
    assert verification["passed"] and verification["preserved_old_assets"] == 23925
    full = cases("full.xml")
    retry = cases("loopback.xml")
    risk = cases("risk-v2.xml")
    failures = {k for k, v in full.items() if v == "failed"}
    assert failures == set(retry) and all(v == "passed" for v in retry.values())
    assert len(full) == 297 and len(risk) == 47 and all(v == "passed" for v in risk.values())
    unique = {**full, **retry}
    assert (
        sum(v == "passed" for v in unique.values()) == 296
        and sum(v == "skipped" for v in unique.values()) == 1
    )
    assert len(unique) == 297 and not any(v == "failed" for v in unique.values())
    stages = (
        read_json(E / "svm-results.json")["candidates"]
        + read_json(E / "ae-results.json")["candidates"]
    )
    recipes = []
    for r in stages:
        run = ROOT / "runs" / r["spec"]["id"]
        assert all(
            sha256(run / n) == d for n, d in read_json(run / "frozen-artifacts.json").items()
        )
        recipes.append(read_json(run / "method-config.json")["method_recipe_hash"])
    assert len(set(recipes)) == 8 and lock["feature_recipe_hash"] not in recipes
    selection = read_json(E / "selection.json")
    assert (
        not selection["recognition_adopted"]
        and selection["selected_new_candidate"] == "ur-ocsvm-06-20261012-001"
    )
    before = {
        str(ROOT / "runs" / r["spec"]["id"] / n): d
        for r in stages
        for n, d in read_json(ROOT / "runs" / r["spec"]["id"] / "frozen-artifacts.json").items()
    }
    denied = []
    for name, operation in [("freeze", freeze), ("svm", svm), ("ae-lock", ae_lock), ("ae", ae)]:
        try:
            operation()
        except ContractError as exc:
            denied.append({"stage": name, "reason": str(exc)})
        else:
            raise AssertionError(f"Completed stage mutation allowed: {name}")
    # The two report/inventory entry points also fail before writing.
    sys.path.insert(0, str(ROOT / "scripts"))
    from normal_methods_inventory import main as inventory
    from report_normal_methods import report

    for name, operation in [("report", report), ("inventory", inventory)]:
        try:
            operation()
        except ContractError as exc:
            denied.append({"stage": name, "reason": str(exc)})
        else:
            raise AssertionError(f"Completed evidence mutation allowed: {name}")
    assert all(sha256(name) == digest for name, digest in before.items())
    training = read_json(ROOT / "runs/ur-temporal-gru-methods-20261012-001/training.json")
    assert sum(training["train_input_counts"].values()) == 3449
    assert sum(training["val_input_counts"].values()) == 1364
    temporal = next(r for r in verification["candidates"] if "temporal" in r["run"])
    assert temporal["score_rows_recomputed"] - 3449 == 2174
    assert lock["val_normal_shape"] == [1427, 512]
    document = (ROOT / "docs/ur-normal-methods-20261012.md").read_text(encoding="utf-8")
    assert "正常Val1427" in document and "Train3449" in document and "全Val2174" in document
    assert "Train3418" not in document and "正常Val1527" not in document
    assert sum(r["events"] for r in verification["candidates"]) == 175
    assert sum(r["score_rows_recomputed"] for r in verification["candidates"]) == 47630
    for i, (name, digest) in enumerate(read_json(E / "before.json")["assets"].items(), 1):
        assert sha256(ROOT / name) == digest, name
        if i % 5000 == 0:
            print(f"Final preservation check: {i}/23925", flush=True)
    hashes = {}
    for folder in [O, E, C]:
        for path in folder.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                hashes[path.relative_to(ROOT).as_posix()] = sha256(path)
    for name in [
        "AGENTS.md",
        "Fall_Detection_Codex_Implementation_Spec.md",
        "docs/course-alignment-plan.md",
        "docs/current-progress.md",
        "docs/acceptance-results.md",
        "docs/ur-normal-methods-20261012.md",
        "docs/ur-normal-methods-20261012-plan.md",
        "scripts/ur_normal_methods.py",
        "scripts/normal_methods_inventory.py",
        "scripts/diagnose_normal_methods.py",
        "scripts/normal_methods_geometry.py",
        "scripts/verify_normal_methods.py",
        "scripts/report_normal_methods.py",
        "scripts/finalize_normal_methods.py",
        "src/fall_detect/models/normal_alternatives.py",
        "tests/test_normal_alternatives.py",
        "tests/test_normal_methods_scope.py",
        "src/fall_detect/training/experiment.py",
        "src/fall_detect/detection/events.py",
        "src/fall_detect/training/scaler.py",
        "src/fall_detect/evaluation/metrics.py",
        "src/fall_detect/visualization/reports.py",
        "src/fall_detect/data/cache.py",
        "src/fall_detect/models/stgcn.py",
        "src/fall_detect/models/weights.py",
        "src/fall_detect/models/autoencoder.py",
        "src/fall_detect/skeleton/preprocessing.py",
        "src/fall_detect/skeleton/windows.py",
        "src/fall_detect/config.py",
        "src/fall_detect/io_utils.py",
    ]:
        hashes[name] = sha256(ROOT / name)
    for r in stages:
        path = ROOT / "configs" / f"{r['spec']['id']}.json"
        hashes[path.relative_to(ROOT).as_posix()] = sha256(path)
    save(
        E / "final-qa.json",
        {
            "passed": True,
            "synthetic": {
                "related_passed": 47,
                "full_initial_passed": 294,
                "full_initial_failed_loopback": 2,
                "full_initial_skipped": 1,
                "loopback_retry_passed": 2,
                "unique_passed": 296,
                "unique_skipped": 1,
                "unique_total": 297,
                "unresolved_failures": 0,
            },
            "real_feature_smoke": {
                "strict_stgcn_windows": 3,
                "feature_parity_max_abs": 0,
                "svm_models_fitted": 6,
                "ae_models_trained": 2,
                "ae_epochs": 62,
                "validation_sequence_results": 184,
                "model_scores_recomputed": 47630,
                "complete_events_verified": 175,
            },
            "real_end_to_end": "not_run",
            "test": "unknown; no new Test preparation or scoring",
            "preserved_old_assets": 23925,
            "self_supervised": "not_run: prior geometry-support rule not met",
            "no_overwrite_probes": denied,
            "static_checks": "ruff check src tests scripts and compileall passed; new-code format checks passed",
            "visual_review": "teacher example and benchmark, sample AE page, comparison and fall19 plots inspected; matrices independently recomputed",
            "final_artifact_hashes": hashes,
            "recognition_adopted": False,
            "demo_default": "ur-sample-human-s1-001",
        },
    )
    print(
        {
            "passed": True,
            "unique_tests_passed": 296,
            "skipped": 1,
            "protected_stage_probes": len(denied),
            "final_artifacts": len(hashes),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
