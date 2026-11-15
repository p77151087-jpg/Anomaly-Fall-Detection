"""Close and freeze the completed single-candidate experiment without additional fitting."""

import subprocess
import sys
import xml.etree.ElementTree as ET

from ur_motion_novelty import (
    IMPLEMENTATIONS,
    ROOT,
    RUN,
    TAG,
    C,
    E,
    O,
    save,
    validated_index,
    verify_lock,
)

from fall_detect.io_utils import ContractError, read_json, sha256


def main():
    if (E / "final-qa.json").exists():
        raise ContractError("Preserve completed final QA")
    lock = verify_lock()
    index = validated_index()
    audit = read_json(E / "verification.json")
    report = read_json(E / "report.json")
    result = read_json(RUN / "result.json")
    assert audit["passed"] and audit["old_assets_unchanged"] == 24620
    assert audit["motion_rows"] == 6001 and audit["three_branch_scores_verified"] == 18003
    assert audit["test_evaluations"] == audit["test_feature_reads"] == 0
    assert result["recognition_adoptable"] is False and report["adopted"] is False
    assert result["metrics"] == lock["reference_metrics"]
    assert (
        report["timing"]["event_timing_and_tracks_match"]
        and report["timing"]["window_state_changes"] == report["timing"]["first_alarm_changes"] == 0
    )
    assert len([s for s in index["split_manifest"]["sequences"] if s["split"] == "test"]) == 23
    for name, digest in read_json(RUN / "frozen-artifacts.json").items():
        assert sha256(RUN / name) == digest
    assert sha256(ROOT / "docs/ur-motion-novelty-20261012.md") == report["report_sha256"]
    for field, path in [
        ("verification_sha256", E / "verification.json"),
        ("candidate_comparison_sha256", O / "candidate-comparison.csv"),
    ]:
        assert report[field] == sha256(path)
    retry = ET.parse(E / "risk-retry.xml").getroot().findall(".//testcase")
    initial = ET.parse(E / "risk.xml").getroot().findall(".//testcase")
    assert len(retry) == len(initial) == 60
    assert all(c.find("failure") is None and c.find("error") is None for c in retry)
    assert sum(c.find("error") is not None for c in initial) == 11
    assert report["phase_max"]["fused_score"] < result["threshold"]
    assert audit["min_normal_fp_to_recover_fall19_with_same_score"] == 8
    protected = {}
    for folder in [RUN, C, O]:
        for path in folder.rglob("*"):
            if path.is_file():
                protected[path.relative_to(ROOT).as_posix()] = sha256(path)
    checks = []
    for script, stage in [
        ("ur_motion_novelty.py", "inventory"),
        ("ur_motion_novelty.py", "prepare"),
        ("ur_motion_novelty.py", "fit"),
        ("report_motion_novelty.py", None),
    ]:
        command = [sys.executable, "-X", "utf8", str(ROOT / "scripts" / script)]
        if stage:
            command.append(stage)
        completed = subprocess.run(
            command, cwd=ROOT, text=True, encoding="utf-8", capture_output=True, check=False
        )
        assert completed.returncode != 0 and "ContractError" in completed.stderr
        checks.append(
            {
                "script": script,
                "stage": stage,
                "rejected": True,
                "reason": completed.stderr.strip().splitlines()[-1],
            }
        )
    for name, digest in protected.items():
        assert sha256(ROOT / name) == digest
    sources = IMPLEMENTATIONS + [
        "scripts/verify_motion_novelty.py",
        "scripts/report_motion_novelty.py",
        "scripts/finalize_motion_novelty.py",
        "tests/test_motion_novelty.py",
    ]
    artifacts = {**protected}
    for name in sources + [
        "AGENTS.md",
        "Fall_Detection_Codex_Implementation_Spec.md",
        "docs/current-progress.md",
        "docs/acceptance-results.md",
        "docs/course-alignment-plan.md",
        "docs/ur-motion-novelty-20261012-plan.md",
        "docs/ur-motion-novelty-20261012.md",
        f"configs/{TAG}.json",
    ]:
        artifacts[name] = sha256(ROOT / name)
    for path in E.rglob("*"):
        if path.is_file() and "tmp-risk-retry" not in path.parts:
            artifacts[path.relative_to(ROOT).as_posix()] = sha256(path)
    save(
        E / "final-qa.json",
        {
            "passed": True,
            "candidate": TAG,
            "adopted": False,
            "demo_changed": False,
            "test": "unknown_not_evaluated",
            "risk_passed": 60,
            "initial_setup_errors": 11,
            "setup_error_resolution": "unique writable project basetemp, same 60 cases passed",
            "independent_audit_sha256": sha256(E / "verification.json"),
            "old_assets_verified": 24620,
            "score_values_verified": 18003,
            "train_video_exclusion_values_verified": 3638,
            "window_rows": 2917,
            "events": 14,
            "timing_unchanged": True,
            "guard_checks": checks,
            "artifacts": artifacts,
            "scope": "synthetic and real_feature_smoke; no real_end_to_end, new Test, demo, Notebook, video or public upload",
            "graph": "Tier2 best-effort coverage; ignored assets directly read/hash verified",
        },
    )
    print(
        {
            "passed": True,
            "risk_tests": 60,
            "frozen_artifacts": len(artifacts),
            "overwrite_rejections": len(checks),
            "adopted": False,
            "test": "unknown",
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
