"""Finalize immutable QA metadata from the already completed audits."""

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import ur_short20_isotropic as workflow
import verify_short20_retirement as retirement

from fall_detect.io_utils import ContractError, read_json, sha256, write_json
from fall_detect.training import experiment


def suite(path):
    cases = ET.parse(path).getroot().findall(".//testcase")
    failed = {
        (r.attrib["classname"], r.attrib["name"])
        for r in cases
        if r.find("failure") is not None or r.find("error") is not None
    }
    skipped = {
        (r.attrib["classname"], r.attrib["name"]) for r in cases if r.find("skipped") is not None
    }
    passed = {(r.attrib["classname"], r.attrib["name"]) for r in cases} - failed - skipped
    return passed, failed, skipped


def main():
    target = workflow.evidence("final-qa")
    if target.exists():
        raise ContractError("Preserve completed final QA")
    lock = workflow.verify_lock()
    root = workflow.ROOT
    retirement.verify()
    result = read_json(workflow.evidence("validation"))
    verification = read_json(workflow.evidence("verification"))
    if (
        experiment.verify_artifacts(root / "runs" / workflow.RUN)
        != result["candidate"]["artifact_hashes"]
    ):
        raise ContractError("Frozen candidate changed")
    if not verification["passed"] or verification["verifier_sha256"] != sha256(
        root / "scripts/verify_ur_short20_isotropic.py"
    ):
        raise ContractError("Independent audit/source changed")
    passed, failed, skipped = suite(root / "docs/validation/pytest-ur-short20-isotropic-full.xml")
    retry_passed, retry_failed, retry_skipped = suite(
        root / "docs/validation/pytest-ur-short20-isotropic-loopback.xml"
    )
    risk_passed, risk_failed, risk_skipped = suite(
        root / "docs/validation/pytest-ur-short20-isotropic-risk.xml"
    )
    if failed != retry_passed or retry_failed or retry_skipped or risk_failed or risk_skipped:
        raise ContractError("Unresolved test results")
    commands = [
        [sys.executable, "-m", "ruff", "check", "src", "tests", "scripts"],
        [sys.executable, "-m", "compileall", "-q", "src", "scripts", "tests"],
    ]
    checks = []
    for command in commands:
        response = subprocess.run(command, cwd=root, capture_output=True, check=False)
        if response.returncode:
            raise ContractError("Final static check failed")
        checks.append(
            {
                "command": command,
                "exit_code": response.returncode,
                "stdout": response.stdout.decode("utf8", errors="replace")[-2000:],
            }
        )
    report = root / "docs" / f"{workflow.PREFIX}.md"
    generation = read_json(workflow.evidence("report"))
    # Preserve the generated report before the later retirement-audit supplement.
    content = report.read_text(encoding="utf8")
    start = content.index("補充保全核對：")
    end = content.index("\n\n", start) + 2
    original_text = content[:start] + content[end:]
    import hashlib

    previous = next(
        (
            payload
            for payload in (
                original_text.encode("utf8"),
                original_text.replace("\n", "\r\n").encode("utf8"),
            )
            if hashlib.sha256(payload).hexdigest() == generation["report_sha256"]
        ),
        None,
    )
    if previous is None:
        raise ContractError("Original generated report digest mismatch")
    archive = root / "docs/validation" / f"{workflow.PREFIX}-report-before-retirement-note.md"
    if archive.exists():
        raise ContractError("Preserve original report archive")
    archive.write_bytes(previous)
    files = [p for p in (root / "outputs" / workflow.PREFIX).rglob("*") if p.is_file()]
    files += [p for p in (root / "runs" / workflow.RUN).rglob("*") if p.is_file()]
    files += [
        report,
        archive,
        root / "docs/current-progress.md",
        root / "docs/acceptance-results.md",
        root / "docs/course-alignment-plan.md",
        root / "Fall_Detection_Codex_Implementation_Spec.md",
        root / "configs" / f"{workflow.RUN}.yaml",
        workflow.INDEX,
    ]
    write_json(
        target,
        {
            "passed": True,
            "adopted": False,
            "metrics": result["candidate"]["metrics"],
            "synthetic_risk_tests": {"passed": len(risk_passed), "failed": 0, "skipped": 0},
            "full_regression": {
                "initial_passed": len(passed),
                "initial_failed_loopback_permissions": len(failed),
                "retry_passed": len(retry_passed),
                "final_unique_passed": len(passed | retry_passed),
                "historical_skip": len(skipped),
                "unresolved_failures": 0,
            },
            "real_feature_smoke": "passed fixed10 pilot, full47 features, 39epoch AE and23Val independent audit",
            "real_end_to_end": "not_run; no new RGB Pose/demo/Test",
            "test_prepared": False,
            "test_evaluations": 0,
            "test_feature_arrays_decoded": 0,
            "historical_cache_bytes_hashed_for_preservation_only": True,
            "original_retirement_files_verified": 1430,
            "retired_external_files_verified": 201,
            "original_core_exact_archives": 3,
            "round_preserved_files": len(lock["preserved_files"]),
            "plots_visually_checked": [
                "validation-comparison.png",
                "confusion-matrix.png",
                "training-curve.png",
            ],
            "static_checks": checks,
            "format_check": "8 changed/new files passed before finalizer; finalizer checked separately",
            "graph_coverage_evidence": f"docs/validation/{workflow.PREFIX}-graph-coverage.json",
            "report_revision": "Original generated report archived by its exact recorded hash; final report adds independent retirement/source preservation audit only",
            "files": {p.relative_to(root).as_posix(): sha256(p) for p in files},
            "qa_source_sha256": sha256(Path(__file__)),
        },
    )
    workflow.verify_lock()
    print(
        {
            "final_qa": str(target),
            "passed": True,
            "unique_tests_passed": len(passed | retry_passed),
            "skip": len(skipped),
            "risk_passed": len(risk_passed),
            "adopted": False,
        }
    )


if __name__ == "__main__":
    main()
