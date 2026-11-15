"""Preserve pre-comparison assets and read the supplied course attachments."""

import json
import shutil
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import psutil

from fall_detect.io_utils import ContractError, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
TAG = "ur-normal-methods-20261012-001"
EVIDENCE = ROOT / "docs/validation" / TAG
MUTABLE = [
    "AGENTS.md",
    "Fall_Detection_Codex_Implementation_Spec.md",
    "docs/course-alignment-plan.md",
    "docs/current-progress.md",
    "docs/acceptance-results.md",
]


def main():
    if EVIDENCE.exists():
        raise ContractError("Inventory exists; preserve previous snapshot")
    active = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = process.info["cmdline"] or []
            if "python" in (process.info["name"] or "").lower() and any(
                word in " ".join(cmd) for word in [" train ", "training", "normal_methods.py"]
            ):
                active.append(process.info)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    if active:
        raise ContractError(f"Concurrent research process: {active}")
    EVIDENCE.mkdir(parents=True)
    for name in MUTABLE:
        dest = EVIDENCE / "prior-contract" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    assets = {}
    start = time.monotonic()
    for scope in [
        "runs",
        "configs",
        "data/manifests",
        "data/cache",
        "data/raw",
        "data/retired",
        "weights",
        "outputs",
    ]:
        for path in sorted((ROOT / scope).rglob("*")):
            if path.is_file():
                assets[path.relative_to(ROOT).as_posix()] = sha256(path)
                if len(assets) % 3000 == 0:
                    print(
                        f"Preserved {len(assets)} files in {time.monotonic() - start:.1f}s",
                        flush=True,
                    )
    statuses = {}
    for path in (ROOT / "runs").glob("*/status.json"):
        statuses[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
    write_json(
        EVIDENCE / "before.json",
        {
            "assets": assets,
            "mutable_contracts": {name: sha256(ROOT / name) for name in MUTABLE},
            "active_training_processes": active,
            "run_statuses": statuses,
            "test_bytes_hash_only": True,
            "test_features_decoded": 0,
        },
    )
    folder = ROOT / "outputs/course-requirements-review-20261012"
    with zipfile.ZipFile(folder / "teacher-admin-full.pptx") as archive:
        names = sorted(
            [
                n
                for n in archive.namelist()
                if n.startswith("ppt/slides/slide") and n.endswith(".xml")
            ],
            key=lambda n: int(n.rsplit("slide", 1)[1].split(".")[0]),
        )
        slides = [
            " | ".join(
                t.text or ""
                for t in ET.fromstring(archive.read(n)).findall(
                    ".//{http://schemas.openxmlformats.org/drawingml/2006/main}t"
                )
            )
            for n in names
        ]
    # Read-only pure-Python document dependency; project execution remains .venv.
    sys.path.append(
        str(
            Path.home()
            / ".cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages"
        )
    )
    from pypdf import PdfReader

    pages = [page.extract_text() for page in PdfReader(folder / "sample-report.pdf").pages]
    requested = [
        "AGENTS.md",
        "Fall_Detection_Codex_Implementation_Spec.md",
        "docs/course-alignment-plan.md",
        "docs/current-progress.md",
        "docs/acceptance-results.md",
        "docs/ur-only-shift-20261012.md",
        "docs/ur-isotropic-wide-test-20261012.md",
        "docs/ur-short20-isotropic-20261012.md",
        "docs/ur-short20-isotropic-20261012-plan.md",
    ]
    write_json(
        EVIDENCE / "reading.json",
        {
            "documents": {
                n: {
                    "sha256": sha256(ROOT / n),
                    "lines": len((ROOT / n).read_text(encoding="utf-8").splitlines()),
                }
                for n in requested
            },
            "teacher_slides": slides,
            "sample_pages": pages,
            "teacher_sha256": sha256(folder / "teacher-admin-full.pptx"),
            "sample_sha256": sha256(folder / "sample-report.pdf"),
            "interpretation": "Slides 37-40: normal-data anomaly detection, one-class and AE, ST-GCN example. Sample p3-4 claims AE pipeline, p6 reports 70-video metrics; attachment lacks complete AE training/evaluation evidence. Do not attribute all results to the action classifier.",
        },
    )
    print(f"Snapshot complete: {len(assets)} assets, 54 slides, {len(pages)} PDF pages", flush=True)


if __name__ == "__main__":
    main()
