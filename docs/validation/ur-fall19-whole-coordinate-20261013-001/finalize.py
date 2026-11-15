"""Create or read-only verify this stage's final artifact manifest."""

import json
from datetime import datetime, timezone
from pathlib import Path

from fall_detect.config import ROOT
from fall_detect.io_utils import sha256

ID = "ur-fall19-whole-coordinate-20261013-001"
QA = ROOT / "docs/validation" / ID
TARGET = QA / "final-qa.json"

if TARGET.exists():
    result = json.loads(TARGET.read_text(encoding="utf-8"))
    for name, digest in result["artifact_sha256"].items():
        assert sha256(ROOT / name) == digest, name
    print(json.dumps({"pass": True, "artifacts": len(result["artifact_sha256"])}))
else:
    files = set()
    for folder in (QA, ROOT / "outputs" / ID, ROOT / "data/cache" / ID):
        files.update(p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    lock = json.loads((QA / "lock.json").read_text(encoding="utf-8"))
    files.update(ROOT / p for p in lock["files"])
    files.update(ROOT / p for p in (
        "AGENTS.md", "Fall_Detection_Codex_Implementation_Spec.md",
        "docs/current-progress.md", "docs/acceptance-results.md",
        "docs/ur-fall19-whole-coordinate-20261013.md", "scripts/audit_whole_pose_outputs.py",
    ))
    result = {
        "id": ID, "finalized_at": datetime.now(timezone.utc).isoformat(),
        "human_stage": "completed_self_reported", "adopted": False,
        "fall19": "still_false_negative", "test": "not_loaded_not_evaluated",
        "artifact_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(files)},
    }
    with TARGET.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    for name, digest in result["artifact_sha256"].items():
        assert sha256(ROOT / name) == digest, name
    print(json.dumps({"pass": True, "artifacts": len(result["artifact_sha256"])}))
