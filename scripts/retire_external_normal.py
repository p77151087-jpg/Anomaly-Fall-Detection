"""Remove acquired external inputs from active paths, preserving bytes and audit history."""

import argparse
from pathlib import Path

from fall_detect.io_utils import ContractError, read_json, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "external-normal-retired-20261012"
LOCK = ROOT / f"docs/validation/{PREFIX}-lock.json"
RESULT = ROOT / f"docs/validation/{PREFIX}.json"
ARCHIVE = "data/retired/external-normal-20261012"
MAPPINGS = {
    "data/raw/external-normal-20261012": f"{ARCHIVE}/raw/external-normal-20261012",
    "data/cache/external-normal-human-isotropic-20261012-001": f"{ARCHIVE}/cache/external-normal-human-isotropic-20261012-001",
}


def bounded_path(root, relative):
    root = Path(root).resolve()
    candidate = Path(root / relative).resolve()
    if candidate == root or not candidate.is_relative_to(root):
        raise ContractError("Archive path must remain strictly inside the workspace")
    return candidate


def mapped_path(relative, mappings):
    for source, destination in mappings.items():
        if relative == source or relative.startswith(source + "/"):
            return destination + relative[len(source) :]
    return relative


def save_new(path, value):
    if path.exists():
        raise ContractError("Preserve completed retirement evidence")
    write_json(path, value)


def archive():
    if RESULT.exists():
        raise ContractError("External inputs already retired; use status")
    if not LOCK.exists():
        proof = ROOT / "docs/validation/external-normal-20261012-final-qa.json"
        preserved = read_json(proof)["files"]
        for relative, digest in preserved.items():
            if sha256(bounded_path(ROOT, relative)) != digest:
                raise ContractError(f"Prior evidence changed: {relative}")
        for source, destination in MAPPINGS.items():
            src, dst = bounded_path(ROOT, source), bounded_path(ROOT, destination)
            if not src.is_dir() or dst.exists() or src.is_symlink():
                raise ContractError("Require existing original and new archive directories")
        save_new(
            LOCK,
            {
                "source_sha256": sha256(__file__),
                "prior_qa_sha256": sha256(proof),
                "files": preserved,
                "mappings": MAPPINGS,
                "authorized_action": "Retire downloaded external normal inputs; return to UR-only",
                "permanent_deletion": False,
                "original_runs_configs_results_unchanged": True,
            },
        )
    record = read_json(LOCK)
    if record["source_sha256"] != sha256(__file__):
        raise ContractError("Retirement implementation changed")
    for source, destination in record["mappings"].items():
        src, dst = bounded_path(ROOT, source), bounded_path(ROOT, destination)
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            # Same-workspace directory rename; both absolute paths bounded above.
            src.rename(dst)
        elif src.exists() or not dst.is_dir():
            raise ContractError("Ambiguous archive state; refuse overwrite")
    status_record = verify(record)
    save_new(
        RESULT,
        {
            **status_record,
            "retirement_lock_sha256": sha256(LOCK),
            "active_training_scope": "original UR Train24 ADL only; optional Train derivatives",
            "external_runs": ["ur-external128-20261012-001", "ur-external256-20261012-001"],
            "external_runs_status": "historical comparison only, never adopted",
            "old_external_verifiers": "Original source paths are retired; use this mapped preservation audit",
            "new_test_evaluations": 0,
            "test_feature_arrays_loaded": 0,
        },
    )
    print(status_record)


def verify(record):
    if record["source_sha256"] != sha256(__file__):
        raise ContractError("Retirement source changed")
    moved = 0
    for relative, digest in record["files"].items():
        actual = mapped_path(relative, record["mappings"])
        if sha256(bounded_path(ROOT, actual)) != digest:
            raise ContractError(f"Archived or preserved artifact changed: {relative}")
        moved += actual != relative
    for source, destination in record["mappings"].items():
        if bounded_path(ROOT, source).exists() or not bounded_path(ROOT, destination).is_dir():
            raise ContractError("External input still active or archive missing")
    return {
        "external_active_paths_absent": True,
        "preserved_files": len(record["files"]),
        "relocated_files": moved,
        "mappings": record["mappings"],
        "permanent_deletion": False,
    }


def status():
    result = read_json(RESULT)
    if sha256(LOCK) != result["retirement_lock_sha256"]:
        raise ContractError("Retirement lock changed")
    print(verify(read_json(LOCK)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("archive", "status"))
    globals()[parser.parse_args().stage]()
