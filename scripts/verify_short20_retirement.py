"""Preserve exact prior core bytes and audit retired inputs after authorized scope changes."""

import argparse
import hashlib
import re
from pathlib import Path

from retire_external_normal import mapped_path

from fall_detect.io_utils import ContractError, read_json, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "ur-short20-isotropic-20261012"
SIDECAR = ROOT / f"docs/validation/{PREFIX}-prior-source.json"
RETIREMENT = ROOT / "docs/validation/external-normal-retired-20261012-lock.json"


def replace(text, old, new, count=1):
    if text.count(old) != count:
        raise ContractError("Prior source reconstruction pattern mismatch")
    return text.replace(old, new)


def undo_cli(text):
    text, count = re.subn(
        r'            sub.add_argument\(\n                "--train-validation-only",\n                action="store_true",\n                help="[^"\n]*",\n            \)\n',
        "",
        text,
    )
    if count != 2:
        raise ContractError("CLI opt-in reconstruction mismatch")
    text, count = re.subn(
        r"            manifest = dataset_split\(config\)\n.*?            report = raw_dataset_report\(config, manifest, selected\)\n",
        "            report = raw_dataset_report(config, dataset_split(config), args.sequence_ids)\n",
        text,
        flags=re.DOTALL,
    )
    if count != 1:
        raise ContractError("CLI inventory reconstruction mismatch")
    text = replace(text, "            train_validation_only=args.train_validation_only,\n", "")
    text = replace(
        text,
        """        selected = args.sequence_ids or [
            s["sequence_id"]
            for s in index["split_manifest"]["sequences"]
            if not args.train_validation_only or s["split"] in ("train", "validation")
        ]
""",
        '        selected = args.sequence_ids or list(index["sequences"])\n',
    )
    return replace(text, ", train_validation_only=args.train_validation_only", "", count=2)


def undo_pipeline(text):
    for old in (
        "    train_validation_only=False,\n",
        """    if train_validation_only and (selected_ids or limit_frames):
        raise ContractError("Train/Validation-only cannot be combined with bounded smoke")
""",
        '        **({"preparation_scope": ["train", "validation"]} if train_validation_only else {}),\n',
        """        if train_validation_only and source["split"] == "test":
            index["sequences"][sid] = {
                "status": "not_prepared",
                "reason": "Test excluded by explicit Train/Validation-only execution",
            }
            continue
""",
    ):
        text = replace(text, old, "")
    return text


def undo_training(text):
    for old in (
        """    if "preparation_scope" in index:
        if index["preparation_scope"] != ["train", "validation"]:
            raise ContractError("Unsupported explicit preparation scope")
        for sid, source in expected.items():
            if source["split"] == "test" and (
                index["sequences"][sid].get("status") != "not_prepared"
                or any(k in index["sequences"][sid] for k in ("feature_path", "pose_path"))
            ):
                raise ContractError("Scoped Test must remain unprepared without cache paths")
""",
        """    scope = index.get("preparation_scope")
    if scope is not None and source["split"] not in scope:
        raise ContractError(f"{sid}: split outside explicit preparation scope")
""",
        """    if index.get("preparation_scope") is not None and not train_validation_only:
        raise ContractError("Scoped index requires explicit Train/Validation-only training")
""",
        '            "execution_scope": "train_validation_only" if train_validation_only else "all_splits",\n',
        """    if scorer.index.get("preparation_scope") is not None:
        raise ContractError("Test evaluation unavailable for a Train/Validation-only run")
""",
    ):
        text = replace(text, old, "")
    text = replace(
        text,
        "def train(config, index_path, run, synthetic=False, *, train_validation_only=False):",
        "def train(config, index_path, run, synthetic=False):",
    )
    text = replace(
        text,
        "return _train(config, index_path, run, synthetic, train_validation_only)",
        "return _train(config, index_path, run, synthetic)",
    )
    text = replace(
        text,
        "def _train(config, index_path, run, synthetic=False, train_validation_only=False):",
        "def _train(config, index_path, run, synthetic=False):",
    )
    start = text.index('    if train_validation_only and index.get("preparation_scope")')
    end = text.index("    run = Path(run)\n", start)
    return (
        text[:start]
        + """    if not synthetic and not all(
        row["status"] == "prepared" for row in index["sequences"].values()
    ):
        raise ContractError(
            "Formal training requires all planned sequences prepared; incomplete data is smoke only"
        )
"""
        + text[end:]
    )


def archive_prior():
    if SIDECAR.exists():
        raise ContractError("Preserve completed prior source archive")
    record = read_json(RETIREMENT)
    transformations = {
        "src/fall_detect/cli.py": undo_cli,
        "src/fall_detect/pipeline.py": undo_pipeline,
        "src/fall_detect/training/experiment.py": undo_training,
    }
    archived = {}
    for name, undo in transformations.items():
        value = undo((ROOT / name).read_text(encoding="utf8"))
        payloads = [value.encode("utf8"), value.replace("\n", "\r\n").encode("utf8")]
        payload = next(
            (p for p in payloads if hashlib.sha256(p).hexdigest() == record["files"][name]), None
        )
        if payload is None:
            raise ContractError(f"Prior source bytes do not match historical SHA256: {name}")
        destination = ROOT / "docs/validation" / f"{PREFIX}-prior-source" / name
        if destination.exists():
            if sha256(destination) != record["files"][name]:
                raise ContractError("Archive already exists with a different digest")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        archived[name] = {
            "original_sha256": record["files"][name],
            "original_source": destination.relative_to(ROOT).as_posix(),
            "authorized_current_sha256": sha256(ROOT / name),
        }
    write_json(
        SIDECAR,
        {
            "retirement_lock_sha256": sha256(RETIREMENT),
            "reason": "Authorized Train/Validation-only core implementation. Original core reconstructed by reversing this turn's edits and verified byte-for-byte against preexisting retirement hashes before saving. Old retirement lock/source remain unchanged.",
            "archived_sources": archived,
        },
    )


def verify(write=False):
    sidecar, record = read_json(SIDECAR), read_json(RETIREMENT)
    if sidecar["retirement_lock_sha256"] != sha256(RETIREMENT):
        raise ContractError("Original retirement lock changed")
    if record["source_sha256"] != sha256(ROOT / "scripts/retire_external_normal.py"):
        raise ContractError("Old retirement implementation changed")
    checked, moved = 0, 0
    for relative, digest in record["files"].items():
        actual = mapped_path(relative, record["mappings"])
        entry = sidecar["archived_sources"].get(relative)
        if entry:
            if (
                digest != entry["original_sha256"]
                or sha256(ROOT / relative) != entry["authorized_current_sha256"]
            ):
                raise ContractError("Authorized core source changed")
            actual = entry["original_source"]
        if sha256(ROOT / actual) != digest:
            raise ContractError(f"Preserved bytes changed: {relative}")
        moved += mapped_path(relative, record["mappings"]) != relative
        checked += 1
    for source, destination in record["mappings"].items():
        if (ROOT / source).exists() or not (ROOT / destination).is_dir():
            raise ContractError("External input active or archive absent")
    result = {
        "passed": True,
        "preserved_historical_files": checked,
        "retired_external_files": moved,
        "exact_original_core_archives": len(sidecar["archived_sources"]),
        "external_active_paths_absent": True,
        "archive_restored_or_redownloaded": False,
        "original_retirement_lock_unchanged": True,
        "source_snapshot_sha256": sha256(SIDECAR),
        "test_arrays_decoded": 0,
        "hash_checks_only": True,
    }
    if write:
        target = ROOT / f"docs/validation/{PREFIX}-retirement-verification.json"
        if target.exists():
            raise ContractError("Preserve completed retirement audit")
        write_json(target, result)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-prior-source", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.archive_prior_source:
        archive_prior()
    verify(args.write)
