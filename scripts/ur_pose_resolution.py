"""One locked higher-resolution Pose experiment; no Test operation, no old writes."""

import argparse
import copy
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import psutil
import torch
import yaml

from fall_detect import pipeline
from fall_detect.config import load_config, project_path, recipe_fingerprint
from fall_detect.data.cache import load_pose_cache
from fall_detect.detection.events import score_sequence
from fall_detect.evaluation.metrics import sequence_metrics
from fall_detect.evaluation.pose_comparison import (
    assert_resolution_only,
    human_phase_summary,
    paired_pose_rows,
    resolution_config,
    scoped_source,
)
from fall_detect.io_utils import ContractError, read_json, sha256, write_csv, write_json
from fall_detect.training import experiment
from fall_detect.visualization.reports import evaluation_report

ROOT = project_path(".")
TAG = "ur-pose-resolution-20261013-001"
BASE = ROOT / "runs/ur-isotropic-wide-20261012-001"
E = ROOT / "docs/validation" / TAG
C = ROOT / "data/cache" / TAG
O = ROOT / "outputs" / TAG
RUN = ROOT / "runs" / TAG
CONFIG = ROOT / "configs" / f"{TAG}.yaml"
PLAN = ROOT / "docs/ur-pose-resolution-20261013-plan.md"
HUMAN = ROOT / "outputs/ur-fall19-human-amendment-20261013-001/derived-annotation.json"
PILOT_IDS = ["adl-05", "adl-24", "adl-04", "fall-19"]
MUTABLE = [
    "AGENTS.md",
    "Fall_Detection_Codex_Implementation_Spec.md",
    "docs/course-alignment-plan.md",
    "docs/current-progress.md",
    "docs/acceptance-results.md",
]
SOURCES = [
    "scripts/ur_pose_resolution.py",
    "src/fall_detect/evaluation/pose_comparison.py",
    "tests/test_pose_resolution.py",
    "src/fall_detect/pipeline.py",
    "src/fall_detect/pose/human.py",
    "src/fall_detect/config.py",
    "src/fall_detect/skeleton/windows.py",
    "src/fall_detect/skeleton/preprocessing.py",
    "src/fall_detect/models/stgcn.py",
    "src/fall_detect/models/weights.py",
    "src/fall_detect/data/cache.py",
    "src/fall_detect/training/experiment.py",
    "src/fall_detect/training/scaler.py",
    "src/fall_detect/detection/events.py",
    "src/fall_detect/evaluation/metrics.py",
]


def save(path, value):
    if path.exists():
        raise ContractError(f"Preserve completed evidence: {path}")
    write_json(path, value)


def verify_lock():
    lock = read_json(E / "lock.json")
    for category in ["source_hashes", "input_hashes"]:
        for name, value in lock[category].items():
            if sha256(ROOT / name) != value:
                raise ContractError(f"Locked source/input changed: {name}")
    config = load_config(CONFIG)
    assert_resolution_only(load_config(BASE / "resolved-config.yaml"), config)
    return lock, config


@contextmanager
def exclusive():
    path = E / "active.lock"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    try:
        yield
    finally:
        if path.read_text(encoding="utf-8") == str(os.getpid()):
            path.unlink()


def inventory():
    if any(p.exists() for p in [E, C, O, RUN, CONFIG]):
        raise ContractError("Preserve existing experiment directories; no repeated inventory")
    active = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = " ".join(proc.info["cmdline"] or [])
            if (
                proc.pid not in {os.getpid(), os.getppid()}
                and "python" in (proc.info["name"] or "").lower()
                and any(
                    s in cmd
                    for s in [
                        " train ",
                        "training",
                        "ur_pose_resolution.py",
                        "ur_normal_methods.py ae",
                        " prepare ",
                    ]
                )
            ):
                active.append(proc.info)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    if active:
        raise ContractError(f"Another preparation/training process is active: {active}")
    E.mkdir(parents=True)
    contracts = {}
    for name in MUTABLE:
        destination = E / "prior-contract" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
        contracts[name] = sha256(ROOT / name)
    assets = {}
    for scope in ["runs", "configs", "data", "weights", "outputs"]:
        for path in sorted((ROOT / scope).rglob("*")):
            if path.is_file():
                assets[path.relative_to(ROOT).as_posix()] = sha256(path)
                if len(assets) % 5000 == 0:
                    print(f"Preserved {len(assets)} files", flush=True)
    for scope in [
        "docs/validation/ur-fall19-human-amendment-20261013-001",
        "docs/validation/ur-fall19-human-review-20261012-001",
    ]:
        for path in (ROOT / scope).rglob("*"):
            if path.is_file():
                assets[path.relative_to(ROOT).as_posix()] = sha256(path)
    save(
        E / "before.json",
        {
            "assets": assets,
            "contracts": contracts,
            "active_training": active,
            "test_bytes_hash_only": True,
            "test_features_decoded": 0,
            "process_inventory": "psutil; CIM unavailable",
        },
    )
    base = load_config(BASE / "resolved-config.yaml")
    weights = E / "weights-manifest.json"
    shutil.copyfile(project_path(base["pose"]["weights_manifest"]), weights)
    config = resolution_config(
        base,
        str(C.relative_to(ROOT)),
        str((C / "manifests").relative_to(ROOT)),
        str(weights.relative_to(ROOT)),
    )
    CONFIG.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    resources = read_json(BASE / "dataset-index.json")["weights"]
    hashes = {role: entry["sha256"] for role, entry in resources.items()}
    recipe = recipe_fingerprint(config, hashes)
    if recipe == read_json(BASE / "dataset-index.json")["recipe_hash"]:
        raise ContractError("New Pose input must use a distinct recipe")
    inputs = [
        PLAN,
        CONFIG,
        HUMAN,
        BASE / "resolved-config.yaml",
        BASE / "dataset-index.json",
        project_path(base["pose"]["weights_manifest"]),
    ]
    save(
        E / "lock.json",
        {
            "source_hashes": {n: sha256(ROOT / n) for n in SOURCES},
            "input_hashes": {p.relative_to(ROOT).as_posix(): sha256(p) for p in inputs},
            "candidate": TAG,
            "candidate_count": 1,
            "pilot_ids": PILOT_IDS,
            "recipe_hash": recipe,
            "weight_hashes": hashes,
            "normal_train_videos": 24,
            "validation_videos": 23,
            "scope": ["train", "validation"],
            "test_evaluations": 0,
            "calibration": "same wide balanced_sequence_accuracy; fewer FP; higher threshold",
        },
    )
    print({"preserved_assets": len(assets), "recipe_hash": recipe}, flush=True)


@contextmanager
def scoped_access(index, audit):
    allowed = {
        s["sequence_id"]
        for s in index["split_manifest"]["sequences"]
        if s["split"] in ("train", "validation")
    }
    read_source, identity, cache_for = (
        pipeline.read_source,
        pipeline.source_identity,
        experiment.cache_for,
    )

    def checked_read(directory, *args, **kwargs):
        sid = kwargs.get("sequence_id", Path(directory).name)
        if sid not in allowed or Path(directory).name not in allowed:
            raise ContractError("Test/raw source access forbidden")
        audit.append({"kind": "raw", "sequence_id": sid})
        return read_source(directory, *args, **kwargs)

    def checked_identity(directory, *args, **kwargs):
        if Path(directory).name not in allowed:
            raise ContractError("Test source identity access forbidden during preparation")
        audit.append({"kind": "identity", "sequence_id": Path(directory).name})
        return identity(directory, *args, **kwargs)

    def checked_cache(actual, sid):
        source = scoped_source(actual, sid)
        audit.append({"kind": "features", "sequence_id": sid, "split": source["split"]})
        return cache_for(actual, sid)

    with (
        patch.object(pipeline, "read_source", side_effect=checked_read),
        patch.object(pipeline, "source_identity", side_effect=checked_identity),
        patch.object(experiment, "cache_for", side_effect=checked_cache),
    ):
        yield


def prepared_rows(index, ids):
    for sid in ids:
        if index["sequences"][sid]["status"] != "prepared":
            raise ContractError(f"Incomplete real preparation: {sid}: {index['sequences'][sid]}")


def prepare_stage(pilot=False):
    lock, config = verify_lock()
    target = E / ("pilot.json" if pilot else "prepare.json")
    if target.exists():
        raise ContractError("Preparation stage already completed")
    if not pilot and not read_json(E / "pilot.json")["technical_gate_passed"]:
        raise ContractError("Pilot technical gate required")
    old_index = experiment.load_index(BASE / "dataset-index.json")
    cfg = copy.deepcopy(config)
    if pilot:
        cfg["data"]["cache_dir"] = str((C / "pilot").relative_to(ROOT))
    destination = C / ("pilot-index.json" if pilot else "dataset-index.json")
    audit, reports = [], []
    original_create = pipeline.create_backends

    def recorded_backends(*args):
        pose, extractor = original_create(*args)
        record = {
            "strict_loads": {**pose.load_reports, "stgcn": extractor.load_report},
            "heatmap_shapes": [],
        }
        if any(
            not record["strict_loads"][role]["strict"]
            or record["strict_loads"][role]["sha256"] != lock["weight_hashes"][role]
            for role in ["detector", "pose", "stgcn"]
        ):
            raise ContractError("Strict real pretrained weights required")

        def heatmap_hook(_model, inputs, output):
            if (
                tuple(inputs[0].shape[1:]) != (3, 512, 384)
                or tuple(output.shape[1:]) != (17, 128, 96)
                or not torch.isfinite(output).all()
            ):
                raise ContractError("512x384 crop / finite 17x128x96 heatmap required")
            shape = list(output.shape)
            if not record["heatmap_shapes"]:
                record["heatmap_shapes"].append(shape)

        pose.pose.register_forward_hook(heatmap_hook)
        reports.append(record)
        return pose, extractor

    with (
        exclusive(),
        scoped_access(old_index, audit),
        patch.object(pipeline, "create_backends", side_effect=recorded_backends),
    ):
        _, index = pipeline.prepare(
            cfg,
            selected_ids=PILOT_IDS if pilot else None,
            output_index=destination,
            reuse_index=None if pilot else C / "pilot-index.json",
            train_validation_only=not pilot,
        )
    ids = (
        PILOT_IDS
        if pilot
        else [
            s["sequence_id"]
            for s in index["split_manifest"]["sequences"]
            if s["split"] in ("train", "validation")
        ]
    )
    prepared_rows(index, ids)
    if (
        index["recipe_hash"] != lock["recipe_hash"]
        or index["split_manifest"]["split_hash"] != old_index["split_manifest"]["split_hash"]
    ):
        raise ContractError("New recipe or original split mismatch")
    stats = []
    for sid in ids:
        cache = experiment.cache_for(index, sid)
        stats.append(
            {
                "sequence_id": sid,
                "features": len(cache.features),
                "dim": cache.features.shape[1],
                "finite": bool(np.isfinite(cache.features).all()),
                "source_hash": cache.metadata["source_hash"],
            }
        )
        if cache.metadata["source_hash"] != old_index["sequences"][sid]["source_hash"]:
            raise ContractError("Original source bytes changed")
    if pilot and not all(r["features"] > 0 for r in stats if r["sequence_id"].startswith("adl")):
        raise ContractError("Train and normal Validation pilot need actual finite features")
    save(
        target,
        {
            "technical_gate_passed": True,
            "source_kind": "real",
            "evidence_level": "real_feature_smoke",
            "index_sha256": sha256(destination),
            "stats": stats,
            "load_reports": reports,
            "verification": index["verification"],
            "accesses": audit,
            "test_feature_reads": 0,
            "test_image_reads": 0,
            "test_evaluations": 0,
        },
    )
    print(
        {
            "stage": "pilot" if pilot else "prepare",
            "sources": len(ids),
            "features": sum(r["features"] for r in stats),
        },
        flush=True,
    )


def train():
    _, config = verify_lock()
    if not (E / "prepare.json").exists() or RUN.exists():
        raise ContractError("Require complete preparation and a new run; no repeated training")
    index = experiment.load_index(C / "dataset-index.json")
    audit = []
    with exclusive(), scoped_access(index, audit):
        experiment.train(config, C / "dataset-index.json", RUN, train_validation_only=True)
    save(
        E / "training.json",
        {
            **read_json(RUN / "status.json"),
            "accesses": audit,
            "model_sha256": sha256(RUN / "autoencoder.pt"),
            "test_feature_reads": 0,
        },
    )


def evaluate_one(run, output, audit):
    scorer = experiment.Scorer(run)
    rows, windows, events = [], [], []
    with scoped_access(scorer.index, audit):
        for source in scorer.index["split_manifest"]["sequences"]:
            if source["split"] != "validation":
                continue
            cache = experiment.cache_for(scorer.index, source["sequence_id"])
            summary, sw, se = score_sequence(
                cache, scorer.score(cache.features), scorer.threshold["threshold"]
            )
            rows.append({**summary, "label": source["label"]})
            windows.extend(sw)
            events.extend(se)
    if len(rows) != 23:
        raise ContractError("Every original Validation video required")
    metrics = {"source_kind": "real", **sequence_metrics(rows)}
    evaluation_report(output, metrics, rows, windows, events)
    return {
        "metrics": metrics,
        "rows": rows,
        "windows": windows,
        "events": events,
        "threshold": scorer.threshold["threshold"],
        "run": str(run.relative_to(ROOT)),
    }


def validation():
    _, config = verify_lock()
    if O.exists() or (E / "validation.json").exists():
        raise ContractError("Validation already completed; no additional calibration")
    index = experiment.load_index(C / "dataset-index.json")
    audit = []
    with exclusive(), scoped_access(index, audit):
        experiment.calibrate(RUN)
        baseline = evaluate_one(BASE, O / "baseline", audit)
        candidate = evaluate_one(RUN, O / "candidate", audit)
    old_index = experiment.load_index(BASE / "dataset-index.json")
    human = read_json(HUMAN)
    all_pose_rows = []
    for source in index["split_manifest"]["sequences"]:
        if source["split"] not in ("train", "validation"):
            continue
        sid = source["sequence_id"]
        old, _ = load_pose_cache(project_path(old_index["sequences"][sid]["pose_path"]), {})
        new, _ = load_pose_cache(project_path(index["sequences"][sid]["pose_path"]), {})
        all_pose_rows.extend(
            paired_pose_rows(
                old, new, config["quality"], human["frames"] if sid == "fall-19" else None
            )
        )
        if sid == "fall-19":
            for result, observations in [(baseline, old), (candidate, new)]:
                result["fall19_human_phases"] = human_phase_summary(
                    [w for w in result["windows"] if w["sequence_id"] == sid],
                    [e for e in result["events"] if e["sequence_id"] == sid],
                    observations,
                    human["frames"],
                    config["quality"],
                )
    write_csv(O / "paired-pose-frames.csv", all_pose_rows)
    old_rows = {r["sequence_id"]: r for r in baseline["rows"]}
    changes = [
        {
            "sequence_id": r["sequence_id"],
            "label": r["label"],
            "old_prediction": old_rows[r["sequence_id"]]["prediction"],
            "new_prediction": r["prediction"],
            "old_alarm_ms": old_rows[r["sequence_id"]]["first_alarm_ms"],
            "new_alarm_ms": r["first_alarm_ms"],
            "old_coverage": old_rows[r["sequence_id"]]["window_coverage"],
            "new_coverage": r["window_coverage"],
        }
        for r in candidate["rows"]
    ]
    write_csv(O / "video-comparison.csv", changes)
    save(
        E / "validation.json",
        {
            "baseline": baseline,
            "candidate": candidate,
            "video_comparison": changes,
            "accesses": audit,
            "precise_time_ground_truth": False,
            "human_new_pose_verified": False,
            "test_feature_reads": 0,
            "test_evaluations": 0,
            "adoption": "pending independent metrics/time/quality verification; default unchanged",
        },
    )
    print(
        {
            "baseline": baseline["metrics"],
            "candidate": candidate["metrics"],
            "fall19": candidate["fall19_human_phases"],
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["inventory", "pilot", "prepare", "train", "validation", "status"]
    )
    stage = parser.parse_args().stage
    if stage == "status":
        verify_lock()
        print(
            {n: (E / f"{n}.json").exists() for n in ["pilot", "prepare", "training", "validation"]}
        )
    elif stage == "pilot":
        prepare_stage(True)
    elif stage == "prepare":
        prepare_stage(False)
    else:
        globals()[stage]()
