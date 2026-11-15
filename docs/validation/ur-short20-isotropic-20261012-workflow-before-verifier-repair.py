"""One preregistered UR-only short20/isotropic experiment. No Test operation."""

import argparse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml
from human_isotropic_refinement import train_validation_only, validation_result

from fall_detect.config import load_config, project_path, recipe_fingerprint
from fall_detect.data.cache import validate_feature_cache
from fall_detect.detection.events import score_sequence
from fall_detect.io_utils import ContractError, read_json, sha256, write_json
from fall_detect.pipeline import prepare
from fall_detect.training import experiment

ROOT = project_path(".")
PREFIX = "ur-short20-isotropic-20261012"
RUN = "ur-short20-isotropic-20261012-001"
BASE = "ur-isotropic-wide-20261012-001"
CONFIG = ROOT / "configs" / f"{RUN}.yaml"
CACHE = ROOT / "data/cache" / RUN
INDEX = CACHE / "dataset-index.json"
SOURCE_INDEX = ROOT / "runs" / BASE / "dataset-index.json"
FALL_IDS = ["fall-09", "fall-19", "fall-29", "fall-25", "fall-18", "fall-30"]


def evidence(kind):
    return ROOT / "docs/validation" / f"{PREFIX}-{kind}.json"


def save(kind, value):
    target = evidence(kind)
    if target.exists():
        raise ContractError(f"Preserve completed evidence: {target}")
    write_json(target, value)


def assert_new(stage):
    if evidence(stage).exists() or evidence("validation").exists():
        raise ContractError(f"Completed {stage}; no overwrite or additional candidate")


def verify_lock():
    lock = read_json(evidence("lock"))
    for name, digest in lock["preserved_files"].items():
        if sha256(ROOT / name) != digest:
            raise ContractError(f"Preserved file changed: {name}")
    for name, digest in lock["implementation_hashes"].items():
        if sha256(ROOT / name) != digest:
            repair_path = evidence("dispatch-repair")
            repair = read_json(repair_path) if repair_path.exists() else {}
            if not (
                name == "scripts/ur_short20_isotropic.py"
                and repair.get("original_sha256") == digest
                and repair.get("repaired_sha256") == sha256(ROOT / name)
                and sha256(ROOT / repair["original_source"]) == digest
                and repair.get("before_training") is True
            ):
                raise ContractError(f"Locked implementation changed: {name}")
    if sha256(CONFIG) != lock["config_sha256"]:
        raise ContractError("Locked config changed")
    return lock


def window_stats(index, sid, interval=None, scorer=None):
    cache = validate_feature_cache(experiment.cache_for(index, sid))
    scores = scorer.score(cache.features) if scorer else np.zeros(len(cache.features))
    out, windows, _ = score_sequence(cache, scores, scorer.threshold["threshold"] if scorer else 1)
    phase = interval["fall_phase_ms"] if interval else None
    phase_rows = [w for w in windows if phase and phase[0] <= w["available_at_ms"] <= phase[1]]
    durations = [r.end_ms - r.start_ms for r in cache.records]
    return {
        **out,
        "valid_features": len(cache.features),
        "feature_shape": list(cache.features.shape),
        "finite": bool(np.isfinite(cache.features).all()),
        "quality_counts": cache.metadata["quality_counts"],
        "window_size": cache.metadata["window_size"],
        "stride": cache.metadata["stride"],
        "duration_ms_range": [min(durations), max(durations)] if durations else None,
        "phase_scheduled_endpoints": len({w["sample_index"] for w in phase_rows}),
        "phase_valid_endpoints": len(
            {w["sample_index"] for w in phase_rows if w["raw_score"] is not None}
        ),
        "phase_legal_k3": sum(w["event_score"] is not None for w in phase_rows),
        "phase_max_raw_score": max(
            (w["raw_score"] for w in phase_rows if w["raw_score"] is not None), default=None
        ),
        "phase_max_event_score": max(
            (w["event_score"] for w in phase_rows if w["event_score"] is not None), default=None
        ),
        "phase_windows": phase_rows if scorer else [],
        "pose_sha256": sha256(project_path(index["sequences"][sid]["pose_path"])),
        "feature_sha256": cache.metadata["array_sha256"],
        "causal_timestamps_verified": all(
            r.available_at_ms >= r.end_ms >= r.start_ms for r in cache.records
        ),
    }


def lock():
    assert_new("lock")
    if CONFIG.exists() or CACHE.exists() or (ROOT / "runs" / RUN).exists():
        raise ContractError("New candidate paths must be unused")
    original = read_json(SOURCE_INDEX)
    old_lock = read_json(ROOT / "docs/validation/isotropic-refinement-20261012-lock.json")
    intervals = old_lock["coarse_intervals"]
    accesses = []
    with train_validation_only(accesses):
        scorer = experiment.Scorer(ROOT / "runs" / BASE)
        reference = validation_result(BASE, {**old_lock, "reference_fn": ["fall-19"]}, export=False)
        diagnosis = window_stats(original, "fall-19", intervals["fall-19"], scorer)
    config = load_config(ROOT / "runs" / BASE / "resolved-config.yaml")
    config["data"]["cache_dir"] = f"data/cache/{RUN}"
    config["skeleton"].update(
        window_size=20,
        temporal_context="short_20",
        stride=1,
        coordinate_normalization="per_frame_isotropic",
    )
    CONFIG.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf8")
    config = load_config(CONFIG)
    hashes = {k: v["sha256"] for k, v in original["weights"].items()}
    train_ids = sorted(
        s["sequence_id"] for s in original["split_manifest"]["sequences"] if s["split"] == "train"
    )
    val_adl = sorted(
        s["sequence_id"]
        for s in original["split_manifest"]["sequences"]
        if s["split"] == "validation" and s["label"] == 0
    )
    pilot_ids = train_ids[:2] + val_adl[:2] + FALL_IDS
    files = set()
    for folder in ("runs", "configs", "data/manifests"):
        files.update(p for p in (ROOT / folder).rglob("*") if p.is_file())
    files.add(ROOT / "weights/sample-human-manifest.json")
    files.update(project_path(v["local_path"]) for v in original["weights"].values())
    for s in original["split_manifest"]["sequences"]:
        if s["split"] == "test":
            continue
        row = original["sequences"][s["sequence_id"]]
        for key in ("pose_path", "feature_path"):
            p = project_path(row[key])
            files.update((p, p.with_suffix(".json")))
    files.discard(CONFIG)
    # Existing outputs are preserved as bytes; their scores never enter selection.
    for folder in ("outputs/isotropic-refinement-20261012", "outputs/ur-train-shift-20261012"):
        files.update(p for p in (ROOT / folder).rglob("*") if p.is_file())
    implementations = list((ROOT / "src/fall_detect").rglob("*.py")) + [
        Path(__file__),
        ROOT / "scripts/human_isotropic_refinement.py",
        ROOT / "scripts/human_residual.py",
        ROOT / "scripts/verify_ur_short20_isotropic.py",
        ROOT / "tests/test_train_validation_scope.py",
    ]
    save(
        "diagnosis",
        {
            "baseline": BASE,
            "fall19": diagnosis,
            "limitation": "Coarse AI intervals are diagnostic, not confirmed temporal GT",
            "feature_accesses": accesses,
        },
    )
    save(
        "lock",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "run": RUN,
            "baseline": BASE,
            "config_sha256": sha256(CONFIG),
            "recipe_hash": recipe_fingerprint(config, hashes),
            "source_recipe_hash": original["recipe_hash"],
            "split_hash": original["split_manifest"]["split_hash"],
            "source_index_sha256": sha256(SOURCE_INDEX),
            "pilot_ids": pilot_ids,
            "reference_metrics": reference["metrics"],
            "reference_fn": ["fall-19"],
            "reference_phases": reference["phases"],
            "coarse_intervals": intervals,
            "hypothesis": "Shorter windows may reduce pre-fall normal-history dilution and cropping-related invalid windows; causal contribution is unproven.",
            "fixed": "Exactly one: 20frames/stride1/per_frame_isotropic, genuine frozen Human512D, 24 original UR Train ADL without augmentation/external data. Same AE256/64 seed42 lr.001 wd.001 noise0 batch32 sequence sampling max100 patience10 min_delta1e-6. Scaler/AE Train ADL; normal Validation checkpoint; original23 Validation balanced_sequence_accuracy threshold, strict > K3; quality .2/9/.8 and causal missing policy unchanged.",
            "pilot_gate": "All10 fixed sequences finite genuine512D, exact observed-pose hash parity, causal timestamps and complete schedules. All legalK3; normal mean coverage >=90% baseline; each6 coarse phases >=1 legalK3 and total >=80% baseline; distinct30/per_frame20/isotropic20 fingerprints. No classification selection at pilot.",
            "selection": "Accuracy strictly >22/23, F1>=28/29, no extra FP/unknown, no window coverage regression, no baseline correct video regresses, >=4 preexisting coarse phase-event hits, no early events, no loss of reference phase/current-good-pose hits. Recovered fall19 requires current-good-pose alarm in its preexisting interval. Keep default unless every guard passes; diagnostic intervals cannot establish true latency.",
            "stop": "One candidate only; stop on failed pilot/nonfinite/integrity error. <=100epochs patience10; no extra capacity/seed/normalization/quality/K/calibration objectives. Stop after Validation irrespective of outcome. No Test prepare/read/evaluate, no Notebook/video/public upload.",
            "preserved_files": {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(files)},
            "implementation_hashes": {
                p.relative_to(ROOT).as_posix(): sha256(p) for p in implementations
            },
            "historical_test_seen": True,
            "repeated_validation_not_independent": True,
            "graph_evidence": "Tier2 generation2026-10-12T07:52:22Z; source paths metadata_match/no_recorded_issue. Ignored cache/upstream read directly; best effort only.",
        },
    )
    print(
        {"locked_run": RUN, "recipe": recipe_fingerprint(config, hashes), "pilot_ids": pilot_ids},
        flush=True,
    )


def pilot():
    assert_new("pilot")
    record = verify_lock()
    accesses = []
    with train_validation_only(accesses):
        _, index = prepare(
            load_config(CONFIG), selected_ids=record["pilot_ids"], reuse_poses_index=SOURCE_INDEX
        )
        base = read_json(SOURCE_INDEX)
        old = {
            sid: window_stats(base, sid, record["coarse_intervals"].get(sid))
            for sid in record["pilot_ids"]
        }
        new = {
            sid: window_stats(index, sid, record["coarse_intervals"].get(sid))
            for sid in record["pilot_ids"]
        }
    normal = record["pilot_ids"][:4]
    ratio = np.mean([new[s]["window_coverage"] for s in normal]) / np.mean(
        [old[s]["window_coverage"] for s in normal]
    )
    phase_ratio = sum(new[s]["phase_legal_k3"] for s in FALL_IDS) / sum(
        old[s]["phase_legal_k3"] for s in FALL_IDS
    )
    hashes = {k: v["sha256"] for k, v in index["weights"].items()}
    per_frame20 = load_config(CONFIG)
    per_frame20["skeleton"]["coordinate_normalization"] = "per_frame"
    checks = {
        "genuine_strict_stgcn": bool(index["model_load_reports"]["stgcn"]["strict"]),
        "real_20_forward_head_parity": index["verification"]["feature_forward"],
        "finite_512": all(
            r["finite"] and r["feature_shape"][1] == 512 and r["valid_features"] > 0
            for r in new.values()
        ),
        "pose_hash_parity": all(new[s]["pose_sha256"] == old[s]["pose_sha256"] for s in new),
        "timestamps": all(r["causal_timestamps_verified"] for r in new.values()),
        "all_legal_k3": all(r["sequence_score"] is not None for r in new.values()),
        "normal_coverage": bool(ratio >= 0.9),
        "phase_coverage": bool(phase_ratio >= 0.8),
        "all_phases": all(new[s]["phase_legal_k3"] >= 1 for s in FALL_IDS),
        "cache_isolation": len(
            {
                record["source_recipe_hash"],
                index["recipe_hash"],
                recipe_fingerprint(per_frame20, hashes),
            }
        )
        == 3,
    }
    save(
        "pilot",
        {
            "evidence_tier": "real_feature_smoke",
            "checks": checks,
            "passed": all(checks.values()),
            "baseline": old,
            "candidate": new,
            "normal_coverage_ratio": float(ratio),
            "phase_legal_ratio": float(phase_ratio),
            "strict_load_reports": index["model_load_reports"],
            "verification": index["verification"],
            "test_feature_reads": 0,
            "classification_not_performed": True,
            "feature_accesses": accesses,
        },
    )
    verify_lock()
    print(checks, flush=True)
    if not all(checks.values()):
        raise ContractError("Preregistered pilot failed; stop without AE")


def full_prepare():
    assert_new("prepared")
    record = verify_lock()
    if not read_json(evidence("pilot"))["passed"]:
        raise ContractError("Pilot gate required")
    _, index = prepare(
        load_config(CONFIG), reuse_poses_index=SOURCE_INDEX, train_validation_only=True
    )
    required = [s for s in index["split_manifest"]["sequences"] if s["split"] != "test"]
    if not all(index["sequences"][s["sequence_id"]]["status"] == "prepared" for s in required):
        raise ContractError("Incomplete Train/Validation features; stop")
    save(
        "prepared",
        {
            "index_sha256": sha256(INDEX),
            "recipe_hash": record["recipe_hash"],
            "prepared_sequences": len(required),
            "test_status": "not_prepared",
            "test_features_read": 0,
            "verification": index["verification"],
            "counts": {
                s["sequence_id"]: index["sequences"][s["sequence_id"]]["valid_features"]
                for s in required
            },
        },
    )
    verify_lock()


def training():
    assert_new("validation")
    record = verify_lock()
    if sha256(INDEX) != read_json(evidence("prepared"))["index_sha256"]:
        raise ContractError("Prepared index changed")
    accesses = []
    with train_validation_only(accesses):
        run = experiment.train(
            load_config(CONFIG), INDEX, ROOT / "runs" / RUN, train_validation_only=True
        )
        experiment.calibrate(run)
        result = validation_result(RUN, record, export=False)
        scorer = experiment.Scorer(run)
        rows, windows, events = [], [], []
        for s in scorer.index["split_manifest"]["sequences"]:
            if s["split"] != "validation":
                continue
            cache = experiment.cache_for(scorer.index, s["sequence_id"])
            out, w, e = score_sequence(cache, scorer.score(cache.features), result["threshold"])
            rows.append({**out, "label": s["label"]})
            windows.extend(w)
            events.extend(e)
        from fall_detect.visualization.reports import evaluation_report

        folder = ROOT / "outputs" / PREFIX / RUN
        if folder.exists():
            raise ContractError("Preserve Validation output")
        folder.mkdir(parents=True)
        evaluation_report(
            folder,
            {
                **result["metrics"],
                "split": "validation",
                "source_kind": "real",
                "protocol_id": scorer.index["split_manifest"]["protocol_id"],
                "recipe_hash": scorer.index["recipe_hash"],
                "split_hash": record["split_hash"],
            },
            rows,
            windows,
            events,
        )
        reference = validation_result(BASE, record, export=False)
        ref_by_id = {r["sequence_id"]: r for r in reference["predictions"]}
        regressions = [
            r["sequence_id"]
            for r in rows
            if ref_by_id[r["sequence_id"]]["prediction"] == r["label"]
            and r["prediction"] != r["label"]
        ]
        ref_phases = {p["sequence_id"]: p for p in reference["phases"]}
        result["adoption_checks"].update(
            {
                "window_coverage_not_lower": result["metrics"]["window_coverage"]
                >= reference["metrics"]["window_coverage"],
                "no_video_regression": not regressions,
                "no_phase_hit_loss": all(
                    not ref_phases[p["sequence_id"]]["phase_event_starts"]
                    or p["phase_event_starts"]
                    for p in result["phases"]
                ),
                "no_visible_phase_loss": all(
                    not ref_phases[p["sequence_id"]]["visible_phase_alert_ms"]
                    or p["visible_phase_alert_ms"]
                    for p in result["phases"]
                ),
            }
        )
        result["eligible"] = all(result["adoption_checks"].values())
        result["regressions"] = regressions
        result["fall19_diagnosis"] = window_stats(
            scorer.index, "fall-19", record["coarse_intervals"]["fall-19"], scorer
        )
        result["exports"] = {p.relative_to(ROOT).as_posix(): sha256(p) for p in folder.iterdir()}
    save(
        "validation",
        {
            "lock_sha256": sha256(evidence("lock")),
            "reference": reference,
            "candidate": result,
            "selected_candidate": RUN if result["eligible"] else None,
            "default_run": "ur-sample-human-s1-001",
            "feature_accesses": accesses,
            "evidence_tier": "real_feature_smoke",
            "real_end_to_end": "not_run",
            "test_feature_reads": 0,
            "test_evaluations": 0,
            "test_prepared": False,
        },
    )
    verify_lock()
    print(
        {
            "metrics": result["metrics"],
            "adoption_checks": result["adoption_checks"],
            "adopted": result["eligible"],
        },
        flush=True,
    )


def status():
    verify_lock()
    result = read_json(evidence("validation")) if evidence("validation").exists() else None
    if result:
        experiment.verify_artifacts(ROOT / "runs" / RUN)
        for filename, digest in result["candidate"]["exports"].items():
            if sha256(ROOT / filename) != digest:
                raise ContractError("Validation output changed")
    print(
        {
            "run": RUN,
            "completed": result is not None,
            "adopted": result["candidate"]["eligible"] if result else None,
            "new_test_evaluations": 0,
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["lock", "pilot", "prepare", "training", "status"])
    command = parser.parse_args().command
    (full_prepare if command == "prepare" else globals()[command])()
