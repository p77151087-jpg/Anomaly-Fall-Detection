"""Preregistered pose flip pilot and Train-only normal-pattern sampling experiment."""

import argparse
import math
import shutil
from datetime import UTC, datetime

import numpy as np
import yaml
from ae_generalization import result
from human_residual import visible_pose_keys

from fall_detect.config import load_config, project_path, recipe_fingerprint
from fall_detect.detection.events import score_sequence
from fall_detect.evaluation.metrics import sequence_metrics
from fall_detect.io_utils import ContractError, read_json, sha256, write_csv, write_json
from fall_detect.pipeline import prepare
from fall_detect.training.experiment import (
    Scorer,
    cache_for,
    calibrate,
    evaluate,
    train,
    verify_artifacts,
)

ROOT = project_path(".")
INDEX = ROOT / "data/cache/ur-sample-human-s1-001/dataset-index.json"
FILES = [
    "src/fall_detect/config.py",
    "src/fall_detect/pose/human.py",
    "src/fall_detect/training/experiment.py",
    "src/fall_detect/training/patterns.py",
    "src/fall_detect/visualization/loss_viewer.py",
    "scripts/human_quality.py",
    "scripts/human_residual.py",
    "scripts/ae_generalization.py",
    "tests/test_human_quality.py",
]


def path(stage):
    return ROOT / f"docs/validation/human-quality-{stage}.json"


def save(stage, value):
    if path(stage).exists():
        raise ContractError(f"Preserve existing evidence: {stage}")
    write_json(path(stage), value)


def verify():
    lock = read_json(path("lock"))
    before_path = ROOT / "docs/validation/human-flip-before.json"
    before = read_json(before_path)
    if sha256(before_path) != lock["before_sha256"]:
        raise ContractError("Original snapshot changed")
    for run, hashes in before["old_runs"].items():
        if verify_artifacts(ROOT / "runs" / run) != hashes:
            raise ContractError("Old frozen run changed")
    for group in ("configs", "cache_files", "manifests"):
        for name, digest in before[group].items():
            if sha256(ROOT / name) != digest:
                raise ContractError(f"Original data/config changed: {name}")
    extra = path("extra-source-before")
    if sha256(extra) != lock["extra_source_sha256"]:
        raise ContractError("Extra source snapshot changed")
    for name, digest in {**before["source_files"], **read_json(extra)}.items():
        if sha256(ROOT / "docs/validation/human-flip-legacy-source" / name) != digest:
            raise ContractError("Archived source changed")
    for name, key in [
        ("human-flip-normal-diagnosis", "diagnosis_sha256"),
        ("human-quality-phase-diagnosis", "phase_diagnosis_sha256"),
    ]:
        if sha256(ROOT / f"docs/validation/{name}.json") != lock[key]:
            raise ContractError("Diagnosis changed")
    baseline = load_config(ROOT / "runs" / lock["baseline"] / "resolved-config.yaml")
    index = read_json(INDEX)
    hashes = {k: v["sha256"] for k, v in index["weights"].items()}
    if recipe_fingerprint(baseline, hashes) != index["recipe_hash"]:
        raise ContractError("Legacy default recipe hash changed")
    for resource in index["weights"].values():
        if sha256(project_path(resource["local_path"])) != resource["sha256"]:
            raise ContractError("Weight changed")
    if path("implementation").exists():
        frozen = read_json(path("implementation"))
        if sha256(path("lock")) != frozen["lock_sha256"]:
            raise ContractError("Candidate rules changed")
        for name, digest in frozen["files"].items():
            if sha256(ROOT / name) != digest:
                raise ContractError(f"Locked implementation changed: {name}")
    return lock


def freeze():
    lock = verify()
    if path("implementation").exists():
        raise ContractError("Implementation already frozen")
    files = list(FILES)
    for c in lock["candidates"]:
        name = ROOT / f"configs/{c['run']}.yaml"
        if name.exists() or (ROOT / "runs" / c["run"]).exists():
            raise ContractError("Candidate already exists")
        config = load_config(ROOT / "runs" / lock["baseline"] / "resolved-config.yaml")
        if c["kind"] == "flip_tta":
            config["pose"]["flip_tta"] = True
            manifest = ROOT / "weights/human-flip-manifest.json"
            if manifest.exists():
                raise ContractError("Preserve existing manifest")
            shutil.copyfile(project_path(config["pose"]["weights_manifest"]), manifest)
            config["pose"]["weights_manifest"] = str(manifest.relative_to(ROOT))
            config["data"]["cache_dir"] = f"data/cache/{c['run']}"
        else:
            config["autoencoder"]["sampling"] = "sequence_patterns"
        name.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf8")
        load_config(name)
        files.append(str(name.relative_to(ROOT)))
    save(
        "implementation",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "lock_sha256": sha256(path("lock")),
            "files": {f: sha256(ROOT / f) for f in files},
            "candidate_validation_not_yet_run": True,
        },
    )


def pilot_stats(index, config, lock):
    from types import SimpleNamespace

    rows = []
    scaffold = SimpleNamespace(config=config, index=index)
    for source in index["split_manifest"]["sequences"]:
        sid = source["sequence_id"]
        if sid not in lock["pilot_ids"]:
            continue
        if source["split"] == "test":
            raise ContractError("Pilot must not include Test")
        cache = cache_for(index, sid)
        out, _, _ = score_sequence(cache, np.zeros(len(cache.features)), 1.0)
        row = {**source, **out, "feature_windows": len(cache.features)}
        if sid in lock["coarse_intervals"]:
            lo, hi = lock["coarse_intervals"][sid]["fall_phase_ms"]
            good = visible_pose_keys(scaffold, sid)
            phase = [r for r in cache.records if lo <= r.available_at_ms <= hi]
            row["valid_phase_windows"] = sum(r.row_index is not None for r in phase)
            row["visible_good_phase_endpoints"] = len(
                {r.sample_index for r in phase if (r.sample_index, r.track_id) in good}
            )
        rows.append(row)
    return rows


def pilot():
    lock = verify()
    if not path("implementation").exists() or path("pilot").exists():
        raise ContractError("Freeze first; preserve completed pilot")
    c = lock["candidates"][0]
    config = load_config(ROOT / f"configs/{c['run']}.yaml")
    index_path, index = prepare(config, selected_ids=lock["pilot_ids"])
    if any(index["sequences"][sid]["status"] != "prepared" for sid in lock["pilot_ids"]):
        raise ContractError("Pilot preparation incomplete; preserve partial cache")
    base_config = load_config(ROOT / "runs" / lock["baseline"] / "resolved-config.yaml")
    baseline = pilot_stats(read_json(INDEX), base_config, lock)
    rows = pilot_stats(index, config, lock)
    mean_coverage = lambda data: float(
        np.mean([r["window_coverage"] for r in data if r["label"] == 0])
    )
    sum_misses = lambda data, key: sum(r[key] for r in data if r["sequence_id"] in lock["missed"])
    coverage_ratio = mean_coverage(rows) / mean_coverage(baseline)
    old_good = sum_misses(baseline, "visible_good_phase_endpoints")
    new_good = sum_misses(rows, "visible_good_phase_endpoints")
    gate = (
        all(r["sequence_score"] is not None for r in rows)
        and coverage_ratio >= 0.95
        and new_good >= max(old_good + 3, math.ceil(1.10 * old_good))
        and sum_misses(rows, "valid_phase_windows") >= sum_misses(baseline, "valid_phase_windows")
    )
    save(
        "pilot",
        {
            "source_kind": "real_feature_smoke",
            "index": str(index_path),
            "index_sha256": sha256(index_path),
            "baseline": baseline,
            "rows": rows,
            "adl_coverage_ratio": coverage_ratio,
            "missed_fall_good_phase_before": old_good,
            "missed_fall_good_phase_after": new_good,
            "gate_passed": bool(gate),
            "test_scored": False,
            "accuracy_measured": False,
        },
    )
    verify()
    print(
        {
            "pilot_gate": bool(gate),
            "adl_coverage_ratio": coverage_ratio,
            "fall_phase_good": (old_good, new_good),
        },
        flush=True,
    )


def validation_row(name, lock):
    scorer = Scorer(ROOT / "runs" / name)
    rows, windows_all, events_all, phases = [], [], [], []
    for source in scorer.index["split_manifest"]["sequences"]:
        if source["split"] != "validation":
            continue
        sid = source["sequence_id"]
        cache = cache_for(scorer.index, sid)
        out, windows, events = score_sequence(
            cache, scorer.score(cache.features), scorer.threshold["threshold"]
        )
        rows.append({**out, "label": source["label"]})
        windows_all.extend(windows)
        events_all.extend(events)
        if sid in lock["coarse_intervals"]:
            interval = lock["coarse_intervals"][sid]
            lo, hi = interval["fall_phase_ms"]
            good = visible_pose_keys(scorer, sid)
            phases.append(
                {
                    "sequence_id": sid,
                    "prediction": out["prediction"],
                    "phase_event_starts": [
                        e["confirmed_start_ms"]
                        for e in events
                        if lo <= e["confirmed_start_ms"] <= hi
                    ],
                    "early_event_starts": [
                        e["confirmed_start_ms"]
                        for e in events
                        if e["confirmed_start_ms"] < interval["normal_until_ms"]
                    ],
                    "visible_phase_alert_ms": [
                        w["available_at_ms"]
                        for w in windows
                        if w["state"] == "suspected_fall"
                        and lo <= w["available_at_ms"] <= hi
                        and (w["sample_index"], w["track_id"]) in good
                    ],
                }
            )
    row = result(name)
    m = sequence_metrics(rows)
    if any(row["validation"][k] != v for k, v in m.items()):
        raise ContractError("Recomputed metrics differ")
    row.update(
        predictions=rows,
        phases=phases,
        phase_hits=sum(bool(p["phase_event_starts"]) for p in phases),
        early_events=sum(len(p["early_event_starts"]) for p in phases),
        fn=[r["sequence_id"] for r in rows if r["label"] == 1 and r["prediction"] == 0],
        fp=[r["sequence_id"] for r in rows if r["label"] == 0 and r["prediction"] == 1],
    )
    folder = ROOT / "outputs/human-quality-validation" / name
    if folder.exists():
        raise ContractError("Validation export exists")
    folder.mkdir(parents=True)
    write_json(folder / "metrics.json", m)
    write_csv(folder / "sequences.csv", rows)
    write_csv(folder / "windows.csv", windows_all)
    write_csv(folder / "events.csv", events_all)
    row["exports"] = {str(p.relative_to(ROOT)): sha256(p) for p in folder.iterdir()}
    return row


def rank(row):
    m = row["validation"]
    return m["operational_accuracy"], m["sequence_fall_f1"], m["end_to_end_fall_recall"], -m["FP"]


def training():
    lock = verify()
    if path("validation").exists():
        raise ContractError("Completed experiment; preserve results")
    pilot_report = read_json(path("pilot"))
    if sha256(pilot_report["index"]) != pilot_report["index_sha256"]:
        raise ContractError("Pilot changed")
    rows = []
    skipped = []
    for c in [{"run": lock["baseline"], "kind": "baseline"}, *lock["candidates"]]:
        if c["kind"] == "flip_tta" and not pilot_report["gate_passed"]:
            skipped.append(c["run"])
            continue
        stage = "result-" + c["run"]
        if path(stage).exists():
            row = read_json(path(stage))
            if verify_artifacts(ROOT / "runs" / c["run"]) != row["hashes"]:
                raise ContractError("Completed candidate changed")
        else:
            if c["kind"] != "baseline":
                run = ROOT / "runs" / c["run"]
                if run.exists():
                    raise ContractError("Partial run exists; inspect before continuing")
                config = load_config(ROOT / f"configs/{c['run']}.yaml")
                index_path = INDEX
                if c["kind"] == "flip_tta":
                    index_path, _ = prepare(config, reuse_index=pilot_report["index"])
                train(config, index_path, run)
                calibrate(run)
            row = validation_row(c["run"], lock)
            if not rows:
                row["guard_passed"] = True
            else:
                b = rows[0]
                m = row["validation"]
                row["guard_passed"] = (
                    row["phase_hits"] >= b["phase_hits"]
                    and row["early_events"] <= b["early_events"]
                    and m["operational_accuracy"] >= b["validation"]["operational_accuracy"]
                    and m["sequence_fall_f1"] >= b["validation"]["sequence_fall_f1"]
                    and m["n_unknown"] <= b["validation"]["n_unknown"]
                    and all(
                        p["prediction"] != 1 or p["visible_phase_alert_ms"]
                        for p in row["phases"]
                        if p["sequence_id"] in lock["missed"]
                    )
                )
            save(stage, row)
        rows.append(row)
        print(
            {k: row[k] for k in ("run", "threshold", "validation", "guard_passed", "fn", "fp")},
            flush=True,
        )
    selected = max((r for r in rows if r["guard_passed"]), key=rank)
    save(
        "validation",
        {
            "source_kind": "real_end_to_end",
            "rows": rows,
            "pilot_rejected": skipped,
            "selected": selected["run"],
            "target_reached": all(
                selected["validation"][k] >= v for k, v in lock["targets"].items()
            ),
            "test_scored": False,
            "historical_test_seen": True,
            "repeated_validation_not_independent": True,
        },
    )
    verify()


def test():
    verify()
    report = read_json(path("validation"))
    if not report["target_reached"]:
        raise ContractError("Validation targets/guard failed; no Test scoring")
    run = ROOT / "runs" / report["selected"]
    save("before-test", {"run": report["selected"], "hashes": verify_artifacts(run)})
    output, metrics = evaluate(run)
    save("test", {"output": str(output), "metrics": metrics, "historical_test_seen": True})


def status():
    verify()
    if path("pilot").exists():
        p = read_json(path("pilot"))
        print(
            {
                k: p[k]
                for k in (
                    "gate_passed",
                    "adl_coverage_ratio",
                    "missed_fall_good_phase_before",
                    "missed_fall_good_phase_after",
                )
            }
        )
    if path("validation").exists():
        r = read_json(path("validation"))
        for row in r["rows"]:
            if verify_artifacts(ROOT / "runs" / row["run"]) != row["hashes"]:
                raise ContractError("Result changed")
            for name, digest in row["exports"].items():
                if sha256(ROOT / name) != digest:
                    raise ContractError("Export changed")
            print(
                {
                    "run": row["run"],
                    "accuracy": row["validation"]["operational_accuracy"],
                    "f1": row["validation"]["sequence_fall_f1"],
                    "guard": row["guard_passed"],
                    "fn": row["fn"],
                    "fp": row["fp"],
                }
            )
        print({"selected": r["selected"], "target_reached": r["target_reached"]})
    print({"old_runs_preserved": 39, "test_scored": path("test").exists()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["freeze", "pilot", "training", "test", "status"])
    globals()[parser.parse_args().stage]()
