"""Two preregistered Train-residual weighting profiles; no Test without Validation gate."""

import argparse
from datetime import UTC, datetime

import yaml
from ae_generalization import result
from verify_action_display import verify_preserved

from fall_detect.config import load_config, project_path
from fall_detect.data.cache import load_pose_cache
from fall_detect.detection.events import score_sequence
from fall_detect.evaluation.metrics import sequence_metrics
from fall_detect.io_utils import ContractError, read_json, sha256, write_csv, write_json
from fall_detect.recipes import recipe_spec
from fall_detect.skeleton.preprocessing import good_frame
from fall_detect.skeleton.windows import CausalResampler
from fall_detect.training.experiment import Scorer, cache_for, calibrate, evaluate, verify_artifacts
from fall_detect.training.residual import derive_residual_run

ROOT = project_path(".")
FILES = [
    "src/fall_detect/config.py",
    "src/fall_detect/models/autoencoder.py",
    "src/fall_detect/training/experiment.py",
    "src/fall_detect/training/residual.py",
    "src/fall_detect/visualization/loss_viewer.py",
    "scripts/human_residual.py",
    "scripts/ae_generalization.py",
    "scripts/verify_action_display.py",
    "tests/test_residual_profiles.py",
]


def path(stage):
    return ROOT / f"docs/validation/human-residual-{stage}.json"


def save(stage, data):
    if path(stage).exists():
        raise ContractError(f"Preserve existing evidence: {stage}")
    write_json(path(stage), data)


def verify():
    lock = read_json(path("lock"))
    before = read_json(path("before"))
    for stage in ("before", "diagnosis"):
        if sha256(path(stage)) != lock[stage + "_sha256"]:
            raise ContractError(f"Preregistered evidence changed: {stage}")
    verify_preserved()
    for name, digest in before["source_files"].items():
        if sha256(ROOT / "docs/validation/human-residual-legacy-source" / name) != digest:
            raise ContractError("Archived scoring source changed")
    if path("implementation").exists():
        frozen = read_json(path("implementation"))
        if sha256(path("lock")) != frozen["lock_sha256"]:
            raise ContractError("Selection protocol changed")
        for name, digest in frozen["files"].items():
            if sha256(ROOT / name) != digest:
                raise ContractError(f"Frozen implementation changed: {name}")
    return lock


def freeze():
    lock = verify()
    if path("implementation").exists():
        raise ContractError("Implementation already frozen")
    files = list(FILES)
    for c in lock["candidates"]:
        filename = ROOT / f"configs/{c['run']}.yaml"
        if filename.exists() or (ROOT / "runs" / c["run"]).exists():
            raise ContractError("Candidate already exists")
        config = load_config(ROOT / "runs" / lock["baseline"] / "resolved-config.yaml")
        config["anomaly_score"]["mode"] = c["mode"]
        filename.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf8")
        files.append(str(filename.relative_to(ROOT)))
    save(
        "implementation",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "lock_sha256": sha256(path("lock")),
            "files": {name: sha256(ROOT / name) for name in files},
            "candidate_validation_not_yet_run": True,
        },
    )
    print("Implementation frozen before candidate calibration", flush=True)


def visible_pose_keys(scorer, sid):
    config = scorer.config
    observations, _ = load_pose_cache(
        project_path(scorer.index["sequences"][sid]["pose_path"]),
        {
            "sequence_id": sid,
            "recipe_hash": scorer.index["recipe_hash"],
            "source_hash": scorer.index["sequences"][sid]["source_hash"],
        },
    )
    spec = recipe_spec(config)
    resampler = CausalResampler(
        config["skeleton"]["target_fps"],
        config["skeleton"]["resampling_hold_policy"],
        config["skeleton"]["timestamp_tolerance_ms"],
    )
    return {
        (sample.sample_index, pose.track_id)
        for frame, poses in observations
        for sample in resampler.push(frame, poses)
        for pose in sample.poses
        if good_frame(
            pose.keypoints, config["quality"], spec["joints"], spec.get("derived_joints", ())
        )
    }


def validation_row(run, lock):
    scorer = Scorer(ROOT / "runs" / run)
    rows, all_windows, all_events, phases = [], [], [], []
    for source in scorer.index["split_manifest"]["sequences"]:
        if source["split"] != "validation":
            continue
        sid = source["sequence_id"]
        cache = cache_for(scorer.index, sid)
        out, windows, events = score_sequence(
            cache, scorer.score(cache.features), scorer.threshold["threshold"]
        )
        rows.append({**out, "label": source["label"]})
        all_windows.extend(windows)
        all_events.extend(events)
        if sid in lock["coarse_intervals"]:
            phase = lock["coarse_intervals"][sid]
            lo, hi = phase["fall_phase_ms"]
            visible = visible_pose_keys(scorer, sid)
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
                        if e["confirmed_start_ms"] < phase["normal_until_ms"]
                    ],
                    "visible_phase_alert_ms": [
                        w["available_at_ms"]
                        for w in windows
                        if w["state"] == "suspected_fall"
                        and lo <= w["available_at_ms"] <= hi
                        and (w["sample_index"], w["track_id"]) in visible
                    ],
                }
            )
    row = result(run)
    metrics = sequence_metrics(rows)
    for key, value in metrics.items():
        if row["validation"][key] != value:
            raise ContractError(f"Recomputed Validation mismatch: {key}")
    row.update(
        predictions=rows,
        phases=phases,
        phase_hits=sum(bool(p["phase_event_starts"]) for p in phases),
        early_events=sum(len(p["early_event_starts"]) for p in phases),
        visible_phase_hits=sum(bool(p["visible_phase_alert_ms"]) for p in phases),
        fn=[r["sequence_id"] for r in rows if r["label"] == 1 and r["prediction"] == 0],
        fp=[r["sequence_id"] for r in rows if r["label"] == 0 and r["prediction"] == 1],
    )
    folder = ROOT / "outputs/human-residual-validation" / run
    if folder.exists():
        raise ContractError("Preserve existing Validation exports")
    folder.mkdir(parents=True)
    write_json(folder / "metrics.json", metrics)
    write_csv(folder / "sequences.csv", rows)
    write_csv(folder / "windows.csv", all_windows)
    write_csv(folder / "events.csv", all_events)
    row["exports"] = {str(p.relative_to(ROOT)): sha256(p) for p in folder.iterdir()}
    return row


def rank(row):
    m = row["validation"]
    return m["operational_accuracy"], m["sequence_fall_f1"], m["end_to_end_fall_recall"], -m["FP"]


def run():
    lock = verify()
    if not path("implementation").exists() or path("validation").exists():
        raise ContractError("Freeze implementation first; do not overwrite completed results")
    rows = []
    for c in [{"run": lock["baseline"]}, *lock["candidates"]]:
        stage = "result-" + c["run"]
        if path(stage).exists():
            row = read_json(path(stage))
            if verify_artifacts(ROOT / "runs" / c["run"]) != row["hashes"]:
                raise ContractError("Completed candidate changed")
        else:
            if "mode" in c:
                target = ROOT / "runs" / c["run"]
                derive_residual_run(ROOT / "runs" / lock["baseline"], target, c["mode"])
                if load_config(target / "resolved-config.yaml") != load_config(
                    ROOT / f"configs/{c['run']}.yaml"
                ):
                    raise ContractError("Candidate config differs from preregistration")
                calibrate(target)
            row = validation_row(c["run"], lock)
            if not rows:
                row["guard_passed"] = True
            else:
                b, m = rows[0], row["validation"]
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
            "selected": selected["run"],
            "raw_metric_winner": max(rows, key=rank)["run"],
            "target_reached": all(
                selected["validation"][k] >= v for k, v in lock["targets"].items()
            ),
            "test_scored": False,
            "historical_test_seen": True,
            "repeated_validation_not_independent": True,
            "new_ae_epochs": 0,
            "coarse_timing_is_not_ground_truth_accuracy": True,
        },
    )
    verify()


def test():
    verify()
    report = read_json(path("validation"))
    if not report["target_reached"]:
        raise ContractError("Validation targets/guard failed; no Test scoring")
    selected = ROOT / "runs" / report["selected"]
    save("before-test", {"run": report["selected"], "hashes": verify_artifacts(selected)})
    output, metrics = evaluate(selected)
    save("test", {"output": str(output), "metrics": metrics, "historical_test_seen": True})


def status():
    verify()
    if path("validation").exists():
        report = read_json(path("validation"))
        for r in report["rows"]:
            if verify_artifacts(ROOT / "runs" / r["run"]) != r["hashes"]:
                raise ContractError("Candidate artifacts changed")
            for filename, digest in r["exports"].items():
                if sha256(ROOT / filename) != digest:
                    raise ContractError("Validation export changed")
            print(
                {
                    "run": r["run"],
                    "accuracy": r["validation"]["operational_accuracy"],
                    "f1": r["validation"]["sequence_fall_f1"],
                    "FN": r["fn"],
                    "FP": r["fp"],
                    "guard_passed": r["guard_passed"],
                }
            )
        print(
            {
                "selected": report["selected"],
                "target_reached": report["target_reached"],
                "test_scored": path("test").exists(),
            }
        )
    print("Old runs, inputs and locked scoring implementation verified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["freeze", "run", "test", "status"])
    globals()[parser.parse_args().stage]()
