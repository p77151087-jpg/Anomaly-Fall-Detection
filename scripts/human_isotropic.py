"""One preregistered aspect-preserving normalization experiment; no Test tuning."""

import argparse
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
from fall_detect.training.experiment import Scorer, cache_for, calibrate, train, verify_artifacts

ROOT = project_path(".")
BASE = "ur-sample-human-s1-001"
RUN = "ur-human-isotropic-001"
INDEX = ROOT / f"data/cache/{BASE}/dataset-index.json"
PILOT = ["adl-11", "adl-31", "adl-35", "adl-37", "fall-19", "fall-18", "fall-30", "fall-09"]
FILES = [
    "src/fall_detect/config.py",
    "src/fall_detect/skeleton/preprocessing.py",
    "src/fall_detect/data/pose_reuse.py",
    "scripts/human_isotropic.py",
    "scripts/human_keyframes.py",
    "tests/test_human_isotropic.py",
]


def path(stage):
    return ROOT / f"docs/validation/human-isotropic-{stage}.json"


def save(stage, value):
    if path(stage).exists():
        raise ContractError(f"Preserve completed evidence: {stage}")
    write_json(path(stage), value)


def lock():
    if path("lock").exists():
        raise ContractError("Experiment already locked")
    before = read_json(path("before"))
    for f, digest in before["source_files"].items():
        if sha256(ROOT / "docs/validation/human-isotropic-legacy-source" / f) != digest:
            raise ContractError("Archived source differs")
    config = load_config(ROOT / f"runs/{BASE}/resolved-config.yaml")
    config["skeleton"]["coordinate_normalization"] = "per_frame_isotropic"
    config["data"]["cache_dir"] = f"data/cache/{RUN}"
    filename = ROOT / f"configs/{RUN}.yaml"
    if filename.exists() or (ROOT / "runs" / RUN).exists():
        raise ContractError("Candidate exists")
    filename.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf8")
    load_config(filename)
    old = read_json(ROOT / "docs/validation/human-quality-lock.json")
    save(
        "lock",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "baseline": BASE,
            "candidate": RUN,
            "before_sha256": sha256(path("before")),
            "index_sha256": sha256(INDEX),
            "diagnosis_sha256": sha256(ROOT / "docs/validation/human-keyframes-diagnosis.json"),
            "annotations_sha256": sha256(ROOT / "docs/validation/human-keyframes-annotations.json"),
            "files": {f: sha256(ROOT / f) for f in FILES + [str(filename.relative_to(ROOT))]},
            "hypothesis": "Independent per-frame X/Y scaling discards pixel bounding-box aspect. Center the observed per-frame pixel bbox and divide both XY by its largest pixel span, to retain aspect before quality is lost. Recognition benefit is unproven; pretrained distribution changes.",
            "fixed": "One candidate, original30/stride1, same observed joints/confidence/tracks, no interpolation or hidden-joint fabrication, same .2/9/.8 quality, same K3. Frozen genuine512D ST-GCN plus Train ADL scaler/AE128/32, seed42 sequence sampling wd.001 noise0 max100/patience10. NormalVal checkpoint and balanced sequence Accuracy threshold on Val only.",
            "pilot_ids": PILOT,
            "pilot_gate": "All eight finite genuine512D features change measurably, exact same pose arrays and window valid/invalid schedules, each retains legal K3. Diagnostic only, no Accuracy selection.",
            "coarse_intervals": old["coarse_intervals"],
            "missed": old["missed"],
            "selection": "Candidate only if ordinary/all Accuracy strictly improves, F1 not lower, unknown not higher, coarse phase-hit count not lower, no extra coarse early event, and each newly recovered old FN has a CURRENT good-pose visible AE alert inside old prelocked phase. Ties retain baseline. New visual onset brackets are separate diagnostics and do not replace old scoring intervals.",
            "stop": "Exactly one AE and normalization candidate; stop after result. No second variant, confidence setting, seed or threshold objective. No new Test in this diagnostic experiment; freeze selection before any later Test request. Formal Notebook/film remain held until quality demonstrated.",
            "targets": old["targets"],
            "historical_test_seen": True,
            "repeated_validation_not_independent": True,
        },
    )
    print({"locked": RUN, "old_runs": len(before["old_runs"])})


def verify(deep=False):
    record = read_json(path("lock"))
    for name, expected in [
        (path("before"), record["before_sha256"]),
        (INDEX, record["index_sha256"]),
        (ROOT / "docs/validation/human-keyframes-diagnosis.json", record["diagnosis_sha256"]),
        (ROOT / "docs/validation/human-keyframes-annotations.json", record["annotations_sha256"]),
    ]:
        if sha256(name) != expected:
            raise ContractError(f"Frozen evidence changed: {name}")
    for name, digest in record["files"].items():
        if sha256(ROOT / name) != digest:
            raise ContractError(f"Locked source/config changed: {name}")
    before = read_json(path("before"))
    for name, hashes in before["old_runs"].items():
        if verify_artifacts(ROOT / "runs" / name) != hashes:
            raise ContractError(f"Old run changed: {name}")
    for group in ["configs", "manifests"] + (["cache_files"] if deep else []):
        for name, digest in before[group].items():
            if sha256(ROOT / name) != digest:
                raise ContractError(f"Old input changed: {name}")
    baseline = load_config(ROOT / f"runs/{BASE}/resolved-config.yaml")
    index = read_json(INDEX)
    if (
        recipe_fingerprint(baseline, {k: v["sha256"] for k, v in index["weights"].items()})
        != index["recipe_hash"]
    ):
        raise ContractError("Default recipe fingerprint changed")
    return record


def pilot():
    verify()
    if path("pilot").exists():
        raise ContractError("Pilot already completed")
    index_path, index = prepare(
        load_config(ROOT / f"configs/{RUN}.yaml"), selected_ids=PILOT, reuse_poses_index=INDEX
    )
    baseline = read_json(INDEX)
    rows = []
    for sid in PILOT:
        old, new = cache_for(baseline, sid), cache_for(index, sid)
        signature = lambda c: [
            (r.sample_index, r.track_id, r.quality, r.row_index) for r in c.records
        ]
        same = signature(old) == signature(new)
        pose_same = sha256(project_path(baseline["sequences"][sid]["pose_path"])) == sha256(
            project_path(index["sequences"][sid]["pose_path"])
        )
        out, _, _ = score_sequence(new, np.zeros(len(new.features)), 1.0)
        difference = float(np.max(np.abs(old.features - new.features))) if same else None
        rows.append(
            {
                "sequence_id": sid,
                "records_identical": same,
                "poses_identical": pose_same,
                "valid_features": len(new.features),
                "legal_k3": out["sequence_score"] is not None,
                "max_feature_difference": difference,
            }
        )
    gate = all(
        r["records_identical"]
        and r["poses_identical"]
        and r["legal_k3"]
        and r["max_feature_difference"] > 1e-6
        for r in rows
    )
    save(
        "pilot",
        {
            "evidence": "real_feature_smoke",
            "gate_passed": gate,
            "rows": rows,
            "index": str(index_path),
            "index_sha256": sha256(index_path),
            "test_scored": False,
            "lock_sha256": sha256(path("lock")),
            "model_reports": index["model_load_reports"],
        },
    )
    print({"pilot_passed": gate, "rows": rows}, flush=True)


def validation_row(name, record):
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
        if sid in record["coarse_intervals"]:
            interval = record["coarse_intervals"][sid]
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
    metrics = sequence_metrics(rows)
    if any(row["validation"][k] != v for k, v in metrics.items()):
        raise ContractError("Validation recompute mismatch")
    folder = ROOT / "outputs/human-isotropic-validation" / name
    if folder.exists():
        raise ContractError("Preserve Validation exports")
    folder.mkdir(parents=True)
    write_json(folder / "metrics.json", metrics)
    for filename, values in [("sequences", rows), ("windows", windows_all), ("events", events_all)]:
        write_csv(folder / f"{filename}.csv", values)
    row.update(
        predictions=rows,
        phases=phases,
        fn=[r["sequence_id"] for r in rows if r["label"] == 1 and r["prediction"] == 0],
        fp=[r["sequence_id"] for r in rows if r["label"] == 0 and r["prediction"] == 1],
        phase_hits=sum(bool(p["phase_event_starts"]) for p in phases),
        early_events=sum(len(p["early_event_starts"]) for p in phases),
        exports={str(p.relative_to(ROOT)): sha256(p) for p in folder.iterdir()},
    )
    return row


def training():
    record = verify()
    if path("validation").exists() or (ROOT / "runs" / RUN).exists():
        raise ContractError("Preserve completed/partial candidate")
    p = read_json(path("pilot"))
    if not p["gate_passed"] or sha256(p["index"]) != p["index_sha256"]:
        raise ContractError("Pilot failed or changed")
    config = load_config(ROOT / f"configs/{RUN}.yaml")
    index_path, index = prepare(config, reuse_poses_index=INDEX)
    if not all(r["status"] == "prepared" for r in index["sequences"].values()):
        raise ContractError("Preparation incomplete")
    train(config, index_path, ROOT / "runs" / RUN)
    calibrate(ROOT / "runs" / RUN)
    b = validation_row(BASE, record)
    c = validation_row(RUN, record)
    bm, cm = b["validation"], c["validation"]
    guard = (
        cm["operational_accuracy"] > bm["operational_accuracy"]
        and cm["sequence_fall_f1"] >= bm["sequence_fall_f1"]
        and cm["n_unknown"] <= bm["n_unknown"]
        and c["phase_hits"] >= b["phase_hits"]
        and c["early_events"] <= b["early_events"]
        and all(
            p["prediction"] != 1 or p["visible_phase_alert_ms"]
            for p in c["phases"]
            if p["sequence_id"] in record["missed"]
        )
    )
    selected = c if guard else b
    save(
        "validation",
        {
            "evidence": "real_end_to_end",
            "rows": [b, c],
            "candidate_guard_passed": bool(guard),
            "selected": selected["run"],
            "target_reached": all(
                selected["validation"][k] >= v for k, v in record["targets"].items()
            ),
            "test_scored": False,
            "historical_test_seen": True,
            "repeated_validation_not_independent": True,
        },
    )
    print(
        {"selected": selected["run"], "candidate_validation": cm, "guard": bool(guard)}, flush=True
    )


def status():
    verify()
    if path("validation").exists():
        v = read_json(path("validation"))
        print(
            {
                "selected": v["selected"],
                "target_reached": v["target_reached"],
                "rows": [
                    {"run": r["run"], "metrics": r["validation"], "fn": r["fn"], "fp": r["fp"]}
                    for r in v["rows"]
                ],
            }
        )
    else:
        print({"locked": True, "pilot_complete": path("pilot").exists()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["lock", "pilot", "training", "status", "verify"])
    parser.add_argument("--deep", action="store_true")
    args = parser.parse_args()
    if args.stage == "verify":
        verify(args.deep)
        print({"preservation_verified": True, "deep": args.deep})
    else:
        globals()[args.stage]()
