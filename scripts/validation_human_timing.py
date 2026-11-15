"""Validate three self-reported human reviews against saved Validation alerts only."""

import argparse
import base64
import bisect
import hashlib
import json
from pathlib import Path

import validation_alert_quality as prior

from fall_detect.evaluation.human_timing import frame_time_bounds, temporal_relation
from fall_detect.io_utils import ContractError

ROOT = Path(__file__).resolve().parents[1]
TAG = "ur-validation-human-timing-20261105-001"
OUT = ROOT / "outputs" / TAG
QA = ROOT / "docs/validation" / TAG
OLD = ROOT / "outputs/ur-validation-alert-quality-20261013-001"
OLD_QA = ROOT / "docs/validation/ur-validation-alert-quality-20261013-001"
WIDE = ROOT / "outputs/isotropic-refinement-20261012/ur-isotropic-wide-20261012-001"
DEMO = ROOT / "outputs/human-isotropic-validation/ur-sample-human-s1-001"
SELECTED = ("fall-18", "fall-09", "fall-30")
MUTABLE = ("docs/current-progress.md", "docs/acceptance-results.md")
save, read, sha = prior.save, prior.read, prior.digest


def old_inventory():
    result = {
        str((OLD / name).relative_to(ROOT)): digest
        for name, digest in read(OLD / "artifact-manifest.json")["files"].items()
    }
    result.update(read(OLD_QA / "final-qa.json")["artifact_sha256"])
    for path in (
        OLD_QA / "final-qa.json",
        ROOT / "AGENTS.md",
        ROOT / "Fall_Detection_Codex_Implementation_Spec.md",
    ):
        result[str(path.relative_to(ROOT))] = sha(path)
    return result


def check_hashes(files, archived=False):
    for name, digest in files.items():
        normalized = name.replace("\\", "/")
        path = (
            QA / "prior-contracts" / normalized
            if archived and normalized in MUTABLE
            else ROOT / normalized
        )
        if sha(path) != digest:
            raise ContractError(f"Preserved file changed: {name}")


def verify_bundle(bundle):
    claimed = bundle.get("bundle_sha256")
    computed = hashlib.sha256(
        json.dumps(
            {k: v for k, v in bundle.items() if k != "bundle_sha256"},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf8")
    ).hexdigest()
    if claimed != computed or bundle.get("split") != "validation":
        raise ContractError("Original bundle content checksum/split mismatch")
    if [c["sequence_id"] for c in bundle["clips"]] != list(SELECTED):
        raise ContractError("Unexpected original bundle selection/order")
    manifest = read(OLD_QA / "source-manifest.json")
    if not set(SELECTED).issubset(manifest["allowed_sequences"]):
        raise ContractError("Selected clip is outside original Validation")
    metadata, paths, frame_count = {}, [], 0
    for clip in bundle["clips"]:
        sid = clip["sequence_id"]
        candidates = [
            name
            for name in manifest["files"]
            if "poses" in name and name.endswith(".json") and f"{sid}-" in Path(name).name
        ]
        if len(candidates) != 1:
            raise ContractError("Ambiguous original Validation pose metadata")
        path = prior.local_path(candidates[0])
        if sha(path) != manifest["files"][candidates[0]]:
            raise ContractError("Original pose metadata changed")
        paths.append(path)
        meta = read(path)
        for key, expected in {
            "split": "validation",
            "sequence_id": sid,
            "recipe_hash": manifest["baseline_recipe"],
            "split_hash": manifest["split_hash"],
        }.items():
            if meta.get(key) != expected:
                raise ContractError(f"Pose metadata {key} mismatch")
        original = {f["source_frame_id"]: f for f in meta["frames"]}
        if len(original) != len(meta["frames"]):
            raise ContractError("Duplicate original frame identities")
        identity = {
            f["path"].replace("\\", "/"): f["sha256"] for f in meta["source_identity"]["files"]
        }
        ids, previous_time = set(), -1.0
        for frame in clip["frames"]:
            fid, time = frame["source_frame_id"], frame["timestamp_ms"]
            if type(fid) is not int or fid in ids or time <= previous_time or fid not in original:
                raise ContractError("Duplicate/decreasing/unobserved review frame")
            if original[fid]["timestamp_ms"] != time:
                raise ContractError("Review timestamps differ from source metadata")
            if (original[fid]["width"], original[fid]["height"]) != (
                frame["width"],
                frame["height"],
            ):
                raise ContractError("Review dimensions differ from source metadata")
            raw = base64.b64decode(frame["image"].split(",", 1)[1], validate=True)
            image_sha = hashlib.sha256(raw).hexdigest()
            key = f"{sid}-cam0-rgb/{sid}-cam0-rgb-{fid:03}.png"
            if image_sha != frame["source_sha256"] or image_sha != identity.get(key):
                raise ContractError("Review PNG differs from original cached RGB identity")
            if sha(OLD / "review/frames" / f"{sid}-{fid:03}.png") != image_sha:
                raise ContractError("Lightweight review PNG differs from embedded original")
            ids.add(fid)
            previous_time = time
            frame_count += 1
        metadata[sid] = meta
    return (
        metadata,
        paths,
        {
            "bundle_sha256": computed,
            "original_png_checks": frame_count,
            "pose_metadata_files": len(paths),
            "split": "validation",
        },
    )


def formal_metrics():
    rows = prior.csv_read(WIDE / "sequences.csv")
    if len(rows) != 23 or len({r["sequence_id"] for r in rows}) != 23:
        raise ContractError("Expected original 23 Validation sequences")
    allowed = set(read(OLD_QA / "source-manifest.json")["allowed_sequences"])
    if {r["sequence_id"] for r in rows} != allowed:
        raise ContractError("Formal results not exactly the original Validation split")
    for row in rows:
        row["label"] = int(row["label"])
        row["prediction"] = int(row["prediction"]) if row["prediction"] else None
    result = prior.metrics(rows)
    reported = read(WIDE / "metrics.json")
    for key in result:
        if key in reported and result[key] != reported[key]:
            raise ContractError(f"Formal metric mismatch: {key}")
    return rows, result


def build(annotation_path):
    if OUT.exists() or QA.exists():
        raise ContractError("Preserve completed human timing outputs/QA; refuse overwrite")
    annotation_bytes = annotation_path.read_bytes()
    annotation = json.loads(annotation_bytes.decode("utf-8-sig"))
    if annotation.get("draft") is True:
        raise ContractError("Draft is not a completed human review")
    bundle = read(OLD / "review-bundle.json")
    schema_audit = prior.validate_review(annotation, bundle)
    metadata, meta_paths, bundle_audit = verify_bundle(bundle)
    previous = old_inventory()
    check_hashes(previous)
    formal, metrics = formal_metrics()
    QA.mkdir(parents=True)
    OUT.mkdir(parents=True)
    for name in MUTABLE:
        dest = QA / "prior-contracts" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as stream:
            stream.write((ROOT / name).read_bytes())
    save(
        QA / "preservation.json",
        {"files": previous, "old_outputs_count": 231, "previous_final_qa_artifact_count": 29},
    )
    with (OUT / "human-review-original.json").open("xb") as stream:
        stream.write(annotation_bytes)
    save(
        OUT / "annotation-audit.json",
        {
            **schema_audit,
            **bundle_audit,
            "annotation_sha256": hashlib.sha256(annotation_bytes).hexdigest(),
            "semantics": "onset/impact instant uncertainty bounds, not whole posture phases",
            "independent_multi_reviewer_ground_truth": False,
        },
    )
    enriched = prior.csv_read(OLD / "windows-with-quality.csv")
    original_windows = prior.parse_windows(prior.csv_read(WIDE / "windows.csv"))
    original_events = prior.csv_read(WIDE / "events.csv")
    demo = {r["sequence_id"]: r for r in prior.csv_read(DEMO / "sequences.csv")}
    threshold_path = ROOT / "runs/ur-isotropic-wide-20261012-001/threshold.json"
    threshold = read(threshold_path)
    if (
        threshold["threshold"] != 0.5341113805770874
        or threshold["comparison"] != "gt"
        or threshold["consecutive_windows"] != 3
    ):
        raise ContractError("Frozen wide threshold/event rule changed")
    comparisons, active, selected_windows, event_rows, frame_rows = [], [], [], [], []
    for reviewed in annotation["clips"]:
        sid = reviewed["sequence_id"]
        clip = next(c for c in bundle["clips"] if c["sequence_id"] == sid)
        onset = frame_time_bounds(reviewed["onset"], clip["frames"])
        impact = frame_time_bounds(reviewed["impact"], clip["frames"])
        source = next(r for r in formal if r["sequence_id"] == sid)
        formal_events = [e for e in original_events if e["sequence_id"] == sid]
        saved_windows = [w for w in original_windows if w["sequence_id"] == sid]
        raw_score, replay = prior.audit_events(
            saved_windows, threshold["threshold"], metadata[sid]["frames"][-1]["timestamp_ms"]
        )
        if raw_score != float(source["sequence_score"]) or len(replay) != len(formal_events):
            raise ContractError("Saved K3/sequence score changed")
        for a, b in zip(replay, formal_events, strict=True):
            for key, value in a.items():
                expected = (
                    int(b[key])
                    if key == "track_id"
                    else prior.number(b[key])
                    if key.endswith("_ms") or key == "peak_score"
                    else b[key]
                )
                if value != expected:
                    raise ContractError(
                        "Original event differs from independent saved-score replay"
                    )
        first = prior.number(source["first_alarm_ms"])
        relation = temporal_relation(first, onset, impact)
        selected = [w for w in enriched if w["sequence_id"] == sid]
        all_times = [f["timestamp_ms"] for f in metadata[sid]["frames"]]
        original_by_id = {f["source_frame_id"]: f for f in metadata[sid]["frames"]}
        by_identity = {(w["sample_index"], w["track_id"]): w for w in saved_windows}
        selected_active = []
        for w in selected:
            time = float(w["available_at_ms"])
            identity = (int(w["sample_index"]), int(w["track_id"]) if w["track_id"] else None)
            original = by_identity[identity]
            for key in ("start_ms", "end_ms", "available_at_ms", "raw_score", "event_score"):
                if prior.number(w[key]) != original[key]:
                    raise ContractError("Enriched window differs from original formal CSV")
            if w["state"] != original["state"]:
                raise ContractError("Saved active state differs from formal CSV")
            diagnosis = temporal_relation(time, onset, impact)
            selected_windows.append({**w, **{f"human_{k}": v for k, v in diagnosis.items()}})
            if w["state"] != "suspected_fall":
                continue
            fid = int(w["source_frame_id_at_tick"])
            nearest = bisect.bisect_right(all_times, time + 1e-7) - 1
            if (
                metadata[sid]["frames"][nearest]["source_frame_id"] != fid
                or original_by_id[fid]["timestamp_ms"] > time + 1e-7
            ):
                raise ContractError("Alert source-frame mapping is not past-only")
            r = {
                "sequence_id": sid,
                "track_id": identity[1],
                "sample_index": identity[0],
                "available_at_ms": time,
                "causal_source_frame_id": fid,
                "causal_source_timestamp_ms": original_by_id[fid]["timestamp_ms"],
                "raw_score": float(w["raw_score"]),
                "event_score": float(w["event_score"]),
                "current_same_track_confidence_good": w["current_same_track_confidence_good"]
                == "True",
                **diagnosis,
            }
            selected_active.append(r)
            active.append(r)
        if (min((w["available_at_ms"] for w in selected_active), default=None)) != first:
            raise ContractError("First alarm does not equal first active endpoint")
        for e in replay:
            starts = temporal_relation(e["confirmed_start_ms"], onset, impact)
            event_rows.append({**e, **{f"human_start_{k}": v for k, v in starts.items()}})
        for f in clip["frames"]:
            frame_rows.append(
                {
                    "sequence_id": sid,
                    "split": "validation",
                    "source_frame_id": f["source_frame_id"],
                    "timestamp_ms": f["timestamp_ms"],
                    "source_sha256": f["source_sha256"],
                }
            )
        demo_first = prior.number(demo[sid]["first_alarm_ms"])
        comparisons.append(
            {
                "sequence_id": sid,
                "split": "validation",
                "wide_prediction": source["prediction"],
                "wide_video_outcome": "TP"
                if source["label"] == source["prediction"] == 1
                else "FN",
                "onset_bounds_frame_ids": reviewed["onset"]["bounds_frame_ids"],
                "impact_bounds_frame_ids": reviewed["impact"]["bounds_frame_ids"],
                "onset_bounds_ms": onset,
                "impact_bounds_ms": impact,
                "first_alarm_ms": first,
                **relation,
                "events": len(replay),
                "last_active_ms": max((e["last_active_ms"] for e in replay), default=None),
                "event_end_ms": max((e["end_ms"] for e in replay), default=None),
                "active_endpoint_count": len(selected_active),
                "good_active_endpoint_count": sum(
                    w["current_same_track_confidence_good"] for w in selected_active
                ),
                "possible_fall_active_endpoints": sum(
                    w["inside_possible_fall_envelope"] is True for w in selected_active
                ),
                "guaranteed_fall_active_endpoints": sum(
                    w["inside_guaranteed_fall_interval"] is True for w in selected_active
                ),
                "after_latest_impact_active_endpoints": sum(
                    w["definitely_after_impact"] is True for w in selected_active
                ),
                "possible_fall_good_active_endpoints": sum(
                    w["inside_possible_fall_envelope"] is True
                    and w["current_same_track_confidence_good"]
                    for w in selected_active
                ),
                "human_skeleton_quality": reviewed["skeleton_quality"],
                "human_notes": reviewed.get("notes", ""),
                "demo_prediction": int(demo[sid]["prediction"])
                if demo[sid]["prediction"]
                else None,
                "demo_first_alarm_ms": demo_first,
                "demo_human_first_alarm_relation": temporal_relation(demo_first, onset, impact)[
                    "relation"
                ],
            }
        )
    known = [
        r
        for r in comparisons
        if r["onset_bounds_ms"] is not None and r["impact_bounds_ms"] is not None
    ]
    summary = {
        "id": TAG,
        "reference_run": "ur-isotropic-wide-20261012-001",
        "human_status": "three_self_reported_reviews_validated",
        "reviewed_fall_videos": 3,
        "full_validation_fall_videos": 15,
        "full_validation_videos": 23,
        "fall_videos_without_this_onset_impact_review": 12,
        "timing_evaluable_reviewed_videos": len(known),
        "wide_selected_video_TP": sum(r["wide_video_outcome"] == "TP" for r in comparisons),
        "first_alarm_possible_envelope_hits": sum(
            r["inside_possible_fall_envelope"] is True for r in comparisons
        ),
        "first_alarm_guaranteed_interval_hits": sum(
            r["inside_guaranteed_fall_interval"] is True for r in comparisons
        ),
        "first_alarm_inside_impact_uncertainty": sum(
            r["inside_impact_uncertainty"] is True for r in comparisons
        ),
        "first_alarm_definitely_before_impact": sum(
            r["definitely_before_impact"] is True for r in comparisons
        ),
        "first_alarm_definitely_after_impact": sum(
            r["definitely_after_impact"] is True for r in comparisons
        ),
        "possible_envelope_good_active_videos": sum(
            r["possible_fall_good_active_endpoints"] > 0 for r in comparisons
        ),
        "selected_active_endpoints": len(active),
        "selected_after_latest_impact_active_endpoints": sum(
            r["definitely_after_impact"] is True for r in active
        ),
        "formal_validation_metrics_unchanged": metrics,
        "new_model_inference": 0,
        "new_training": 0,
        "new_calibration": 0,
        "test_artifact_access_evaluation": 0,
        "exact_time_truth_invented": False,
        "adoption": "unchanged",
        "timing_accuracy_improvement_claimed": False,
        "limitation": "Selected post-hoc diagnostic subset; human instant bounds are not exact or multi-reviewer ground truth",
    }
    prior.csv_save(OUT / "comparisons.csv", comparisons)
    prior.csv_save(OUT / "active-endpoints.csv", active)
    prior.csv_save(OUT / "events-with-human-time.csv", event_rows)
    prior.csv_save(OUT / "selected-windows-with-human-time.csv", selected_windows)
    prior.csv_save(OUT / "source-frame-times.csv", frame_rows)
    save(OUT / "comparisons.json", comparisons)
    save(OUT / "summary.json", summary)
    overview, reviewed_by_id = (
        prior.csv_read(OLD / "validation-23.csv"),
        {r["sequence_id"]: r for r in comparisons},
    )
    for row in overview:
        r = reviewed_by_id.get(row["sequence_id"])
        row.update(
            human_onset_impact_review="self_reported_validated"
            if r
            else "not_applicable_ADL"
            if row["label"] == "0"
            else "not_reviewed_here",
            human_onset_bounds_ms=r["onset_bounds_ms"] if r else None,
            human_impact_bounds_ms=r["impact_bounds_ms"] if r else None,
            human_first_alarm_relation=r["relation"] if r else None,
            human_possible_envelope_hit=r["inside_possible_fall_envelope"] if r else None,
            human_guaranteed_interval_hit=r["inside_guaranteed_fall_interval"] if r else None,
        )
    prior.csv_save(OUT / "validation-23-human-supplement.csv", overview)
    paths = [
        *meta_paths,
        WIDE / "sequences.csv",
        WIDE / "windows.csv",
        WIDE / "events.csv",
        WIDE / "metrics.json",
        DEMO / "sequences.csv",
        threshold_path,
        ROOT / "docs/ur-validation-human-timing-20261105-plan.md",
        ROOT / "scripts/validation_human_timing.py",
        ROOT / "src/fall_detect/evaluation/human_timing.py",
        ROOT / "tests/test_validation_human_timing.py",
        ROOT / "scripts/validation_alert_quality.py",
    ]
    save(
        QA / "inputs.json",
        {
            "files": {str(p.relative_to(ROOT)): sha(p) for p in paths},
            "annotation_external_path": str(annotation_path),
            "annotation_sha256": hashlib.sha256(annotation_bytes).hexdigest(),
            "model_feature_arrays_loaded": False,
            "test_access": False,
        },
    )
    print(json.dumps({"completed": True, "summary": summary}, ensure_ascii=False))


def verify():
    check_hashes(read(QA / "preservation.json")["files"], archived=True)
    check_hashes(read(QA / "inputs.json")["files"])
    if sha(OUT / "human-review-original.json") != read(QA / "inputs.json")["annotation_sha256"]:
        raise ContractError("Preserved uploaded annotation changed")
    if (OUT / "artifact-manifest.json").exists():
        for name, digest in read(OUT / "artifact-manifest.json")["files"].items():
            if sha(OUT / name) != digest:
                raise ContractError(f"New output changed: {name}")
    print(
        json.dumps(
            {
                "verified": True,
                "old_hashes": len(read(QA / "preservation.json")["files"]),
                "summary": read(OUT / "summary.json"),
            },
            ensure_ascii=False,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("build", "verify"))
    parser.add_argument("--annotation", type=Path)
    args = parser.parse_args()
    if args.stage == "build":
        if args.annotation is None:
            parser.error("--annotation is required")
        build(args.annotation)
    else:
        verify()


if __name__ == "__main__":
    main()
